"""A bake that outlived its drain is published, not baked again.

The serve leg is a long bake followed by a short publish, and only the publish
runs in the drain process. Killed in between, the row used to say the bake was
interrupted — false, and expensive, since re-running re-bakes everything. A
marker records the finished bake, and these pin what reads it: the recovery
publish, the failures that must not claim the bake died, and the orderings that
keep a second kill from publishing the wrong archive.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from autogeoref import drain_lock
from autogeoref.queue import publish as qpublish
from autogeoref.queue import run as qrun
from autogeoref.queue import store as qstore
from autogeoref.queue.command import DrainContext
from autogeoref.viewer import publish as viewer_publish
from queue_support import CITY, _FakeProc, _publication, _spy, _volume

pytestmark = pytest.mark.usefixtures("_publish_without_viewer_io")


def _owed(work: Path, volume: str) -> None:
    """The state a drain killed after its bake but before its publish leaves.

    ``persist``, not ``save_queue``: this is called more than once per test and
    ``save_queue`` writes the whole file, so the second call would erase the first.
    """
    entry = qstore.add(work, volume, "serve")
    entry.status = "running"
    entry.started = time.time()
    qstore.persist(work, [entry])
    viewer_publish.record_publish_owed(
        volume, work, baked_at=entry.started, log="work/queue/logs/x.serve.log"
    )


def test_a_dead_drains_completed_bake_is_published_not_re_baked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: seconds of publishing, not hours of GDAL."""
    _volume(tmp_path, "vol_a", results=True)
    _owed(tmp_path, "vol_a")
    published: list[str] = []
    monkeypatch.setattr(viewer_publish, "publish_volume", lambda vol, _cfg: published.append(vol))

    touched = qrun.run_queue(
        DrainContext(work=tmp_path, city=CITY, publication=_publication(tmp_path)), track="serve"
    )

    assert published == ["vol_a"], "the owed publish must be the recovery"
    recovered = qstore.load_queue(tmp_path)[0]
    assert recovered.status == "done" and recovered.exit_code == 0
    assert "published on recovery" in (recovered.note or "")
    assert [e.volume for e in touched] == ["vol_a"], "a recovery is work, not 'nothing to do'"


def test_each_recovered_publish_is_persisted_before_the_next_one_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-entry, because the publish has already cleared the marker.

    A kill after publishing but before the write leaves a published archive behind
    a `running` row with no evidence left — the original incident, re-created by
    its own recovery. Persisting after the whole batch would lose every earlier
    entry when a later one dies.
    """
    _volume(tmp_path, "vol_a", results=True)
    _volume(tmp_path, "vol_b", results=True)
    _owed(tmp_path, "vol_a")
    _owed(tmp_path, "vol_b")

    def _publish_then_die(vol: str, _cfg: object) -> None:
        if vol == "vol_b":
            raise KeyboardInterrupt("the host slept again")

    monkeypatch.setattr(viewer_publish, "publish_volume", _publish_then_die)

    with pytest.raises(KeyboardInterrupt):
        qrun.run_queue(
            DrainContext(work=tmp_path, city=CITY, publication=_publication(tmp_path)),
            track="serve",
        )

    by_volume = {e.volume: e for e in qstore.load_queue(tmp_path)}
    assert by_volume["vol_a"].status == "done", "the publish that landed must be on disk"
    assert by_volume["vol_b"].status == "running", "the one that died is the next drain's"


def test_a_broken_publication_config_fails_the_row_instead_of_wedging_the_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """publish_volume reaches config loading and lock creation OUTSIDE its own guard.

    Letting one of those escape would abort run_queue inside the drain lock, before
    fail_interrupted — so the marker and the `running` row survive, every later
    drain dies at the same line, and nothing on the queue ever drains again.
    """
    _volume(tmp_path, "vol_a", results=True)
    _owed(tmp_path, "vol_a")
    _volume(tmp_path, "vol_b", results=True)
    qstore.add(tmp_path, "vol_b", "serve")

    def _config_error(vol: str, _cfg: object) -> None:
        if vol == "vol_a":  # the RECOVERY publish is the one that explodes
            raise OSError("deploy/tiles is read-only")

    monkeypatch.setattr(viewer_publish, "publish_volume", _config_error)
    spawned = _spy(monkeypatch)

    qrun.run_queue(
        DrainContext(work=tmp_path, city=CITY, publication=_publication(tmp_path)), track="serve"
    )

    by_volume = {e.volume: e for e in qstore.load_queue(tmp_path)}
    assert by_volume["vol_a"].status == "failed"
    assert "recovery publish failed" in (by_volume["vol_a"].note or "")
    assert by_volume["vol_b"].status == "done", "the rest of the queue must still drain"
    assert spawned, "and its bake must actually have run"


def test_a_hand_publish_closes_the_row_it_recovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented recovery must not leave the row for a later drain to mislabel."""
    _volume(tmp_path, "vol_a", results=True)
    _owed(tmp_path, "vol_a")

    settled = qpublish.settle_published(tmp_path, "vol_a")

    assert settled is not None and settled.status == "done"
    assert qstore.load_queue(tmp_path)[0].status == "done"
    # and the next drain has nothing left to call interrupted
    assert qstore.fail_interrupted(tmp_path, "serve") == []


def test_a_hand_publish_leaves_a_live_drains_row_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `running` row under a LIVE drain is a leg in flight, and the drain owns it."""
    _volume(tmp_path, "vol_a", results=True)
    _owed(tmp_path, "vol_a")
    monkeypatch.setattr(drain_lock, "live_drain", lambda _w, _t: 4321)

    assert qpublish.settle_published(tmp_path, "vol_a") is None
    assert qstore.load_queue(tmp_path)[0].status == "running"


def test_the_stale_marker_is_gone_before_the_row_says_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ORDER, not just occurrence: no kill may expose `running` beside a stale claim.

    Cleared after the persist instead, a kill in that one-unlink window leaves the
    exact pairing finish_owed_publishes acts on — and it would publish the PREVIOUS
    bake's archive as though the new one had finished.
    """
    _volume(tmp_path, "vol_a", results=True)
    viewer_publish.record_publish_owed("vol_a", tmp_path, baked_at=1.0)
    qstore.add(tmp_path, "vol_a", "serve")
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _FakeProc(2))
    real_persist = qstore.persist
    owed_when_persisted: list[bool] = []

    def _spy_persist(work: Path, mine: list[qstore.QueueEntry]) -> None:
        if any(e.status == "running" for e in mine):
            owed_when_persisted.append(viewer_publish.publish_owed("vol_a", tmp_path) is not None)
        real_persist(work, mine)

    monkeypatch.setattr(qstore, "persist", _spy_persist)

    qrun.run_queue(
        DrainContext(work=tmp_path, city=CITY, publication=_publication(tmp_path)), track="serve"
    )

    assert owed_when_persisted == [False], "the row must never be persisted running while owed"


def test_a_hand_publish_does_not_close_a_queued_re_bake(tmp_path: Path) -> None:
    """A `queued` serve row is a re-bake someone asked for; publishing today's
    archive by hand does not discharge it."""
    _volume(tmp_path, "vol_a", results=True)
    qstore.add(tmp_path, "vol_a", "serve")

    assert qpublish.settle_published(tmp_path, "vol_a") is None
    assert qstore.load_queue(tmp_path)[0].status == "queued"


def test_an_orphan_with_no_completed_bake_is_still_reported_as_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No marker means the bake itself died: the pessimistic note is the true one."""
    _volume(tmp_path, "vol_a", results=True)
    entry = qstore.add(tmp_path, "vol_a", "serve")
    entry.status = "running"
    qstore.save_queue(tmp_path, [entry])
    published: list[str] = []
    monkeypatch.setattr(viewer_publish, "publish_volume", lambda vol, _cfg: published.append(vol))

    qrun.run_queue(
        DrainContext(work=tmp_path, city=CITY, publication=_publication(tmp_path)), track="serve"
    )

    recovered = qstore.load_queue(tmp_path)[0]
    assert recovered.status == "failed"
    assert "interrupted before completion" in (recovered.note or "")
    assert published == [], "nothing was baked, so nothing may be published"


def test_a_recovery_publish_failure_is_recorded_without_claiming_the_bake_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A marker whose archive has since gone missing fails HERE, as it would have there."""
    _volume(tmp_path, "vol_a", results=True)
    _owed(tmp_path, "vol_a")

    def _boom(_vol: str, _cfg: object) -> None:
        raise viewer_publish.PublicationError("missing or empty PMTiles archive")

    monkeypatch.setattr(viewer_publish, "publish_volume", _boom)

    qrun.run_queue(
        DrainContext(work=tmp_path, city=CITY, publication=_publication(tmp_path)), track="serve"
    )

    recovered = qstore.load_queue(tmp_path)[0]
    assert recovered.status == "failed"
    assert "recovery publish failed" in (recovered.note or "")
    assert "interrupted before completion" not in (recovered.note or "")


def test_stopping_a_drain_names_the_command_that_finishes_an_owed_publish(
    tmp_path: Path,
) -> None:
    """The console instructs rather than publishing, deliberately.

    Publishing is a large copy plus a manifest rebuild, and the console would be
    doing it in an HTTP handler without the drain lock — the same reason
    ``/api/apply`` passes ``do_warp=False``. So Stop labels the row with the
    command that finishes it, and a drain or a human runs that.
    """
    _volume(tmp_path, "vol_a", results=True)
    _owed(tmp_path, "vol_a")

    stopped = qstore.fail_interrupted(tmp_path, "serve")

    assert stopped[0].status == "failed"
    assert "autogeoref publish vol_a" in (stopped[0].note or "")
    assert "interrupted before completion" not in (stopped[0].note or "")


def test_a_place_orphan_is_never_read_as_an_owed_publish(tmp_path: Path) -> None:
    """A place leg publishes nothing; a stray marker must not relabel its failure."""
    _volume(tmp_path, "vol_a")
    entry = qstore.add(tmp_path, "vol_a")
    entry.status = "running"
    qstore.save_queue(tmp_path, [entry])
    viewer_publish.record_publish_owed("vol_a", tmp_path)

    interrupted = qstore.fail_interrupted(tmp_path, "place")

    assert "interrupted before completion" in (interrupted[0].note or "")


def test_the_debt_is_recorded_before_the_publish_is_attempted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordering IS the fix: a kill during publish must still find the marker."""
    _volume(tmp_path, "vol_a", results=True)
    qstore.add(tmp_path, "vol_a", "serve")
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _FakeProc(0))
    seen: list[bool] = []
    monkeypatch.setattr(
        viewer_publish,
        "publish_volume",
        lambda vol, _cfg: seen.append(viewer_publish.publish_owed(vol, tmp_path) is not None),
    )

    qrun.run_queue(
        DrainContext(work=tmp_path, city=CITY, publication=_publication(tmp_path)), track="serve"
    )

    assert seen == [True], "publish_volume must run with the debt already on disk"


def test_a_bake_that_never_finished_records_no_debt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed serve leg leaves nothing for a later drain to publish blind."""
    _volume(tmp_path, "vol_a", results=True)
    qstore.add(tmp_path, "vol_a", "serve")
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _FakeProc(2))

    qrun.run_queue(
        DrainContext(work=tmp_path, city=CITY, publication=_publication(tmp_path)), track="serve"
    )

    assert viewer_publish.publish_owed("vol_a", tmp_path) is None


def test_a_re_bake_disowns_the_previous_bakes_debt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A marker must describe THIS run's bake, or a recovery publishes the wrong archive.

    An earlier bake whose publish FAILED leaves its marker on disk. Re-queue that
    volume and kill the drain mid-bake, and a marker-trusting recovery would mark
    the entry done and serve the previous archive as though the new bake had
    finished. The bake clears the claim it is about to invalidate.
    """
    _volume(tmp_path, "vol_a", results=True)
    viewer_publish.record_publish_owed("vol_a", tmp_path, baked_at=1.0)
    qstore.add(tmp_path, "vol_a", "serve")

    def _killed_mid_bake(*_a: object, **_k: object) -> _FakeProc:
        gone = viewer_publish.publish_owed("vol_a", tmp_path) is None
        assert gone, "the stale claim must be gone by now"
        return _FakeProc(2)

    monkeypatch.setattr(subprocess, "run", _killed_mid_bake)

    qrun.run_queue(
        DrainContext(work=tmp_path, city=CITY, publication=_publication(tmp_path)), track="serve"
    )

    assert viewer_publish.publish_owed("vol_a", tmp_path) is None


def test_a_damaged_marker_is_not_a_debt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed: a truncated marker must not assert a bake nobody verified."""
    _volume(tmp_path, "vol_a", results=True)
    _owed(tmp_path, "vol_a")
    (tmp_path / "vol_a" / "publish-owed.json").write_text("{not json")
    published: list[str] = []
    monkeypatch.setattr(viewer_publish, "publish_volume", lambda vol, _cfg: published.append(vol))

    qrun.run_queue(
        DrainContext(work=tmp_path, city=CITY, publication=_publication(tmp_path)), track="serve"
    )

    assert published == []
    assert qstore.load_queue(tmp_path)[0].status == "failed"
