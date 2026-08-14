"""Draining a queue: what runs, in how many lanes, and what the locks allow.

A place entry that is end to end promotes itself to the serve queue once it has
placed something; a failed place promotes nothing. Place and fetch are forced to
one lane whatever is asked for, and the per-track locks let a serve drain run
while a place drain holds its own. ``run_all`` walks the queues together and
must never wedge when a worker dies mid-drain.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from autogeoref import drain_lock
from autogeoref.queue import command as qcommand
from autogeoref.queue import render as qrender
from autogeoref.queue import run as qrun
from autogeoref.queue import store as qstore
from autogeoref.queue.command import DrainContext
from autogeoref.viewer import publish as viewer_publish
from conftest import antedate
from queue_support import CITY, _by, _FakeProc, _publication, _spy, _volume

pytestmark = pytest.mark.usefixtures("_publish_without_viewer_io")


def _spy_placing(monkeypatch: pytest.MonkeyPatch, work: Path) -> list[list[str]]:
    """Record commands, and make a PLACE run produce results/ — so the promotion to the
    serve queue (which requires placements to exist) can actually fire, as it would in a
    real run. A serve run (``--warp-only``) produces nothing."""
    spawned: list[list[str]] = []

    def _fake(cmd: list[str], **_k: object) -> _FakeProc:
        spawned.append(list(cmd))
        if "--warp-only" not in cmd:
            vol = cmd[cmd.index("run") + 1]  # `... run <vol> ...` (a `nice` prefix may lead)
            r = work / vol / "results"
            r.mkdir(parents=True, exist_ok=True)
            (r / "p1.json").write_text(
                json.dumps({"page": "1", "status": "OK", "gcps_geojson": {"type": "FC"}})
            )
        return _FakeProc(0)

    monkeypatch.setattr(subprocess, "run", _fake)
    return spawned


def test_placing_an_end_to_end_volume_promotes_it_to_the_serve_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "End to end" is a PLACE entry that enqueues a SERVE entry on success.

    Draining the place queue places the volume (one invocation, no --warp-only) and
    leaves a queued serve entry behind — which a serve drain then bakes. The place run
    never bakes anything itself.
    """
    _volume(tmp_path, "vol_a")
    qstore.add(tmp_path, "vol_a")  # place, then_serve
    spawned = _spy_placing(monkeypatch, tmp_path)

    placed = qrun.run_queue(DrainContext(work=tmp_path, city=CITY), track="place")

    assert [e.status for e in placed] == ["done"]
    assert len(spawned) == 1 and "--warp-only" not in spawned[0], "the place run places"
    assert _by(qstore.load_queue(tmp_path)) == {
        ("vol_a", "place"): "done",
        ("vol_a", "serve"): "queued",
    }

    served = qrun.run_queue(
        DrainContext(work=tmp_path, city=CITY, publication=_publication(tmp_path)), track="serve"
    )
    assert [e.status for e in served] == ["done"]
    assert len(spawned) == 2 and "--warp-only" in spawned[1], "the serve run only bakes"


def test_review_first_places_but_does_not_promote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """then_serve=False (the --review diagnostic): place it, park it, promote nothing."""
    _volume(tmp_path, "vol_a")
    qstore.add(tmp_path, "vol_a", then_serve=False)
    spawned = _spy(monkeypatch)

    placed = qrun.run_queue(DrainContext(work=tmp_path, city=CITY), track="place")

    assert [e.status for e in placed] == ["needs-review"]
    assert len(spawned) == 1, "nothing is baked, and no serve entry is created"
    assert not any(e.track == "serve" for e in qstore.load_queue(tmp_path))
    assert "review --volume vol_a" in (placed[0].note or "")
    assert "NEEDS YOU" in qrender.render_text(tmp_path)


def test_a_failed_place_never_promotes_to_serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _volume(tmp_path, "vol_a")
    qstore.add(tmp_path, "vol_a")
    spawned = _spy(monkeypatch, code=2)

    placed = qrun.run_queue(DrainContext(work=tmp_path, city=CITY), track="place")

    assert placed[0].status == "failed" and "place failed" in (placed[0].note or "")
    assert len(spawned) == 1
    served = [e for e in qstore.load_queue(tmp_path) if e.track == "serve"]
    assert not served, "a failed place serves nothing"


def test_a_served_volume_is_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _volume(tmp_path, "vol_a", results=True)
    qstore.add(tmp_path, "vol_a", "serve")
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _FakeProc(0))
    served = qrun.run_queue(
        DrainContext(work=tmp_path, city=CITY, publication=_publication(tmp_path)), track="serve"
    )
    assert [e.status for e in served] == ["done"]


def test_a_failed_run_is_recorded_with_its_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _volume(tmp_path, "vol_a")
    qstore.add(tmp_path, "vol_a", then_serve=False)
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _FakeProc(2))
    done = qrun.run_queue(DrainContext(work=tmp_path, city=CITY), track="place")
    assert done[0].status == "failed" and done[0].exit_code == 2
    assert done[0].log is not None and Path(done[0].log).exists()
    assert "FAILED" in qrender.render_text(tmp_path)


def test_a_new_drain_marks_an_orphaned_running_entry_failed(tmp_path: Path) -> None:
    """An interrupted old drain must not leave an entry permanently un-retryable."""
    _volume(tmp_path, "vol_a")
    entry = qstore.add(tmp_path, "vol_a")
    entry.status = "running"
    qstore.save_queue(tmp_path, [entry])

    assert qrun.run_queue(DrainContext(work=tmp_path, city=CITY), track="place") == []
    recovered = qstore.load_queue(tmp_path)[0]
    assert recovered.status == "failed"
    assert "interrupted before completion" in (recovered.note or "")


def test_the_place_queue_is_forced_to_one_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two concurrent place runs would write one volume's annotation cache twice."""
    _volume(tmp_path, "vol_a")
    _volume(tmp_path, "vol_b")
    qstore.add(tmp_path, "vol_a", then_serve=False)
    qstore.add(tmp_path, "vol_b", then_serve=False)
    live = peak = 0

    def _fake(*_a: object, **_k: object) -> _FakeProc:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        live -= 1
        return _FakeProc(0)

    monkeypatch.setattr(subprocess, "run", _fake)
    qrun.run_queue(
        DrainContext(work=tmp_path, city=CITY), track="place", lanes=4
    )  # asked for 4; forced to 1
    assert peak == 1


def test_the_fetch_queue_is_forced_to_one_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LOC conduct is ONE request lane at >= 5 s; a second fetch lane would break it."""
    qstore.add(tmp_path, "vol_a", "fetch")
    qstore.add(tmp_path, "vol_b", "fetch")
    live = peak = 0

    def _fake(*_a: object, **_k: object) -> _FakeProc:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        live -= 1
        return _FakeProc(0)

    monkeypatch.setattr(subprocess, "run", _fake)
    qrun.run_queue(
        DrainContext(work=tmp_path, city=CITY), track="fetch", lanes=4
    )  # asked for 4; forced to 1
    assert peak == 1


def test_a_bare_run_drains_both_queues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare `--run` drains place AND serve; a pre-existing backlog is still drained."""
    from autogeoref.cli.entry import main

    _volume(tmp_path, "vol_old")
    qstore.add(tmp_path, "vol_old", "place", then_serve=False)  # what an older tool enqueued
    spawned = _spy(monkeypatch)

    rc = main(["queue", "--work", str(tmp_path), "--run", "--city", "configs/chicago/chicago.toml"])

    assert rc == 0
    assert len(spawned) == 1, "the pre-existing place entry is drained"
    assert _by(qstore.load_queue(tmp_path)) == {("vol_old", "place"): "needs-review"}


def test_run_all_places_and_serves_in_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline: one `run_all` places a volume and bakes it, across the queues.

    The serve worker polls while the place worker is live, so it picks up the volume
    the place worker promotes — without a second command, and without the place worker
    ever baking anything itself.
    """
    _volume(tmp_path, "vol_a")
    qstore.add(tmp_path, "vol_a")  # end to end
    spawned = _spy_placing(monkeypatch, tmp_path)

    touched = qrun.run_all(tmp_path, CITY, poll_s=0.02, publication=_publication(tmp_path))

    kinds = sorted(("--warp-only" in c) for c in spawned)
    assert kinds == [False, True], "exactly one place run and one serve run"
    assert _by(qstore.load_queue(tmp_path)) == {
        ("vol_a", "place"): "done",
        ("vol_a", "serve"): "done",
    }
    assert {(e.volume, e.track) for e in touched} == {("vol_a", "place"), ("vol_a", "serve")}


def test_run_all_rejects_a_forbidden_place_arg_instead_of_hanging(tmp_path: Path) -> None:
    """`--run-arg --warp` must fail FAST, not deadlock.

    `--warp` is forbidden as a passthrough. Routed through `run_all` it used to make
    the place worker raise mid-drain while the serve worker polled forever on a queue
    nothing would feed — a hang that also stranded the serve lock. It is rejected up
    front now.
    """
    _volume(tmp_path, "vol_a")
    qstore.add(tmp_path, "vol_a")
    with pytest.raises(qstore.QueueError, match="cannot override"):
        qrun.run_all(
            tmp_path,
            CITY,
            extra=["--warp"],
            poll_s=0.02,
            publication=_publication(tmp_path),
        )


def test_a_dying_place_worker_does_not_hang_the_serve_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The backstop: if the place worker raises MID-DRAIN (any bug), the serve worker
    must stop following and `run_all` must return/raise — never wedge.

    The place worker's `finally` flips the follow predicate, so the serve worker's empty
    queue becomes genuinely empty and it exits. Run in a thread with a join deadline so a
    regression FAILS (deadlock) rather than hanging the whole suite.
    """
    _volume(tmp_path, "vol_a")  # place, will blow up mid-run
    _volume(tmp_path, "vol_b", results=True)
    qstore.add(tmp_path, "vol_a", "place", then_serve=False)
    qstore.add(tmp_path, "vol_b", "serve")  # an independent serve item to drain

    def _boom(cmd: list[str], **_k: object) -> _FakeProc:
        if "--warp-only" not in cmd:  # a place run
            raise RuntimeError("place blew up")
        return _FakeProc(0)  # serve runs fine

    monkeypatch.setattr(subprocess, "run", _boom)

    result: dict[str, object] = {}

    def _go() -> None:
        try:
            qrun.run_all(tmp_path, CITY, poll_s=0.02, publication=_publication(tmp_path))
        except Exception as exc:
            result["error"] = exc

    t = threading.Thread(target=_go)
    t.start()
    t.join(timeout=15)
    assert not t.is_alive(), "run_all hung: the serve worker never stopped following"
    assert isinstance(result.get("error"), RuntimeError), "the place failure surfaced"
    # and serve still did its independent work before returning
    assert _by(qstore.load_queue(tmp_path))[("vol_b", "serve")] == "done"


def test_a_serve_drain_runs_while_a_place_drain_holds_its_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of per-track locks: place and serve do not block each other."""
    _volume(tmp_path, "vol_a", results=True)
    qstore.add(tmp_path, "vol_a", "serve")
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _FakeProc(0))
    with drain_lock.drain_lock(tmp_path, "place"):  # a place drain is holding its lock
        served = qrun.run_queue(
            DrainContext(work=tmp_path, city=CITY, publication=_publication(tmp_path)),
            track="serve",
        )
    assert [e.status for e in served] == ["done"]


def test_a_publish_failure_marks_the_serve_entry_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _volume(tmp_path, "vol_a", results=True)
    qstore.add(tmp_path, "vol_a", "serve")
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _FakeProc(0))
    monkeypatch.setattr(
        viewer_publish,
        "publish_volume",
        lambda *_a, **_k: (_ for _ in ()).throw(viewer_publish.PublicationError("bad manifest")),
    )

    entry = qrun.run_queue(
        DrainContext(work=tmp_path, city=CITY, publication=_publication(tmp_path)), track="serve"
    )[0]

    assert entry.status == "failed"
    assert "published nothing" in (entry.note or "")


def test_a_successful_publish_refreshes_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report written at placement says 'serve pass needed'; the publish it
    asks for resolves that, so the serve leg must rewrite the report — or the
    artifact keeps demanding a serve pass that already happened."""
    from autogeoref.paths import VolumePaths
    from autogeoref.stages.report import stage_report

    vol = _volume(tmp_path, "vol_a", results=True)
    archive = tmp_path / "tiles" / "autogeoref" / "vol_a.pmtiles"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"x")
    # the record predates the re-bake below; pin it rather than trusting the
    # wall clock to record the two writes in order
    antedate(*sorted((vol / "results").glob("p*.json")))
    committed = (vol / "results" / "p1.json").stat().st_mtime
    os.utime(archive, (committed - 100, committed - 100))
    stage_report(VolumePaths(root=vol), "vol_a", tiles_root=tmp_path / "tiles")
    assert "serve pass needed" in (vol / "report.md").read_text()

    def _publish(volume: str, publication: object, **_k: object) -> Path:
        archive.write_bytes(b"xx")  # the bake lands: the archive is now newest
        return archive

    monkeypatch.setattr(viewer_publish, "publish_volume", _publish)
    qstore.add(tmp_path, "vol_a", "serve")
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _FakeProc(0))

    entry = qrun.run_queue(
        DrainContext(work=tmp_path, city=CITY, publication=_publication(tmp_path)), track="serve"
    )[0]

    assert entry.status == "done"
    assert "serve pass needed" not in (vol / "report.md").read_text()


def test_a_serve_drain_requires_publication_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _volume(tmp_path, "vol_a", results=True)
    qstore.add(tmp_path, "vol_a", "serve")
    spawned = _spy(monkeypatch)

    with pytest.raises(qstore.QueueError, match="needs a publication configuration"):
        qrun.run_queue(DrainContext(work=tmp_path, city=CITY), track="serve")

    assert not spawned


def test_two_place_drains_cannot_run_at_once(tmp_path: Path) -> None:
    """The one exclusion that is correctness, not budget: two writers of one cache."""
    _volume(tmp_path, "vol_a")
    qstore.add(tmp_path, "vol_a", "place")
    refusal = pytest.raises(qstore.QueueError, match="another place drain")
    with drain_lock.drain_lock(tmp_path, "place"), refusal:
        qrun.run_queue(DrainContext(work=tmp_path, city=CITY), track="place")
    with drain_lock.drain_lock(tmp_path, "place"):  # released, so a later drain is fine
        pass


def test_a_concurrent_add_survives_a_drain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A volume queued WHILE a drain runs must not be erased by the drain's final save.

    With two drains and the console all writing queue.json, the merge in `persist`
    holds `queue_write_lock` so no update is silently dropped. The looping drain then
    goes on to drain the newcomer too — which is the point, not a hazard.
    """
    _volume(tmp_path, "vol_a")
    _volume(tmp_path, "vol_b")
    qstore.add(tmp_path, "vol_a", "place", then_serve=False)
    added = threading.Event()

    def _fake(*_a: object, **_k: object) -> _FakeProc:
        if not added.is_set():  # exactly once, when placing vol_a
            qstore.add(tmp_path, "vol_b", "place", then_serve=False)  # "another agent", mid-drain
            added.set()
        return _FakeProc(0)

    monkeypatch.setattr(subprocess, "run", _fake)
    qrun.run_queue(DrainContext(work=tmp_path, city=CITY), track="place")
    # vol_b was not erased by vol_a's drain, and the loop went on to drain it too
    assert _by(qstore.load_queue(tmp_path)) == {
        ("vol_a", "place"): "needs-review",
        ("vol_b", "place"): "needs-review",
    }


def test_a_direct_run_is_refused_while_a_queued_child_owns_the_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The queued child is a plain `autogeoref run`, so it takes the volume
    lock itself; a manual run of the SAME volume mid-drain must be refused
    before it can duplicate the child's model reads."""
    from autogeoref.cli.entry import main
    from autogeoref.paths import VolumePaths, volume_lock

    _volume(tmp_path, "vol_a")
    qstore.add(tmp_path, "vol_a", "place", then_serve=False)
    refused: list[str] = []

    def child_holding_the_volume(
        leg: str, entry: qstore.QueueEntry, ctx: qcommand.DrainContext
    ) -> int:
        work, city = ctx.work, ctx.city
        # exactly what the spawned `autogeoref run` does first: own the volume
        with volume_lock(VolumePaths(root=work / entry.volume), operation="run"):
            with pytest.raises(SystemExit, match="is busy") as exc_info:
                main(["run", entry.volume, "--city", str(city), "--work", str(work)])
            refused.append(str(exc_info.value))
        return 0

    monkeypatch.setattr(qcommand, "_run_leg", child_holding_the_volume)
    qrun.run_queue(DrainContext(work=tmp_path, city=CITY), track="place")
    assert refused and "vol_a is busy" in refused[0]
