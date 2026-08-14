"""The fetch queue: the one track that creates a work tree instead of consuming one.

Acquisition has a single implementation, so the fetch leg shells out to it
rather than running the pipeline, and no run argument reaches it. A fetched
volume promotes to the place queue carrying whatever intent it was enqueued
with, a failed fetch promotes nothing, and ``run_all`` carries one volume from
fetch through serve without any worker hanging on a queue nothing will feed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from autogeoref.queue import command as qcommand
from autogeoref.queue import run as qrun
from autogeoref.queue import store as qstore
from autogeoref.queue.command import DrainContext
from autogeoref.viewer import publish as viewer_publish
from queue_support import CITY, _by, _ctx, _FakeProc, _publication, _spy, _volume

pytestmark = pytest.mark.usefixtures("_publish_without_viewer_io")


def _spy_fetching(monkeypatch: pytest.MonkeyPatch, work: Path) -> list[list[str]]:
    """Record commands, and make a FETCH run build the work tree it downloads into —
    so the promotion to the place queue (which requires one) can actually fire, as it
    would in a real run."""
    spawned: list[list[str]] = []

    def _fake(cmd: list[str], **_k: object) -> _FakeProc:
        spawned.append(list(cmd))
        if str(qcommand.FETCH_SCRIPT) in " ".join(cmd):
            _volume(work, cmd[cmd.index("--work") - 1])
        return _FakeProc(0)

    monkeypatch.setattr(subprocess, "run", _fake)
    return spawned


def test_a_fetch_enqueue_is_the_one_that_needs_no_work_tree(tmp_path: Path) -> None:
    """Every other track CONSUMES a work tree; fetch creates one. Requiring it would
    make the acquisition queue impossible to enqueue on."""
    entry = qstore.add(tmp_path, "vol_new", "fetch")
    assert entry.track == "fetch" and entry.status == "queued"
    assert not (tmp_path / "vol_new").exists(), "enqueuing fetched nothing"


def test_place_and_serve_still_refuse_a_volume_with_no_work_tree(tmp_path: Path) -> None:
    """And the refusal names the track that would build one."""
    for track in ("place", "serve"):
        with pytest.raises(qstore.QueueError, match="no work tree"):
            qstore.add(tmp_path, "vol_new", track)
    with pytest.raises(qstore.QueueError, match="--track fetch"):
        qstore.add(tmp_path, "vol_new", "place")


def test_fetching_a_volume_promotes_it_to_the_place_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first promotion in the chain: a fetched volume lands on the place queue,
    carrying its end-to-end intent, and nothing places inside the fetch leg."""
    qstore.add(tmp_path, "vol_a", "fetch")
    spawned = _spy_fetching(monkeypatch, tmp_path)

    fetched = qrun.run_queue(DrainContext(work=tmp_path, city=CITY), track="fetch")

    assert [e.status for e in fetched] == ["done"]
    assert len(spawned) == 1 and "run" not in spawned[0], "the fetch leg only fetches"
    assert _by(qstore.load_queue(tmp_path)) == {
        ("vol_a", "fetch"): "done",
        ("vol_a", "place"): "queued",
    }
    promoted = next(e for e in qstore.load_queue(tmp_path) if e.track == "place")
    assert promoted.then_serve is True, "end to end survives the first promotion"


def test_a_review_first_fetch_carries_that_intent_to_the_place_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--review` on a fetch enqueue must still park the volume at needs-review after
    placement — the intent has to survive the promotion, or the diagnostic is lost."""
    qstore.add(tmp_path, "vol_a", "fetch", then_serve=False)
    _spy_fetching(monkeypatch, tmp_path)

    qrun.run_queue(DrainContext(work=tmp_path, city=CITY), track="fetch")

    promoted = next(e for e in qstore.load_queue(tmp_path) if e.track == "place")
    assert promoted.then_serve is False


def test_a_failed_fetch_never_promotes_to_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-fetched volume must not reach a model. The fetcher exits nonzero when any
    page failed, and that is what keeps the promotion from firing."""
    qstore.add(tmp_path, "vol_a", "fetch")
    _spy(monkeypatch, code=1)

    fetched = qrun.run_queue(DrainContext(work=tmp_path, city=CITY), track="fetch")

    assert fetched[0].status == "failed" and "fetch failed" in (fetched[0].note or "")
    assert not any(e.track == "place" for e in qstore.load_queue(tmp_path))


def test_the_fetch_command_runs_the_one_fetcher_and_not_autogeoref_run(tmp_path: Path) -> None:
    """The fetch leg is the one leg that is NOT `autogeoref run`: acquisition has a
    single implementation, and the queue shells out to it."""
    entry = qstore.QueueEntry(volume="vol_a", track="fetch")
    cmd = qcommand._command("fetch", entry, _ctx(tmp_path))

    assert cmd[0] == sys.executable
    assert Path(cmd[1]).is_file() and Path(cmd[1]).name == qcommand.FETCH_SCRIPT.name
    assert cmd[2:] == ["vol_a", "--work", str(tmp_path), "--cache", str(qcommand._LOC_CACHE)]
    assert "--city" not in cmd, "the fetcher takes no city: LOC keys on the item id"


def test_the_fetch_command_uses_the_publications_loc_cache(tmp_path: Path) -> None:
    """One polite lane means one cache directory, and the configured one wins."""
    publication = viewer_publish.PublicationConfig(
        work=tmp_path,
        city_toml=CITY,
        manifest=tmp_path / "viewer" / "chicago-ill" / "manifest.json",
        loc_cache=tmp_path / "elsewhere" / "loc",
    )
    cmd = qcommand._command(
        "fetch",
        qstore.QueueEntry(volume="vol_a", track="fetch"),
        _ctx(tmp_path, publication=publication),
    )
    assert cmd[-2:] == ["--cache", str(publication.loc_cache)]


def test_run_arg_extras_never_reach_the_fetcher(tmp_path: Path) -> None:
    """`--run-arg` carries `autogeoref run` flags. Forwarding them to a different
    program would fail every fetch in a drain configured for placement — but a
    forbidden override is still refused for the whole batch."""
    entry = qstore.QueueEntry(volume="vol_a", track="fetch")
    cmd = qcommand._command(
        "fetch", entry, _ctx(tmp_path, extra=("--no-escalate", "--annotate-jobs", "6"))
    )
    assert "--no-escalate" not in cmd and "--annotate-jobs" not in cmd

    with pytest.raises(qstore.QueueError, match="cannot override"):
        qcommand._command("fetch", entry, _ctx(tmp_path, extra=("--warp",)))


def test_the_enqueue_message_names_where_the_volume_actually_goes_next(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fetch entry promotes to PLACE, and only from there to serve. Printing "will
    promote to serve" on a fetch enqueue describes two hops as one — and `--review`
    printed nothing at all, which is the one intent the chain goes to trouble to
    carry."""
    from autogeoref.cli.entry import main

    base = ["queue", "--work", str(tmp_path), "--track", "fetch", "--add"]
    assert main([*base, "vol_a"]) == 0
    assert "queued vol_a on the fetch queue (will promote to place, then serve)" in (
        capsys.readouterr().out
    )

    assert main([*base, "vol_b", "--review"]) == 0
    assert "queued vol_b on the fetch queue (stops at needs-review)" in capsys.readouterr().out

    _volume(tmp_path, "vol_c")
    assert main(["queue", "--work", str(tmp_path), "--add", "vol_c"]) == 0
    assert "queued vol_c on the place queue (will promote to serve)" in capsys.readouterr().out


def test_run_all_carries_a_volume_from_fetch_through_serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline for the three-worker pool: one `run_all` fetches a volume, places
    it, and bakes it, with each worker following the one feeding it."""
    qstore.add(tmp_path, "vol_a", "fetch")
    spawned: list[list[str]] = []

    def _fake(cmd: list[str], **_k: object) -> _FakeProc:
        spawned.append(list(cmd))
        if str(qcommand.FETCH_SCRIPT) in " ".join(cmd):
            _volume(tmp_path, "vol_a")
        elif "--warp-only" not in cmd:  # a place run writes the records serve needs
            r = tmp_path / "vol_a" / "results"
            r.mkdir(parents=True, exist_ok=True)
            (r / "p1.json").write_text(
                json.dumps({"page": "1", "status": "OK", "gcps_geojson": {"type": "FC"}})
            )
        return _FakeProc(0)

    monkeypatch.setattr(subprocess, "run", _fake)

    touched = qrun.run_all(tmp_path, CITY, poll_s=0.02, publication=_publication(tmp_path))

    assert _by(qstore.load_queue(tmp_path)) == {
        ("vol_a", "fetch"): "done",
        ("vol_a", "place"): "done",
        ("vol_a", "serve"): "done",
    }
    assert {e.track for e in touched} == {"fetch", "place", "serve"}
    assert len(spawned) == 3, "one leg per track, no leg run twice"


def test_a_dying_fetch_worker_does_not_hang_the_place_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same backstop the serve worker has, one link up the chain: a fetch worker
    that raises MID-DRAIN must not leave place polling forever on a queue nothing will
    feed — which would also strand the place lock."""
    _volume(tmp_path, "vol_b")
    qstore.add(tmp_path, "vol_a", "fetch")
    qstore.add(tmp_path, "vol_b", "place", then_serve=False)  # independent place work

    def _boom(cmd: list[str], **_k: object) -> _FakeProc:
        if str(qcommand.FETCH_SCRIPT) in " ".join(cmd):
            raise RuntimeError("fetch blew up")
        return _FakeProc(0)

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
    assert not t.is_alive(), "run_all hung: the place worker never stopped following"
    assert isinstance(result.get("error"), RuntimeError), "the fetch failure surfaced"
    assert _by(qstore.load_queue(tmp_path))[("vol_b", "place")] == "needs-review"
