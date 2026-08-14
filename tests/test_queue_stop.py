"""Stopping a drain on the first failure, so a backlog does not run into one wall.

Opt-in, and it leaves everything behind the failure ``queued`` for a re-run —
including the entries a multi-lane track had not started, and including the
fetch lane, which would otherwise keep downloading for hours after the drain
"stopped". What already succeeded stays done: a stop is not a rollback.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from autogeoref.queue import command as qcommand
from autogeoref.queue import run as qrun
from autogeoref.queue import store as qstore
from autogeoref.queue.command import DrainContext
from queue_support import CITY, _by, _FakeProc, _publication, _spy, _volume

pytestmark = pytest.mark.usefixtures("_publish_without_viewer_io")


def test_a_failure_stops_the_drain_and_leaves_the_rest_queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: a model limit that fails volume 1 must not have volume 2 walk
    into the same wall. Everything behind the failure stays `queued`, so a re-run after
    the limit resets picks up exactly where it stopped."""
    for name in ("vol_a", "vol_b", "vol_c"):
        _volume(tmp_path, name)
        qstore.add(tmp_path, name, then_serve=False)
    spawned = _spy(monkeypatch, code=1)

    touched = qrun.run_queue(
        DrainContext(work=tmp_path, city=CITY, stop_on_failure=True), track="place"
    )

    assert len(spawned) == 1, "exactly one doomed attempt, not one per volume"
    assert [e.volume for e in touched] == ["vol_a"]
    assert _by(qstore.load_queue(tmp_path)) == {
        ("vol_a", "place"): "failed",
        ("vol_b", "place"): "queued",
        ("vol_c", "place"): "queued",
    }


def test_without_the_flag_the_drain_still_walks_the_whole_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default is unchanged — opt-in, so an unrelated failure does not park a
    backlog somebody is watching."""
    for name in ("vol_a", "vol_b"):
        _volume(tmp_path, name)
        qstore.add(tmp_path, name, then_serve=False)
    spawned = _spy(monkeypatch, code=1)

    qrun.run_queue(DrainContext(work=tmp_path, city=CITY), track="place")

    assert len(spawned) == 2
    assert set(_by(qstore.load_queue(tmp_path)).values()) == {"failed"}


def test_stopping_keeps_what_already_succeeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop is not a rollback: the volumes drained before the failure stay done, and
    only the failure and what follows it are left for the operator."""
    for name in ("vol_a", "vol_b", "vol_c"):
        _volume(tmp_path, name)
        qstore.add(tmp_path, name, then_serve=False)

    def _fake(cmd: list[str], **_k: object) -> _FakeProc:
        return _FakeProc(1 if "vol_b" in cmd else 0)

    monkeypatch.setattr(subprocess, "run", _fake)

    qrun.run_queue(DrainContext(work=tmp_path, city=CITY, stop_on_failure=True), track="place")

    assert _by(qstore.load_queue(tmp_path)) == {
        ("vol_a", "place"): "needs-review",
        ("vol_b", "place"): "failed",
        ("vol_c", "place"): "queued",
    }


def test_a_multi_lane_track_stops_too_and_leaves_the_unstarted_queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The serve track runs N lanes, and every pending entry is submitted at once —
    so withholding the work at the START of a task is the only place an unstarted
    entry can be left `queued` instead of run into the same wall. Without that,
    `--stop-on-failure` was a no-op here and the whole queue burned."""
    for name in ("vol_a", "vol_b", "vol_c", "vol_d", "vol_e", "vol_f"):
        _volume(tmp_path, name, results=True)
        qstore.add(tmp_path, name, "serve")
    spawned = _spy(monkeypatch, code=1)

    qrun.run_queue(
        DrainContext(
            work=tmp_path, city=CITY, publication=_publication(tmp_path), stop_on_failure=True
        ),
        track="serve",
        lanes=2,
    )

    statuses = _by(qstore.load_queue(tmp_path))
    failed = [v for (v, _), s in statuses.items() if s == "failed"]
    queued = [v for (v, _), s in statuses.items() if s == "queued"]
    # at most `lanes` entries can have been in flight when the first failure landed
    assert 1 <= len(failed) <= 2, f"stopped after the in-flight lanes, not the queue: {statuses}"
    assert len(spawned) == len(failed), "nothing ran that was not accounted for"
    assert len(queued) == 6 - len(failed) and queued, "the rest are still queued"


def test_one_tracks_failure_stops_the_fetch_lane_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stop is SHARED across run_all's workers. A model limit that fails the
    place drain must also stop fetch, or the drain "stops" while a download lane
    runs for hours at 5 s a page — filling the disk that only a publish reclaims."""
    _volume(tmp_path, "vol_place")
    qstore.add(tmp_path, "vol_place", "place", then_serve=False)
    for name in ("vol_f1", "vol_f2", "vol_f3"):
        qstore.add(tmp_path, name, "fetch")

    fetches = 0

    def _fake(cmd: list[str], **_k: object) -> _FakeProc:
        nonlocal fetches
        if str(qcommand.FETCH_SCRIPT) in " ".join(cmd):
            fetches += 1
            time.sleep(0.05)  # give the place failure time to land first
            _volume(tmp_path, cmd[cmd.index("--work") - 1])
            return _FakeProc(0)
        return _FakeProc(1)  # the place leg hits the wall

    monkeypatch.setattr(subprocess, "run", _fake)

    qrun.run_all(
        tmp_path,
        CITY,
        poll_s=0.02,
        publication=_publication(tmp_path),
        stop_on_failure=True,
    )

    statuses = _by(qstore.load_queue(tmp_path))
    assert statuses[("vol_place", "place")] == "failed"
    assert fetches <= 1, f"the fetch lane kept downloading after the stop: {fetches} legs"
    assert [v for (v, t), s in statuses.items() if t == "fetch" and s == "queued"], (
        f"nothing was left for the re-run: {statuses}"
    )


def test_stop_on_failure_reaches_the_drain_from_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag is only worth having if `queue --run --stop-on-failure` sets it."""
    from autogeoref.cli.entry import main

    for name in ("vol_a", "vol_b"):
        _volume(tmp_path, name)
        qstore.add(tmp_path, name, "place", then_serve=False)
    spawned = _spy(monkeypatch, code=1)

    rc = main(
        [
            "queue",
            "--work",
            str(tmp_path),
            "--run",
            "--track",
            "place",
            "--city",
            str(CITY),
            "--stop-on-failure",
        ]
    )

    assert rc == 0  # the drain stopped cleanly; the FAILED row is the report
    assert len(spawned) == 1
    assert _by(qstore.load_queue(tmp_path))[("vol_b", "place")] == "queued"
