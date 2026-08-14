"""The acting console: every button is a call into the queue, never a second queue.

Enqueueing, dequeueing and clearing failed-read markers all go through
`queue.store`, so the page and the CLI cannot drift about what is allowed. A
drain is spawned with the roots the board was started against, with bounded
dials, into a rotated log, and handed to a reaper — an unreaped child reads as
live forever. Stop follows the kill, and "running" is read from the lock.
"""

from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path

import pytest

from autogeoref import drain_lock, logfiles
from autogeoref.console import actions as console_actions
from autogeoref.console import payload as console_payload
from autogeoref.queue import store as qstore
from autogeoref.status import build_status
from console_support import _tree


def test_the_board_acts_only_through_the_queues_own_functions(tmp_path: Path) -> None:
    """The acting console got buttons, not a SECOND queue.

    The page now writes — that is the item. What it must never do is grow its own
    copy of "what a track means" or "when a volume may be enqueued": every act is a
    call into `queue`, so the CLI and the page cannot drift into disagreeing about
    what is allowed. This pins the delegation, which is the only reason the buttons
    are safe to have.
    """
    roots = _tree(tmp_path)
    (roots["work"] / "v1").mkdir(parents=True)
    actions = console_actions.ConsoleActions(work=roots["work"], city=Path("configs/x.toml"))

    entry = actions.enqueue("v1", "place")
    assert entry["track"] == "place" and entry["then_serve"] is True
    # it landed in the REAL queue, not somewhere of the console's own
    assert [e.volume for e in qstore.load_queue(roots["work"])] == ["v1"]

    # and the queue's rules still bind it: a second live entry is refused, by `queue.store.add`
    with pytest.raises(qstore.QueueError, match="already on the place queue"):
        actions.enqueue("v1", "place")

    assert actions.dequeue("v1")["removed"] == 1
    assert qstore.load_queue(roots["work"]) == []


def test_retry_failed_reads_clears_the_markers_and_requeues(tmp_path: Path) -> None:
    """The one-click version of the documented manual fix (delete markers, re-run).

    A ``*.failed.json`` marker deliberately survives a plain retry — a marked page
    counts as cached, so nothing re-spends silently — which means the retry fails
    the same way instantly. This action IS the human decision to pay: it clears the
    markers (and only the markers — landed reads keep replaying free) and re-adds
    the volume through ``queue.store.add``, never around it.
    """
    roots = _tree(tmp_path)
    ann = roots["work"] / "v1" / "annotations"
    ann.mkdir(parents=True)
    (ann / "p1.failed.json").write_text("{}")
    (ann / "p2.failed.json").write_text("{}")
    (ann / "p2.escalated.claude-opus-4-8.failed.json").write_text("{}")
    (ann / "p3.json").write_text("{}")  # a read that LANDED: budget spent, must survive
    actions = console_actions.ConsoleActions(work=roots["work"], city=Path("configs/x.toml"))

    # the terminal failed entry from the run that minted the markers does not block
    dead = qstore.add(roots["work"], "v1", "place")
    dead.status = "failed"
    qstore.save_queue(roots["work"], [dead])

    out = actions.retry_failed_reads("v1")
    assert out == {"ok": True, "volume": "v1", "track": "place", "cleared": 3}
    assert list(ann.glob("*.failed.json")) == []
    assert (ann / "p3.json").exists()
    live = [e for e in qstore.load_queue(roots["work"]) if not e.terminal]
    assert [(e.volume, e.track, e.status) for e in live] == [("v1", "place", "queued")]


def test_retry_failed_reads_refuses_while_the_volume_is_live(tmp_path: Path) -> None:
    """Clearing markers under a running (or queued) leg would race its planner, so
    the action refuses BEFORE deleting anything — the markers must still be there."""
    roots = _tree(tmp_path)
    ann = roots["work"] / "v1" / "annotations"
    ann.mkdir(parents=True)
    (ann / "p1.failed.json").write_text("{}")
    actions = console_actions.ConsoleActions(work=roots["work"], city=Path("configs/x.toml"))
    qstore.add(roots["work"], "v1", "place")  # queued: non-terminal

    with pytest.raises(qstore.QueueError, match="would race"):
        actions.retry_failed_reads("v1")
    assert (ann / "p1.failed.json").exists()


def test_a_console_without_a_city_cannot_run_anything(tmp_path: Path) -> None:
    """No city => no drain, and the page is told so rather than offering dead buttons.

    A run needs a city config (`autogeoref run --city`), and so does the address-era
    check that decides which volumes are even runnable. A console that rendered a
    "Run it" button and then 409'd every press would be worse than one that admits it
    is a viewer.
    """
    roots = _tree(tmp_path)
    actions = console_actions.ConsoleActions(work=roots["work"], city=None)
    assert not actions.can_act
    with pytest.raises(qstore.QueueError, match="without --city"):
        actions.start_drain("both")

    rows = build_status(work=roots["work"], fixtures=roots["fixtures"], tiles=roots["tiles"])
    assert console_payload.board(work=roots["work"], rows=rows, can_act=False)["can_act"] is False


def test_console_drain_preserves_all_publication_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A console drain must publish to the roots the board was started against."""
    from autogeoref.viewer.publish import PublicationConfig

    roots = _tree(tmp_path)
    publication = PublicationConfig(
        work=roots["work"],
        city_toml=tmp_path / "city.toml",
        tiles_root=tmp_path / "served",
        manifest=tmp_path / "viewer" / "custom-manifest.json",
        loc_catalog=tmp_path / "catalog.json",
    )
    spawned: list[list[str]] = []

    class Proc:
        pid = 42

        def wait(self) -> int:  # the reaper thread parks here
            return 0

    def fake_popen(cmd: list[str], **_kwargs: object) -> Proc:
        spawned.append(cmd)
        return Proc()

    monkeypatch.setattr(console_actions.subprocess, "Popen", fake_popen)
    console_actions.ConsoleActions(
        work=roots["work"], city=publication.city_toml, publication=publication
    ).start_drain()

    command = spawned[0]
    for flag, path in (
        ("--work", publication.work),
        ("--city", publication.city_toml),
        ("--tiles", publication.tiles_root),
        ("--viewer-manifest", publication.manifest),
        ("--loc-catalog", publication.loc_catalog),
    ):
        assert command[command.index(flag) + 1] == str(path)


def test_console_drain_composes_the_parallelism_the_page_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dials are passthroughs to the flags an operator would type, so prove the
    composed command round-trips the REAL parsers: `queue` carries `--annotate-jobs 10`
    intact inside --run-arg, and `run` accepts what falls out the other side."""
    from autogeoref.cli.parser import build_parser
    from autogeoref.viewer.publish import PublicationConfig

    roots = _tree(tmp_path)
    publication = PublicationConfig(
        work=roots["work"],
        city_toml=tmp_path / "city.toml",
        tiles_root=roots["tiles"],
        manifest=tmp_path / "viewer" / "manifest.json",
    )
    spawned: list[list[str]] = []

    class Proc:
        pid = 44

        def wait(self) -> int:
            return 0

    def fake_popen(cmd: list[str], **_kwargs: object) -> Proc:
        spawned.append(cmd)
        return Proc()

    monkeypatch.setattr(console_actions.subprocess, "Popen", fake_popen)
    actions = console_actions.ConsoleActions(
        work=roots["work"], city=publication.city_toml, publication=publication
    )
    result = actions.start_drain("both", annotate_jobs=10, serve_lanes=2)
    assert result["annotate_jobs"] == 10 and result["serve_lanes"] == 2

    cmd = spawned[0]
    assert cmd[cmd.index("--serve-lanes") + 1] == "2"
    args = build_parser().parse_args(cmd[1:])  # cmd[0] is the binary
    assert args.run_arg == ["--annotate-jobs", "10"]
    run_args = build_parser().parse_args(["run", "vol_a", "--city", "c.toml", *args.run_arg])
    assert run_args.annotate_jobs == 10

    # a serve-only drain annotates nothing, so the dial does not ride along
    actions.start_drain("serve", annotate_jobs=5)
    assert not any(a.startswith("--run-arg") for a in spawned[1])

    # nor does a fetch-only drain: it downloads, it never reaches a model
    result = actions.start_drain("fetch", annotate_jobs=5, serve_lanes=2)
    assert result["annotate_jobs"] is None and result["serve_lanes"] is None
    assert not any(a.startswith("--run-arg") for a in spawned[2])
    assert spawned[2][spawned[2].index("--track") + 1] == "fetch"


def test_the_console_can_start_a_drain_on_every_track(tmp_path: Path) -> None:
    """A track the queue drains and the console cannot start is a track nobody uses.
    Spelled out against literals, so this pins the surface rather than restating it."""
    assert set(console_actions.DRAIN_TARGETS) == {"both", "fetch", "place", "serve"}
    with pytest.raises(qstore.QueueError, match="unknown drain target"):
        console_actions.ConsoleActions(work=tmp_path, city=Path("c.toml")).start_drain("nope")


def test_console_drain_rotates_an_oversized_log_before_appending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drain logs are the only append-mode logs, so a multi-day history is capped
    HERE: an oversized log is moved aside to `.1` (still readable on disk) and the
    new drain's header starts a fresh file. A small log keeps appending in place."""
    from autogeoref.viewer.publish import PublicationConfig

    roots = _tree(tmp_path)
    publication = PublicationConfig(
        work=roots["work"],
        city_toml=tmp_path / "city.toml",
        tiles_root=roots["tiles"],
        manifest=tmp_path / "viewer" / "manifest.json",
    )

    class Proc:
        pid = 45

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(console_actions.subprocess, "Popen", lambda *_a, **_k: Proc())
    monkeypatch.setattr(logfiles, "DRAIN_LOG_ROTATE_BYTES", 1024)
    actions = console_actions.ConsoleActions(
        work=roots["work"], city=publication.city_toml, publication=publication
    )

    log = qstore.log_path(roots["work"], "drain", "both")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("previous drain " * 100)  # well past the (patched) cap

    actions.start_drain("both")
    rotated = log.with_name(log.name + ".1")
    assert rotated.read_text().startswith("previous drain ")
    fresh = log.read_text()
    assert "previous drain" not in fresh and "$ " in fresh  # header only

    actions.start_drain("both")  # still small: appends, no second rotation
    assert log.read_text().count("$ ") == 2
    assert rotated.read_text().startswith("previous drain ")


def test_console_drain_refuses_out_of_range_parallelism(tmp_path: Path) -> None:
    """The dials are bounded BEFORE anything spawns — an over-eager or garbled value
    is a refusal with the bound in it, not a drain with a surprise inside."""
    actions = console_actions.ConsoleActions(work=_tree(tmp_path)["work"], city=Path("c.toml"))
    for bad in (0, console_actions.MAX_ANNOTATE_JOBS + 1, "ten"):
        with pytest.raises(qstore.QueueError, match="annotate jobs"):
            actions.start_drain("both", annotate_jobs=bad)
    with pytest.raises(qstore.QueueError, match="serve lanes"):
        actions.start_drain("both", serve_lanes=0)


def test_the_console_reaps_the_drain_it_spawned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The console is a long-lived parent: a drain it never wait()s on stays a
    ZOMBIE when it exits, and a zombie still signals as alive — so the board showed
    a finished (or stopped) drain as "running" forever, and its lock could never be
    read as stale. start_drain must hand every child to a reaper."""
    import threading

    from autogeoref.viewer.publish import PublicationConfig

    roots = _tree(tmp_path)
    publication = PublicationConfig(
        work=roots["work"],
        city_toml=tmp_path / "city.toml",
        tiles_root=roots["tiles"],
        manifest=tmp_path / "viewer" / "manifest.json",
    )
    reaped = threading.Event()

    class Proc:
        pid = 43

        def wait(self) -> int:
            reaped.set()
            return 0

    monkeypatch.setattr(console_actions.subprocess, "Popen", lambda *_a, **_k: Proc())
    console_actions.ConsoleActions(
        work=roots["work"], city=publication.city_toml, publication=publication
    ).start_drain()
    assert reaped.wait(2.0), "no reaper ever wait()ed on the drain"


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs Linux /proc")
def test_stop_clears_what_it_stopped(tmp_path: Path) -> None:
    """Stop must leave a board the operator believes: the drain gone, and the
    volumes it was mid-way through marked failed-interrupted (retryable) — not
    saying `running` until some future drain's orphan reset gets around to it."""
    import subprocess
    import sys

    work = _tree(tmp_path)["work"]
    (work / "vol_a").mkdir()
    entry = qstore.add(work, "vol_a", "place")
    entry.status = "running"
    qstore.save_queue(work, [entry])

    # a stand-in drain: detached exactly as start_drain detaches (its own session)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        lock = drain_lock.drain_lock_path(work, "place")
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(f"{proc.pid} {drain_lock._proc_start(proc.pid) or ''}".strip())

        result = console_actions.ConsoleActions(work=work).stop_drain("place")
        assert result["stopped"] is True and result["interrupted"] == ["vol_a"]
        assert drain_lock.live_drain(work, "place") is None, "the stopped drain still reads as live"
        recovered = qstore.load_queue(work)[0]
        assert recovered.status == "failed"
        assert "interrupted" in (recovered.note or "")
        # and stopping again is a no-op, not an internal error
        assert console_actions.ConsoleActions(work=work).stop_drain("place")["stopped"] is False
    finally:
        with suppress(ProcessLookupError):
            proc.kill()
        proc.wait()


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs Linux /proc")
def test_a_single_track_stop_of_a_both_drain_clears_both_tracks(tmp_path: Path) -> None:
    """A one-process both-drain holds both locks under one pid, so SIGTERMing it for
    `place` takes the serve worker down too. The reset must follow the KILL, not the
    request — resetting only the named track would strand the other track's rows at
    `running` with no owner, the exact state stop_drain exists to end."""
    import subprocess
    import sys

    work = _tree(tmp_path)["work"]
    entries = []
    for volume, track in (("vol_p", "place"), ("vol_s", "serve")):
        (work / volume).mkdir()
        results = work / volume / "results"
        results.mkdir()
        (results / "p1.json").write_text("{}")  # serve enqueue demands placements
        entry = qstore.add(work, volume, track)
        entry.status = "running"
        entries.append(entry)
    qstore.save_queue(work, entries)

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        for track in qstore.TRACKS:  # one pid, both locks: a `queue --run` both-drain
            lock = drain_lock.drain_lock_path(work, track)
            lock.parent.mkdir(parents=True, exist_ok=True)
            lock.write_text(f"{proc.pid} {drain_lock._proc_start(proc.pid) or ''}".strip())

        result = console_actions.ConsoleActions(work=work).stop_drain("place")
        assert sorted(result["interrupted"]) == ["vol_p", "vol_s"]
        assert all(e.status == "failed" for e in qstore.load_queue(work))
    finally:
        with suppress(ProcessLookupError):
            proc.kill()
        proc.wait()


def test_the_drain_is_reported_from_the_lock_not_from_a_queue_status(tmp_path: Path) -> None:
    """ "Running" is a fact about a PROCESS, and the lock is where that fact lives.

    A drain killed at 3am leaves its entry saying `running` forever, so a console that
    read the queue file would offer a Stop button for a process that does not exist and
    refuse to start the one you need. It asks `drain_lock.live_drain`, which reads the lock
    the drain actually holds — and it reports a drain it could not safely signal as
    unstoppable rather than pretending.
    """
    work = _tree(tmp_path)["work"]
    idle = {"running": False, "pid": None, "stoppable": False}
    # every track reports, so a track added to `queue.store.TRACKS` cannot go unrendered
    assert console_actions.drain_state(work) == {
        "running": False,
        **dict.fromkeys(qstore.TRACKS, idle),
    }

    # a place lock naming a process that is not a drain
    lock = drain_lock.drain_lock_path(work, "place")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("999999999")  # a pid that cannot exist
    assert console_actions.drain_state(work)["running"] is False, "a dead holder is not a drain"

    with drain_lock.drain_lock(work, "place"):  # a REAL place drain: this process holds its lock
        state = console_actions.drain_state(work)
        assert state["running"] and state["place"]["pid"] == os.getpid()
        assert state["serve"]["running"] is False, "a place drain is not a serve drain"
        assert state["fetch"]["running"] is False, "nor a fetch drain"
        # this test process is not its own session leader, so the console refuses to
        # signal it — killing that group could take the shell down with it
        assert state["place"]["stoppable"] is (os.getsid(os.getpid()) == os.getpid())
        with pytest.raises(qstore.QueueError, match="already has a drain running"):
            console_actions.ConsoleActions(work=work, city=Path("c.toml")).start_drain("place")
