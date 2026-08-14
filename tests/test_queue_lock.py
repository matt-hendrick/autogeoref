"""The drain lock: one drain per track, and a dead holder never wedges a queue.

The lock file is permanent and the flock is the authority, so a stale holder is
taken over, a live one is obeyed, and a recycled pid or an unreaped zombie is
neither. The last two tests barrier-start real subprocesses over one stale
legacy lock, because the reclamation this replaced had a window where both
racers could proceed.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from autogeoref import drain_lock
from autogeoref.queue import store as qstore
from queue_support import CITY, _by, _volume

pytestmark = pytest.mark.usefixtures("_publish_without_viewer_io")


def _lockfile(tmp_path: Path, track: str = "place") -> Path:
    lock = drain_lock.drain_lock_path(tmp_path, track)
    lock.parent.mkdir(parents=True, exist_ok=True)
    return lock


def test_a_stale_lock_is_taken_not_obeyed(tmp_path: Path) -> None:
    lock = _lockfile(tmp_path)
    lock.write_text("999999")  # a PID that is not running
    with drain_lock.drain_lock(tmp_path, "place"):
        holder, _ = drain_lock._lock_holder(lock.read_text())
        assert holder == os.getpid(), "we took it"
    assert lock.exists(), "the lock file is permanent — acquisition never unlinks it"
    assert lock.read_text() == "", "released: metadata cleared, never someone else's bytes"


def test_a_live_holder_is_still_obeyed(tmp_path: Path) -> None:
    lock = _lockfile(tmp_path)
    me = os.getpid()
    lock.write_text(f"{me} {drain_lock._proc_start(me) or ''}".strip())  # this process, right now
    with (
        pytest.raises(qstore.QueueError, match="another place drain"),
        drain_lock.drain_lock(tmp_path, "place"),
    ):
        pass
    assert lock.exists(), "a live holder's lock must not be deleted"


def test_live_drain_reports_per_track(tmp_path: Path) -> None:
    """The console and the serve poll ask per-track: is THIS queue's drain live?"""
    assert drain_lock.live_drain(tmp_path, "place") is None
    with drain_lock.drain_lock(tmp_path, "serve"):
        assert drain_lock.live_drain(tmp_path, "serve") == os.getpid()
        not_place = drain_lock.live_drain(tmp_path, "place") is None
        assert not_place, "a serve drain is not a place drain"
    released = drain_lock.live_drain(tmp_path, "serve") is None
    assert released, "released: the permanent file names nobody"


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs Linux /proc")
def test_a_recycled_pid_does_not_re_wedge_the_queue(tmp_path: Path) -> None:
    lock = _lockfile(tmp_path)
    lock.write_text(f"{os.getpid()} 1")  # our PID, but a start time that is not ours
    with drain_lock.drain_lock(tmp_path, "place"):
        pass
    assert lock.read_text() == ""


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs Linux /proc")
def test_a_zombie_holder_is_a_dead_drain(tmp_path: Path) -> None:
    """An unreaped drain must not read as live: it signals as alive and keeps its
    /proc entry, but it has EXITED. Reading it as live wedged the console — "running"
    forever, Stop pointed at a corpse, and every new drain refused by its lock."""
    import subprocess
    import sys
    import time

    # the child dies immediately; until wait() below it is this process's zombie
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        lock = _lockfile(tmp_path)
        lock.write_text(f"{proc.pid} {drain_lock._proc_start(proc.pid) or ''}".strip())
        deadline = time.monotonic() + 5.0
        while drain_lock._proc_fields(proc.pid)[0] != "Z":
            assert time.monotonic() < deadline, "child never became a zombie"
            time.sleep(0.01)
        assert drain_lock.live_drain(tmp_path, "place") is None, "a zombie is not a drain"
        with drain_lock.drain_lock(tmp_path, "place"):  # and its lock is stale, and taken
            pass
        assert lock.read_text() == ""
    finally:
        proc.wait()


def test_a_lock_from_an_older_version_still_refuses_while_live(tmp_path: Path) -> None:
    lock = _lockfile(tmp_path)
    lock.write_text(str(os.getpid()))  # OLD format: bare pid, no start time
    with (
        pytest.raises(qstore.QueueError, match="another place drain"),
        drain_lock.drain_lock(tmp_path, "place"),
    ):
        pass
    assert lock.exists()


def test_an_unreadable_lock_does_not_wedge_the_queue(tmp_path: Path) -> None:
    lock = _lockfile(tmp_path)
    lock.write_text("")
    with drain_lock.drain_lock(tmp_path, "place"):
        pass
    assert lock.read_text() == ""


# The old create/unlink/recreate reclamation had a TOCTOU: two drains could both
# judge a lock stale, one unlink the other's freshly created lock, and both
# proceed. The flock is the authority now, so a barrier-started race must admit
# exactly one — these drivers hold the lock until the sibling has been refused,
# so the loser meets a HELD lock, never a released one.

_DRAIN_LOCK_RACE_DRIVER = """
import os, sys, time
from pathlib import Path

from autogeoref import drain_lock
from autogeoref.queue import store as qstore

sync, work = Path(sys.argv[1]), Path(sys.argv[2])

(sync / f"ready-{os.getpid()}").touch()
deadline = time.monotonic() + 120.0
while not (sync / "go").exists():
    if time.monotonic() > deadline:
        raise RuntimeError("no go signal")
    time.sleep(0.01)

try:
    with drain_lock.drain_lock(work, "place"):
        (sync / f"entered-{os.getpid()}").touch()
        deadline = time.monotonic() + 120.0
        while not list(sync.glob("refused-*")):
            if time.monotonic() > deadline:
                raise RuntimeError("the sibling was never refused")
            time.sleep(0.02)
except qstore.QueueError:
    (sync / f"refused-{os.getpid()}").touch()
    sys.exit(3)
sys.exit(0)
"""


_DRAIN_RACE_DRIVER = """
import os, subprocess, sys, time, types
from pathlib import Path

from autogeoref.queue import run as qrun
from autogeoref.queue import store as qstore
from autogeoref.queue.command import DrainContext

sync, work, city = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def fake_run(cmd, **kwargs):
    # the mocked child `autogeoref run`: record the launch, then hold the leg
    # open until the sibling drain has been refused by the held lock
    with open(sync / "spawned", "a") as fh:
        fh.write(f"{os.getpid()}\\n")
    deadline = time.monotonic() + 120.0
    while not list(sync.glob("refused-*")):
        if time.monotonic() > deadline:
            raise RuntimeError("the sibling was never refused")
        time.sleep(0.02)
    return types.SimpleNamespace(returncode=0)


subprocess.run = fake_run

(sync / f"ready-{os.getpid()}").touch()
deadline = time.monotonic() + 120.0
while not (sync / "go").exists():
    if time.monotonic() > deadline:
        raise RuntimeError("no go signal")
    time.sleep(0.01)

try:
    qrun.run_queue(DrainContext(work=work, city=city), track="place")
except qstore.QueueError:
    (sync / f"refused-{os.getpid()}").touch()
    sys.exit(3)
sys.exit(0)
"""


def _await(
    condition: Callable[[], object], deadline_s: float = 120.0, what: str = "condition"
) -> None:
    import time

    deadline = time.monotonic() + deadline_s
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.05)


def _race(tmp_path: Path, driver_source: str, *args: str) -> tuple[Path, list[int]]:
    """Barrier-start two subprocess drivers over one stale legacy lock."""
    import sys as _sys

    lock = _lockfile(tmp_path)
    lock.write_text("999999")  # a dead legacy holder: stale under the old reclamation
    sync = tmp_path / "sync"
    sync.mkdir()
    driver = tmp_path / "driver.py"
    driver.write_text(driver_source)
    procs = [
        subprocess.Popen(
            [_sys.executable, str(driver), str(sync), str(tmp_path), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    try:
        _await(lambda: len(list(sync.glob("ready-*"))) == 2, what="both children ready")
        (sync / "go").touch()  # the barrier drops: both race for the drain lock
        for p in procs:
            p.communicate(timeout=120)
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    return sync, sorted(p.returncode for p in procs)


def test_barrier_started_drains_over_a_stale_legacy_lock_admit_exactly_one(
    tmp_path: Path,
) -> None:
    """Two simultaneous drain_lock takers over one stale legacy lock: exactly one
    enters the critical section; the other is refused by the HELD flock."""
    sync, codes = _race(tmp_path, _DRAIN_LOCK_RACE_DRIVER)
    assert len(list(sync.glob("entered-*"))) == 1, "exactly ONE drain may enter"
    assert codes == [0, 3], f"one winner, one refusal (got {codes})"
    lock = drain_lock.drain_lock_path(tmp_path, "place")
    assert lock.exists() and lock.read_text() == "", "permanent file, cleared on release"


def test_barrier_started_place_drains_launch_exactly_one_child(tmp_path: Path) -> None:
    """The same race through a real place drain: exactly one mocked child
    `autogeoref run` launches for the queued volume."""
    _volume(tmp_path, "vol_a")
    qstore.add(tmp_path, "vol_a", "place", then_serve=False)
    sync, codes = _race(tmp_path, _DRAIN_RACE_DRIVER, str(CITY))
    spawned = (sync / "spawned").read_text().splitlines() if (sync / "spawned").exists() else []
    assert len(spawned) == 1, f"exactly ONE child command may launch (got {len(spawned)})"
    assert codes == [0, 3], f"one winner, one refusal (got {codes})"
    assert _by(qstore.load_queue(tmp_path))[("vol_a", "place")] == "needs-review"
