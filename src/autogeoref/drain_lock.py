"""One drainer per (work tree, track): POSIX process liveness and the drain lock.

The liveness half (``/proc`` fields, PID-recycling and zombie detection) is general process
introspection; the lock half applies it to the queue's per-track drain files. Top-level rather
than under ``queue/`` because the introspection is not queue-shaped — see
:func:`_is_a_live_drain`. It imports :mod:`.queue.store` for the shared path and type
definitions (``TRACKS``, ``queue_path``, ``QueueError``) but is otherwise queue-agnostic.
"""

from __future__ import annotations

import fcntl
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .queue.store import TRACKS, QueueError, queue_path

logger = logging.getLogger(__name__)


def _proc_fields(pid: int) -> tuple[str | None, str | None]:
    """The process's STATE and START TIME (``/proc/<pid>/stat``), or ``(None, None)``.

    The start time is what makes a PID identify a PROCESS rather than a slot:
    without it, a recycled PID is indistinguishable from the drain that used to
    own it. The state is what makes "the PID exists" mean "it is running": a
    zombie (``Z``) passes ``os.kill(pid, 0)`` and keeps its ``/proc`` entry, but
    it is an EXITED process its parent has not reaped.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None, None  # not Linux, or gone
    # the comm field can contain spaces AND parentheses, so split after the LAST ')'
    tail = stat.rpartition(")")[2].split()
    state = tail[0] if tail else None  # field 3
    return state, (tail[19] if len(tail) > 19 else None)  # field 22


def _proc_start(pid: int) -> str | None:
    return _proc_fields(pid)[1]


def _lock_holder(text: str) -> tuple[int, str | None]:
    """Parse a lockfile: ``"<pid>"`` (old) or ``"<pid> <starttime>"`` (current).

    An older version of this tool wrote the PID alone, and one of its drains may be
    running RIGHT NOW — the queue file, and the lock beside it, outlive the code.
    A bare PID still parses; it simply carries no start time to check.
    """
    parts = text.split()
    if not parts:
        return -1, None
    try:
        return int(parts[0]), (parts[1] if len(parts) > 1 else None)
    except ValueError:
        return -1, None


def _is_a_live_drain(pid: int, started: str | None) -> bool:
    """Is that lock held by a process that still EXISTS and is the SAME one?

    "Alive" alone is not enough: PIDs are recycled, so reading ``os.kill(pid, 0)`` as "a drain
    holds the lock" would wedge the queue permanently. The lock records the holder's START TIME,
    and a PID whose process started at a different moment is a different process.

    Where the start time is unavailable, liveness is all we have and the refusal stands: a false
    "busy" is recoverable, a false "free" means two drains against one model budget.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # someone else's process: alive, and not ours to interrogate
    state, now = _proc_fields(pid)
    if state == "Z":
        # A zombie is an EXITED drain whose parent has not reaped it. It signals as
        # alive and keeps its /proc entry, but it is draining nothing — reading it as
        # live wedged the console permanently: "running" forever, and every new drain
        # refused by a lock held by a corpse.
        return False
    if started is None or now is None:
        return True
    return started == now


def drain_lock_path(work: Path, track: str) -> Path:
    """The lock file for ONE track's drain. Per-track, so place and serve run at once.

    A single ``drain.lock`` for the whole tree is what forced place and serve to take
    turns — the serialization the two-queue model exists to end. Place and serve
    touch disjoint volumes and disjoint resources (money vs CPU), so their drains
    hold different locks and neither blocks the other. Two PLACE drains still can't
    coexist (one place lock), which is the only exclusion that is about correctness
    rather than budget: two writers of one annotation cache.
    """
    return queue_path(work).with_name(f"drain.{track}.lock")


def live_drain(work: Path, track: str) -> int | None:
    """The PID of the drain on ``track`` running RIGHT NOW, or None if none is.

    The same question :func:`drain_lock` asks before it refuses, asked WITHOUT
    taking the lock — the console must render "a drain is running" on a page that
    can never itself become a drainer.

    Advisory and racy by nature: a drain can start or die between this call and
    whatever the caller does next, and :func:`drain_lock` remains the only
    authority. Do not reimplement the staleness rule; a second copy would drift.
    """
    try:
        holder, started = _lock_holder(drain_lock_path(work, track).read_text())
    except OSError:
        return None
    return holder if _is_a_live_drain(holder, started) else None


def _drain_refusal(track: str, lock: Path, holder: int) -> QueueError:
    who = f"pid {holder}" if holder > 0 else "pid unreadable"
    return QueueError(
        f"another {track} drain is running ({who}, {lock}). Two {track} drains "
        "would run the same volume twice. Wait for it, or stop it."
    )


@contextmanager
def drain_lock(work: Path, track: str) -> Iterator[None]:
    """One drainer per (work tree, TRACK) — nonblocking, kernel-held.

    Two place drains would each spawn `autogeoref run` for the SAME volume: two writers of one
    results tree. That is the exclusion, and it is per-track. Same shape as
    :func:`paths.volume_lock`: a nonblocking ``flock`` on a PERMANENT per-track file, so a
    holder killed by any signal releases automatically, and the ``<pid> <starttime>`` metadata
    is written only after the flock is held. One inherited case remains: a drain from an older
    version holds the file by metadata alone, so bytes naming a LIVE process still refuse.
    """
    if track not in TRACKS:
        raise QueueError(f"unknown track {track!r}")
    lock = drain_lock_path(work, track)
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o644)
    try:

        def _holder() -> tuple[int, str | None]:
            try:
                return _lock_holder(os.pread(fd, 1024, 0).decode(errors="replace"))
            except OSError:
                return -1, None

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise _drain_refusal(track, lock, _holder()[0]) from None
        # The flock is ours, but an older version's drain may hold this file by
        # metadata alone. Live legacy bytes still refuse; dead ones are a crashed
        # holder's leftovers, and the kernel released that exclusion with the
        # process.
        holder, started = _holder()
        if _is_a_live_drain(holder, started):
            raise _drain_refusal(track, lock, holder)
        if holder > 0:
            logger.warning(
                "%s: leftover holder metadata (pid %s is not a running drain — it died "
                "without its cleanup, and the kernel released any lock with it). Proceeding.",
                lock,
                holder,
            )
        # PID *and* start time, written only now that the descriptor is held: it
        # only ever describes the actual holder, and a PID alone names a slot the
        # kernel will hand out again, not a process (see _is_a_live_drain).
        me = os.getpid()
        os.ftruncate(fd, 0)
        os.write(fd, f"{me} {_proc_start(me) or ''}".strip().encode())
        try:
            yield
        finally:
            os.ftruncate(fd, 0)  # clear while still holding — never someone else's bytes
    finally:
        os.close(fd)  # releases the flock
