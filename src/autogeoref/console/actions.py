"""The acting half: enqueue, dequeue, and drain control, all somebody else's calls."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import drain_lock, logfiles
from ..annotate_volume import clear_failed_markers
from ..paths import VolumePaths
from ..queue import store as queue_store
from ..queue.command import autogeoref_bin
from ..queue.store import TRACKS, QueueError
from ..validation import volume_id

if TYPE_CHECKING:
    from ..viewer.publish import PublicationConfig

logger = logging.getLogger(__name__)


def _stoppable(pid: int | None) -> bool:
    """Can this console safely SIGTERM ``pid``'s process group?

    Only if the process is its own session LEADER (``getsid(pid) == pid``), which a
    drain this console spawned is (``start_new_session``). A drain started from a
    terminal lives in the operator's session, and signalling THAT group could take
    the shell down with it — so the console reports it but refuses to kill it.
    """
    if pid is None:
        return False
    try:
        return os.getsid(pid) == pid
    except (OSError, AttributeError):  # no getsid (Windows), or it died just now
        return False


def drain_state(work: Path) -> dict[str, Any]:
    """The per-QUEUE drain state, read off the per-track locks.

    Each track has its own drain (:func:`drain_lock.drain_lock`), and all of them
    can be live at once — a ``queue --run`` process that drains every track holds
    every lock under one pid, and separate ``--run --track`` processes hold one
    each. The board shows each queue's own drain; ``running`` is the OR, for the
    one-glance "is anything spending" indicator.
    """
    tracks = {}
    for t in TRACKS:
        pid = drain_lock.live_drain(work, t)
        tracks[t] = {"running": pid is not None, "pid": pid, "stoppable": _stoppable(pid)}
    return {"running": any(v["running"] for v in tracks.values()), **tracks}


#: What the console's "Start drain" can launch: EVERY queue in one process (the
#: default end-to-end parallel drain), or one queue on its own. ``both`` is the
#: historical name for "every track" and stays the wire value the page sends.
DRAIN_TARGETS = ("both", *TRACKS)

#: The most pages one place run may annotate or escalate concurrently from this
#: console (``autogeoref run --annotate-jobs``). Each call is its own backend process, so
#: jobs multiply RATE, never total spend — but a budget cap or a rate limit lands
#: with up to N-1 calls already in flight, so the overshoot grows with N. Ten is
#: the ceiling the operator asked for; it is not a recommendation.
MAX_ANNOTATE_JOBS = 10

#: Ceiling for a serve-lane override. `queue.run.default_serve_lanes` derives a
#: conservative default (half the cores, capped at 4) because GDAL is internally
#: multithreaded and baking is memory-heavy — this box has OOM'd before. Eight is
#: "I have measured my box", not a starting point.
MAX_SERVE_LANES = 8


def _bounded_int(name: str, value: object, ceiling: int) -> int | None:
    """A lane/jobs override off the wire: an int in [1, ceiling], None, or a refusal."""
    if value is None:
        return None
    try:
        out = int(str(value))  # the page's <select> sends a number, but trust nothing
    except ValueError:
        raise QueueError(f"{name} must be a whole number, not {value!r}") from None
    if not 1 <= out <= ceiling:
        raise QueueError(f"{name} must be between 1 and {ceiling} (asked for {out})")
    return out


def _drain_command(
    target: str,
    publication: PublicationConfig,
    *,
    place_lanes: int,
    nice: int,
    serve_lanes: int | None,
    annotate_jobs: int | None,
) -> list[str]:
    """The `autogeoref queue --run` argv a drain spawns — the command a human would type."""
    cmd = [autogeoref_bin(), "queue", "--run"]
    if target != "both":
        cmd += ["--track", target]
    cmd += [
        "--city",
        str(publication.city_toml),
        "--work",
        str(publication.work),
        "--place-lanes",
        str(place_lanes),
        "--nice",
        str(nice),
        "--tiles",
        str(publication.tiles_root),
        "--viewer-manifest",
        str(publication.manifest),
    ]
    if publication.loc_catalog is not None:
        cmd += ["--loc-catalog", str(publication.loc_catalog)]
    if serve_lanes is not None:
        cmd += ["--serve-lanes", str(serve_lanes)]
    if annotate_jobs is not None and target not in ("fetch", "serve"):
        # `=`-joined, so argparse reads the inner flag as --run-arg's VALUE and not
        # as an option of `queue` itself. A fetch- or serve-only drain gets no copy:
        # neither annotates anything (one downloads, the other is --warp-only).
        cmd += ["--run-arg=--annotate-jobs", f"--run-arg={annotate_jobs}"]
    return cmd


def _spawn_detached(cmd: list[str], log: Path) -> subprocess.Popen[bytes]:
    """Launch ``cmd`` detached in its own session, appending its output to ``log``."""
    log.parent.mkdir(parents=True, exist_ok=True)
    # Append-mode logs are the one unbounded growth path; cap them per drain
    # start by rotating the oversized file aside (its tail stays readable).
    if logfiles.rotate_log(log):
        logger.info("rotated oversized drain log aside to %s.1", log.name)
    with log.open("a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    # REAP it when it exits. This console is a long-lived parent: an unwaited
    # child stays a zombie, and a zombie still signals as alive — so the board
    # showed a finished (or stopped) drain as "running" forever, and its lock
    # could never be read as stale. One daemon thread parked in wait() is the
    # whole fix; the drain itself stays detached and outlives the console.
    threading.Thread(target=proc.wait, daemon=True, name=f"reap-drain-{proc.pid}").start()
    return proc


def _reset_interrupted(work: Path, stopped: Sequence[str]) -> list[str]:
    """Mark the killed drains' rows failed-interrupted, once each drain actually dies.

    The killed drain persists nothing on its way out, so its volumes would say
    `running` until the NEXT drain's orphan reset — a stop that leaves the board
    claiming the work is live is the stop the operator does not believe. Wait
    (briefly) for the drain to actually die, then run that reset here.
    """
    interrupted: list[str] = []
    for t in stopped:
        deadline = time.monotonic() + 5.0
        while drain_lock.live_drain(work, t) is not None and time.monotonic() < deadline:
            time.sleep(0.1)
        if drain_lock.live_drain(work, t) is None:
            interrupted += [e.volume for e in queue_store.fail_interrupted(work, t)]
        else:  # it outlived SIGTERM; leave its rows to the next drain's reset
            logger.warning("%s drain is still alive after SIGTERM; not touching its entries", t)
    return interrupted


@dataclass
class ConsoleActions:
    """The things the page can DO. Each one is a call into somebody else.

    Every act here is available on the CLI and is *implemented* on the CLI: this
    class enqueues with :func:`queue.store.add`, dequeues with :func:`queue.store.remove`, and
    drains by spawning the very command an operator would type. It does not know
    what a track means, what a run costs, or when a volume is placed — three things
    it would have had to learn to do any of this a second time.
    """

    work: Path
    #: The city TOML. A drain shells out to `autogeoref run --city <this>`, so
    #: without it there is nothing to spawn — the console is read-only, and says so.
    city: Path | None = None
    publication: PublicationConfig | None = None
    nice: int = 10
    #: Volumes the PLACE queue drains at once (model-bound; 1 keeps spend legible).
    place_lanes: int = 1
    #: Volumes the SERVE queue bakes at once (CPU-bound). None -> the derived default.
    serve_lanes: int | None = None

    @property
    def can_act(self) -> bool:
        return self.city is not None and self.publication is not None

    def _need_city(self) -> Path:
        if self.city is None:
            raise QueueError(
                "this console was started without --city, so it cannot run anything: a run "
                "needs a city config, and the address-era check that decides what is even "
                "runnable needs one too. Restart it with --city <city.toml>."
            )
        return self.city

    def enqueue(self, volume: str, track: str = "place", then_serve: bool = True) -> dict[str, Any]:
        """Add a volume to a queue. `queue.store.add` owns every rule about whether it may.

        A place enqueue defaults to end-to-end (``then_serve`` — it promotes itself to
        the serve queue when placed). ``then_serve=False`` is the "review first"
        diagnostic: place it and park it at ``needs-review``.
        """
        entry = queue_store.add(self.work, volume, track, then_serve=then_serve)
        return {
            "ok": True,
            "volume": entry.volume,
            "track": entry.track,
            "then_serve": entry.then_serve,
        }

    def dequeue(self, volume: str, track: str | None = None) -> dict[str, Any]:
        removed = queue_store.remove(self.work, volume, track)
        return {"ok": True, "removed": removed}

    def retry_failed_reads(self, volume: str, track: str = "place") -> dict[str, Any]:
        """Clear a volume's ``*.failed.json`` retry markers, then re-enqueue it.

        The markers are deliberately sticky (``annotate_volume.plan`` treats a
        marked page as cached), so a plain Retry re-runs without re-reading and
        hits the same unread-pages gate instantly, at zero spend. This is the
        documented manual fix — delete the markers, re-run — behind one explicit
        action that says how many re-reads it authorizes. Refused while the
        volume is live on either queue: clearing markers under a running leg
        would race its planner.
        """
        try:
            volume = volume_id(volume)
        except ValueError as exc:
            raise QueueError(f"invalid volume: {exc}") from exc
        entry = next(
            (e for e in queue_store.load_queue(self.work) if e.volume == volume and not e.terminal),
            None,
        )
        if entry is not None:
            raise QueueError(
                f"{volume} is {entry.status} on the {entry.track} queue — clearing its retry "
                "markers now would race that run. Let it finish (or stop the drain) first."
            )
        cleared = clear_failed_markers(VolumePaths(self.work / volume))
        try:
            added = queue_store.add(self.work, volume, track)
        except QueueError as exc:
            # the markers are already gone (that part is done and safe); say so
            # rather than reporting a refusal that looks like nothing happened
            raise QueueError(
                f"cleared {len(cleared)} retry marker(s), but the re-enqueue was refused: {exc}"
            ) from exc
        return {
            "ok": True,
            "volume": added.volume,
            "track": added.track,
            "cleared": len(cleared),
        }

    def start_drain(
        self,
        target: str = "both",
        *,
        annotate_jobs: int | None = None,
        serve_lanes: int | None = None,
    ) -> dict[str, Any]:
        """Spawn a DETACHED drain: the same `autogeoref queue --run` you would type.

        ``target`` is ``both`` (every queue in parallel, one process) or one
        queue on its own. Detached (``start_new_session``) so it outlives the
        console and owns a session this console can later signal without
        reaching anything else. ``annotate_jobs`` and ``serve_lanes`` are the
        page's parallelism dials for THIS drain only, and neither changes what
        it spends. The per-track lock is still the authority — this refuses
        early as a better error message, never as a guarantee.
        """
        if target not in DRAIN_TARGETS:
            raise QueueError(f"unknown drain target {target!r} (want one of {DRAIN_TARGETS})")
        annotate_jobs = _bounded_int("annotate jobs", annotate_jobs, MAX_ANNOTATE_JOBS)
        serve_lanes = _bounded_int("serve lanes", serve_lanes, MAX_SERVE_LANES)
        self._need_city()
        wanted = list(TRACKS) if target == "both" else [target]
        for t in wanted:
            live = drain_lock.live_drain(self.work, t)
            if live is not None:
                raise QueueError(f"the {t} queue already has a drain running (pid {live}).")
        if self.publication is None:
            raise QueueError("this console has no publication configuration")
        publication = self.publication
        if publication.work != self.work:
            raise QueueError("console work root differs from its publication configuration")
        lanes = serve_lanes if serve_lanes is not None else self.serve_lanes
        cmd = _drain_command(
            target,
            publication,
            place_lanes=self.place_lanes,
            nice=self.nice,
            serve_lanes=lanes,
            annotate_jobs=annotate_jobs,
        )
        log = queue_store.log_path(self.work, "drain", target)
        proc = _spawn_detached(cmd, log)
        logger.info("drain started: pid %d, target %s, log %s", proc.pid, target, log)
        return {
            "ok": True,
            "pid": proc.pid,
            "target": target,
            "log": str(log),
            "annotate_jobs": annotate_jobs if target not in ("fetch", "serve") else None,
            "serve_lanes": lanes if target in ("both", "serve") else None,
        }

    def stop_drain(self, target: str = "both") -> dict[str, Any]:
        """SIGTERM the drain process group(s) for ``target`` — the drain AND its run.

        The group, not the pid: a drain's whole job is a child ``autogeoref
        run``, and signalling only the parent would orphan a process that keeps
        working with nothing to report to. Model reads are cached per page and
        publication is atomic, so nothing already paid for is lost — but the
        current stage is left unfinished. A real stop, not a graceful one. The
        kernel releases a killed drain's flock with the process.
        """
        if target not in DRAIN_TARGETS:
            raise QueueError(f"unknown drain target {target!r} (want one of {DRAIN_TARGETS})")
        wanted = list(TRACKS) if target == "both" else [target]
        owners = {t: drain_lock.live_drain(self.work, t) for t in TRACKS}
        pids: dict[int, bool] = {}  # pid -> stoppable
        for t in wanted:
            pid = owners[t]
            if pid is not None:
                pids[pid] = _stoppable(pid)
        if not pids:
            return {"ok": True, "stopped": False, "reason": "no drain is running"}
        unstoppable = [pid for pid, ok in pids.items() if not ok]
        if unstoppable:
            raise QueueError(
                f"a drain (pid {unstoppable[0]}) was not started by this console — it lives in "
                "a terminal's session, and signalling its group could take that terminal down. "
                "Stop it where it was started (ctrl-c)."
            )
        for pid in pids:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue  # the group died between the lock read and now: already stopped
            logger.warning("drain pid %d: SIGTERM (operator pressed stop)", pid)
        # Reset the tracks the kill actually STOPPED, not those the request
        # named: a one-process both-drain holds both locks under one pid, so
        # killing it for one track takes the other down with it, and resetting
        # only the named track strands the other's rows at `running` with no
        # owner. A track whose drain was never ours to kill keeps its rows.
        interrupted = _reset_interrupted(self.work, [t for t in TRACKS if owners[t] in pids])
        return {"ok": True, "stopped": True, "pids": list(pids), "interrupted": interrupted}
