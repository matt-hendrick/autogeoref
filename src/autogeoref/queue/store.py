"""The queue file: schema, validation on load, atomic persistence, membership."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ..paths import VolumePaths, atomic_write_text
from ..validation import volume_id
from ..viewer import publish as viewer_publish

logger = logging.getLogger(__name__)

#: The three queues, in pipeline order. There is no ``all`` track: end-to-end is
#: a FETCH entry that promotes to place and a PLACE entry with ``then_serve`` set
#: (see the package docstring), each promoting itself forward on success — not a
#: fourth queue.
TRACKS = ("fetch", "place", "serve")

#: The single stage-run each track's entry executes.
_LEG_OF = {"fetch": "fetch", "place": "place", "serve": "serve"}

#: Terminal states. ``needs-review`` is terminal FOR THE RUNNER — the queue is
#: done with it and a human is not. It is now only ever reached by an entry that
#: ASKED for it (``--review``); nothing else parks a volume there.
_TERMINAL = {"needs-review", "done", "failed"}

#: Every status this version can hold. Anything else in a persisted file was
#: written by different code (or by hand) and cannot be run here: drains select
#: only ``queued``, while a nonterminal stranger blocks a re-add — so
#: :func:`load_queue` quarantines it instead of keeping a zombie.
_STATUSES = {"queued", "running"} | _TERMINAL

#: Persisted field types, checked on load. ``volume`` and ``track`` are the
#: entry's identity and are validated separately (an entry that cannot even be
#: named is dropped, not quarantined).
_FIELD_TYPES: dict[str, tuple[type, ...]] = {
    "status": (str,),
    "added": (int, float),
    "started": (int, float, type(None)),
    "finished": (int, float, type(None)),
    "exit_code": (int, type(None)),
    "log": (str, type(None)),
    "note": (str, type(None)),
    "then_serve": (bool,),
}

#: What a malformed field becomes inside a quarantined entry. ``added`` is a
#: constant, not ``time.time()``: it keys :func:`persist`'s merge, so the same
#: file must load to the same key twice.
_SANITIZED: dict[str, Any] = {
    "status": "failed",
    "added": 0.0,
    "started": None,
    "finished": None,
    "exit_code": None,
    "log": None,
    "note": None,
    "then_serve": False,
}


class QueueError(RuntimeError):
    """The queue was asked for something that would break a contract."""


@dataclass
class QueueEntry:
    volume: str
    track: str
    status: str = "queued"  # queued | running | needs-review | done | failed
    added: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    exit_code: int | None = None
    log: str | None = None
    #: Why it is where it is — a failure's first line, or the reviewer's note.
    note: str | None = None
    #: Promote to the serve queue when placing succeeds. This is what "end to end"
    #: means — a place that auto-enqueues a serve (see the package docstring), not a
    #: separate track. ``False`` parks the placed volume at ``needs-review`` instead
    #: (the ``--review`` diagnostic). A FETCH entry carries the flag through its own
    #: promotion so the intent survives the whole chain; it is ignored on a serve
    #: entry, which has nothing left to promote to.
    then_serve: bool = True

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL


def queue_path(work: Path) -> Path:
    return work / "queue" / "queue.json"


def log_path(work: Path, volume: str, track: str) -> Path:
    _safe_volume(volume)
    return work / "queue" / "logs" / f"{volume}.{track}.log"


def _safe_volume(value: object) -> str:
    try:
        return volume_id(value)
    except ValueError as exc:
        raise QueueError(f"invalid volume: {exc}") from exc


def load_queue(work: Path) -> list[QueueEntry]:
    """Read the queue, tolerating fields this version does not know about.

    The queue file outlives the code that wrote it, so a plain ``QueueEntry(**e)`` would make an
    operator's whole queue unloadable over one unknown field. Unknown keys are dropped, missing
    ones take their defaults.

    An entry whose ``track`` this version does not know is DROPPED with a loud warning: kept, it
    would match no queue while still blocking a re-add. An unknown status or mistyped field is
    QUARANTINED instead — a terminal ``failed`` row whose note names what was wrong.
    """
    p = queue_path(work)
    if not p.exists():
        return []
    known = {f.name for f in fields(QueueEntry)}
    out: list[QueueEntry] = []
    for e in json.loads(p.read_text()).get("entries", []):
        entry = _coerce_entry(e, p, known)
        if entry is not None:
            out.append(entry)
    return out


def _coerce_entry(e: object, p: Path, known: set[str]) -> QueueEntry | None:
    """Validate one raw queue-file entry per :func:`load_queue`'s rules, or drop it."""
    if not isinstance(e, dict):
        logger.warning("%s: dropping a non-object queue entry (%r)", p, e)
        return None
    extra = set(e) - known
    if extra:
        logger.warning("%s: ignoring unknown queue field(s) %s", p, ", ".join(sorted(extra)))
    data = {k: v for k, v in e.items() if k in known}
    try:
        _safe_volume(data.get("volume"))
    except QueueError:
        logger.warning("%s: dropping invalid volume id %r", p, data.get("volume"))
        return None
    if data.get("track") not in TRACKS:
        logger.warning(
            "%s: dropping %s — unknown queue %r (the schema changed under this file; "
            "re-add it with `autogeoref queue --add`)",
            p,
            data.get("volume"),
            data.get("track"),
        )
        return None
    wrong: list[str] = []
    malformed = sorted(
        name for name, ok in _FIELD_TYPES.items() if name in data and not isinstance(data[name], ok)
    )
    if malformed:
        wrong.append("malformed field(s) " + ", ".join(malformed))
        for name in malformed:
            data[name] = _SANITIZED[name]
    status = data.get("status", "queued")
    if "status" not in malformed and status not in _STATUSES:
        wrong.append(f"unknown status {status!r}")
    entry = QueueEntry(**data)
    if wrong:
        why = "; ".join(wrong)
        entry.status = "failed"
        entry.note = (
            f"quarantined on load: {why}. This row is terminal and does not block "
            "the volume — re-add it to run again."
        )
        logger.warning("%s: quarantining %s/%s — %s", p, entry.volume, entry.track, why)
    return entry


def save_queue(work: Path, entries: Sequence[QueueEntry]) -> None:
    """Atomic replace: a crash mid-write must not truncate the queue."""
    doc = json.dumps({"entries": [asdict(e) for e in entries]}, indent=2)
    atomic_write_text(queue_path(work), doc)


@contextmanager
def queue_write_lock(work: Path) -> Iterator[None]:
    """Serialize every read-modify-write of ``queue.json`` across processes/threads.

    ``save_queue`` is atomic, but the mutations here are read-MODIFY-write, and
    two writers can each load, then each save, with the second silently dropping
    the first's change. This ``flock`` makes load+save one critical section.
    Advisory and per open-file-description, so two threads in one process
    contend correctly too. **Never nest it.**
    """
    p = queue_path(work).with_name("queue.json.lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _key(e: QueueEntry) -> tuple[str, str, float]:
    return (e.volume, e.track, e.added)


def persist(work: Path, mine: Sequence[QueueEntry]) -> None:
    """Write back ONLY the entries this process owns, keeping everyone else's.

    A drain used to hold the whole list in memory and rewrite it wholesale, so a volume someone
    queued WHILE it ran was erased by the drainer's stale snapshot on the next save. Several
    processes share this tree; a queue that silently forgets an entry is worse than no queue.
    Now that two drains write concurrently, the load+save is held under
    :func:`queue_write_lock` so the merge cannot be clobbered mid-flight.
    """
    updated = {_key(e): e for e in mine}
    with queue_write_lock(work):
        merged: list[QueueEntry] = []
        seen: set[tuple[str, str, float]] = set()
        for e in load_queue(work):  # re-read: whatever is on disk RIGHT NOW
            k = _key(e)
            seen.add(k)
            merged.append(updated.get(k, e))
        # an entry of ours the file has never seen (removed under us) is re-added
        merged.extend(e for k, e in updated.items() if k not in seen)
        save_queue(work, merged)


def add(
    work: Path,
    volume: str,
    track: str = "place",
    *,
    then_serve: bool = True,
    note: str | None = None,
) -> QueueEntry:
    """Enqueue a volume on the fetch, place, or serve queue.

    A FETCH enqueue downloads the volume from LOC; it is the one enqueue that
    does NOT require a work tree, because building one is what it does, and it
    promotes to place on success. A PLACE enqueue defaults to ``then_serve``,
    letting the drain promote it to the serve queue; ``--review`` parks it at
    ``needs-review`` instead. A SERVE enqueue bakes an already-placed volume,
    where ``then_serve`` is meaningless and forced off.
    """
    if track not in TRACKS:
        raise QueueError(f"unknown track {track!r} (want one of {', '.join(TRACKS)})")
    volume = _safe_volume(volume)
    paths = VolumePaths(work / volume)
    # Every other track consumes a work tree; fetch CREATES one, so requiring it
    # here would make the acquisition queue impossible to enqueue on.
    if track != "fetch" and not paths.root.exists():
        raise QueueError(
            f"{volume}: no work tree at {paths.root}. Acquire it first: "
            f"autogeoref queue --track fetch --add {volume}"
        )
    # A SERVE enqueue must already have placements — it runs `--warp-only`, which
    # consumes committed records and produces none. (A place entry that will PROMOTE
    # to serve has none yet, and that is the point: its own place run writes them,
    # and the promotion happens after.)
    if track == "serve" and (not paths.results.exists() or not any(paths.results.glob("p*.json"))):
        raise QueueError(
            f"{volume}: nothing to serve — no results/ records. Place it first "
            f"(`autogeoref queue --add {volume}` does both)."
        )
    with queue_write_lock(work):
        for e in load_queue(work):
            if e.volume != volume or e.terminal:
                continue
            if e.track == track:
                raise QueueError(f"{volume} is already on the {track} queue ({e.status})")
            # place + serve at once would bake a funnel that is still moving. The
            # legit end-to-end path never hits this: the place entry goes terminal
            # (done) BEFORE its promotion enqueues the serve entry.
            raise QueueError(
                f"{volume} is on the {e.track} queue ({e.status}); it cannot be on {track} "
                "at the same time. Let it finish, then add it."
            )
        entry = QueueEntry(
            volume=volume, track=track, then_serve=(then_serve and track != "serve"), note=note
        )
        entries = [*load_queue(work), entry]
        save_queue(work, entries)
    return entry


def remove(work: Path, volume: str, track: str | None = None) -> int:
    volume = _safe_volume(volume)
    with queue_write_lock(work):
        entries = load_queue(work)
        keep = [
            e for e in entries if not (e.volume == volume and (track is None or e.track == track))
        ]
        save_queue(work, keep)
    return len(entries) - len(keep)


def fail_interrupted(work: Path, track: str) -> list[QueueEntry]:
    """Mark ``track``'s ``running`` entries failed: their runner is known to be dead.

    A drain that died after persisting "running" cannot clean up its entry, so the row says
    ``running`` forever and the volume can be neither re-added nor served. The caller asserts
    the precondition — no live drain owns these rows. The note distinguishes the two ways a
    serve leg dies, because they call for opposite recoveries: a publish-owed marker means the
    bake LANDED and only the publish is missing, while its absence means the bake did not
    finish. Both callers settle owed publishes FIRST where they can, so a row still carrying a
    marker here gets the instruction rather than the action.
    """
    orphaned = [e for e in load_queue(work) if e.track == track and e.status == "running"]
    if orphaned:
        now = time.time()
        leg = _LEG_OF[track]
        for entry in orphaned:
            entry.status = "failed"
            entry.finished = now
            if track == "serve" and viewer_publish.publish_owed(entry.volume, work) is not None:
                # --work is named explicitly: it defaults to `work`, and a drain
                # against any other tree would otherwise be handed a command that
                # publishes from the wrong one.
                entry.note = (
                    f"bake completed; the publish did not run. Finish it without re-baking: "
                    f"autogeoref publish {entry.volume} --work {work} --city <city.toml>"
                    f" - see {entry.log}"
                )
            else:
                entry.note = f"{leg} interrupted before completion - see {entry.log}"
        persist(work, orphaned)
    return orphaned
