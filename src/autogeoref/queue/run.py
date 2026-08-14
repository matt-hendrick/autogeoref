"""The drains: one entry, one track's queue, or all three tracks concurrently."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..drain_lock import drain_lock
from ..viewer import publish as viewer_publish
from ..viewer.publish import PublicationConfig, PublicationError
from . import command, publish, store
from .command import DrainContext
from .store import _LEG_OF, TRACKS, QueueEntry, QueueError, log_path

logger = logging.getLogger(__name__)


def default_serve_lanes() -> int:
    """A conservative default for how many volumes the serve queue bakes at once.

    NOT ``os.cpu_count()``: GDAL is already internally multithreaded, so one lane per core
    oversubscribes and thrashes; baking full-res sheets is memory-heavy and this workload has
    hit OOM before. So: half the AFFINITY-AWARE core count, floored at 1 and capped to leave
    headroom for GDAL's own threads and RAM. A floor to build up from, not a ceiling —
    ``--serve-lanes N`` overrides it.
    """
    try:
        cores = len(os.sched_getaffinity(0))  # Linux: respects cgroup/affinity
    except AttributeError:  # pragma: no cover - non-Linux
        cores = os.cpu_count() or 2
    return min(4, max(1, cores // 2))


def _finish_fetch(entry: QueueEntry, ctx: DrainContext) -> None:
    """A fetched volume promotes itself to the place queue."""
    entry.status = "done"
    entry.note = "fetched; promoted to the place queue"
    # Same ordering rule as the place promotion: the terminal fetch row
    # lands first, because `add` refuses a place enqueue while a NON-terminal
    # entry for the volume exists.
    store.persist(ctx.work, [entry])
    try:
        store.add(ctx.work, entry.volume, "place", then_serve=entry.then_serve)
    except QueueError as exc:
        # The pages are on disk either way, so a refused promotion is a note.
        # The usual cause is a place entry already queued for this volume.
        entry.note = f"fetched, but auto-promote to place failed: {exc}"


def _finish_place(entry: QueueEntry, ctx: DrainContext) -> None:
    """A placed volume promotes to the serve queue, or parks at needs-review."""
    if entry.then_serve:
        entry.status = "done"
        entry.note = "placed; promoted to the serve queue"
        # PERSIST the terminal place entry BEFORE promoting: `add` refuses a serve
        # enqueue while a NON-terminal place entry for the volume exists (place+serve
        # at once bakes a moving funnel), and until this write lands the on-disk copy
        # still reads "running".
        store.persist(ctx.work, [entry])
        try:
            store.add(ctx.work, entry.volume, "serve")  # promote: the serve drain will bake it
        except QueueError as exc:
            # already serving / a stale serve entry / a race — placed all the same,
            # so this is a note, not a failure. The operator can re-add serve.
            entry.note = f"placed, but auto-promote to serve failed: {exc}"
    else:
        entry.status = "needs-review"
        entry.note = (
            f"placed; you asked to review it: autogeoref review --volume {entry.volume}"
            f"  (then send it to the serve queue)"
        )


def _finish_serve(entry: QueueEntry, ctx: DrainContext) -> None:
    """A finished bake publishes now, with the owed-publish marker as its net."""
    if ctx.publication is None:
        entry.status = "failed"
        entry.note = "serve finished but has no publication configuration"
        return
    # RECORD THE DEBT BEFORE PAYING IT. The bake just finished — hours of GDAL — and the publish
    # ahead takes seconds. A drain killed in that gap used to leave a `running` row that read as
    # "the bake was interrupted", and recovery meant re-baking work that was already on disk
    # (the back half cannot skip it: warp/mask/mosaic declare no outputs, so `dag.is_fresh`
    # never holds). The marker is the only thing that survives a SIGKILL here, so it is written
    # first and `publish_volume` clears it on success.
    viewer_publish.record_publish_owed(
        entry.volume, ctx.work, baked_at=entry.finished, log=entry.log
    )
    try:
        viewer_publish.publish_volume(entry.volume, ctx.publication)
    except PublicationError as exc:
        entry.status = "failed"
        entry.note = f"serve published nothing: {exc} — see {entry.log}"
    else:
        entry.status = "done"
        entry.note = None
        # PERSIST BEFORE THE REFRESH. The publish above cleared the marker,
        # so from here on nothing on disk remembers the debt — and the
        # refresh below takes real seconds. A kill in between would strand a
        # published archive behind a `running` row with no evidence, which is
        # precisely the state finish_owed_publishes can no longer recover.
        store.persist(ctx.work, [entry])
        # the report written at placement carries the serve-staleness
        # note this publish just resolved; refresh so it stops
        # claiming a serve pass is needed
        publish._refresh_report(ctx.work, entry.volume, ctx.publication)


def _execute(entry: QueueEntry, ctx: DrainContext) -> QueueEntry:
    """Run ONE queue entry's single leg, and — except on serve — promote it.

    A fetch entry that succeeds enqueues a place entry for the same volume,
    carrying its ``then_serve`` intent forward. A place entry that succeeds
    either enqueues a serve entry (``then_serve`` — the auto-promote that makes
    "end to end" one action across three queues) or parks at ``needs-review``
    (the ``--review`` diagnostic). A serve entry that succeeds is simply done.
    Each promoted entry is picked up by the next track's drain running alongside
    — that is the parallelism.
    """
    leg = _LEG_OF[entry.track]
    entry.status = "running"
    entry.started = time.time()
    entry.log = str(log_path(ctx.work, entry.volume, leg))
    if entry.track == "serve":
        # A bake ABOUT TO RUN invalidates any standing claim that this tree holds a complete
        # unpublished archive. Clearing here keeps a marker's meaning tied to THIS run:
        # otherwise a drain killed mid-re-bake would find the stale marker on a `running` row
        # and "recover" by publishing the PREVIOUS archive. BEFORE the persist below, so no kill
        # can expose that pairing.
        viewer_publish.clear_publish_owed(entry.volume, ctx.work)
    # PERSIST BEFORE THE LEG. Every view re-reads the queue file, so an entry that is
    # only "running" in this process's memory reads as `queued` for the hour it takes
    # to run, and its log link would point at nothing. A drain is the thing you watch
    # from somewhere else.
    store.persist(ctx.work, [entry])
    try:
        code = command._run_leg(leg, entry, ctx)
    except OSError as exc:
        entry.status = "failed"
        entry.finished = time.time()
        entry.note = f"{leg} could not launch: {exc}"
        store.persist(ctx.work, [entry])
        return entry
    entry.exit_code = code
    entry.finished = time.time()
    if code != 0:
        entry.status = "failed"
        entry.note = f"{leg} failed (exit {code}) — see {entry.log}"
    elif entry.track == "fetch":
        _finish_fetch(entry, ctx)
    elif entry.track == "place":
        _finish_place(entry, ctx)
    else:  # serve leg
        _finish_serve(entry, ctx)
    return entry


#: How often a serve drain re-checks for volumes a place drain has just promoted,
#: while that place drain is still running. A few seconds: this queue drains a
#: handful of volumes an hour, not a second.
_SERVE_POLL_S = 3.0


def _resolve_lanes(track: str, lanes: int | None, publication: PublicationConfig | None) -> int:
    """The lane count a track actually drains at; raises on an unusable track.

    ``fetch`` and ``place`` are forced to one lane whatever the caller asked
    for (see :func:`run_queue`); an explicit override is logged, not honoured.
    """
    if track not in TRACKS:
        raise QueueError(f"unknown track {track!r}")
    if track == "serve" and publication is None:
        raise QueueError("serve queue needs a publication configuration")
    if track in ("fetch", "place"):
        if lanes not in (None, 1):
            logger.info("%s drains one volume at a time (asked for %d lanes)", track, lanes)
        return 1
    return default_serve_lanes() if lanes is None else lanes


def run_queue(
    ctx: DrainContext,
    *,
    track: str,
    lanes: int | None = None,
    follow_while: Callable[[], bool] | None = None,
    poll_s: float = _SERVE_POLL_S,
) -> list[QueueEntry]:
    """Drain ONE queue until it is empty. Returns the entries it touched.

    ``ctx`` carries what every drain thread needs verbatim — work tree, city, passthrough args,
    publication, abort — while ``track``, ``lanes``, ``follow_while`` and ``poll_s`` differ per
    worker. ``fetch`` and ``place`` drain one volume at a time (LOC conduct is one request lane,
    and two concurrent place runs would write one volume's cache twice), while ``serve`` honours
    ``lanes``. ``follow_while`` makes an EMPTY queue keep polling while the predicate holds, so
    a drain launched ALONGSIDE its feeder picks up promotions as they land, and
    ``stop_on_failure`` leaves every unstarted entry ``queued``. Holds the drain lock throughout.
    """
    lanes = _resolve_lanes(track, lanes, ctx.publication)
    abort = ctx.abort

    def _stop_for(entry: QueueEntry) -> None:
        """Raise the shared stop after ``entry`` failed, once and loudly."""
        if not abort.is_set():
            logger.warning(
                "%s queue: stopping on %s's failure — anything not already running is "
                "left queued and a re-run resumes it. %s itself is now `failed`, which no "
                "drain ever selects: re-add it or it leaves the batch silently",
                track,
                entry.volume,
                entry.volume,
            )
        abort.set()

    def _run_one(entry: QueueEntry) -> QueueEntry | None:
        """Run one entry unless the stop beat it to the start line.

        Returning None is what keeps ``--stop-on-failure`` honest on a multi-lane
        track: every pending entry is submitted at once (that is where the
        throughput comes from), so withholding the work at the START of the task
        is the only place an unstarted entry can be left ``queued`` rather than
        run into the same wall.
        """
        if ctx.stop_on_failure and abort.is_set():
            return None
        _execute(entry, ctx)
        if ctx.stop_on_failure and entry.status == "failed":
            _stop_for(entry)
        return entry

    touched: list[QueueEntry] = []
    with drain_lock(ctx.work, track):
        # We hold this track's new drain lock, so no live worker can own a
        # "running" row here — anything still saying so belongs to a dead drain.
        # Settle what that drain merely left unpublished BEFORE failing the rest:
        # its bake is on disk and finishing it costs seconds. They count as touched
        # — this drain published an archive and moved a row to `done`, so returning
        # an empty list would have it report "nothing to do" for real work.
        if track == "serve" and ctx.publication is not None:
            touched += publish.finish_owed_publishes(ctx.work, ctx.publication)
        store.fail_interrupted(ctx.work, track)
        while not (ctx.stop_on_failure and abort.is_set()):
            pending = [
                e for e in store.load_queue(ctx.work) if e.track == track and e.status == "queued"
            ]
            if not pending:
                # keep following only while the sibling upstream worker says it is still
                # running: a promotion it makes lands BEFORE it signals done, so an empty
                # queue + "upstream done" is genuinely done, and an upstream worker that
                # died cannot wedge us (its `finally` flips the predicate — see run_all)
                if follow_while is not None and follow_while():
                    time.sleep(poll_s)
                    continue
                break
            # fail this batch before spending anything if a passthrough arg is illegal
            for entry in pending:
                command._command(_LEG_OF[track], entry, ctx)

            if lanes == 1:
                for entry in pending:
                    if _run_one(entry) is None:
                        break
                    touched.append(entry)
                    # persist after EVERY volume: a drain killed halfway must not
                    # forget what it already did, or a re-run re-spends the budget
                    store.persist(ctx.work, touched)
            else:
                with ThreadPoolExecutor(max_workers=lanes) as pool:
                    futs = {pool.submit(_run_one, e): e for e in pending}
                    for fut in as_completed(futs):
                        if fut.result() is None:
                            continue  # never started: its row is still `queued`
                        touched.append(futs[fut])
                        store.persist(ctx.work, touched)
    if not touched:
        logger.info("%s queue: nothing to do", track)
    return touched


def run_all(
    work: Path,
    city: Path,
    *,
    place_lanes: int = 1,
    serve_lanes: int | None = None,
    extra: Sequence[str] = (),
    nice: int = 10,
    publication: PublicationConfig | None = None,
    poll_s: float = _SERVE_POLL_S,
    stop_on_failure: bool = False,
) -> list[QueueEntry]:
    """Drain every queue CONCURRENTLY, one worker thread per track.

    What a bare ``queue --run`` starts: fetch downloads ahead while place works through the
    middle and serve bakes behind. Each thread takes its own per-track drain lock, so this
    composes with a separately-started drain, which loses the race for its lock. Each downstream
    worker follows its FEEDER via an in-process Event the feeder sets in a ``finally``, so a
    worker that DIES stops the ones below it rather than leaving them polling a queue nothing
    will feed. ``stop_on_failure`` is shared across all three, not per-track: a model limit that
    stops place must also stop FETCH. Work in flight always finishes.
    """
    command._reject_owned_overrides(extra)
    if publication is None:
        raise QueueError("run_all needs a publication configuration for the serve queue")

    fetch_done = threading.Event()
    place_done = threading.Event()
    # ONE context for all three workers: its `abort` is the shared stop.
    ctx = DrainContext(
        work=work,
        city=city,
        extra=extra,
        nice=nice,
        publication=publication,
        stop_on_failure=stop_on_failure,
    )

    def _leading(
        track: str, done: threading.Event, **kwargs: Any
    ) -> Callable[[], list[QueueEntry]]:
        """A worker that signals ``done`` on the way out, however it leaves."""

        def _worker() -> list[QueueEntry]:
            try:
                return run_queue(ctx, track=track, **kwargs)
            finally:
                # even on exception: a downstream worker must not wait on a corpse
                done.set()

        return _worker

    with ThreadPoolExecutor(max_workers=len(TRACKS)) as pool:
        futures = [
            pool.submit(_leading("fetch", fetch_done)),
            pool.submit(
                _leading(
                    "place",
                    place_done,
                    lanes=place_lanes,
                    follow_while=lambda: not fetch_done.is_set(),
                    poll_s=poll_s,
                )
            ),
            pool.submit(
                run_queue,
                ctx,
                track="serve",
                lanes=serve_lanes,
                follow_while=lambda: not place_done.is_set(),
                poll_s=poll_s,
            ),
        ]
        touched: list[QueueEntry] = []
        errors: list[Exception] = []
        for fut in as_completed(futures):
            try:
                touched.extend(fut.result())
            except Exception as exc:  # noqa: BLE001 - settle every worker, surface the first
                errors.append(exc)
    if errors:
        raise errors[0]
    return touched
