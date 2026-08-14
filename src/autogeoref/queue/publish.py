"""Settling the publish a dead or bypassed serve drain left owed."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .. import drain_lock
from ..paths import VolumePaths
from ..viewer import publish as viewer_publish
from . import store

if TYPE_CHECKING:
    from ..viewer.publish import PublicationConfig
    from .store import QueueEntry

logger = logging.getLogger(__name__)


def finish_owed_publishes(work: Path, publication: PublicationConfig) -> list[QueueEntry]:
    """Land the publish a dead serve drain's COMPLETED bake left owed.

    A serve leg is a long bake followed by a short publish, and a drain killed between them
    leaves the archive built and unpublished — hours of finished work reachable by seconds of
    copying. Publishing runs the same transactional landing as the original attempt, so it
    proves the claim. The caller must hold this track's fresh drain lock, which is why this is a
    drain's job and not the console's: it is a large copy plus a manifest rebuild. Must run
    BEFORE ``fail_interrupted``.
    """
    settled: list[QueueEntry] = []
    for entry in store.load_queue(work):
        if entry.track != "serve" or entry.status != "running":
            continue
        if viewer_publish.publish_owed(entry.volume, work) is None:
            continue
        entry.finished = time.time()
        try:
            viewer_publish.publish_volume(entry.volume, publication)
        except Exception as exc:  # noqa: BLE001 - see below
            # NOT just PublicationError: publish_volume also reaches config loading
            # and lock-file creation, which raise ConfigError/OSError outside its own
            # guard. Letting one of those escape would abort run_queue INSIDE the
            # drain lock, before fail_interrupted — leaving the marker and the
            # `running` row in place so every later drain died at the same line and
            # nothing on the queue ever drained again.
            entry.status = "failed"
            entry.note = f"bake completed, but the recovery publish failed: {exc} - see {entry.log}"
        else:
            # exit_code is the BAKE's, and the marker is only ever written after
            # the bake exited 0 — this drain did not watch it, but the artifact
            # is the record that it did.
            entry.exit_code = 0
            entry.status = "done"
            entry.note = "bake completed before the drain stopped; published on recovery"
        # PERSIST PER ENTRY, and before the report refresh — for the same reason
        # run_queue's own loop does it per volume. The publish above already cleared
        # the marker, so a kill before this write would leave a published archive
        # behind a `running` row with no evidence left: the exact incident this
        # function exists to end, re-created by its own recovery.
        store.persist(work, [entry])
        if entry.status == "done":
            _refresh_report(work, entry.volume, publication)
        settled.append(entry)
    return settled


def settle_published(work: Path, volume: str) -> QueueEntry | None:
    """Close the stranded serve row a HAND publish just finished, if there is one.

    The operator recovery is ``autogeoref publish``, and a publish clears the
    marker. Without this, following that instruction would destroy the only
    evidence that the bake had completed, and the next drain would stamp the
    still-``running`` row as interrupted — the operator does the right thing and
    is punished with a bogus re-bake. Only ``running`` rows, and never while a
    serve drain is live: that row may be a leg the drain owns.
    """
    if drain_lock.live_drain(work, "serve") is not None:
        return None
    stranded = [
        e
        for e in store.load_queue(work)
        if e.volume == volume and e.track == "serve" and e.status == "running"
    ]
    if not stranded:
        return None
    # ALL of them, not the newest: `add` cannot create a second live serve row, so
    # more than one means a hand-edited or older-version file — and settling only
    # one would leave the rest `running` forever with nothing to explain them.
    now = time.time()
    for entry in stranded:
        entry.status = "done"
        entry.finished = now
        entry.exit_code = 0
        entry.note = "bake completed before the drain stopped; published by hand"
    store.persist(work, stranded)
    return stranded[-1]


def _refresh_report(work: Path, volume: str, publication: PublicationConfig) -> None:
    from ..config.load import load_city_config
    from ..stages.report import stage_report

    try:
        overview_pages = load_city_config(publication.city_toml).volume(volume).overview_pages
        stage_report(
            VolumePaths(root=work / volume),
            volume,
            tiles_root=publication.tiles_root,
            city_toml=publication.city_toml,
            overview_pages=overview_pages,
        )
    except Exception as exc:  # noqa: BLE001 - see below
        # the publish itself landed; a report refresh failure is a note, not a rollback
        # — and never a reason to abort. This runs inside a drain holding its lock and
        # after the row is already persisted, so an escaping error would kill the drain
        # before it settled anyone else's orphans, for a cosmetic staleness note.
        logger.warning("%s: report refresh after publish failed (%s)", volume, exc)
