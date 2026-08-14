"""Per-volume progress, derived from the work tree — never from the queue file."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..paths import VolumePaths
from ..status import annotation_reads, volume_funnel
from .store import QueueEntry, _safe_volume, load_queue


@dataclass(frozen=True)
class VolumeProgress:
    """Live progress for one volume, derived entirely from its work tree.

    ``accepted``/``flagged`` come from :func:`status.volume_funnel`, which defers
    to ``report.build_report`` — the one owner of what counts as accepted. The
    queue view must never re-derive that: a monitor that disagrees with
    ``autogeoref report`` about how many sheets landed is worse than no monitor.
    """

    volume: str
    pages: int  # addressable pages in the manifest (the denominator)
    reads: int  # pages the model has read == budget already spent
    #: ``*.failed.json`` retry markers on disk — calls that did NOT land. They
    #: make a plain retry fail instantly by design (annotate_volume.plan treats
    #: a marked page as cached), so the view offers to clear them, and this
    #: count is what that clearing would re-spend.
    failed_markers: int
    results: int  # result records written
    accepted: int | None  # per report.build_report; None until results exist
    flagged: int | None
    stage: str | None  # newest marker's stage
    stage_status: str | None  # ok | failed | fresh | disabled | skipped
    stage_at: float | None  # when that marker was written
    error: str | None  # the newest marker's error, if it failed


def _newest_marker(markers: Path) -> tuple[float, dict[str, Any]] | None:
    """The newest stage marker in ``markers``, as ``(when, marker)``."""
    newest: tuple[float, dict[str, Any]] | None = None
    for mp in markers.glob("*.marker.json"):
        try:
            m = json.loads(mp.read_text())
        except json.JSONDecodeError:
            continue
        when = m.get("finished") or m.get("started") or mp.stat().st_mtime
        if newest is None or when > newest[0]:
            newest = (when, m)
    return newest


def volume_progress(work: Path, volume: str) -> VolumeProgress:
    """Read a volume's live state off its work tree. Never trusts the queue file."""
    volume = _safe_volume(volume)
    paths = VolumePaths(work / volume)
    pages = 0
    if paths.manifest.exists():
        try:
            manifest = json.loads(paths.manifest.read_text())
        except (OSError, ValueError):
            manifest = {}
        # "_orientation_normalized" is a policy sentinel, not a page (prep)
        pages = sum(1 for k in manifest if not k.startswith("_"))
    # status.annotation_reads owns the budget-accounting rule (a `*.failed.json`
    # marker is a call that did NOT land, and several per-model sidecars are one
    # page read) — a second copy of it here would drift into flattering itself
    reads = annotation_reads(paths.annotations) or 0
    failed_markers = (
        sum(1 for _ in paths.annotations.glob("*.failed.json")) if paths.annotations.is_dir() else 0
    )
    funnel, _damaged, recorded = volume_funnel(volume, paths.results)
    results = len(recorded)
    accepted = funnel.accepted_total if funnel else None
    flagged = funnel.flagged if funnel else None
    stage = stage_status = error = None
    stage_at: float | None = None
    if paths.markers.exists():
        newest = _newest_marker(paths.markers)
        if newest is not None:
            stage_at, m = newest
            stage = m.get("stage")
            stage_status = m.get("status")
            err = m.get("error")
            error = err.strip().splitlines()[0] if err else None
    return VolumeProgress(
        volume=volume,
        pages=pages,
        reads=reads,
        failed_markers=failed_markers,
        results=results,
        accepted=accepted,
        flagged=flagged,
        stage=stage,
        stage_status=stage_status,
        stage_at=stage_at,
        error=error,
    )


def iter_display(work: Path) -> Iterator[tuple[QueueEntry, VolumeProgress]]:
    """Queue entries paired with their live filesystem progress, for the views."""
    for entry in load_queue(work):
        yield entry, volume_progress(work, entry.volume)


def board(work: Path) -> dict[str, Any]:
    """The whole board as plain data — the one payload both views render.

    The terminal table and the web page are two renderings of THIS, so they
    cannot drift into disagreeing about the same tree.
    """
    rows = []
    for entry, prog in iter_display(work):
        rows.append({**asdict(entry), "progress": asdict(prog)})
    return {"generated": time.time(), "entries": rows}
