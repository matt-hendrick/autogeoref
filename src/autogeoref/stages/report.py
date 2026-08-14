"""The report stage: assemble one volume's report.json and report.md."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ..name_match import alias_gap_note, load_name_match
from ..paths import atomic_write_text
from ..report import build_report, load_results_dir, report_json, report_markdown
from ..scoring import load_scores

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

    from ..paths import VolumePaths


def stage_report(
    paths: VolumePaths,
    volume: str,
    *,
    tiles_root: Path | None = None,
    city_toml: Path | None = None,
    overview_pages: Collection[str] = (),
) -> Path:
    """``city_toml`` names the ``deploy/tiles/`` directory this city publishes
    into, which is what the serve-staleness note is judged against; without it
    the default directory is assumed."""
    from ..mask.qa import load_masks_qa, qa_note
    from ..status import (
        SERVE_STALE,
        newest_committed,
        own_archives,
        serve_staleness,
    )
    from ..status import stale_note as serve_stale_note
    from ..viewer.config import SERVING_DIRS, load_viewer_config

    results = load_results_dir(paths.results)
    seam_record = json.loads(paths.seam_deltas.read_text()) if paths.seam_deltas.exists() else None
    # the bake's mask QA verdict rides along when the volume has been baked:
    # a flagged mask is exactly the kind of "accepted but wrong-looking" state
    # the report exists to keep visible
    qa = load_masks_qa(paths.masks)
    note = qa_note(qa) if qa else None
    notes = [note] if note else []
    if tiles_root is not None:
        # so a report generated after placement says plainly that the served
        # archive predates a record it would serve
        serving = load_viewer_config(city_toml).serving_dirs if city_toml else SERVING_DIRS
        archives = own_archives(tiles_root, tuple(dict.fromkeys((serving[0], *SERVING_DIRS))))
        verdict, record = serve_staleness(newest_committed(paths.results), archives.get(volume))
        if verdict == SERVE_STALE and record:
            notes.append(serve_stale_note(record))
    # advisory only: a volume whose sheets speak a vocabulary the centerline
    # index does not hold has no anchors to fit, and the funnel alone does not
    # say so. A pre-sidecar volume keeps the zero-candidate half and stays
    # silent about the missing match rate.
    gap = alias_gap_note(load_name_match(paths), results)
    if gap:
        notes.append(gap)
    report = build_report(
        volume,
        results,
        seam_record,
        notes=notes or None,
        overview_pages=overview_pages,
        # the four human-pin counters read the scoring pass's sidecar, which a
        # volume nobody has scored simply does not have
        scores=load_scores(paths),
    )
    atomic_write_text(paths.root / "report.json", report_json(report))
    atomic_write_text(paths.root / "report.md", report_markdown(report))
    return paths.root / "report.json"
