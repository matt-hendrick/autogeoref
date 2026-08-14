"""The match stage: fit every sheet's street reads against the centerline index.

Persists the constraints it used to ``volume-constants.json`` so rescue and
escalate pin the same scale and rotation without re-deriving them.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from ..name_match import write_name_match
from ..paths import atomic_write_text, write_result
from ..sheet_inputs import load_sheet_inputs
from ..volume import (
    VolumeConstraints,
    constraints_for_page,
    constraints_from_constants,
    derive_constraints,
    match_sheet,
    status_ok,
)
from ..volume_constants import persisted_constants

if TYPE_CHECKING:
    from ..centerlines import CenterlineIndex
    from ..config.model import VolumeConfig
    from ..paths import VolumePaths

logger = logging.getLogger(__name__)


def stage_match(
    paths: VolumePaths,
    index: CenterlineIndex,
    vol: VolumeConfig,
    skip_committed: bool = True,
) -> dict[str, dict[str, Any]]:
    """Two-pass match; never clobbers a committed record with a rejection.

    The constraints used (pinned or pass-1-derived) are persisted to
    ``volume-constants.json`` so later stages (rescue) can pin the same
    scale/rotation without re-deriving or requiring config entries. A
    resumed run (``skip_committed``) reuses that persisted file instead of
    re-running the pass-1 RANSAC over every sheet: JSON round-trips the
    medians exactly, so the reconstructed windows are the ones the first
    run derived (rescue and escalate already trust the same file).
    """
    sheets = load_sheet_inputs(paths)
    if not sheets:
        # Matching requires at least one readable annotation.
        raise RuntimeError(
            f"{vol.identifier}: no annotations to match — {paths.annotations} has no "
            "readable page annotations, so there is nothing to fit against the "
            "centerlines. The `annotate` stage produces them and runs by default; it "
            "was skipped (--no-annotate?) or every page it tried FAILED (look for "
            "p<N>.failed.json markers, and delete one to retry that page)."
        )
    aliases = index.aliases
    constraints: VolumeConstraints | None = None
    if vol.scale_m_per_px is not None and vol.rotation_deg is not None:
        constraints = constraints_from_constants(vol.scale_m_per_px, vol.rotation_deg)
    elif skip_committed:
        recorded = persisted_constants(paths)
        if recorded is not None:
            constraints = constraints_from_constants(*recorded)
            logger.info(
                "%s: reusing persisted constants scale=%s rot=%s",
                vol.identifier,
                constraints.scale_median,
                constraints.rotation_median,
            )
    if constraints is None:
        constraints = derive_constraints(
            sheets, index, aliases, page_scale_multiples=vol.page_scale_multiples
        )
        logger.info(
            "%s: derived constraints scale=%s rot=%s",
            vol.identifier,
            constraints.scale_median,
            constraints.rotation_median,
        )
    if constraints.scale_median is not None and constraints.rotation_median is not None:
        atomic_write_text(
            paths.constants,
            json.dumps(
                {
                    "scale_m_per_px": constraints.scale_median,
                    "rotation_deg": constraints.rotation_median,
                },
                indent=2,
            ),
        )
    paths.results.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for sheet in sheets:
        out_path = paths.results / f"p{sheet.page}.json"
        if skip_committed and out_path.exists():
            old = json.loads(out_path.read_text())
            if status_ok(str(old.get("status", ""))):
                results[sheet.page] = old
                continue
        # a page declared as printed at another scale gets the volume's window
        # RE-CENTERED (never widened) on its own scale; every other page is
        # untouched — see constraints_for_page
        sheet_constraints = constraints_for_page(sheet.page, constraints, vol.page_scale_multiples)
        record = match_sheet(sheet, index, sheet_constraints, aliases)
        # Preserve an existing committed record.
        if not status_ok(str(record.get("status", ""))) and out_path.exists():
            old = json.loads(out_path.read_text())
            if old.get("layer"):
                results[sheet.page] = old
                continue
        write_result(out_path, record)
        results[sheet.page] = record
    # the alias-gap tripwire: this is the one place that holds both the index
    # and every sheet's reads, so the metric is free here and unaffordable in
    # the report (two of its call sites have no index and must not pay a
    # citywide-GeoJSON parse for one). Recomputed unconditionally — a resumed
    # run still loaded every sheet above, and the count is cheap.
    try:
        write_name_match(paths, vol.identifier, sheets, index)
    except Exception:  # noqa: BLE001 - see below
        # an ADVISORY diagnostic runs last and cannot fail the stage: every
        # result above is already on disk, and losing the whole run's rescue,
        # seam and report over an unwritable metric would be the tail wagging
        # the dog. The report simply omits the metric, exactly as it does for
        # a volume matched before the sidecar existed.
        logger.warning("%s: name-match sidecar not written", vol.identifier, exc_info=True)
    return results
