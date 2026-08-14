"""Advisory junction-snap verification stage.

Runs the drawn-junction placement verifier over every rescue-family placement — rescued,
corroborated, or provisional-revoked — and records the verdict as ``junction_snap``.
ADVISORY ONLY: no status changes. It produces the evidence stream for
:mod:`autogeoref.verified_accept`, which decides whether a verdict becomes an accept vote.

``supports`` is recorded as JSON ``true`` (support) or ``null`` (ABSTAIN — no junction
evidence either way). It is never ``false``: this channel supports or abstains and does
not refute. A ``skipped`` record is a different fact again — extraction failed outright —
and is kept distinct on purpose.

Cheap by construction: only rescue-family pages are scored, never strict accepts.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .affine import TO_3857
from .paths import VolumePaths, iter_results, small_sheet_entry, write_result
from .seam import sheet_fit_from_result
from .volume import REVOKED_PREFIX, STATUS_CORROBORATED, STATUS_RESCUED

logger = logging.getLogger(__name__)

#: Statuses whose placements get an advisory junction verdict.
RESCUE_FAMILY = (STATUS_RESCUED, STATUS_CORROBORATED)


def stage_junction_verify(
    paths: VolumePaths,
    centerline_features: list[dict[str, Any]],
    bounds_4326: tuple[float, float, float, float],
) -> dict[str, dict[str, Any]]:
    """Attach advisory ``junction_snap`` verdicts to rescue-family results.

    paths: The volume's work tree (results/sheets/manifest used). centerline_features:
    Centerline GeoJSON features (4326). bounds_4326: Volume search bounds; converted to 3857 to
    clip the scoring world. Returns ``{page: verdict_record}`` for the pages that were scored.
    """
    # lazy import: cv2/skimage are the optional [cv] extra
    from .junction_snap import (
        JunctionSnapError,
        extract_junctions,
        extraction_in_source_frame,
        verify_placement,
        world_from_centerlines,
    )

    minx, miny = TO_3857.transform(bounds_4326[0], bounds_4326[1])
    maxx, maxy = TO_3857.transform(bounds_4326[2], bounds_4326[3])
    world = world_from_centerlines(centerline_features, bounds_3857=(minx, miny, maxx, maxy))
    manifest = json.loads(paths.manifest.read_text())

    verdicts: dict[str, dict[str, Any]] = {}
    for page, r, rp in iter_results(paths):
        status = str(r.get("status", ""))
        if not (status in RESCUE_FAMILY or status.startswith(REVOKED_PREFIX)):
            continue
        # no rotation skip: extraction_in_source_frame turns junctions read on
        # an orientation-normalized (upright) small back into the source frame
        # the record affine uses, so rotated scans keep this evidence channel
        entry = small_sheet_entry(
            paths, manifest, page, stage="junction verify", skip_rotated=False
        )
        if entry is None:
            continue
        info, small = entry
        fit = sheet_fit_from_result(page, r)
        record: dict[str, Any]
        if fit is None:
            record = {"skipped": "no usable GCPs in record"}
        else:
            small_to_full = 1.0 / float(info["scale"])
            try:
                extraction = extraction_in_source_frame(
                    extract_junctions(small), int(info.get("rotation_applied", 0))
                )
                verdict = verify_placement(
                    extraction,
                    fit.coef,
                    world,
                    small_to_full=small_to_full,
                )
                record = {
                    "supports": verdict.supports,
                    "score_at_prior": round(verdict.score_at_prior, 4),
                    "separation_ratio": round(verdict.separation_ratio, 3),
                    "best_offset_m": round(verdict.best_offset_m, 1),
                    "n_junctions": extraction.n_junctions,
                }
            except JunctionSnapError as exc:
                record = {"skipped": str(exc)}
        r["junction_snap"] = record
        write_result(rp, r)
        verdicts[page] = record
        logger.info("p%s junction_snap: %s", page, record)
    # SKIPPED and ABSTAINED are different facts and are counted apart: a skip is
    # an EXTRACTION failure (too few junctions to score at all), an abstain is a
    # successful extraction that yielded no evidence either way. Collapsing them
    # would destroy the audit trail — a page that scored and said nothing looks
    # identical to a page the verifier never managed to read.
    supported = sum(1 for v in verdicts.values() if v.get("supports") is True)
    skipped = sum(1 for v in verdicts.values() if "skipped" in v)
    abstained = len(verdicts) - supported - skipped
    logger.info(
        "junction verify: %d scored (%d support, %d abstain, %d skipped; never refutes)",
        len(verdicts),
        supported,
        abstained,
        skipped,
    )
    return verdicts
