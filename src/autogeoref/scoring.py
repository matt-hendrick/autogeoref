"""Ground-truth scoring engine, and the sidecar it writes scores to.

A score lives BESIDE a result record, never on it: the record says what the
pipeline did, the score says something about it. Nothing a run reads touches
this file, so a score can take a placement away afterwards and can never hand
one out.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .affine import fit_affine, gcps_from_geojson, grid_rmse_m
from .paths import VolumePaths, atomic_write_text

logger = logging.getLogger(__name__)

#: Commit gate: grid-RMSE vs human GCPs (true ground meters). Reporting only —
#: no run stage reads it and nothing in the product acts on it.
GT_COMMIT_RMSE_M = 15.0


def score_record_vs_ground_truth(
    record: Mapping[str, Any],
    info: Mapping[str, Any] | None,
    layer: Mapping[str, Any] | None,
    correction_lat: float,
) -> float | None:
    """Grid RMSE of a record's exported GCPs against human GCPs, or None.

    ``correction_lat`` comes from :func:`bounds.mercator_correction_lat` over
    the volume's ground truth — the caller holds the full GT map; this leaf
    sees one layer.
    """
    if layer is None or info is None or not record.get("gcps_geojson"):
        return None
    width, height = info["full_size"]
    return grid_rmse_m(
        fit_affine(gcps_from_geojson(dict(record)["gcps_geojson"])),
        fit_affine(gcps_from_geojson(dict(layer)["gcps_geojson"])),
        width,
        height,
        mercator_correction_lat=correction_lat,
    )


def record_digest(result_path: Path) -> str | None:
    """SHA-256 of the result record a score was taken from, or None if unreadable.

    Stamped into every sidecar entry so a score cannot outlive its subject: a
    re-place rewrites the record, the digest stops matching, and the score
    disappears rather than describing a placement that no longer exists.
    """
    try:
        return hashlib.sha256(result_path.read_bytes()).hexdigest()
    except OSError:
        return None


def load_sidecar(paths: VolumePaths) -> dict[str, Any]:
    """The volume's score sidecar, or an empty mapping when it has none."""
    if not paths.scores.exists():
        return {}
    try:
        loaded: Any = json.loads(paths.scores.read_text())
    except (OSError, ValueError):
        logger.warning("%s: unreadable sidecar; the volume reads as unscored", paths.scores)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def load_scores(paths: VolumePaths) -> dict[str, float]:
    """``{page: rmse_vs_human_m}`` for pages whose record is still the scored one.

    An entry whose ``record_sha256`` no longer matches the record on disk is
    DROPPED, so a report run after a re-place says "unscored" instead of
    publishing a median over placements that have since moved. Score again.
    """
    pages = load_sidecar(paths).get("pages") or {}
    out: dict[str, float] = {}
    stale: list[str] = []
    for page, entry in pages.items():
        value = (entry or {}).get("rmse_vs_human_m")
        if value is None:
            continue
        if (entry or {}).get("record_sha256") != record_digest(paths.results / f"p{page}.json"):
            stale.append(str(page))
            continue
        out[str(page)] = float(value)
    if stale:
        logger.warning(
            "%s: %d score(s) describe records that have since changed and are ignored "
            "(re-run `autogeoref score`): %s",
            paths.scores,
            len(stale),
            ", ".join(sorted(stale)),
        )
    return out


def write_sidecar(paths: VolumePaths, payload: Mapping[str, Any]) -> Path:
    """Publish the sidecar atomically, sorted so re-scoring diffs cleanly."""
    return atomic_write_text(paths.scores, json.dumps(payload, indent=2, sort_keys=True))


def drop_score(paths: VolumePaths, page: str) -> bool:
    """Forget one page's score; True when there was one.

    For a caller that has REPLACED the placement the score described — the
    score no longer says anything about what is on disk, and a stale one would
    keep judging a sheet its subject no longer is.
    """
    sidecar = load_sidecar(paths)
    pages = sidecar.get("pages")
    if not isinstance(pages, dict) or page not in pages:
        return False
    del pages[page]
    write_sidecar(paths, sidecar)
    return True


def median_rmse(scores: Mapping[str, float]) -> float | None:
    """Median grid-RMSE (true ground meters) over the scored pages."""
    if not scores:
        return None
    return float(statistics.median(scores.values()))


__all__ = [
    "GT_COMMIT_RMSE_M",
    "drop_score",
    "load_scores",
    "load_sidecar",
    "median_rmse",
    "record_digest",
    "score_record_vs_ground_truth",
    "write_sidecar",
]
