"""Joint translation adjustment ("mini bundle adjustment") for one volume.

Neighboring auto-georeferenced sheets can err in opposite directions, showing street offsets at
seams (~2x per-sheet error). This solves per-sheet translation deltas that pull shared street
corners together while an anchor term keeps the whole volume tied to the absolute fit.
Scale/rotation stay untouched.

A tie is two sheets whose GCP sets contain the SAME centerline node (world coordinates are
identical when both matched the same intersection); the mismatch between the sheets' *warped*
positions of that node is what the viewer shows at a seam. Synthetic rescue-corner GCPs carry no
cross-sheet information and are excluded.

The solve is unconditional: nothing here consults human pins. Whether a solve made the volume
better or worse against them is graded afterwards, by the scoring pass.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .affine import TO_3857, TO_4326, apply_affine, fit_affine_checked, gcps_from_geojson
from .rescue import is_synthetic_gcp

if TYPE_CHECKING:
    from numpy.typing import NDArray

ALPHA = 0.5  # anchor weight
MIN_SHIFT_M = 2.0  # apply threshold

#: (px, py, X3857, Y3857, is_synthetic)
SeamGcp = tuple[float, float, float, float, bool]
#: page -> (dx_m, dy_m)
Deltas = dict[str, tuple[float, float]]
Tie = tuple[str, tuple[float, float], str, tuple[float, float]]


def node_key(x: float, y: float) -> tuple[float, float]:
    """Exact-match identity of a centerline node from its 3857 world coords.

    THE node-identity convention that seam ties, corroboration, and the
    vouch pool all hinge on: two sheets that matched the same centerline
    intersection carry EXACTLY equal world coordinates, so a 0.1 m rounding
    is an exact-match key, never a distance tolerance. Every consumer must
    use this helper — a drifted copy silently breaks tie/vouch matching.
    """
    return (round(x, 1), round(y, 1))


@dataclass
class SheetFit:
    """A committed sheet's GCPs and the affine GDAL derives from them."""

    page: str
    gcps: list[SeamGcp]
    coef: NDArray[np.float64]  # 2x3 affine.py convention: [X, Y] = coef @ [1, px, py]


def sheet_fit_from_result(page: str, result: dict[str, Any]) -> SheetFit | None:
    """Build a :class:`SheetFit` from a recorded result's ``gcps_geojson``.

    The affine comes from :func:`autogeoref.affine.fit_affine_checked` — the
    ONE least-squares owner (byte-identical-replay contract lives there);
    this module only adds the per-GCP synthetic flag the seam/vouch logic
    needs. The shared implementation preserves replay consistency.
    """
    fc = result.get("gcps_geojson") or {}
    feats = fc.get("features") or []
    if len(feats) < 3:
        return None
    base = gcps_from_geojson(fc)
    gcps: list[SeamGcp] = [
        (px, py, x, y, is_synthetic_gcp(ft)) for (px, py, x, y), ft in zip(base, feats, strict=True)
    ]
    coef = fit_affine_checked(base)
    if coef is None:
        return None
    return SheetFit(page=page, gcps=gcps, coef=coef)


def build_ties(sheets: dict[str, SheetFit]) -> list[Tie]:
    """Shared centerline nodes -> pairwise tie observations."""
    node_map: dict[tuple[float, float], list[tuple[str, float, float]]] = {}
    for page, s in sheets.items():
        for px, py, x, y, synthetic in s.gcps:
            if synthetic:
                continue  # model-derived corners carry no cross-sheet information
            node_map.setdefault(node_key(x, y), []).append((page, px, py))
    ties: list[Tie] = []
    for uses in node_map.values():
        pages = {u[0] for u in uses}
        if len(pages) < 2:
            continue
        by_page: dict[str, tuple[float, float]] = {}
        for page, px, py in uses:
            by_page.setdefault(page, (px, py))
        plist = sorted(by_page)
        ties.extend((a, by_page[a], b, by_page[b]) for a, b in itertools.combinations(plist, 2))
    return ties


def solve(
    sheets: dict[str, SheetFit],
    ties: list[Tie],
    alpha: float = ALPHA,
) -> tuple[Deltas, list[float], list[float]]:
    """Per-axis LSQ: ties want warped positions to coincide; anchors want d=0.

    Returns ``(deltas, seam_mismatches_before_m, seam_mismatches_after_m)``.

    Empty-safe: a volume whose committed sheets are all withheld from the tie
    set (an overview-only volume) yields no sheets at all, and ``np.vstack``
    on zero rows raises — so that case answers "no deltas" here rather than
    relying on every caller's own guard.
    """
    if not sheets:
        return {}, [], []
    pages = sorted(sheets)
    idx = {p: i for i, p in enumerate(pages)}
    n = len(pages)
    rows: list[NDArray[np.float64]] = []
    rhs_x: list[float] = []
    rhs_y: list[float] = []
    mism_before: list[float] = []
    for pi, (pxi, pyi), pj, (pxj, pyj) in ties:
        xi, yi = apply_affine(sheets[pi].coef, pxi, pyi)
        xj, yj = apply_affine(sheets[pj].coef, pxj, pyj)
        row = np.zeros(n)
        row[idx[pi]], row[idx[pj]] = 1.0, -1.0
        rows.append(row)
        rhs_x.append(xj - xi)
        rhs_y.append(yj - yi)
        mism_before.append(math.hypot(xj - xi, yj - yi))
    for p in pages:  # anchor to the absolute fit
        row = np.zeros(n)
        row[idx[p]] = math.sqrt(alpha)
        rows.append(row)
        rhs_x.append(0.0)
        rhs_y.append(0.0)

    arr = np.vstack(rows)
    dx, *_ = np.linalg.lstsq(arr, np.array(rhs_x), rcond=None)
    dy, *_ = np.linalg.lstsq(arr, np.array(rhs_y), rcond=None)
    deltas: Deltas = {p: (float(dx[idx[p]]), float(dy[idx[p]])) for p in pages}

    mism_after: list[float] = []
    for pi, (pxi, pyi), pj, (pxj, pyj) in ties:
        xi, yi = apply_affine(sheets[pi].coef, pxi, pyi)
        xj, yj = apply_affine(sheets[pj].coef, pxj, pyj)
        ddx = (xi + deltas[pi][0]) - (xj + deltas[pj][0])
        ddy = (yi + deltas[pi][1]) - (yj + deltas[pj][1])
        mism_after.append(math.hypot(ddx, ddy))
    return deltas, mism_before, mism_after


def rms(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v) / len(v)) if v else 0.0


def shift_gcps_geojson(fc: dict[str, Any], dx_m: float, dy_m: float) -> None:
    """Shift every GCP's world coordinate by (dx, dy) 3857 meters, in place.

    The single owner of the seam-shift transform: the pipeline applies
    deltas with it and replay tooling reverses them with negated arguments.
    """
    for ft in fc.get("features") or []:
        lng, lat = ft["geometry"]["coordinates"]
        x, y = TO_3857.transform(lng, lat)
        ft["geometry"]["coordinates"] = list(TO_4326.transform(x + dx_m, y + dy_m))
