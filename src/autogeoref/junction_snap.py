"""Drawn-junction placement verifier against modern street centerlines.

Street-grid periodicity makes this a verifier, not a placement finder. Given a tightly bounded
proposal, it compares the proposed translation with nearby alternatives and returns independent
support or abstention — never a refute.

Pixel frames: :func:`extract_junctions` operates on the 2000-px-tall "small"/annotation frame,
while result-record affines are FULL-RES pixels -> EPSG:3857, so :func:`verify_placement` takes
an explicit ``small_to_full`` scale. Never assume a ratio — recover it per sheet from the
manifest. When prep normalized orientation the small on disk is UPRIGHT while the affine stays
in the SOURCE scan frame: put the extraction through :func:`extraction_in_source_frame` first.
Scoring runs in ground metres, so the tuned constants keep their meaning.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .affine import TO_3857, AffineMatrix
from .junction_score import (
    GRID_RES_M,
    ScoreWindow,
    apply_affine_pts,
    corr_score,
    impulse_probe,
    line_probe,
    line_raster,
    make_kernel,
    mercator_lat_deg,
    node_raster,
    sat_blur,
    score_window,
)

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

# --- segmentation constants, tuned on the validated volumes (small-frame px) ---
#: Corridor half-width band in small-frame px (streets ~47-71; alleys ~10 excluded).
CORRIDOR_BAND_PX = (28, 80)
#: Distance-transform half-width above which open land is treated as plaza/vacant.
PLAZA_HALFWIDTH_PX = 95
#: Corridor components must span >= this fraction of the sheet's larger side.
SPAN_FRACTION = 0.55
#: Connected ink components below both limits are text/dashes -> removed.
SMALL_COMPONENT_AREA_PX = 400
SMALL_COMPONENT_DIM_PX = 60
#: Sheet border painted as ink so margins do not read as corridors.
BORDER_FRACTION = 0.03
#: Skeleton branch points within this box merge into one junction (street width).
JUNCTION_CLUSTER_PX = 81
#: Ring radii (px) used to count arms when typing a junction as 4-way ("X").
RING_RADII_PX = (65.0, 78.0)
#: Junctions closer than this to the sheet edge are never typed "X".
EDGE_MARGIN_PX = 140

# --- scoring constants, tuned on the validated volumes (ground meters) ---
#: Gaussian blur applied to centerline intersection nodes.
SIGMA_NODE_M = 20.0
#: Gaussian blur applied to the rasterized centerline geometry.
SIGMA_LINE_M = 8.0
#: Diagonal skeleton arcs (>15 deg off both grid axes) weighted 4x —
#: diagonals do not alias one block over (measured: improves separation
#: where present, never hurts).
DIAGONAL_WEIGHT = 4.0
DIAGONAL_ANGLE_DEG = 15.0
#: Skeleton arcs shorter than this (px) are not orientation-classified.
MIN_ARC_PX = 30
#: Skeleton pixels are subsampled by this stride for the line term.
SKELETON_STRIDE = 4
#: Measured: every committed test sheet separates truth from wrong peaks within +-100 m.
PRIOR_RADIUS_M = 100.0
#: "At the prior" = best score within this distance (the measured truth window).
SNAP_TOLERANCE_M = 50.0
#: Accept margin on the separation ratio: worst passing test sheet was 1.15; default 1.10.
MIN_SEPARATION = 1.10
#: Minimum junctions worth scoring (validation skipped sheets below this).
MIN_JUNCTIONS = 3


class JunctionSnapError(RuntimeError):
    """Unusable inputs for junction extraction or placement verification."""


@dataclass(frozen=True)
class JunctionExtraction:
    """Drawn-street skeleton + clustered junctions, in small-frame pixels.

    Attributes:
        junctions_px: ``(K, 2)`` junction centroids ``(x, y)``, small frame.
        junction_types: Per-junction ``"X"`` (4-way, scored against degree>=4
            nodes only) or ``"any"``.
        skeleton: Boolean corridor-skeleton raster (small frame).
        diagnostics: Extraction counters (``n_junctions``, ``n_x``,
            ``n_skeleton_px``, ``ink_threshold``).
    """

    junctions_px: FloatArray
    junction_types: tuple[str, ...]
    skeleton: NDArray[np.bool_]
    diagnostics: Mapping[str, float]

    @property
    def n_junctions(self) -> int:
        return len(self.junction_types)


@dataclass(frozen=True)
class SnapVerdict:
    """Outcome of verifying one placement proposal.

    ``separation_ratio`` is ``score_at_prior / best_wrong_score``, the wrong
    peak being the best score in the annulus ``(snap_tolerance_m, radius_m]``.
    ``supports`` is TRI-STATE: True when the proposal beats every wrong peak by
    the accept margin or is itself the argmax, else None — the channel has no
    evidence either way and ABSTAINS. **Never False: it does not refute.** Kept
    as ``bool | None`` so a reader can tell "no evidence" from "not scored".
    """

    score_at_prior: float
    best_wrong_score: float
    separation_ratio: float
    best_offset_m: float
    supports: bool | None
    n_junctions: int


@dataclass(frozen=True)
class CenterlineWorld:
    """Modern-street reference for scoring: nodes, degrees, and geometry.

    Attributes:
        nodes_3857: ``(N, 2)`` intersection-node coordinates, EPSG:3857.
        node_degrees: ``(N,)`` segment counts per node (degree>=4 = 4-way).
        polylines_3857: Centerline polylines, each ``(M, 2)`` EPSG:3857.
    """

    nodes_3857: FloatArray
    node_degrees: NDArray[np.int64]
    polylines_3857: tuple[FloatArray, ...]


def world_from_centerlines(
    features: Iterable[Mapping[str, Any]],
    bounds_3857: tuple[float, float, float, float] | None = None,
) -> CenterlineWorld:
    """Build the scoring reference from centerline GeoJSON features.

    ``fnode_id``/``tnode_id`` on the features give the intersection graph
    directly, so no geometric node merging is needed: a node is an intersection
    when its degree is >= 3, or 2 with two distinct street names (a corner).
    ``features`` are WGS84 lng/lat; ``bounds_3857`` optionally clips the result.
    """
    node_names: dict[Any, list[str]] = {}
    node_xy: dict[Any, tuple[float, float]] = {}
    polylines: list[FloatArray] = []
    for f in features:
        props = f["properties"]
        geom = f["geometry"]
        if geom is None:
            continue
        if geom["type"] == "MultiLineString":
            parts = [p for p in geom["coordinates"] if p]
        elif geom["type"] == "LineString":
            parts = [geom["coordinates"]] if geom["coordinates"] else []
        else:
            continue
        if not parts:
            continue
        name = str(props.get("street_nam") or "").strip()
        first, last = parts[0][0], parts[-1][-1]
        for nid, coord in ((props.get("fnode_id"), first), (props.get("tnode_id"), last)):
            if nid is None:
                continue
            node_names.setdefault(nid, []).append(name)
            node_xy[nid] = (float(coord[0]), float(coord[1]))
        for part in parts:
            arr = np.asarray(part, dtype=np.float64)[:, :2]
            x, y = TO_3857.transform(arr[:, 0], arr[:, 1])
            line = np.column_stack([x, y])
            if bounds_3857 is not None:
                minx, miny, maxx, maxy = bounds_3857
                if (
                    line[:, 0].max() < minx
                    or line[:, 0].min() > maxx
                    or line[:, 1].max() < miny
                    or line[:, 1].min() > maxy
                ):
                    continue
            polylines.append(line)

    pts: list[tuple[float, float]] = []
    degs: list[int] = []
    for nid, names in node_names.items():
        deg = len(names)
        if deg >= 3 or (deg == 2 and len(set(names)) > 1):
            lng, lat = node_xy[nid]
            x, y = TO_3857.transform(lng, lat)
            pts.append((float(x), float(y)))
            degs.append(deg)
    nodes = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    node_deg = np.asarray(degs, dtype=np.int64)
    if bounds_3857 is not None and len(nodes):
        minx, miny, maxx, maxy = bounds_3857
        keep = (
            (nodes[:, 0] >= minx)
            & (nodes[:, 0] <= maxx)
            & (nodes[:, 1] >= miny)
            & (nodes[:, 1] <= maxy)
        )
        nodes, node_deg = nodes[keep], node_deg[keep]
    logger.info("centerline world: %d nodes, %d polylines", len(nodes), len(polylines))
    return CenterlineWorld(nodes, node_deg, tuple(polylines))


# --------------------------------------------------------------------------
# extraction (small frame)
# --------------------------------------------------------------------------


def _segment_corridors(gray: NDArray[np.uint8]) -> tuple[NDArray[np.bool_], float]:
    """Otsu ink mask -> street-corridor skeleton (validated prototype, verbatim).

    Sanborn scans are dark (paper ~140-175, ink ~24-60) so fixed thresholds
    fail — Otsu is load-bearing. Corridors are the
    distance-transform width band :data:`CORRIDOR_BAND_PX`; components must
    span >= :data:`SPAN_FRACTION` of the sheet (streets cross it, lot-interior
    bands stay inside one block); plazas/vacant land are excluded.
    """
    from skimage.morphology import skeletonize

    h, w = gray.shape
    margin = int(BORDER_FRACTION * h)
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (gray < thr).astype(np.uint8)
    ink[:margin, :] = 1
    ink[-margin:, :] = 1
    ink[:, :margin] = 1
    ink[:, -margin:] = 1
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    small = np.zeros(nlab, bool)
    for i in range(1, nlab):
        if (
            stats[i, cv2.CC_STAT_AREA] < SMALL_COMPONENT_AREA_PX
            and max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            < SMALL_COMPONENT_DIM_PX
        ):
            small[i] = True
    ink_struct = ink.copy()
    ink_struct[small[lab]] = 0
    dist = cv2.distanceTransform((1 - ink_struct).astype(np.uint8), cv2.DIST_L2, 5)
    plaza = (dist >= PLAZA_HALFWIDTH_PX).astype(np.uint8)
    plaza_zone = cv2.dilate(
        plaza,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * PLAZA_HALFWIDTH_PX + 1,) * 2),
    )
    lo, hi = CORRIDOR_BAND_PX
    corridor = (dist >= lo) & (dist <= hi) & (plaza_zone == 0)
    ncomp, comp, cstats, _ = cv2.connectedComponentsWithStats(
        corridor.astype(np.uint8), connectivity=8
    )
    keep = np.zeros(ncomp, bool)
    for i in range(1, ncomp):
        if max(cstats[i, cv2.CC_STAT_WIDTH], cstats[i, cv2.CC_STAT_HEIGHT]) >= SPAN_FRACTION * max(
            h, w
        ):
            keep[i] = True
    corridor = corridor & keep[comp]
    skel_raw = skeletonize(corridor)  # type: ignore[no-untyped-call]
    skel: NDArray[np.bool_] = np.asarray(skel_raw, dtype=np.bool_)
    return skel, float(thr)


def _branch_points(skel: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Skeleton pixels with >= 3 neighbors."""
    k = np.ones((3, 3), np.uint8)
    k[1, 1] = 0
    nb = cv2.filter2D(skel.astype(np.uint8), -1, k)
    return skel & (nb >= 3)


def _find_junctions(skel: NDArray[np.bool_]) -> tuple[FloatArray, tuple[str, ...]]:
    """Cluster skeleton branch points at street width and type them X/any."""
    h, w = skel.shape
    bp = _branch_points(skel)
    ys, xs = np.nonzero(bp)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float64), ()
    blobs = cv2.dilate(bp.astype(np.uint8), np.ones((JUNCTION_CLUSTER_PX,) * 2, np.uint8))
    _, lab = cv2.connectedComponents(blobs, connectivity=8)
    cid = lab[ys, xs]
    pts = np.column_stack([xs, ys]).astype(np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    cents: list[tuple[float, float]] = []
    types: list[str] = []
    r_in, r_out = RING_RADII_PX
    for c in np.unique(cid):
        cx, cy = pts[cid == c].mean(axis=0)
        cents.append((float(cx), float(cy)))
        rr = np.hypot(xx - cx, yy - cy)
        ring = (skel & (rr >= r_in) & (rr <= r_out)).astype(np.uint8)
        narcs, _ = cv2.connectedComponents(ring, connectivity=8)
        near_margin = min(cx, cy, w - cx, h - cy) < EDGE_MARGIN_PX
        types.append("X" if narcs - 1 >= 4 and not near_margin else "any")
    return np.asarray(cents, dtype=np.float64), tuple(types)


def extract_junctions(small_jpg_path: str | Path) -> JunctionExtraction:
    """Extract drawn-street junctions from a small (2000-px-tall) sheet JPEG.

    Segmentation, verbatim from the validated prototype: Otsu ink mask,
    small-component (text/dash) removal, distance-transform width band,
    sheet-spanning component filter, skeleton branch points clustered at street
    width. Returns small-frame pixels; raises ``JunctionSnapError`` if the image
    cannot be read. Extraction is not the weak link in this channel.
    """
    raw = cv2.imread(str(small_jpg_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise JunctionSnapError(f"cannot read image: {small_jpg_path}")
    gray: NDArray[np.uint8] = np.asarray(raw, dtype=np.uint8)
    skel, thr = _segment_corridors(gray)
    cents, types = _find_junctions(skel)
    diag = {
        "n_junctions": float(len(types)),
        "n_x": float(sum(t == "X" for t in types)),
        "n_skeleton_px": float(int(skel.sum())),
        "ink_threshold": thr,
    }
    logger.info(
        "%s: %d junctions (%d X), %d skeleton px",
        small_jpg_path,
        len(types),
        int(diag["n_x"]),
        int(diag["n_skeleton_px"]),
    )
    return JunctionExtraction(cents, types, skel, diag)


def extraction_in_source_frame(
    extraction: JunctionExtraction, rotation_applied: int
) -> JunctionExtraction:
    """Turn an extraction made on the UPRIGHT small back into the SOURCE frame.

    Prep writes orientation-normalized smalls upright and records the applied clockwise
    quarter-turn, while result-record affines stay in the SOURCE-scan pixel frame, so junctions
    read off the upright small must be turned back before :func:`verify_placement` scores them.
    Quarter-turns only, in pixel-index coordinates so the skeleton raster and its centroids stay
    registered. What composition restores is the DECISIVE verdict, not the score: extraction is
    not bit-equivariant under a turn, so a knife-edge page near :data:`MIN_SEPARATION` can move.
    Raises ``JunctionSnapError`` on a non-quarter-turn.
    """
    rot = int(rotation_applied) % 360
    if rot == 0:
        return extraction
    if rot not in (90, 180, 270):
        raise JunctionSnapError(
            f"rotation_applied must be a quarter-turn (0/90/180/270), got {rotation_applied}"
        )
    back = (360 - rot) % 360
    h, w = extraction.skeleton.shape
    x, y = extraction.junctions_px[:, 0], extraction.junctions_px[:, 1]
    if back == 90:
        turned = np.column_stack([h - 1 - y, x])
    elif back == 180:
        turned = np.column_stack([w - 1 - x, h - 1 - y])
    else:
        turned = np.column_stack([y, w - 1 - x])
    # np.rot90's k is counter-clockwise; the correction is clockwise
    skeleton = np.ascontiguousarray(np.rot90(extraction.skeleton, k=-(back // 90)))
    return JunctionExtraction(
        junctions_px=turned.astype(np.float64).reshape(-1, 2),
        junction_types=extraction.junction_types,
        skeleton=skeleton,
        diagnostics=extraction.diagnostics,
    )


# --------------------------------------------------------------------------
# scoring (ground meters)
# --------------------------------------------------------------------------


def _skeleton_weights(skel: NDArray[np.bool_], diagonal_weight: float) -> FloatArray:
    """Per-skeleton-pixel weight map: ``diagonal_weight`` on diagonal arcs.

    An arc is diagonal when its principal orientation is more than :data:`DIAGONAL_ANGLE_DEG`
    off both grid axes — such corridors do not alias one block over, so they carry the
    discriminating signal.
    """
    bp = _branch_points(skel)
    arcs = skel & ~cv2.dilate(bp.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(arcs.astype(np.uint8), connectivity=8)
    wmap = np.ones(skel.shape, dtype=np.float64)
    for i in range(1, nlab):
        if stats[i, cv2.CC_STAT_AREA] < MIN_ARC_PX:
            continue
        x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        ww, hh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        sub = lab[y0 : y0 + hh, x0 : x0 + ww] == i
        sy, sx = np.nonzero(sub)
        pts = np.column_stack([sx, sy]).astype(np.float64)
        pts -= pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        ang = abs(math.degrees(math.atan2(vt[0, 1], vt[0, 0]))) % 90
        if DIAGONAL_ANGLE_DEG < ang < 90 - DIAGONAL_ANGLE_DEG:
            wmap[y0 : y0 + hh, x0 : x0 + ww][sub] = diagonal_weight
    return wmap


def _reject_unverifiable(
    extraction: JunctionExtraction, world: CenterlineWorld, radius_m: float
) -> None:
    """Raise unless there is enough evidence, and a tight enough prior, to verify at all."""
    if radius_m > PRIOR_RADIUS_M:
        # Measured: pure-grid sheets alias at one block (~150 m); the
        # verifier's separation ratios are only valid inside +-100 m. A wider
        # window (e.g. a margins.predict_window N-S radius of 200 m) must be
        # tightened first (second neighbor, street index) — silently accepting
        # it would re-enable exactly that one-block aliasing.
        raise JunctionSnapError(
            f"prior window radius {radius_m:.0f} m exceeds the validated "
            f"{PRIOR_RADIUS_M:.0f} m contract; tighten the prior before verifying"
        )
    if extraction.n_junctions < MIN_JUNCTIONS:
        raise JunctionSnapError(
            f"only {extraction.n_junctions} junctions extracted (need >= {MIN_JUNCTIONS}); "
            "the sheet carries too little drawn-street evidence to verify"
        )
    if len(world.nodes_3857) == 0:
        raise JunctionSnapError("centerline world has no intersection nodes")


@dataclass(frozen=True)
class _SheetEvidence:
    """One sheet's junctions and corridor skeleton, projected to ground metres."""

    #: junction points, and the constellation centre the kernels are built around
    junctions: FloatArray
    center: FloatArray
    #: EPSG:3857 -> ground-metre scale at the proposal's latitude
    k: float
    #: subsampled skeleton points and their diagonal-arc weights
    skeleton: FloatArray
    skeleton_weights: FloatArray


def _sheet_evidence(
    extraction: JunctionExtraction,
    proposed_affine: AffineMatrix,
    small_to_full: float,
    diagonal_weight: float,
) -> _SheetEvidence:
    """Project sheet evidence to ground meters (3857 * cos(lat) at the proposal)."""
    junc_full = extraction.junctions_px * small_to_full
    junc_3857 = apply_affine_pts(proposed_affine, junc_full)
    k = math.cos(math.radians(mercator_lat_deg(float(junc_3857[:, 1].mean()))))
    q = junc_3857 * k

    sy, sx = np.nonzero(extraction.skeleton)
    sub = slice(None, None, SKELETON_STRIDE)
    skel_px = np.column_stack([sx[sub], sy[sub]]).astype(np.float64)
    skel_g = apply_affine_pts(proposed_affine, skel_px * small_to_full) * k
    wmap = _skeleton_weights(extraction.skeleton, diagonal_weight)
    return _SheetEvidence(
        junctions=q,
        center=q.mean(axis=0),
        k=k,
        skeleton=skel_g,
        skeleton_weights=wmap[sy[sub], sx[sub]],
    )


def _blurred_nodes(
    world: CenterlineWorld, k: float, window: ScoreWindow, is_x: NDArray[np.bool_]
) -> tuple[FloatArray, FloatArray | None]:
    """Blurred node rasters: all nodes, and degree>=4 nodes when any junction is X-typed."""
    impulse = impulse_probe()
    sigma = SIGMA_NODE_M / GRID_RES_M
    all_nodes = np.ones(len(world.nodes_3857), bool)
    nb_any = sat_blur(node_raster(world.nodes_3857[all_nodes], k, window), sigma, impulse)
    if not is_x.any():
        return nb_any, None
    deg4 = np.asarray(world.node_degrees >= 4, dtype=bool)
    return nb_any, sat_blur(node_raster(world.nodes_3857[deg4], k, window), sigma, impulse)


def _junction_term(
    q: FloatArray,
    is_x: NDArray[np.bool_],
    center: FloatArray,
    nb_any: FloatArray,
    nb4: FloatArray | None,
    window: ScoreWindow,
) -> FloatArray:
    """X junctions scored on deg>=4 nodes, the rest on any node, weighted by count."""
    cj = np.zeros((window.nr, window.nc), dtype=np.float64)
    wsum = 0.0
    if is_x.any() and nb4 is not None:
        kern = make_kernel(q[is_x], np.ones(int(is_x.sum())), center)
        cj += corr_score(nb4, kern) * float(is_x.sum())
        wsum += float(is_x.sum())
    if (~is_x).any():
        kern = make_kernel(q[~is_x], np.ones(int((~is_x).sum())), center)
        cj += corr_score(nb_any, kern) * float((~is_x).sum())
        wsum += float((~is_x).sum())
    cj /= max(wsum, 1.0)
    return cj


def verify_placement(
    extraction: JunctionExtraction,
    proposed_affine: AffineMatrix,
    world: CenterlineWorld,
    *,
    small_to_full: float,
    radius_m: float = PRIOR_RADIUS_M,
    min_separation: float = MIN_SEPARATION,
    diagonal_weight: float = DIAGONAL_WEIGHT,
    snap_tolerance_m: float = SNAP_TOLERANCE_M,
) -> SnapVerdict:
    """Score a placement proposal against drawn-junction/centerline agreement.

    A VERIFIER, not a finder: the score surface is discriminating only inside a
    ``radius_m`` prior window, beyond which grid periodicity aliases the
    constellation one block over. The combined score, as validated: half a
    junction term (X-typed junctions scored against degree>=4 nodes, others
    against any node) plus half a line term (subsampled corridor skeleton
    against blurred centerline geometry). Raises ``JunctionSnapError`` on too
    few junctions, an empty reference window, or too wide a prior window.
    """
    _reject_unverifiable(extraction, world, radius_m)
    ev = _sheet_evidence(extraction, proposed_affine, small_to_full, diagonal_weight)
    q, center, skel_g = ev.junctions, ev.center, ev.skeleton

    # local scoring window (ground meters)
    pad = radius_m + snap_tolerance_m + 5 * SIGMA_NODE_M + 4 * GRID_RES_M
    window = score_window(q, skel_g, pad=pad)

    is_x = np.array([t == "X" for t in extraction.junction_types])
    nb_any, nb4 = _blurred_nodes(world, ev.k, window, is_x)
    lines = line_raster(world.polylines_3857, ev.k, window)
    cj = _junction_term(q, is_x, center, nb_any, nb4, window)

    # line term (diagonal-weighted), when geometry + skeleton are available
    have_line = bool(len(world.polylines_3857)) and len(skel_g) > 0 and lines.any()
    if have_line:
        lb = sat_blur(lines, SIGMA_LINE_M / GRID_RES_M, line_probe())
        cl = corr_score(lb, make_kernel(skel_g, ev.skeleton_weights, center))
        score = 0.5 * cj + 0.5 * cl
    else:
        logger.warning("no centerline geometry in window; junction-only score")
        score = cj

    gy, gx = np.mgrid[0 : window.nr, 0 : window.nc]
    ex = window.minx + gx * GRID_RES_M
    ny = window.maxy - gy * GRID_RES_M
    dist = np.hypot(ex - center[0], ny - center[1])

    at_prior = dist <= snap_tolerance_m
    wrong = (dist > snap_tolerance_m) & (dist <= radius_m)
    score_at_prior = float(score[at_prior].max()) if at_prior.any() else 0.0
    best_wrong = float(score[wrong].max()) if wrong.any() else 0.0
    ratio = score_at_prior / best_wrong if best_wrong > 1e-12 else math.inf
    within = dist <= radius_m
    masked = np.where(within, score, -np.inf)
    gi = np.unravel_index(int(np.argmax(masked)), score.shape)
    best_offset = float(dist[gi])
    # SUPPORT or ABSTAIN — never refute. Anything the support clause does not
    # catch used to be emitted as a REFUTE, the sole blocker in verified_accept,
    # so the channel cast a blocking vote on sheets it had no evidence about.
    # Measured, that veto never changed an outcome while falsely vetoing many
    # correctly placed human sheets, and no threshold rescues it: correct and
    # misplaced sheets overlap through the whole low-ratio band.
    supports: bool | None = (
        True if (ratio >= min_separation or best_offset <= snap_tolerance_m) else None
    )
    logger.info(
        "snap verdict: score@prior %.3f, wrong %.3f, ratio %.2f, best offset %.0f m, supports=%s",
        score_at_prior,
        best_wrong,
        ratio,
        best_offset,
        "yes" if supports else "abstain",
    )
    return SnapVerdict(
        score_at_prior=score_at_prior,
        best_wrong_score=best_wrong,
        separation_ratio=ratio,
        best_offset_m=best_offset,
        supports=supports,
        n_junctions=extraction.n_junctions,
    )


__all__ = [
    "DIAGONAL_WEIGHT",
    "MIN_SEPARATION",
    "PRIOR_RADIUS_M",
    "CenterlineWorld",
    "JunctionExtraction",
    "JunctionSnapError",
    "SnapVerdict",
    "extract_junctions",
    "extraction_in_source_frame",
    "verify_placement",
    "world_from_centerlines",
]
