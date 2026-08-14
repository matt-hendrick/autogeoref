"""Post-split blank-core move: return a split winner's blank ground to the
neighbour that drew it.

:func:`autogeoref.mask.geometry.split_overlaps` awards contested ground to the
nearest mask centroid and has no notion of what is drawn, so blank paper can
win over drawn content and occlude it. This pass runs between the split and
the heal ladder and moves those cells back, subject to a reachability bar that
keeps a receiver from being stranded as an island inside its neighbour.

Interior moves detach islands and punch holes, so results are MultiPolygon,
and overlay ops on split finals are grid-snapped (:data:`_GRID_M`, areal
operands only). When the moved masks would leave solid ground uncovered, the
whole move reverts. `docs/INTERNALS.md` states the rules and their cost.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import shapely
from shapely import GEOSException
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

from ..affine import AffineMatrix, invert_affine
from ..slugs import page_sort_key
from .geometry import SLIVER_AREA_M2
from .qa import (
    BLANK_CORE_M,
    CONTESTED_INK_FLOOR,
    RawSheetMask,
    _InkRaster,
    ink_raster_for,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: An areal EPSG:3857 geometry — what every mask in this module is.
type _Areal = Polygon | MultiPolygon

#: Sample points per cell axis for the blankness test — 2 m spacing over the
#: :data:`~autogeoref.mask.qa.BLANK_CORE_M` cell, finer than the speck-opened
#: ink raster's smallest surviving feature. Not a free parameter: looser
#: sampling moves far more ground than the census core warrants.
MOVE_SAMPLES = 15

#: In-region sample points needed before a cell is decidable; partial edge
#: cells below it keep their bisector award.
_MIN_DECIDABLE = max(4, MOVE_SAMPLES * MOVE_SAMPLES // 3)

#: Loser-ink floor: a cell moves only when the losing neighbour carries at
#: least this ink fraction there. The census floor, by owner decision — it
#: maximizes recovery; raising it buys immunity to misregistered-ink moves at
#: a measured cost in recovery.
MOVE_INK_FLOOR = CONTESTED_INK_FLOOR

#: Overlay snap grid (m) for every op that touches a split final.
_GRID_M = 0.001

#: Post-move parts smaller than this (m2) are numeric crumbs, dropped.
_CRUMB_M2 = 0.5

#: Solid (0.5 m-eroded) ground the moved masks may leave uncovered before the
#: whole move is reverted. Zero by construction and by corpus measurement;
#: anything past this is a real hole, not cell-edge ribbon.
_MAX_UNCOVERED_SOLID_M2 = 1.0

#: Separation (m) at which two grid-snapped parts still read as one shape:
#: :data:`_GRID_M` snapping and ``_clean_parts``'s simplify leave sub-centimetre
#: separations on an edge two pieces genuinely share.
_TOUCH_M = 0.05

#: Erosion (m) applied to every operand of a reachability test. A
#: ``split_overlaps`` final can pass ``is_valid`` and still carry a ZERO-AREA
#: fold — a hair-thin spike reaching hundreds of metres into a neighbour — so
#: ``distance`` reads 0 between sheets that are nowhere near each other. Same
#: degeneracy class the grid-snapped overlay ops address.
_FOLD_EROSION_M = 0.001


def _solid(geom: shapely.Geometry) -> shapely.Geometry:
    """``geom`` without its zero-area folds; safe for a reachability test.

    A piece thinner than 2 * :data:`_FOLD_EROSION_M` erodes to nothing and is
    returned WITH its folds, which could false-anchor a group. That guard is
    here so the erosion can never delete a claim outright; claim pieces are
    cell-sized and it has never fired.
    """
    eroded = geom.buffer(-_FOLD_EROSION_M)
    return geom if eroded.is_empty else eroded


def _reachable(
    claims: Mapping[
        tuple[str, tuple[float, float]], tuple[tuple[float, float], str, Polygon | MultiPolygon]
    ],
    split: Mapping[str, Polygon],
) -> dict[tuple[str, tuple[float, float]], tuple[tuple[float, float], str, Polygon | MultiPolygon]]:
    """Keep only moved cells that REACH the sheet they are moving to.

    A cell the receiving sheet cannot reach — directly or through other cells
    moving to it — lands as a detached island of that sheet inside its
    neighbour, and takes a matching hole out of the neighbour. Refusing the
    claim, rather than dropping the part afterwards, leaves the cell with its
    bisector winner, so the coverage union is preserved. Measured against the
    PRISTINE split, so it bounds the stranded-island class without
    eliminating it.
    """
    by_loser: dict[str, list[tuple[tuple[str, tuple[float, float]], shapely.Geometry]]] = {}
    for key, (_rank, l_slug, piece) in claims.items():
        by_loser.setdefault(l_slug, []).append((key, _solid(piece)))
    keep: set[tuple[str, tuple[float, float]]] = set()
    for l_slug, items in by_loser.items():
        final = split.get(l_slug)
        if final is None or final.is_empty:
            continue
        # union-find over "shares a boundary", seeded by the receiving final
        parent = list(range(len(items)))

        def find(i: int, parent: list[int] = parent) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i][1].distance(items[j][1]) <= _TOUCH_M:
                    parent[find(i)] = find(j)
        seed = _solid(final)
        anchored = {find(i) for i, (_k, g) in enumerate(items) if g.distance(seed) <= _TOUCH_M}
        keep.update(key for i, (key, _g) in enumerate(items) if find(i) in anchored)
    return {k: v for k, v in claims.items() if k in keep}


def _ink_at_xy(
    raster: _InkRaster, matrix: AffineMatrix, xs: NDArray[np.float64], ys: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Sample a sheet's ink raster at EPSG:3857 points (outside frame = no ink)."""
    inv = invert_affine(matrix)
    sx = (inv[0][0] + inv[0][1] * xs + inv[0][2] * ys) * raster.scale
    sy = (inv[1][0] + inv[1][1] * xs + inv[1][2] * ys) * raster.scale
    col = np.floor(sx).astype(np.intp)
    row = np.floor(sy).astype(np.intp)
    h, w = raster.ink.shape
    ok = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    out = np.zeros(xs.shape, dtype=bool)
    out[ok] = raster.ink[row[ok], col[ok]]
    return out


def _cell_ink_fracs(
    region: Polygon | MultiPolygon,
    winner: tuple[_InkRaster, AffineMatrix],
    loser: tuple[_InkRaster, AffineMatrix],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, float]:
    """Sampled ink fractions per :data:`BLANK_CORE_M` cell over ``region``.

    Returns ``(winner frac, loser frac, in-region counts, x0, y0)`` as
    ``(ny, nx)`` arrays on a grid aligned to ``floor(bounds / cell) * cell``,
    so cells land identically whichever pair claims them.
    """
    minx, miny, maxx, maxy = region.bounds
    x0 = math.floor(minx / BLANK_CORE_M) * BLANK_CORE_M
    y0 = math.floor(miny / BLANK_CORE_M) * BLANK_CORE_M
    nx = max(1, math.ceil((maxx - x0) / BLANK_CORE_M))
    ny = max(1, math.ceil((maxy - y0) / BLANK_CORE_M))
    offs = (np.arange(MOVE_SAMPLES) + 0.5) / MOVE_SAMPLES * BLANK_CORE_M
    px = x0 + np.arange(nx)[None, :, None, None] * BLANK_CORE_M + offs[None, None, None, :]
    py = y0 + np.arange(ny)[:, None, None, None] * BLANK_CORE_M + offs[None, None, :, None]
    px, py = np.broadcast_arrays(px, py)  # (ny, nx, s, s)
    flat_x, flat_y = px.ravel(), py.ravel()
    shapely.prepare(region)
    inside = shapely.contains_xy(region, flat_x, flat_y).reshape(px.shape)
    n_inside = inside.sum(axis=(2, 3)).astype(float)

    def frac(raster: _InkRaster, matrix: AffineMatrix) -> NDArray[np.float64]:
        hit = _ink_at_xy(raster, matrix, flat_x, flat_y).reshape(px.shape)
        return np.where(
            n_inside > 0, (hit & inside).sum(axis=(2, 3)) / np.maximum(n_inside, 1), 0.0
        )

    return frac(*winner), frac(*loser), n_inside, x0, y0


def _only_areal(g: shapely.Geometry) -> shapely.Geometry:
    """Polygonal parts only — grid-snapped overlay ops reject mixed input."""
    if g.geom_type == "GeometryCollection":
        return unary_union([p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")])
    return g


def _clean_parts(geom: shapely.Geometry) -> Polygon | MultiPolygon:
    """Valid areal geometry with numeric crumbs dropped; multi-part allowed."""
    g = _only_areal(geom).buffer(0)
    g = _only_areal(g)
    if g.geom_type == "MultiPolygon":
        parts = [p for p in g.geoms if p.area >= _CRUMB_M2]
        if not parts:
            return Polygon()
        g = parts[0] if len(parts) == 1 else MultiPolygon(parts)
    g = _only_areal(g.simplify(0.02).buffer(0))
    if g.geom_type not in ("Polygon", "MultiPolygon"):
        return Polygon()
    return g


def _pair_claims(
    winner: RawSheetMask,
    loser: RawSheetMask,
    winner_final: Polygon,
    rasters: tuple[_InkRaster, _InkRaster],
    split: Mapping[str, Polygon],
    centroids: Mapping[str, tuple[float, float]],
) -> list[tuple[tuple[str, tuple[float, float]], tuple[tuple[float, float], str, _Areal]]]:
    """One ordered (winner, loser) pair's movable cells, as claim entries.

    Empty when the pair never contested ground, when the overlay fails (the
    pair is skipped, not the pass), or when no cell clears the blank/ink bars.
    """
    rw, rl = rasters
    try:
        raw_inter = shapely.intersection(winner.poly_3857, loser.poly_3857, grid_size=_GRID_M)
        if raw_inter.is_empty or raw_inter.area < SLIVER_AREA_M2:
            return []
        region = _only_areal(
            shapely.difference(
                _only_areal(shapely.intersection(winner_final, loser.poly_3857, grid_size=_GRID_M)),
                split[loser.slug],
                grid_size=_GRID_M,
            )
        )
    except GEOSException:
        logger.warning(
            "blank-core move: %s/%s overlay failed; pair skipped", winner.slug, loser.slug
        )
        return []
    if region.is_empty or region.area < SLIVER_AREA_M2:
        return []
    fw_frac, fl_frac, n_in, x0, y0 = _cell_ink_fracs(
        region, (rw, winner.matrix), (rl, loser.matrix)
    )
    move = (n_in >= _MIN_DECIDABLE) & (fw_frac == 0.0) & (fl_frac >= MOVE_INK_FLOOR)
    out = []
    for y, x in zip(*np.nonzero(move), strict=True):
        cx0 = x0 + float(x) * BLANK_CORE_M
        cy0 = y0 + float(y) * BLANK_CORE_M
        cell = box(cx0, cy0, cx0 + BLANK_CORE_M, cy0 + BLANK_CORE_M)
        try:
            piece = _only_areal(shapely.intersection(cell, region, grid_size=_GRID_M))
        except GEOSException:
            continue
        if piece.is_empty:
            continue
        rank = (
            float(fl_frac[y, x]),
            -math.dist(cell.centroid.coords[0], centroids[loser.slug]),
        )
        out.append(((winner.slug, (cx0, cy0)), (rank, loser.slug, piece)))
    return out


def _collect_claims(
    regular: Sequence[RawSheetMask],
    split: Mapping[str, Polygon],
    centroids: Mapping[str, tuple[float, float]],
    cache: MutableMapping[tuple[str, tuple[int, int, int, int]], _InkRaster | None],
) -> dict[tuple[str, tuple[float, float]], tuple[tuple[float, float], str, _Areal]]:
    """Every winner cell a loser claims, best claim per cell.

    Keyed by ``(winner slug, cell origin)``; the value ranks by the loser's ink
    fraction, then by nearest loser centroid, so ties resolve in page order.
    """
    claims: dict[tuple[str, tuple[float, float]], tuple[tuple[float, float], str, _Areal]] = {}
    for w in regular:
        fw = split[w.slug]
        rw = ink_raster_for(w, cache)
        if fw.is_empty or rw is None or w.slug not in centroids:
            continue
        for lo in regular:
            if lo.slug == w.slug or lo.slug not in centroids:
                continue
            rl = ink_raster_for(lo, cache)
            if rl is None:
                continue
            for key, claim in _pair_claims(w, lo, fw, (rw, rl), split, centroids):
                if key not in claims or claim[0] > claims[key][0]:
                    claims[key] = claim
    return claims


def _apply_moves(
    out: MutableMapping[str, _Areal],
    moves: Sequence[tuple[str, str, _Areal]],
) -> set[str] | None:
    """Shed each winner's claimed pieces, then add them to their losers.

    Mutates ``out``. Returns the winners that actually shed, or ``None`` on a
    geometry failure — the caller then keeps the input split wholesale, since
    ``out`` may be half-applied.
    """
    removals: dict[str, list[_Areal]] = {}
    for w_slug, _l_slug, piece in moves:
        removals.setdefault(w_slug, []).append(piece)
    removed_ok: set[str] = set()
    try:
        for slug, pieces in removals.items():
            trimmed = _clean_parts(
                shapely.difference(out[slug], _only_areal(unary_union(pieces)), grid_size=_GRID_M)
            )
            # the same bar the split's _usable guard holds a cut to: a pass
            # may not hollow a sheet past half of what it has
            if not trimmed.is_empty and trimmed.area > 0.5 * out[slug].area:
                out[slug] = trimmed
                removed_ok.add(slug)
        additions: dict[str, list[_Areal]] = {}
        for w_slug, l_slug, piece in moves:
            if w_slug in removed_ok:
                additions.setdefault(l_slug, []).append(piece)
        for slug, pieces in additions.items():
            merged = _clean_parts(
                shapely.union_all(
                    [_only_areal(out[slug]), _only_areal(unary_union(pieces))], grid_size=_GRID_M
                )
            )
            if merged.is_empty:
                raise GEOSException(f"union voided {slug}")
            out[slug] = merged
    except GEOSException:
        logger.exception("blank-core move: geometry failure; shipped split kept")
        return None
    return removed_ok


def move_blank_cores(
    sheets: Sequence[RawSheetMask],
    split: Mapping[str, Polygon],
    *,
    ink_rasters: MutableMapping[tuple[str, tuple[int, int, int, int]], _InkRaster | None]
    | None = None,
) -> dict[str, Polygon | MultiPolygon]:
    """Move each split winner's blank-over-loser-ink core cells to the loser.

    Takes the volume's regular sheets and slug -> post-``split_overlaps``
    EPSG:3857 polygon; returns the same mapping with winners possibly holed or
    multi-part. Input geometry is not mutated. Moves are computed against the
    pristine split and applied at once: a winner sheds a cell only when the
    removal survives the split's own more-than-half bar. An overlay failure
    measuring one pair skips that pair; a failure while applying reverts the
    whole pass.
    """
    cache: MutableMapping[tuple[str, tuple[int, int, int, int]], _InkRaster | None] = (
        {} if ink_rasters is None else ink_rasters
    )
    regular = sorted(
        (s for s in sheets if s.slug in split and s.style != "mask_px"),
        key=lambda s: page_sort_key(s.slug),
    )
    centroids = {
        s.slug: s.poly_3857.centroid.coords[0] for s in regular if not s.poly_3857.is_empty
    }
    claims = _reachable(_collect_claims(regular, split, centroids, cache), split)
    out: dict[str, Polygon | MultiPolygon] = dict(split)
    if not claims:
        return out

    moves: list[tuple[str, str, Polygon | MultiPolygon]] = [
        (w_slug, l_slug, piece) for (w_slug, _), (_, l_slug, piece) in claims.items()
    ]
    applied = _apply_moves(out, moves)
    if applied is None:
        return dict(split)
    removed_ok = applied
    if not removed_ok:
        return dict(split)

    # the union invariant is CHECKED, never assumed: grid-snapped overlay
    # keeps the answer consistent, and the erosion separates a real hole from
    # sub-half-metre cell-edge ribbon
    try:
        before = unary_union(list(split.values()))
        after = unary_union(list(out.values()))
        hole = _only_areal(shapely.difference(before, after, grid_size=_GRID_M)).buffer(-0.5)
    except GEOSException:
        logger.exception("blank-core move: union check failed; shipped split kept")
        return dict(split)
    if hole.area > _MAX_UNCOVERED_SOLID_M2:
        logger.warning(
            "blank-core move: reverted — moved masks leave %.1f m2 of solid ground uncovered",
            hole.area,
        )
        return dict(split)
    moved_m2 = sum(piece.area for w_slug, _, piece in moves if w_slug in removed_ok)
    logger.info(
        "blank-core move: %.0f m2 of blank-over-ink ground moved off %d sheet(s)",
        moved_m2,
        len(removed_ok),
    )
    return out
