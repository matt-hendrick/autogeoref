"""Bake-time mask QA: per-sheet flags, volume coverage, and a handedness backstop.

Reads only what the bake already wrote to disk — zero model calls, zero
network — and produces the ``masks-qa.json`` document: thresholds, per-sheet
metrics, a ``flagged`` summary, and volume-level findings in ``volume_flags``.
The volume report and ``autogeoref status`` read the same document.

Every flag here is advisory. Nothing in this module refuses a bake; the one
automated remedy lives in ``bake.stage_masks``.

`docs/INTERNALS.md` defines each flag, the frame each metric is measured in, and
why a missing metric is omitted rather than defaulted.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform as shp_transform

from ..affine import (
    TO_3857,
    AffineMatrix,
    invert_affine,
    model_determinant,
    model_rotation_deg,
    model_scales,
)
from ..matching import fold_quadrant_deg
from ..paths import VolumePaths, write_if_changed
from ..pillow import unlimited_image_pixels
from ..slugs import (
    DuplicateCoverage,
    duplicate_coverage_slug,
    mosaic_paint_order,
    overview_slug,
    page_from_slug,
    page_sort_key,
)
from ..volume import ROT_TOL_DEG, SCALE_TOL_FRAC
from .geometry import (
    _CONTENT_SAT_FLOOR,
    _CONTENT_SAT_MARGIN,
    CONTENT_INK_DARK_THRESH,
    CONTENT_SPECK_PX,
    SLIVER_AREA_M2,
    mask_polygon_4326,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: Downscale width for the ink raster — matches the mask detectors' frame.
QA_WORK_WIDTH = 1200

#: "Drawn content" = colored wash (same saturation bar as the content-hull
#: detector) OR dark linework. Saturation alone is blind to exactly the sheets
#: that collapse: near-monochrome linework carries the content there. Shared
#: with :func:`autogeoref.mask.geometry.detect_content_box`'s own-ink guard, which must
#: measure the same pixels this flags on.
INK_DARK_THRESH = CONTENT_INK_DARK_THRESH

#: Morphological opening size (small-frame px) for the ink raster: removes
#: scan noise and paper texture, keeps linework and wash. Shared with the
#: colour box's own-ink guard for the same reason the threshold is.
_INK_SPECK_PX = CONTENT_SPECK_PX

#: Flag ``hull_collapse`` when more than this fraction of a hull-style
#: sheet's drawn content is covered by no regular sheet's final mask.
HULL_COLLAPSE_MIN_UNCOVERED_INK = 0.15

#: Styles that get the same uncovered-ink bar under the ``uncovered_ink`` name.
#: Kept separate from ``hull_collapse`` because the bake's rectangle swap
#: answers a collapsed hull and not a colour box.
BOX_UNCOVERED_INK_STYLES = ("content_box",)

#: Side (m) of the block used for the blank-core measure: a blank patch must
#: be at least this wide in both directions to count.
BLANK_CORE_M = 30.0

#: Flag ``blank_overpaint`` when at least this much contiguous-blank core
#: (m^2) paints over sheets below.
BLANK_OVERPAINT_MIN_CORE_M2 = 4000.0

#: Loser-ink floor per :data:`BLANK_CORE_M` block for the contested-ground
#: core: the ink raster is speck-opened, so surviving ink is real linework,
#: but one stray mark must not turn a blank-vs-blank block into a defect.
CONTESTED_INK_FLOOR = 0.05

#: Channel half-width (m) for the volume coverage metric: uncovered ground in
#: a channel narrower than twice this is an inter-sheet slot, not the outer
#: margin. Same operator and radius as the bake's pre-split expansion
#: (``bake.BOX_EXPAND_SLOT_RADIUS_M``), so the metric measures exactly the
#: ground that pass failed to hand back.
COVERAGE_SLOT_RADIUS_M = 50.0

#: A gap piece under this (m^2) is reprojection noise, not a hole a reader
#: sees.
COVERAGE_MIN_PIECE_M2 = 1.0

#: Flag ``coverage_gaps`` above this much slot ground per 1000 m^2 of the
#: volume's own page footprint.
COVERAGE_GAPS_MAX_SLOT_PER_1K = 0.5


@dataclass(frozen=True)
class RawSheetMask:
    """One committed sheet's mask as built, before overlap splitting.

    ``style`` names the detector that produced the ring: ``mask_px``,
    ``hull``, ``content_box``, ``overview``, or ``page``.

    ``rect`` is always the PAGE rectangle, whatever the style: it is the frame
    the ink raster is measured in, not the mask's own extent.
    """

    slug: str
    image: Path
    matrix: AffineMatrix
    style: str
    rect: tuple[int, int, int, int] | None
    ring_px: tuple[tuple[float, float], ...] | None
    poly_3857: Polygon


@dataclass
class _InkRaster:
    ink: NDArray[np.bool_]  # small-frame drawn-content raster
    scale: float  # small px per full-res px
    rect_small: tuple[int, int, int, int]


def _ink_raster(image_path: Path, rect: tuple[int, int, int, int]) -> _InkRaster | None:
    """Small-frame boolean raster of the sheet's drawn content."""
    try:
        with unlimited_image_pixels(), Image.open(image_path) as img:
            w, h = img.size
            if w < 2 or h < 2:
                return None
            scale = QA_WORK_WIDTH / w
            small = img.convert("RGB").resize((QA_WORK_WIDTH, round(h * scale)))
    except OSError:
        logger.warning("mask QA: cannot open %s; sheet skipped", image_path)
        return None
    arr = np.asarray(small, dtype=np.int16)
    saturation = arr.max(axis=2) - arr.min(axis=2)
    gray = arr.sum(axis=2) // 3
    x0, y0, x1, y1 = (round(v * scale) for v in rect)
    frame = saturation[y0:y1, x0:x1]
    if frame.size == 0:
        return None
    sat_thresh = max(_CONTENT_SAT_FLOOR, int(np.median(frame)) + _CONTENT_SAT_MARGIN)
    ink = (saturation > sat_thresh) | (gray < INK_DARK_THRESH)
    opened = (
        Image.fromarray((ink * 255).astype(np.uint8))
        .filter(ImageFilter.MinFilter(_INK_SPECK_PX))
        .filter(ImageFilter.MaxFilter(_INK_SPECK_PX))
    )
    return _InkRaster(np.asarray(opened) > 0, scale, (x0, y0, x1, y1))


def ink_raster_for(
    sheet: RawSheetMask,
    cache: MutableMapping[tuple[str, tuple[int, int, int, int]], _InkRaster | None],
) -> _InkRaster | None:
    """The sheet's ink raster through a shared cache.

    Keyed by image path and page rect, so one bake builds each raster
    once across the blank-core move, the QA scoring pass, and every
    auto-exemption candidate re-score. A ``mask_px`` sheet has no page
    frame to measure in and yields None.
    """
    if sheet.rect is None:
        return None
    key = (str(sheet.image), sheet.rect)
    if key not in cache:
        cache[key] = _ink_raster(sheet.image, sheet.rect)
    return cache[key]


def _to_small_px(poly_3857: Polygon | MultiPolygon, matrix: AffineMatrix, scale: float) -> Any:
    """Project an EPSG:3857 geometry into a sheet's small pixel frame."""
    inverse = invert_affine(matrix)

    def f(x: Any, y: Any) -> tuple[Any, Any]:
        # apply_affine per point, written array-safe: shapely's transform
        # hands whole coordinate sequences in at once
        xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        px = inverse[0][0] + inverse[0][1] * xs + inverse[0][2] * ys
        py = inverse[1][0] + inverse[1][1] * xs + inverse[1][2] * ys
        return px * scale, py * scale

    return shp_transform(f, poly_3857)


def _ink_at_points(
    src: _InkRaster,
    src_matrix: AffineMatrix,
    dst_matrix: AffineMatrix,
    dst_scale: float,
    rows: NDArray[np.intp],
    cols: NDArray[np.intp],
) -> NDArray[np.bool_]:
    """Sample another sheet's ink raster at this sheet's small-frame pixels.

    Nearest-neighbour through the two affines: dst small px -> dst full px ->
    EPSG:3857 -> src full px -> src small px. Points that land outside the
    source raster sample as no-ink.
    """
    inverse = invert_affine(src_matrix)
    fx = (cols + 0.5) / dst_scale
    fy = (rows + 0.5) / dst_scale
    mx = dst_matrix[0][0] + dst_matrix[0][1] * fx + dst_matrix[0][2] * fy
    my = dst_matrix[1][0] + dst_matrix[1][1] * fx + dst_matrix[1][2] * fy
    sx = (inverse[0][0] + inverse[0][1] * mx + inverse[0][2] * my) * src.scale
    sy = (inverse[1][0] + inverse[1][1] * mx + inverse[1][2] * my) * src.scale
    col = np.floor(sx).astype(np.intp)
    row = np.floor(sy).astype(np.intp)
    h, w = src.ink.shape
    ok = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    out = np.zeros(rows.shape, dtype=bool)
    out[ok] = src.ink[row[ok], col[ok]]
    return out


def _rasterize(geom: Any, shape: tuple[int, ...]) -> NDArray[np.bool_]:
    """Boolean raster of a small-frame polygon on the ink raster's grid."""
    img = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(img)
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        if poly.geom_type != "Polygon" or poly.is_empty:
            continue
        draw.polygon([(x, y) for x, y in poly.exterior.coords], fill=255)
        for interior in poly.interiors:
            draw.polygon([(x, y) for x, y in interior.coords], fill=0)
    return np.asarray(img) > 0


def _areal(geom: Any) -> Any:
    """Polygonal parts only — an intersection can return a mixed collection."""
    from shapely.ops import unary_union

    if geom.geom_type == "GeometryCollection":
        return unary_union([g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")])
    return geom


def _geom_parts(geom: Any) -> list[Any]:
    if geom.is_empty:
        return []
    return list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]


def volume_coverage(
    sheets: Sequence[RawSheetMask],
    detail_union_3857: Any,
    duplicates: DuplicateCoverage,
) -> dict[str, Any]:
    """The volume-level coverage statistics, keyed as in ``masks-qa.json``.

    ``detail_union_3857`` is the union of every mask the DETAIL mosaic serves.
    The footprint is the union of the regular committed sheets' page
    rectangles; ``slot_per_1k`` is the headline. Empty when no sheet has a
    page rectangle to build the footprint from — see `docs/INTERNALS.md`.
    """
    from shapely.ops import unary_union

    pages = [
        shp_transform(TO_3857.transform, mask_polygon_4326(s.rect, s.matrix))
        for s in sheets
        if s.rect is not None and not duplicate_coverage_slug(s.slug, duplicates)
    ]
    if not pages:
        return {}
    footprint = unary_union(pages)
    if not footprint.area:
        return {}
    union = detail_union_3857
    holes = [
        h
        for part in _geom_parts(union)
        for ring in part.interiors
        if (h := Polygon(ring)).area >= COVERAGE_MIN_PIECE_M2
    ]
    enclosed = _areal(unary_union(holes).intersection(footprint)) if holes else Polygon()
    radius = COVERAGE_SLOT_RADIUS_M
    closed = union.buffer(radius, join_style=2).buffer(-radius, join_style=2)
    slots = _areal(_areal(closed.difference(union)).intersection(footprint))
    pieces = [g for g in _geom_parts(slots) if g.area >= COVERAGE_MIN_PIECE_M2]
    slot_m2 = float(sum(g.area for g in pieces))
    return {
        "footprint_m2": round(footprint.area, 1),
        "covered_frac": round(union.area / footprint.area, 4),
        "enclosed_m2": round(enclosed.area, 1),
        "enclosed_n": len(_geom_parts(enclosed)),
        "slot_radius_m": radius,
        "slot_m2": round(slot_m2, 1),
        "slot_n": len(pieces),
        "slot_per_1k": round(slot_m2 / (footprint.area / 1000.0), 3),
    }


# The per-sheet metric helpers below share one contract with qa_masks: each
# mutates ``entry`` in place and writes ONLY the keys it measured — an absent
# key means "not measured", never 0.0. The driver owns the skip order (its
# ``continue``s) and decides which helpers a sheet reaches at all.


@dataclass(frozen=True)
class _SheetFrame:
    """The measurement frame the deep per-sheet metric blocks share.

    The driver builds one only after every bail-out has passed: a frame
    exists exactly when the sheet has a final mask, a page rect, and a
    readable ink raster. ``core_k`` is hoisted out of the blank-core block
    because the contested-ground block consumes the same core size.
    """

    sheet: RawSheetMask
    final: Polygon | MultiPolygon
    raster: _InkRaster
    #: meters per small-frame pixel, from the affine's linear part
    m_per_px: float
    #: BLANK_CORE_M in small-frame pixels; 0 = core unmeasurable
    core_k: int


def _handedness(entry: dict[str, Any], sheet: RawSheetMask) -> tuple[float, float, float]:
    """HANDEDNESS — see the module docstring for what this is doing in a
    mask instrument, and for why the perpendicularity is recorded beside
    the flag rather than folded into it.

    Returns ``(signed_det, sx, sy)``: the window re-check reads the scales,
    and the driver builds ``m_per_px`` from the determinant.
    """
    signed_det = model_determinant(sheet.matrix)
    sx, sy = model_scales(sheet.matrix)
    entry["axis_perpendicularity"] = round(abs(signed_det) / (sx * sy), 6) if sx * sy else 0.0
    if signed_det >= 0:
        entry["mirrored"] = True
        entry["flags"].append("mirrored")
    return signed_det, sx, sy


def _window_deviation(
    entry: dict[str, Any],
    sheet: RawSheetMask,
    sx: float,
    sy: float,
    placement_window: tuple[float, float] | None,
    page_scale_multiples: Mapping[str, float] | None,
) -> None:
    """RECORDED-MODEL WINDOW — the accept-time contract, re-checked on the
    model that is finally warped; see the module docstring."""
    page = page_from_slug(sheet.slug)
    if placement_window is not None and page is not None and sx and sy:
        pin_scale, pin_rot = placement_window
        center = pin_scale * (page_scale_multiples or {}).get(page, 1.0)
        if center > 0:
            lo, hi = center * (1 - SCALE_TOL_FRAC), center * (1 + SCALE_TOL_FRAC)
            scale_devs = (sx / center - 1.0, sy / center - 1.0)
            scale_dev = max(scale_devs, key=abs)
            rot_dev = fold_quadrant_deg(model_rotation_deg(sheet.matrix) - pin_rot)
            entry["window_scale_dev_frac"] = round(scale_dev, 4)
            entry["window_rot_dev_deg"] = round(rot_dev, 3)
            # same boundary semantics as gates_ok: strictly inside the
            # scale window, at-or-inside the rotation tolerance
            if not (lo < sx < hi and lo < sy < hi) or abs(rot_dev) > ROT_TOL_DEG:
                entry["flags"].append("outside_window")


def _hull_page_ratio(entry: dict[str, Any], sheet: RawSheetMask) -> None:
    """The mask ring's area against its page rectangle's."""
    rect_area = None
    if sheet.rect is not None:
        x0, y0, x1, y1 = sheet.rect
        rect_area = max(0, x1 - x0) * max(0, y1 - y0)
    if rect_area and sheet.ring_px is not None:
        ring = Polygon(sheet.ring_px)
        if ring.is_valid and not ring.is_empty:
            entry["hull_page_ratio"] = round(ring.area / rect_area, 4)
    elif rect_area and sheet.style == "page":
        entry["hull_page_ratio"] = 1.0


def _raw_overlap(
    entry: dict[str, Any],
    sheet: RawSheetMask,
    sheets: Sequence[RawSheetMask],
    duplicates: DuplicateCoverage,
) -> None:
    """Pre-split whole-sheet overlap between REGULAR sheets — the
    undeclared-mask-style signal (MOSAIC-QUILTING record): for a volume
    that never declared a style the raw masks ARE page rectangles,
    so this is exactly the anomalous-overlap measurement. Duplicate-
    coverage sheets overlap everything by design and are excluded."""
    if duplicate_coverage_slug(sheet.slug, duplicates) or sheet.poly_3857.is_empty:
        return
    worst, worst_slug = 0.0, None
    for other in sheets:
        if other.slug == sheet.slug or duplicate_coverage_slug(other.slug, duplicates):
            continue
        inter = sheet.poly_3857.intersection(other.poly_3857).area
        frac = inter / sheet.poly_3857.area
        if frac > worst:
            worst, worst_slug = frac, other.slug
    if worst_slug is not None:
        entry["raw_overlap_frac_max"] = round(worst, 4)
        entry["raw_overlap_with"] = worst_slug


def _ink_capture(entry: dict[str, Any], frame: _SheetFrame, regular_union: Any) -> None:
    """Own-mask ink capture and volume-wide uncovered ink; flags
    ``hull_collapse``/``uncovered_ink`` (see the module docstring)."""
    sheet, final, raster = frame.sheet, frame.final, frame.raster
    final_small = _to_small_px(final, sheet.matrix, raster.scale)
    mask_small = _rasterize(final_small, raster.ink.shape)
    mask_px = int(mask_small.sum())
    if mask_px:
        entry["ink_frac_in_mask"] = round(float((raster.ink & mask_small).sum()) / mask_px, 4)
    x0s, y0s, x1s, y1s = raster.rect_small
    ink_rect = raster.ink[y0s:y1s, x0s:x1s]
    ink_total = int(ink_rect.sum())
    if ink_total:
        captured = int((ink_rect & mask_small[y0s:y1s, x0s:x1s]).sum())
        entry["ink_captured_frac"] = round(captured / ink_total, 4)
        coverage = _rasterize(
            _to_small_px(regular_union, sheet.matrix, raster.scale), raster.ink.shape
        )
        lost = int((ink_rect & ~coverage[y0s:y1s, x0s:x1s]).sum())
        entry["ink_uncovered_frac"] = round(lost / ink_total, 4)
        if entry["ink_uncovered_frac"] > HULL_COLLAPSE_MIN_UNCOVERED_INK:
            if sheet.style == "hull":
                entry["flags"].append("hull_collapse")
            elif sheet.style in BOX_UNCOVERED_INK_STYLES:
                entry["flags"].append("uncovered_ink")


def _core_blocks(
    mask: NDArray[np.bool_], core_k: int, extra: NDArray[np.bool_] | None = None
) -> int:
    """Count the fully-True :data:`BLANK_CORE_M`-scale blocks of ``mask``.

    With ``extra``, a block must also carry an ``extra`` block mean at or
    above :data:`CONTESTED_INK_FLOOR` (the contested-ground predicate).
    Requires ``core_k > 0``; callers guard (a zero ``core_k`` means the
    core is unmeasurable and the metric is omitted).
    Small-frame pixel blocks only: :mod:`.move`'s core cells live on a
    world-coordinate grid and deliberately do not share this.
    """
    h, w = mask.shape
    hk, wk = h // core_k, w // core_k
    if not (hk and wk):
        return 0
    blocks = mask[: hk * core_k, : wk * core_k].reshape(hk, core_k, wk, core_k)
    full = blocks.all(axis=(1, 3))
    if extra is not None:
        li = extra[: hk * core_k, : wk * core_k].reshape(hk, core_k, wk, core_k)
        full = full & (li.mean(axis=(1, 3)) >= CONTESTED_INK_FLOOR)
    return int(full.sum())


def _blank_core(
    entry: dict[str, Any],
    frame: _SheetFrame,
    finals_3857: Mapping[str, Polygon | MultiPolygon],
    paint_pos: Mapping[str, int],
    content_masks: bool,
) -> None:
    """Blank paper this mask paints over sheets BELOW it in paint order;
    the core measure keeps only patches >= BLANK_CORE_M wide both ways.
    This is the blank-MARGIN signal (flag + auto-exemption veto); it
    fires only where final masks overlap, i.e. duplicate coverage."""
    sheet, final, raster = frame.sheet, frame.final, frame.raster
    core_k, m_per_px = frame.core_k, frame.m_per_px
    core_m2, covered = 0.0, []
    for other_slug, other_final in finals_3857.items():
        if paint_pos[other_slug] >= paint_pos[sheet.slug]:
            continue
        overlap = final.intersection(other_final)
        if overlap.is_empty or overlap.area < SLIVER_AREA_M2:
            continue
        overlap_small = _rasterize(
            _to_small_px(overlap, sheet.matrix, raster.scale), raster.ink.shape
        )
        if not overlap_small.any():
            continue
        covered.append(other_slug)
        if core_k:
            blank = overlap_small & ~raster.ink
            core_m2 += float(_core_blocks(blank, core_k)) * (core_k * m_per_px) ** 2
    entry["blank_core_m2"] = round(core_m2, 1)
    if covered:
        entry["blank_over"] = sorted(covered, key=page_sort_key)
    if (
        content_masks
        and sheet.style in ("hull", "content_box", "page")
        and core_m2 > BLANK_OVERPAINT_MIN_CORE_M2
    ):
        entry["flags"].append("blank_overpaint")


def _contested_ground(
    entry: dict[str, Any],
    frame: _SheetFrame,
    sheets: Sequence[RawSheetMask],
    finals_3857: Mapping[str, Polygon | MultiPolygon],
    rasters: MutableMapping[tuple[str, tuple[int, int, int, int]], _InkRaster | None],
    duplicates: DuplicateCoverage,
) -> None:
    """Measure blank paper over a neighbour's ink on split-CONTESTED ground.

    Contested ground is where this sheet's FINAL mask covers ground a
    neighbour's RAW mask claimed but lost. Regular sheets only — the split
    partitions only them. Diagnostics only, no flag.
    """
    sheet, final, raster = frame.sheet, frame.final, frame.raster
    core_k, m_per_px = frame.core_k, frame.m_per_px
    contested_blank = np.zeros(raster.ink.shape, dtype=bool)
    loser_ink = np.zeros(raster.ink.shape, dtype=bool)
    losers = []
    for other in sheets:
        other_final = finals_3857.get(other.slug)
        if (
            other.slug == sheet.slug
            or duplicate_coverage_slug(other.slug, duplicates)
            or other_final is None
            or other.poly_3857.is_empty
        ):
            continue
        raw_inter = sheet.poly_3857.intersection(other.poly_3857)
        if raw_inter.is_empty or raw_inter.area < SLIVER_AREA_M2:
            continue
        region = final.intersection(other.poly_3857).difference(other_final)
        if region.is_empty or region.area < SLIVER_AREA_M2:
            continue
        region_small = _rasterize(
            _to_small_px(region, sheet.matrix, raster.scale), raster.ink.shape
        )
        blank = region_small & ~raster.ink
        rows, cols = np.nonzero(blank)
        if rows.size == 0:
            continue
        contested_blank |= blank
        other_raster = ink_raster_for(other, rasters)
        if other_raster is None:
            continue
        ink_b = _ink_at_points(other_raster, other.matrix, sheet.matrix, raster.scale, rows, cols)
        if ink_b.any():
            loser_ink[rows[ink_b], cols[ink_b]] = True
            losers.append(other.slug)
    entry["blank_over_neighbor_m2"] = round(float(loser_ink.sum()) * m_per_px**2, 1)
    contested_core_m2 = 0.0
    if core_k and losers:
        full = _core_blocks(contested_blank, core_k, extra=loser_ink)
        contested_core_m2 = float(full) * (core_k * m_per_px) ** 2
    entry["blank_over_neighbor_core_m2"] = round(contested_core_m2, 1)
    if losers:
        entry["blank_over_neighbor"] = sorted(losers, key=page_sort_key)


def qa_masks(
    volume: str,
    sheets: Sequence[RawSheetMask],
    final_polys_4326: Mapping[str, Polygon | MultiPolygon | None],
    *,
    content_masks: bool = False,
    duplicates: DuplicateCoverage,
    ink_rasters: MutableMapping[tuple[str, tuple[int, int, int, int]], _InkRaster | None]
    | None = None,
    placement_window: tuple[float, float] | None = None,
    page_scale_multiples: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Per-sheet mask metrics and flags for one bake: the ``masks-qa.json`` document.

    ``final_polys_4326`` maps slug to the mask as written, or None for a sheet
    the heal ladder exhausted. ``content_masks`` gates ``blank_overpaint``;
    ``ink_rasters`` is the shared cache the bake passes so rasters build once
    across the move and every scoring call; omitting ``placement_window``
    leaves ``outside_window`` unmeasured, never clean. A metric that cannot be
    measured for a sheet is omitted rather than guessed — `docs/INTERNALS.md`.
    """
    from shapely.ops import unary_union

    order = mosaic_paint_order(sorted(set(final_polys_4326) | {s.slug for s in sheets}), duplicates)
    paint_pos = {slug: i for i, slug in enumerate(order)}
    finals_3857: dict[str, Polygon | MultiPolygon] = {}
    for slug, final in final_polys_4326.items():
        if final is not None:
            finals_3857[slug] = shp_transform(TO_3857.transform, final)
    regular_union = unary_union(
        [p for slug, p in finals_3857.items() if not duplicate_coverage_slug(slug, duplicates)]
    )

    out_sheets: dict[str, dict[str, Any]] = {}
    # shared ink-raster cache: the contested-ground metric reads each
    # neighbour's raster as well as the sheet's own
    rasters = {} if ink_rasters is None else ink_rasters

    # The ``continue``s below are the skip order — a sheet a bail-out stops
    # never reaches the helpers past it (helper contract: see their header).
    for sheet in sorted(sheets, key=lambda s: page_sort_key(s.slug)):
        entry: dict[str, Any] = {"style": sheet.style, "flags": []}
        out_sheets[sheet.slug] = entry
        signed_det, sx, sy = _handedness(entry, sheet)
        _window_deviation(entry, sheet, sx, sy, placement_window, page_scale_multiples)
        _hull_page_ratio(entry, sheet)
        _raw_overlap(entry, sheet, sheets, duplicates)

        final = finals_3857.get(sheet.slug)
        if final is None:
            entry["unmasked"] = True
            continue
        entry["mask_area_m2"] = round(final.area, 1)

        if sheet.rect is None:
            continue
        raster = ink_raster_for(sheet, rasters)
        if raster is None:
            continue
        m_per_px = math.sqrt(abs(signed_det)) / raster.scale
        core_k = max(1, round(BLANK_CORE_M / m_per_px)) if m_per_px > 0 else 0
        frame = _SheetFrame(sheet, final, raster, m_per_px, core_k)
        _ink_capture(entry, frame, regular_union)
        _blank_core(entry, frame, finals_3857, paint_pos, content_masks)

        if duplicate_coverage_slug(sheet.slug, duplicates) or sheet.poly_3857.is_empty:
            continue
        _contested_ground(entry, frame, sheets, finals_3857, rasters, duplicates)

    flagged = {slug: e["flags"] for slug, e in out_sheets.items() if e["flags"]}
    coverage_stats: dict[str, Any] = {}
    try:
        # the union the DETAIL mosaic serves: regular sheets plus skeleton
        # twins (their fallback paint under a slot is content a reader sees);
        # declared overviews bake into a separate archive nothing serves and
        # stay out of both sides of the metric
        detail_union = unary_union(
            [p for slug, p in finals_3857.items() if not overview_slug(slug, duplicates)]
        )
        coverage_stats = volume_coverage(sheets, detail_union, duplicates)
    except Exception:
        # omitted rather than guessed, like any other unmeasurable metric —
        # but never at the cost of the per-sheet document
        logger.exception("mask QA: volume coverage metric failed; omitted")
    doc: dict[str, Any] = {
        "volume": volume,
        "work_width": QA_WORK_WIDTH,
        "thresholds": {
            "hull_collapse_min_uncovered_ink": HULL_COLLAPSE_MIN_UNCOVERED_INK,
            "uncovered_ink_min_frac": HULL_COLLAPSE_MIN_UNCOVERED_INK,
            "blank_overpaint_min_core_m2": BLANK_OVERPAINT_MIN_CORE_M2,
            "coverage_gaps_max_slot_per_1k": COVERAGE_GAPS_MAX_SLOT_PER_1K,
        },
        "flagged": flagged,
        "sheets": out_sheets,
    }
    if coverage_stats:
        # absent keys mean "not measured" to every reader; an unmeasurable
        # volume must not publish "measured clean"
        doc["coverage"] = coverage_stats
        volume_flags: list[str] = []
        if coverage_stats["slot_per_1k"] > COVERAGE_GAPS_MAX_SLOT_PER_1K:
            volume_flags.append("coverage_gaps")
        doc["volume_flags"] = volume_flags
    return doc


def write_masks_qa(
    paths: VolumePaths,
    volume: str,
    sheets: Sequence[RawSheetMask],
    final_polys_4326: Mapping[str, Polygon | MultiPolygon | None],
    *,
    content_masks: bool = False,
    duplicates: DuplicateCoverage,
    doc: dict[str, Any] | None = None,
    auto_exempted: Sequence[str] = (),
) -> Path:
    """Persist ``masks/masks-qa.json``; log any flags loudly.

    ``doc`` skips recomputation when the caller already ran :func:`qa_masks`
    on exactly the ``sheets``/``final_polys_4326`` being persisted (the bake's
    auto-exemption loop does). ``auto_exempted`` names sheets whose collapsed
    hull the bake replaced with the page rectangle after the QA re-check
    accepted the swap; recorded in the document so the remedy is auditable.
    """
    if doc is None:
        doc = qa_masks(
            volume,
            sheets,
            final_polys_4326,
            content_masks=content_masks,
            duplicates=duplicates,
        )
    doc["auto_exempted"] = sorted(auto_exempted, key=page_sort_key)
    if doc["flagged"]:
        logger.warning(
            "mask QA: %d flagged sheet(s) in %s — %s (see masks/masks-qa.json)",
            len(doc["flagged"]),
            volume,
            ", ".join(f"{slug}: {'+'.join(flags)}" for slug, flags in doc["flagged"].items()),
        )
    if doc.get("volume_flags"):
        logger.warning(
            "mask QA: volume flag(s) on %s — %s (see masks/masks-qa.json)",
            volume,
            ", ".join(doc["volume_flags"]),
        )
    return write_if_changed(paths.masks / "masks-qa.json", json.dumps(doc, indent=2))


def qa_note(qa_doc: Mapping[str, Any]) -> str | None:
    """One-line mask-QA summary for the volume report and status, or None."""
    flagged = qa_doc.get("flagged") or {}
    auto_exempted = qa_doc.get("auto_exempted") or []
    volume_flags = qa_doc.get("volume_flags") or []
    if not flagged and not auto_exempted and not volume_flags:
        return None

    def page(slug: str) -> str:
        p = page_from_slug(slug)
        return f"p{p}" if p else slug

    parts = []
    if flagged:
        detail = ", ".join(
            f"{page(slug)} ({'+'.join(flags)})"
            for slug, flags in sorted(flagged.items(), key=lambda kv: page_sort_key(kv[0]))
        )
        parts.append(
            f"mask QA flags on {len(flagged)} sheet(s): {detail} — see masks/masks-qa.json"
        )
    if auto_exempted:
        pages = ", ".join(page(slug) for slug in auto_exempted)
        parts.append(
            f"mask QA auto-exempted {len(auto_exempted)} collapsed hull(s): {pages} "
            "— page rectangle passed the QA re-check"
        )
    if volume_flags:
        slot = (qa_doc.get("coverage") or {}).get("slot_per_1k")
        detail = ", ".join(volume_flags)
        parts.append(
            f"mask QA volume flag(s): {detail}"
            + (
                f" — {slot} m² of uncovered inter-sheet slot per 1000 m² of page footprint"
                if slot is not None
                else ""
            )
            + " — see masks/masks-qa.json"
        )
    return "; ".join(parts)


def load_masks_qa(masks_dir: Path) -> dict[str, Any] | None:
    """The persisted QA document, or None when absent or unreadable."""
    qa_path = masks_dir / "masks-qa.json"
    if not qa_path.is_file():
        return None
    try:
        doc = json.loads(qa_path.read_text())
        if not isinstance(doc, dict):
            raise ValueError("masks-qa.json is not a JSON object")
    except (OSError, ValueError) as exc:
        logger.warning("%s: unreadable mask QA document (%s)", qa_path, exc)
        return None
    return doc
