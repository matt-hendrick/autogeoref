"""Mask chain: extent detection, overlap splitting, and the heal ladder.

A regular sheet is bounded by :func:`detect_content_box` — the bbox of its
printed colour, padded outward — falling back to :func:`detect_page_bounds`'s
page rectangle when the sheet carries no colour. :func:`split_overlaps` gives
each sheet the overlapping ground nearest its own mask centroid.
:func:`heal` then walks an escalating ladder until a rung passes an injected
acceptance test: the real ``gdalwarp`` cutline dry-run, because shapely and
OGR ``IsValid`` do not predict GDAL cutline acceptance.

Frames: detection is in pixels, splitting and healing in EPSG:3857 planar
metres, stored masks in EPSG:4326. A raw or split mask is one Polygon;
everything downstream of the blank-core move takes MultiPolygon and holed
masks too. `docs/INTERNALS.md` explains the chain and why the colour box.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageFilter
from shapely import wkt as shapely_wkt
from shapely.geometry import MultiPoint, MultiPolygon, Polygon, box
from shapely.ops import transform as shp_transform

from autogeoref.affine import TO_3857, TO_4326, AffineMatrix, apply_affine
from autogeoref.pillow import unlimited_image_pixels
from autogeoref.slugs import page_sort_key

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: Overlaps smaller than this (m^2, EPSG:3857) are slivers and left alone.
SLIVER_AREA_M2 = 25.0

#: Heal-ladder inward buffer distances in meters (0 = clean only).
HEAL_SHRINK_METERS = (0.0, 0.02, 0.1, 0.5, 2.0)


class MaskError(Exception):
    """Base error for the mask module."""


class PageBoundsError(MaskError):
    """Page-bounds detection could not run on the given image."""


def _longest_true_runs(rows: NDArray[np.bool_]) -> NDArray[np.intp]:
    """Longest run of True along axis 1, per row (vectorized)."""
    n = rows.shape[1]
    idx = np.arange(n)
    # last_false[i, j] = index of the last False at or before column j (-1 if none),
    # so idx - last_false is the length of the True run ending at j.
    last_false = np.maximum.accumulate(np.where(~rows, idx, -1), axis=1)
    runs: NDArray[np.intp] = (idx[np.newaxis, :] - last_false).max(axis=1)
    return runs


def _first_long_run(runs: NDArray[np.intp], threshold: float, limit: int, start: int = 0) -> int:
    """First index in ``runs[start..limit]`` with run >= threshold, else ``start``."""
    if start > limit:
        return start
    hits = np.nonzero(runs[start : limit + 1] >= threshold)[0]
    return start + int(hits[0]) if hits.size else start


#: A scan-border row/column is dark almost everywhere; printed linework never
#: is (after the bezel the strongest column in a sampled sheet's search band is
#: 9% dark). Leading rows/columns at or above this dark fraction are the LOC
#: scanner bezel, and the page-bounds search starts past them.
_BEZEL_DARK_FRAC = 0.9

#: Hard cap on the skipped band, as a fraction of the cross dimension. The
#: bezel is a scan border a few pixels wide, so the cap bounds a mis-read: the
#: worst the skip can do is start the search this far in, never deep enough to
#: let a long dark rail corridor inside the map pass for the page's edge. Too
#: big beats truncated, and that asymmetry is the one worth keeping.
_BEZEL_MAX_FRAC = 0.05


def _skip_dark_band(dark_frac: NDArray[np.float64], limit: int) -> int:
    """First index in ``dark_frac`` below :data:`_BEZEL_DARK_FRAC`.

    Searched over the leading :data:`_BEZEL_MAX_FRAC` of the dimension, capped
    by ``limit``. Returns 0 when every index in that band is that dark: there
    is then no non-bezel index to start from, and searching from the edge is
    better than refusing to search.
    """
    band = min(limit, int(len(dark_frac) * _BEZEL_MAX_FRAC))
    misses = np.nonzero(dark_frac[: band + 1] < _BEZEL_DARK_FRAC)[0]
    return int(misses[0]) if misses.size else 0


def detect_page_bounds(
    image_path: Path,
    work_width: int = 1200,
    dark_thresh: int = 120,
    run_frac: float = 0.55,
    inset_frac: float = 0.25,
    pad_px: int = 3,
) -> tuple[int, int, int, int]:
    """Bound the scanned page; return ``(x0, y0, x1, y1)`` full-res pixels.

    Downscaled to ``work_width``; pixels darker than ``dark_thresh`` count.
    From each side the search skips the bezel band, then takes the first
    row/column whose longest dark run covers ``run_frac`` of the cross
    dimension, within the inner ``inset_frac``. **This is the page, not a
    content bound** — long dark linework inside the map can truncate it, and
    regular sheets use :func:`detect_content_box` instead. Raises
    ``PageBoundsError`` when the image is unopenable or degenerate.
    """
    # Full-res Sanborn scans exceed PIL's decompression-bomb default; lift the
    # cap only around this open instead of process-globally at import time.
    try:
        with unlimited_image_pixels(), Image.open(image_path) as img:
            gray = img.convert("L")
    except OSError as exc:
        raise PageBoundsError(f"cannot open {image_path}: {exc}") from exc
    w, h = gray.size
    if w < 2 or h < 2:
        raise PageBoundsError(f"degenerate image {w}x{h}: {image_path}")
    scale = work_width / w
    small = gray.resize((work_width, round(h * scale)))
    dark = np.asarray(small) < dark_thresh
    sh, sw = dark.shape

    col_runs = _longest_true_runs(dark.T)  # vertical dark run per column
    row_runs = _longest_true_runs(dark)  # horizontal dark run per row
    col_dark = dark.mean(axis=0)  # dark fraction per column
    row_dark = dark.mean(axis=1)  # dark fraction per row
    lim_x, lim_y = int(sw * inset_frac), int(sh * inset_frac)

    left = _first_long_run(col_runs, run_frac * sh, lim_x, _skip_dark_band(col_dark, lim_x))
    right = (
        sw
        - 1
        - _first_long_run(
            col_runs[::-1], run_frac * sh, lim_x, _skip_dark_band(col_dark[::-1], lim_x)
        )
    )
    top = _first_long_run(row_runs, run_frac * sw, lim_y, _skip_dark_band(row_dark, lim_y))
    bottom = (
        sh
        - 1
        - _first_long_run(
            row_runs[::-1], run_frac * sw, lim_y, _skip_dark_band(row_dark[::-1], lim_y)
        )
    )

    # small inward nudge so a line the search stopped on is itself trimmed
    padded = (left + pad_px, top + pad_px, right - pad_px, bottom - pad_px)
    x0, y0, x1, y1 = (round(v / scale) for v in padded)
    return (x0, y0, x1, y1)


#: "Drawn in color" = channel spread above the frame median plus this margin,
#: never below the floor (blank paper itself spreads ~25-30).
_CONTENT_SAT_FLOOR = 40
_CONTENT_SAT_MARGIN = 15

#: Opening sizes (small-frame px): the speck pass removes scan noise and dot
#: symbols; the blob pass removes stamps and stray marks that would drag the
#: hull into blank corners. Content blocks are far larger and survive both.
CONTENT_SPECK_PX = 5
_CONTENT_BLOB_PX = 15

#: Hull padding as a fraction of the page rectangle's width — about half a
#: street, so adjacent sheets' masks meet mid-street and the split leaves
#: no gap.
CONTENT_HULL_MARGIN_FRAC = 0.025

#: Above this hull/page-rectangle area ratio the rectangle serves better.
_CONTENT_HULL_KEEP_RATIO = 0.95

#: "Drawn content" = coloured wash OR linework this dark. Later Sanborn eras
#: carry much of the map in near-monochrome line: saturation alone is blind to
#: exactly the sheets a colour box would gut. Mask QA measures ink against the
#: same bar (:mod:`.qa`'s ``INK_DARK_THRESH`` is this constant).
CONTENT_INK_DARK_THRESH = 120


def _opened(mask: NDArray[np.bool_], size: int) -> NDArray[np.bool_]:
    """Morphological opening (erode then dilate) of a boolean raster."""
    image = Image.fromarray((mask * 255).astype(np.uint8))
    eroded = image.filter(ImageFilter.MinFilter(size))
    return np.asarray(eroded.filter(ImageFilter.MaxFilter(size))) > 0


def _content_signals(
    image_path: Path, rect: tuple[int, int, int, int], work_width: int
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], float, tuple[int, int, int, int]] | None:
    """The sheet's printed colour and its drawn content, inside ``rect``.

    Shared by :func:`detect_content_hull` and :func:`detect_content_box` so
    they can never measure different pixels. Returns ``(colour blobs, ink,
    small-per-full-res scale, small-frame rect)``, both rasters indexed from
    that rect's origin; None when the image is unusable or has no colour.

    Raises:
        PageBoundsError: If the image cannot be opened.
    """
    try:
        with unlimited_image_pixels(), Image.open(image_path) as img:
            w, h = img.size
            if w < 2 or h < 2:
                return None
            scale = work_width / w
            small = img.convert("RGB").resize((work_width, round(h * scale)))
    except OSError as exc:
        raise PageBoundsError(f"cannot open {image_path}: {exc}") from exc
    x0, y0, x1, y1 = (round(v * scale) for v in rect)
    frame = np.asarray(small, dtype=np.int16)[y0:y1, x0:x1]
    if frame.size == 0:
        return None
    saturation = frame.max(axis=2) - frame.min(axis=2)
    threshold = max(_CONTENT_SAT_FLOOR, int(np.median(saturation)) + _CONTENT_SAT_MARGIN)
    colored = saturation > threshold

    blobs = _opened(colored, CONTENT_SPECK_PX)
    if blobs.any():
        blobs = _opened(blobs, _CONTENT_BLOB_PX)
    if not blobs.any():
        return None
    ink = _opened(colored | (frame.sum(axis=2) // 3 < CONTENT_INK_DARK_THRESH), CONTENT_SPECK_PX)
    return blobs, ink, scale, (x0, y0, x1, y1)


#: Outward pad on the printed-colour box, as a fraction of the rectangle's
#: width. Load-bearing and not to be shrunk: the colour bounds coloured
#: BUILDINGS, but the mapped area runs out to the bounding streets, roughly
#: half a block further, and a much smaller pad opens visible gaps between
#: sheets.
CONTENT_BOX_PAD_FRAC = 0.08

#: A content box may never keep less than this share of either page dimension;
#: it is expanded symmetrically about its centre and re-clipped when it would.
#: This is what stops a sparse sheet collapsing onto its densest corner — the
#: hull-collapse hazard, which a bbox reduces but does not remove.
CONTENT_BOX_MIN_KEEP = 0.60

#: A colour box is REFUSED unless it still contains this share of the sheet's
#: own drawn content. Colour is not where the map is on every era — the later
#: ones draw much of it in near-monochrome line, and a box there can cut most
#: of a sheet's content away. Below the bar the page rectangle is served
#: instead: the sheet stays untidy, never truncated.
CONTENT_BOX_MIN_INK_KEPT = 0.85


def detect_content_box(
    image_path: Path,
    rect: tuple[int, int, int, int],
    work_width: int = 1200,
    pad_frac: float = CONTENT_BOX_PAD_FRAC,
    min_keep: float = CONTENT_BOX_MIN_KEEP,
    min_ink_kept: float = CONTENT_BOX_MIN_INK_KEPT,
) -> tuple[int, int, int, int] | None:
    """Bbox of the sheet's printed colour, padded outward, in full-res pixels.

    The DEFAULT bound for a regular sheet: the scanned page is mostly blank
    margin, so masking by the page makes neighbours overlap on blank paper and
    the split cuts mid-block diagonals. A **bbox**, deliberately not a hull —
    a hull contracts onto the densest cluster of a sparse sheet. ``min_keep``
    bounds what is left of that hazard; ``min_ink_kept`` refuses the box
    outright when it would cut the sheet's own drawn content. Returns the box
    clipped to ``rect``, or None when refused — the caller keeps the page.
    """
    found = _content_signals(image_path, rect, work_width)
    if found is None:
        return None
    blobs, ink, scale, (sx0, sy0, _sx1, _sy1) = found
    ys, xs = np.nonzero(blobs)
    box_px = [
        (sx0 + float(xs.min())) / scale,
        (sy0 + float(ys.min())) / scale,
        (sx0 + float(xs.max())) / scale,
        (sy0 + float(ys.max())) / scale,
    ]
    rx0, ry0, rx1, ry1 = rect
    pad = pad_frac * (rx1 - rx0)
    box_px = [box_px[0] - pad, box_px[1] - pad, box_px[2] + pad, box_px[3] + pad]
    for low, high, floor in (
        (0, 2, min_keep * (rx1 - rx0)),
        (1, 3, min_keep * (ry1 - ry0)),
    ):
        if box_px[high] - box_px[low] < floor:
            middle = (box_px[low] + box_px[high]) / 2
            box_px[low], box_px[high] = middle - floor / 2, middle + floor / 2
    x0 = max(rx0, round(box_px[0]))
    y0 = max(ry0, round(box_px[1]))
    x1 = min(rx1, round(box_px[2]))
    y1 = min(ry1, round(box_px[3]))
    if x0 >= x1 or y0 >= y1:  # unreachable for a non-degenerate rect; never serve a void
        return None
    drawn = int(ink.sum())
    if drawn:
        kept = int(
            ink[
                round(y0 * scale) - sy0 : round(y1 * scale) - sy0,
                round(x0 * scale) - sx0 : round(x1 * scale) - sx0,
            ].sum()
        )
        if kept < min_ink_kept * drawn:
            logger.info(
                "content box refused for %s: keeps %.3f of the sheet's drawn content",
                image_path.name,
                kept / drawn,
            )
            return None
    return (x0, y0, x1, y1)


def detect_content_hull(
    image_path: Path,
    rect: tuple[int, int, int, int],
    work_width: int = 1200,
) -> list[tuple[float, float]] | None:
    """Convex hull of the sheet's colored content, in full-res pixels.

    For volumes declaring ``VolumeConfig.content_masks``, where each sheet
    details one block in a mostly-blank frame and page rectangles would bury
    every block under a neighbour's paper. **Never inferred per sheet**: on a
    sparse-but-fully-mapped standard sheet the hull collapses onto the densest
    cluster and deletes served content, and no pixel statistic separates the
    formats. Returns an open ring of pixel vertices clipped to ``rect``.
    """
    found = _content_signals(image_path, rect, work_width)
    if found is None:
        return None
    blobs, _ink, scale, (x0, y0, x1, y1) = found
    ys, xs = np.nonzero(blobs)
    hull = MultiPoint(np.column_stack([xs, ys])).convex_hull
    margin = max(1.0, CONTENT_HULL_MARGIN_FRAC * (x1 - x0))
    hull = hull.buffer(margin, join_style="mitre").convex_hull
    hull = hull.intersection(box(0, 0, x1 - x0, y1 - y0))
    if hull.geom_type != "Polygon" or hull.is_empty:
        return None
    if hull.area >= _CONTENT_HULL_KEEP_RATIO * (x1 - x0) * (y1 - y0):
        return None
    return [((px + x0) / scale, (py + y0) / scale) for px, py in hull.exterior.coords[:-1]]


#: Pad around an overview sheet's inlier-GCP hull, EPSG:3857 meters — about a
#: street width, so the boundary streets the outermost GCPs pin still render
#: whole instead of being cut mid-carriageway.
GCP_HULL_MARGIN_M = 30.0


def clip_to_gcp_hull(
    mask: Polygon,
    gcp_points_m: Sequence[tuple[float, float]],
    margin_m: float = GCP_HULL_MARGIN_M,
) -> Polygon:
    """Clip an overview sheet's mask to the hull its inlier GCPs earned.

    An overview sheet is committed full-page as fallback coverage, but its fit
    constrains only the ground between its inliers: outside their convex hull
    it extrapolates across ground other sheets — including a NEIGHBOURING
    volume's, where paint order cannot arbitrate — actually earned. All
    EPSG:3857. Returns the input unchanged when the hull is degenerate or the
    clip would void the mask; keeping fallback paint beats serving nothing.
    """
    hull = MultiPoint([(x, y) for x, y in gcp_points_m]).convex_hull
    if hull.geom_type != "Polygon":
        logger.warning(
            "clip_to_gcp_hull: degenerate GCP hull (%s); mask kept whole", hull.geom_type
        )
        return mask
    # round join: a mitre would overshoot an acute hull vertex by up to the
    # mitre limit, retaining exactly the unearned ground the clip removes
    clipped = mask.intersection(hull.buffer(margin_m))
    if clipped.geom_type != "Polygon" or clipped.is_empty:
        logger.warning("clip_to_gcp_hull: clip voided the mask; mask kept whole")
        return mask
    clipped = _clean(clipped)
    if clipped.is_valid and not clipped.is_empty:
        return clipped
    logger.warning("clip_to_gcp_hull: clipped mask failed cleaning; mask kept whole")
    return mask


def mask_polygon_4326(
    rect: tuple[float, float, float, float], affine_matrix: AffineMatrix
) -> Polygon:
    """Project a full-res pixel rectangle through the GCP affine into 4326.

    ``affine_matrix`` is the 2x3 poly1 matrix with ``[X, Y] = M @ [1, px, py]``
    mapping pixels to EPSG:3857 metres. Returns the 4-corner rectangle as an
    EPSG:4326 polygon — poly1 is affine, so corners suffice.
    """
    x0, y0, x1, y1 = rect
    corners_px = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    ring = [TO_4326.transform(*apply_affine(affine_matrix, px, py)) for px, py in corners_px]
    return Polygon(ring)


def _half_plane_toward(
    keep: tuple[float, float], other: tuple[float, float], big: float = 100000.0
) -> Polygon | None:
    """Half-plane of points nearer ``keep`` than ``other`` (both (x, y) meters)."""
    mx, my = (keep[0] + other[0]) / 2, (keep[1] + other[1]) / 2
    dx, dy = other[0] - keep[0], other[1] - keep[1]
    n = math.hypot(dx, dy)
    if n == 0:
        return None
    dx, dy = dx / n, dy / n  # unit vector keep -> other
    px, py = -dy, dx  # perpendicular (bisector direction)
    # quad: bisector segment through midpoint, extruded toward `keep`
    return Polygon(
        [
            (mx + px * big, my + py * big),
            (mx - px * big, my - py * big),
            (mx - px * big - dx * big, my - py * big - dy * big),
            (mx + px * big - dx * big, my + py * big - dy * big),
        ]
    )


def _clean(geom: Polygon) -> Polygon:
    """Force a difference result into a valid, sliver-free single polygon.

    Bisector cuts can leave self-intersections and micro-slivers that GDAL
    later rejects ('Cutline polygon is invalid'). ``buffer(0)`` heals
    self-intersections; ``simplify(2cm)`` drops collinear noise; for a
    MultiPolygon the dominant part is kept only if it holds >= 98% of the
    area (otherwise the cut was a real split and the caller will reject it).
    """
    g = geom.buffer(0)
    if g.geom_type == "MultiPolygon":
        parts = sorted(g.geoms, key=lambda p: p.area, reverse=True)
        if not parts or parts[0].area < 0.98 * g.area:
            return geom  # a real split, not slivers — caller will reject
        g = parts[0]
    result: Polygon = g.simplify(0.02).buffer(0)
    return result


def _usable(new: Polygon, old: Polygon) -> bool:
    """A trim must never disconnect, hollow out, or invalidate a mask."""
    return bool(
        new.geom_type == "Polygon"
        and new.is_valid
        and not new.is_empty
        and new.area > 0.5 * old.area
    )


def split_overlaps(masks: dict[str, Polygon]) -> dict[str, Polygon]:
    """Award overlapping ground to the sheet whose mask centroid is nearest.

    Sanborn sheets deliberately repeat their neighbours' edge content. Every
    half-plane comes from the ORIGINAL centroids and all are removed in ONE
    pass, so a point under three or more masks is kept by exactly the
    nearest-centroid sheet: a Voronoi partition whose corners close by
    construction and whose result ignores visit order. Overlaps under
    :data:`SLIVER_AREA_M2` are left alone; :func:`_usable` refuses a cut taking
    half of what a sheet has left. slug -> EPSG:3857 polygon (`docs/INTERNALS.md`).
    """
    slugs = sorted(masks, key=page_sort_key)
    centroids = {slug: masks[slug].centroid.coords[0] for slug in slugs}
    out: dict[str, Polygon] = {}
    total_trim = 0.0
    for slug_a in slugs:
        a = masks[slug_a]
        keep = a
        for slug_b in slugs:
            if slug_b == slug_a:
                continue
            inter = a.intersection(masks[slug_b])
            if inter.is_empty or inter.area < SLIVER_AREA_M2:
                continue
            half_a = _half_plane_toward(centroids[slug_a], centroids[slug_b])
            if half_a is None:  # coincident centroids: no bisector exists
                continue
            lost = inter.difference(half_a)
            if lost.is_empty:
                continue
            trimmed = _clean(keep.difference(lost))
            # never let a trim disconnect, hollow out, or invalidate a mask
            if _usable(trimmed, keep):
                keep = trimmed
            else:
                logger.info(
                    "split_overlaps: %s keeps its ground against %s (trim not usable)",
                    slug_a,
                    slug_b,
                )
        total_trim += a.area - keep.area
        out[slug_a] = keep
    logger.info("split_overlaps: %.1f m^2 of overlap resolved", total_trim)
    return out


def _significant_parts(multi: MultiPolygon) -> Polygon | MultiPolygon:
    """Parts of a MultiPolygon above sliver scale (the largest if none are)."""
    parts = [p for p in multi.geoms if p.area >= SLIVER_AREA_M2]
    if not parts:
        return max(multi.geoms, key=lambda p: p.area)
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def _shrink(polygon: Polygon | MultiPolygon, meters: float) -> Polygon | MultiPolygon | None:
    """One heal-ladder rung: clean in 3857, optionally shrink, round-trip.

    Buffers/simplifies in EPSG:3857 meters, transforms back to 4326, and
    rounds coordinates to 7 decimals (near-duplicate vertices from
    reprojection rounding are a known GDAL trip-up). A Polygon input keeps its
    dominant part exactly as before the convention widened; a multi-part
    input (a blank-core-move result) keeps every part above sliver scale.
    Returns None when the result is not non-empty areal geometry.
    """
    keep_parts = polygon.geom_type == "MultiPolygon"
    try:
        m = shp_transform(TO_3857.transform, polygon).buffer(0)
        if meters:
            m = m.buffer(-meters).buffer(0)
        m = m.simplify(max(meters, 0.02))
        if m.geom_type == "MultiPolygon":
            m = _significant_parts(m) if keep_parts else max(m.geoms, key=lambda p: p.area)
        back = shp_transform(TO_4326.transform, m).buffer(0)
        cand = shapely_wkt.loads(shapely_wkt.dumps(back, rounding_precision=7)).buffer(0)
        if cand.geom_type == "MultiPolygon" and not keep_parts:
            cand = max(cand.geoms, key=lambda p: p.area)
    except Exception:  # noqa: BLE001 - hopeless geometry is a normal rung failure, not a crash
        logger.debug("heal shrink(%.2f m) failed", meters, exc_info=True)
        return None
    if cand.geom_type in ("Polygon", "MultiPolygon") and not cand.is_empty:
        return cand
    return None


def snap_clean(polygon: Polygon | MultiPolygon) -> Polygon | MultiPolygon:
    """Unconditional ~1 cm snap-clean, applied to EVERY mask before use.

    The mosaic trim intersects masks with FRESHLY-warped extents; a mask that
    passes every validator against the stored file can still go invalid there,
    a path no dry-run against the stored raster can see. Rounding to 7 decimals
    (~1 cm) plus ``buffer(0)`` removes the near-degenerate vertices that break
    that intersection. A fragmented Polygon keeps its dominant part; a
    MultiPolygon keeps its parts. Falls back to the input if snapping degrades
    it.
    """
    keep_parts = polygon.geom_type == "MultiPolygon"
    try:
        g = polygon.buffer(0)
        snapped = shapely_wkt.loads(shapely_wkt.dumps(g, rounding_precision=7)).buffer(0)
        if snapped.geom_type == "MultiPolygon" and not keep_parts:
            snapped = max(snapped.geoms, key=lambda p: p.area)
    except Exception:  # noqa: BLE001 - hopeless geometry keeps the original
        logger.debug("snap_clean failed; keeping original", exc_info=True)
        return polygon
    ok_types = ("Polygon", "MultiPolygon") if keep_parts else ("Polygon",)
    if snapped.geom_type in ok_types and snapped.is_valid and not snapped.is_empty:
        return snapped
    return polygon


def _survives_reprojection(polygon: Polygon | MultiPolygon) -> bool:
    """Cheap pre-check: shapely-valid in 4326 AND after the 3857 transform."""
    if not polygon.is_valid:
        return False
    m3857 = shp_transform(TO_3857.transform, polygon)
    return bool(m3857.is_valid)


def heal(
    polygon: Polygon | MultiPolygon,
    accepts: Callable[[Polygon | MultiPolygon], bool],
) -> Polygon | MultiPolygon | None:
    """Escalating heal ladder for a mask GDAL rejects.

    Rungs: clean -> shrink 2 cm -> 10 cm -> 50 cm -> 2 m (each re-cleaned) ->
    convex hull, the terminal fallback — generous but always simple, and for a
    multi-part mask it spans the parts. Each rung is pre-checked for shapely
    validity in both CRSs, then handed to ``accepts`` — THE REAL TEST, in
    production the ``gdalwarp`` cutline dry-run, because static validity does
    not predict GDAL cutline acceptance. Both geometries are EPSG:4326.
    Returns the first accepted candidate, or None when the ladder is spent.
    """
    candidates = [_shrink(polygon, m) for m in HEAL_SHRINK_METERS]
    base = candidates[0]
    if base is not None:
        hull = base.convex_hull
        if hull.geom_type == "Polygon":
            candidates.append(hull)
    for cand in candidates:
        if cand is None or not _survives_reprojection(cand):
            continue
        if not accepts(cand):
            continue
        return cand
    return None
