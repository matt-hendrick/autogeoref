"""Sheet masks: detect the drawn content, heal the quilt, and write the cutlines.

The largest part of the bake by far, and one cluster — `stage_masks` owns every
helper here, and the auto-exemption verdict is the one thing it shares with the
QA pass that reads its output.
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import PipelineError
from ..paths import VolumePaths, regions_by_page, write_if_changed
from ..rescue import is_synthetic_gcp
from ..review.materialize import mask_ring_4326
from ..slugs import DuplicateCoverage, duplicate_coverage_slug, overview_slug, page_sort_key
from ..volume import is_reviewer_verified, reviewer_result_key
from .layers import committed_layers

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from shapely.geometry import MultiPolygon, Polygon

    from ..mask.qa import RawSheetMask

logger = logging.getLogger(__name__)


def raw_sheet_mask(
    image: Path,
    slug: str,
    page: str,
    record: dict[str, Any],
    *,
    content_masks: bool,
    content_mask_exempt: Collection[str],
    duplicates: DuplicateCoverage,
) -> RawSheetMask:
    """One committed sheet's mask ring, style, and raw EPSG:3857 polygon.

    The mask stage's per-sheet policy in one place, in precedence order: a
    reviewer ``mask_px`` ring wins outright, else the declared ``content_masks``
    hull, else the printed-colour box, else the page rectangle when no colour is
    found; overview sheets clip to the inlier-GCP hull. ``duplicates`` names the
    sheets that repeat other pages' ground; they and ``content_mask_exempt``
    pages skip the colour rungs and take the rectangle. Shared with the
    standalone mask-QA survey so the two cannot diverge.
    """
    from shapely.geometry import Polygon
    from shapely.ops import transform as shp_transform

    from ..affine import TO_3857, apply_affine, fit_affine, gcps_from_geojson
    from ..mask.geometry import (
        clip_to_gcp_hull,
        detect_content_box,
        detect_content_hull,
        detect_page_bounds,
        mask_polygon_4326,
    )
    from ..mask.qa import RawSheetMask

    matrix = fit_affine(gcps_from_geojson(record["gcps_geojson"]))
    ring_px = reviewer_result_key(record, "mask_px") or None
    if ring_px is not None:
        # the reviewer's dry-run-validated crop wins outright: page-bounds
        # detection must not even run here (its rect once clobbered the ring),
        # so the QA metrics that need a page denominator are skipped too
        poly = Polygon([(x, y) for x, y in mask_ring_4326(ring_px, matrix)])
        return RawSheetMask(
            slug=slug,
            image=image,
            matrix=matrix,
            style="mask_px",
            rect=None,
            ring_px=tuple((float(x), float(y)) for x, y in ring_px),
            poly_3857=shp_transform(TO_3857.transform, poly),
        )
    rect = detect_page_bounds(image)
    style = "page"
    # a declared exemption excuses the page from ALL colour-derived masking,
    # not only the hull: the sheets it names are the ones whose drawn ground a
    # colour bound would chop, which is the whole reason it exists
    exempt = page in content_mask_exempt
    if not duplicate_coverage_slug(slug, duplicates) and not exempt:
        if content_masks:
            ring_px = detect_content_hull(image, rect)
            if ring_px is not None:
                style = "hull"
        if ring_px is None:
            box_rect = detect_content_box(image, rect)
            if box_rect is not None:
                bx0, by0, bx1, by1 = box_rect
                ring_px = [(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)]
                style = "content_box"
    # `rect` stays the page rectangle throughout: it is the QA denominator and
    # frame the ink raster is measured in, not necessarily the mask's extent
    poly = (
        Polygon([(x, y) for x, y in mask_ring_4326(ring_px, matrix)])
        if ring_px
        else mask_polygon_4326(rect, matrix)
    )
    poly_3857 = shp_transform(TO_3857.transform, poly)
    # The hull must be the one the EVIDENCE earned, so synthetic corners are
    # excluded: they lie exactly on the placement model and carry none. Every
    # rescue record now carries three, and taking them at face value clips a
    # rescued overview sheet to their triangle instead of its anchors' hull.
    # A reviewer-verified sheet is skipped outright — a human placed it, and a
    # reviewer bounds coverage with a mask_px ring.
    if overview_slug(slug, duplicates) and not is_reviewer_verified(str(record.get("status", ""))):
        anchors = [
            apply_affine(matrix, *f["properties"]["image"])
            for f in record["gcps_geojson"]["features"]
            if not is_synthetic_gcp(f)
        ]
        clipped = clip_to_gcp_hull(poly_3857, anchors)
        if clipped is not poly_3857:
            style = "overview"
        poly_3857 = clipped
    return RawSheetMask(
        slug=slug,
        image=image,
        matrix=matrix,
        style=style,
        rect=rect,
        ring_px=tuple((float(x), float(y)) for x, y in ring_px) if ring_px else None,
        poly_3857=poly_3857,
    )


#: Half-width (m) of the channel the pre-split box expansion treats as an
#: inter-sheet slot rather than the volume's outer margin. The largest radius
#: on the measured plateau: much wider and the closing starts handing back the
#: outer blank margin the colour box exists to trim.
BOX_EXPAND_SLOT_RADIUS_M = 50.0


def expand_content_boxes(
    raws: list[RawSheetMask], duplicates: DuplicateCoverage
) -> list[RawSheetMask]:
    """Grow each colour box that retreated past an inter-sheet slot.

    Each ``content_box`` sheet takes the smallest rectangle in its own pixel
    frame containing its box plus the slot ground its own page covers, clipped
    to its page rectangle — so it can never claim ground its page did not
    serve. A slot is a channel narrower than twice
    :data:`BOX_EXPAND_SLOT_RADIUS_M`, plus any enclosed hole. Runs BEFORE the
    split, and the raw mask itself grows, so QA still describes what is served.
    """
    from dataclasses import replace

    from shapely.geometry import Polygon
    from shapely.ops import transform as shp_transform
    from shapely.ops import unary_union

    from ..affine import TO_3857, invert_affine
    from ..mask.geometry import SLIVER_AREA_M2, mask_polygon_4326

    if not any(raw.style == "content_box" for raw in raws):
        return raws

    def areal(geom: Any) -> Any:
        if geom.geom_type == "GeometryCollection":
            return unary_union(
                [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            )
        return geom

    def parts(geom: Any) -> list[Any]:
        if geom.is_empty:
            return []
        return list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]

    regular = [r for r in raws if not duplicate_coverage_slug(r.slug, duplicates)]
    # a `mask_px` sheet (rect None) contributes no page rectangle — its crop is
    # deliberate — but its mask still counts as coverage below
    pages = {
        r.slug: shp_transform(TO_3857.transform, mask_polygon_4326(r.rect, r.matrix))
        for r in regular
        if r.rect is not None
    }
    if not pages:
        return raws
    union = unary_union([r.poly_3857 for r in regular])
    radius = BOX_EXPAND_SLOT_RADIUS_M
    closed = union.buffer(radius, join_style=2).buffer(-radius, join_style=2)
    ground = areal(closed.difference(union))
    holes = [Polygon(ring) for part in parts(union) for ring in part.interiors]
    if holes:
        ground = unary_union([ground, *holes])
    ground = areal(areal(ground).intersection(unary_union(list(pages.values()))))
    if ground.is_empty:
        return raws
    out: list[RawSheetMask] = []
    for raw in raws:
        page = pages.get(raw.slug)
        if raw.style != "content_box" or page is None or raw.rect is None:
            out.append(raw)
            continue
        gain = areal(ground.intersection(page))
        if gain.is_empty or gain.area < SLIVER_AREA_M2:
            out.append(raw)
            continue
        inverse = invert_affine(raw.matrix)
        xs = [float(p[0]) for p in raw.ring_px or ()]
        ys = [float(p[1]) for p in raw.ring_px or ()]
        for part in parts(gain):
            for x, y in part.exterior.coords:
                xs.append(inverse[0][0] + inverse[0][1] * x + inverse[0][2] * y)
                ys.append(inverse[1][0] + inverse[1][1] * x + inverse[1][2] * y)
        rx0, ry0, rx1, ry1 = raw.rect
        box = (
            max(float(rx0), min(xs)),
            max(float(ry0), min(ys)),
            min(float(rx1), max(xs)),
            min(float(ry1), max(ys)),
        )
        ring = ((box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3]))
        poly = shp_transform(TO_3857.transform, mask_polygon_4326(box, raw.matrix))
        out.append(replace(raw, ring_px=ring, poly_3857=poly))
    return out


def _resolve_masks(
    raws: list[RawSheetMask],
    paths: VolumePaths,
    *,
    duplicates: DuplicateCoverage,
    heal_cache: dict[tuple[str, bytes], Polygon | MultiPolygon | None],
    ink_rasters: dict[tuple[str, tuple[int, int, int, int]], Any],
    timeout_s: float,
) -> tuple[list[RawSheetMask], dict[str, Polygon | MultiPolygon | None]]:
    """expand -> split -> blank-core move -> heal for one candidate
    raw-mask set. Returns the raw masks actually used (grown colour boxes
    included) with the finals: QA must score the raws the finals came from, or
    an expansion arm reports the base arm's ratios. None final = unmaskable."""
    from shapely.ops import transform as shp_transform

    from ..affine import TO_4326
    from ..mask.geometry import heal, snap_clean, split_overlaps
    from ..mask.move import move_blank_cores
    from ..warp import gdalwarp_cutline_dryrun

    try:
        raws = expand_content_boxes(raws, duplicates)
    except Exception:
        # an improvement pass, not a gate: the unexpanded boxes serve and
        # the residual gap stays measured by the coverage metric
        logger.exception("masks: content-box expansion failed; unexpanded boxes kept")
    detail_raws = [r for r in raws if not duplicate_coverage_slug(r.slug, duplicates)]
    fallback = {r.slug: r.poly_3857 for r in raws if duplicate_coverage_slug(r.slug, duplicates)}
    moved = split_overlaps({r.slug: r.poly_3857 for r in detail_raws})
    try:
        moved = move_blank_cores(detail_raws, moved, ink_rasters=ink_rasters)
    except Exception:
        # the move is an improvement pass, not a gate: on failure the
        # shipped split serves and the defect stays measured by mask QA
        logger.exception("masks: blank-core move failed; shipped split kept")
    split = {**moved, **fallback}
    finals: dict[str, Polygon | MultiPolygon | None] = {}
    for slug in sorted(split, key=page_sort_key):
        cleaned = snap_clean(shp_transform(TO_4326.transform, split[slug]))
        key = (slug, cleaned.wkb)
        if key in heal_cache:
            finals[slug] = heal_cache[key]
            continue
        cog = paths.warped / f"{slug}.tif"

        def accepts(candidate: Any, cog: Path = cog) -> bool:
            return gdalwarp_cutline_dryrun(cog, candidate, timeout_s=timeout_s, crs_epsg=4326)

        poly: Polygon | MultiPolygon | None = cleaned
        if not accepts(cleaned):
            poly = heal(cleaned, accepts)
            if poly is None:
                logger.warning("masks: %s heal ladder exhausted — sheet joins unmasked", slug)
        heal_cache[key] = poly
        finals[slug] = poly
    return raws, finals


def _placement_window(paths: VolumePaths) -> tuple[float, float] | None:
    """The volume's pinned ``(scale_m_per_px, rotation_deg)`` from
    ``volume-constants.json``, or None when not persisted or unreadable
    (see the window-metric note in :func:`stage_masks`)."""
    placement_window: tuple[float, float] | None = None
    try:
        if paths.constants.is_file():
            consts = json.loads(paths.constants.read_text())
            placement_window = (consts["scale_m_per_px"], consts["rotation_deg"])
    except Exception:
        # same shield as QA itself: a malformed constants file costs the
        # window metric, never the bake
        logger.exception("masks: unreadable volume constants; outside_window not measured")
    return placement_window


def _write_mask_features(
    paths: VolumePaths, finals: dict[str, Polygon | MultiPolygon | None]
) -> Path:
    """Write the per-slug mask files and rebuild ``masks.geojson`` from
    ``finals``. An unmaskable sheet (None final) joins ``unmasked`` and
    loses its per-slug file. Returns the collection path — the mask
    stage's return value."""
    from shapely.geometry import mapping

    features: list[dict[str, Any]] = []
    unmasked: list[str] = []
    for slug in sorted(finals, key=page_sort_key):
        poly = finals[slug]
        if poly is None:
            unmasked.append(slug)
            (paths.masks / f"{slug}.geojson").unlink(missing_ok=True)
            continue
        feature = {"type": "Feature", "properties": {"slug": slug}, "geometry": mapping(poly)}
        features.append(feature)
        write_if_changed(
            paths.masks / f"{slug}.geojson",
            json.dumps({"type": "FeatureCollection", "features": [feature]}, indent=2),
        )
    collection = {"type": "FeatureCollection", "features": features, "unmasked": unmasked}
    return write_if_changed(paths.masks / "masks.geojson", json.dumps(collection, indent=2))


def stage_masks(
    paths: VolumePaths,
    volume: str,
    *,
    content_masks: bool = False,
    content_mask_exempt: Collection[str] = (),
    overview_pages: Collection[str] = (),
    page_scale_multiples: Mapping[str, float] | None = None,
    timeout_s: float = 600.0,
) -> Path:
    """Masks for every warped sheet: detect -> expand -> split -> move -> heal.

    ``page_scale_multiples`` feeds QA's ``outside_window`` re-check; the window
    comes from the volume's persisted constants, so the flag is not measured
    when none were persisted. Duplicate-coverage sheets are never
    bisector-split — a whole-sheet overlap would halve both along a mid-sheet
    diagonal — and paint order resolves them instead. Skeleton twins come from
    the whole page inventory, not from what placed. Declared overviews clip to
    their inlier-GCP hull unless reviewer-verified. See `docs/INTERNALS.md`.
    """
    from ..mask.qa import qa_masks, write_masks_qa

    if isinstance(content_mask_exempt, str):
        raise TypeError("content_mask_exempt takes a collection of page ids, not a bare string")
    images = regions_by_page(paths.regions)
    duplicates = DuplicateCoverage.resolve(images, overview_pages)
    committed = committed_layers(paths, volume)
    raw_masks: list[RawSheetMask] = []
    polys_3857 = {}
    for page, slug, record in committed:
        image = images.get(page)
        cog = paths.warped / f"{slug}.tif"
        if image is None or not cog.is_file():
            continue
        raw = raw_sheet_mask(
            image,
            slug,
            page,
            record,
            content_masks=content_masks,
            content_mask_exempt=content_mask_exempt,
            duplicates=duplicates,
        )
        raw_masks.append(raw)
        polys_3857[slug] = raw.poly_3857
    if not polys_3857:
        raise PipelineError(f"masks: no warped sheet with a full-res image under {paths.root}")
    unmatched = set(content_mask_exempt) - {page for page, _, _ in committed}
    if unmatched:
        # a typo here silently resurrects the hull collapse the exemption
        # exists to prevent — say so instead
        logger.warning(
            "masks: content_mask_exempt pages match no committed sheet: %s", sorted(unmatched)
        )
    # cutline dry-run/heal results survive across auto-exemption candidates:
    # a rejected swap must not re-run gdalwarp on the polygons it left alone
    heal_cache: dict[tuple[str, bytes], Polygon | MultiPolygon | None] = {}
    # ink rasters build once per sheet, shared by the blank-core move and
    # every QA scoring call (including auto-exemption candidate re-scores)
    ink_rasters: dict[tuple[str, tuple[int, int, int, int]], Any] = {}
    resolve = functools.partial(
        _resolve_masks,
        paths=paths,
        duplicates=duplicates,
        heal_cache=heal_cache,
        ink_rasters=ink_rasters,
        timeout_s=timeout_s,
    )
    raw_masks, finals = resolve(raw_masks)
    policy = _MaskPolicy(
        content_masks=content_masks,
        content_mask_exempt=content_mask_exempt,
        duplicates=duplicates,
        ink_rasters=ink_rasters,
        placement_window=_placement_window(paths),
        page_scale_multiples=page_scale_multiples,
    )
    qa_doc: dict[str, Any] | None = None
    auto_exempted: list[str] = []
    try:
        qa_doc = qa_masks(
            volume,
            raw_masks,
            finals,
            content_masks=policy.content_masks,
            duplicates=policy.duplicates,
            ink_rasters=policy.ink_rasters,
            placement_window=policy.placement_window,
            page_scale_multiples=policy.page_scale_multiples,
        )
    except Exception:
        # QA is detect-and-flag only: a metric bug must not take down a bake
        # it merely measures — the baseline masks below still get written.
        logger.exception("masks: QA metrics failed; masks written without masks-qa.json")
        qa_doc = None
    if qa_doc is not None and content_masks:
        try:
            raw_masks, finals, qa_doc, auto_exempted = _auto_exempt_collapsed(
                volume,
                raw_masks,
                finals,
                qa_doc,
                committed,
                images,
                resolve,
                policy=policy,
            )
        except Exception:
            # remedy failure must not cost the measurement: keep the baseline
            # masks AND the baseline QA doc (partial swaps are discarded)
            logger.exception("masks: auto-exemption loop failed; baseline masks and QA kept")
            auto_exempted = []

    out = _write_mask_features(paths, finals)
    if qa_doc is not None:
        try:
            write_masks_qa(
                paths,
                volume,
                raw_masks,
                finals,
                content_masks=content_masks,
                duplicates=duplicates,
                doc=qa_doc,
                auto_exempted=auto_exempted,
            )
        except Exception:
            # a failed QA write-out never costs the bake: the masks above are
            # already on disk, only the measurement document goes unwritten
            logger.exception("masks: QA write-out failed; masks kept, masks-qa.json not written")
    return out


#: Auto-exemption severity bar: a hull_collapse sheet is a rectangle-swap
#: candidate only when MORE THAN HALF of its drawn content sits in the quilt
#: hole. Lower-severity flags can be fringe spill or coverage already supplied
#: by neighboring sheets, so they do not justify a rectangle swap.
AUTO_EXEMPT_MIN_UNCOVERED = 0.5


def auto_exempt_verdict(
    slug: str, baseline_doc: dict[str, Any], candidate_doc: dict[str, Any]
) -> tuple[bool, str]:
    """Should the bake adopt this sheet's rectangle swap?

    Pure decision over two QA documents. Every guard must pass: the baseline
    ``ink_uncovered_frac`` clears :data:`AUTO_EXEMPT_MIN_UNCOVERED`; the
    candidate shows the collapse cleared; no sheet gains a flag; no sheet's
    blank core grows past ``max(its baseline, flag threshold)``, since "no NEW
    flag" alone is blind to an already-flagged sheet getting worse; and the
    candidate measured at least what the baseline did, because a measurement
    failure must abstain rather than accept.
    """
    from ..mask.qa import BLANK_OVERPAINT_MIN_CORE_M2

    base_sheet = baseline_doc["sheets"].get(slug, {})
    uncovered = base_sheet.get("ink_uncovered_frac", 0.0)
    if uncovered <= AUTO_EXEMPT_MIN_UNCOVERED:
        return False, (
            f"collapse below the catastrophic bar ({uncovered} <= {AUTO_EXEMPT_MIN_UNCOVERED})"
        )
    if "hull_collapse" in candidate_doc["flagged"].get(slug, []):
        return False, "rectangle does not clear the collapse"
    base_flags = baseline_doc["flagged"]
    new_flags = {
        s: [f for f in fl if f not in base_flags.get(s, [])]
        for s, fl in candidate_doc["flagged"].items()
    }
    new_flags = {s: fl for s, fl in new_flags.items() if fl}
    if new_flags:
        return False, f"introduces {new_flags}"
    for s, base_entry in baseline_doc["sheets"].items():
        cand_entry = candidate_doc["sheets"].get(s, {})
        if cand_entry.get("unmasked") and not base_entry.get("unmasked"):
            return False, f"{s} becomes unmasked (heal exhausted)"
        if "blank_core_m2" in base_entry and "blank_core_m2" not in cand_entry:
            return False, f"{s} blank core became unmeasurable"
    for s, entry in candidate_doc["sheets"].items():
        after = entry.get("blank_core_m2", 0.0)
        before = baseline_doc["sheets"].get(s, {}).get("blank_core_m2", 0.0)
        if after > max(before, BLANK_OVERPAINT_MIN_CORE_M2):
            return False, f"{s} blank core grows {before:.0f} -> {after:.0f} m2"
    return True, "collapse cleared; no new flag; no blank-core growth"


@dataclass(frozen=True)
class _MaskPolicy:
    """What every mask-building and mask-scoring call in one bake shares."""

    content_masks: bool
    content_mask_exempt: Collection[str]
    duplicates: DuplicateCoverage
    #: shared across the pass so each sheet's ink raster is built once
    ink_rasters: dict[tuple[str, tuple[int, int, int, int]], Any]
    placement_window: tuple[float, float] | None
    page_scale_multiples: Mapping[str, float] | None


def _auto_exempt_collapsed(
    volume: str,
    raw_masks: list[RawSheetMask],
    finals: dict[str, Any],
    qa_doc: dict[str, Any],
    committed: list[tuple[str, str, dict[str, Any]]],
    images: dict[str, Path],
    resolve: Any,
    *,
    policy: _MaskPolicy,
) -> tuple[list[RawSheetMask], dict[str, Any], dict[str, Any], list[str]]:
    """Retry each catastrophically collapsed hull as its page rectangle.

    The candidate is built by exempting the page, and an exemption means no
    colour bound at all — so it is the WHOLE rectangle, not the colour box a
    regular sheet would get. The remedy was qualified as "restore everything
    the hull dropped", and a box need not restore it. Candidates and adoption
    are decided by :func:`auto_exempt_verdict`, qualified against labeled QA cases.
    Sheets are tried in page order against the current adopted state, so each
    accepted swap is part of the baseline the next candidate must not degrade.
    """
    from ..mask.qa import qa_masks

    by_slug = {slug: (page, record) for page, slug, record in committed}
    auto_exempted: list[str] = []
    # the candidate list is a snapshot of the BASELINE flags; qa_doc rebinds
    # after each accepted swap, so re-check against the current doc — an
    # earlier rectangle may already have cleared a later sheet's flag
    for slug in sorted(qa_doc["flagged"], key=page_sort_key):
        if "hull_collapse" not in qa_doc["flagged"].get(slug, []):
            continue
        uncovered = qa_doc["sheets"].get(slug, {}).get("ink_uncovered_frac", 0.0)
        if uncovered <= AUTO_EXEMPT_MIN_UNCOVERED:
            # baseline-only veto: skip the expensive candidate rebuild
            logger.info(
                "masks: %s keeps its hull and its flag — collapse below the "
                "catastrophic bar (%s <= %s)",
                slug,
                uncovered,
                AUTO_EXEMPT_MIN_UNCOVERED,
            )
            continue
        page, record = by_slug[slug]
        image = images.get(page)
        if image is None:
            continue
        swapped = raw_sheet_mask(
            image,
            slug,
            page,
            record,
            content_masks=policy.content_masks,
            content_mask_exempt=(*policy.content_mask_exempt, page),
            duplicates=policy.duplicates,
        )
        cand_raws = [swapped if r.slug == slug else r for r in raw_masks]
        cand_raws, cand_finals = resolve(cand_raws)
        cand_qa = qa_masks(
            volume,
            cand_raws,
            cand_finals,
            content_masks=policy.content_masks,
            duplicates=policy.duplicates,
            ink_rasters=policy.ink_rasters,
            placement_window=policy.placement_window,
            page_scale_multiples=policy.page_scale_multiples,
        )
        accept, reason = auto_exempt_verdict(slug, qa_doc, cand_qa)
        if accept:
            raw_masks, finals, qa_doc = cand_raws, cand_finals, cand_qa
            auto_exempted.append(slug)
            logger.info("masks: auto-exempted %s — %s", slug, reason)
        else:
            logger.info("masks: %s keeps its hull and its flag — %s", slug, reason)
    return raw_masks, finals, qa_doc, auto_exempted
