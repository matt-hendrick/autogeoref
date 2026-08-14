"""Derive search bounds from sampled sheet annotations and centerlines.

Declared bounds take precedence. Derived bounds are coarse search areas,
persisted with their evidence, and never determine acceptance.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .bounds import Bounds, BoundsError
from .budget import estimate_spend
from .names import Aliases, load_aliases, normalize
from .paths import VolumePaths, atomic_write_text

if TYPE_CHECKING:
    from .config.model import CityConfig, VolumeConfig

logger = logging.getLogger(__name__)

#: Localization grid edge in degrees.
GRID_DEG = 0.01

#: Evenly distributed sample size.
SAMPLE_PAGES = 12

#: Minimum distinct names required to localize a page.
MIN_NAMES_PER_PAGE = 3

#: Minimum mutually consistent page localizations.
MIN_LOCALIZED_PAGES = 3

#: Maximum center distance from the median for a consistent localization.
MAX_CENTER_SPREAD_DEG = 0.08

#: A page whose qualifying cells spread wider than this (degrees) names
#: streets that meet in more than one part of the city — it abstains rather
#: than vote for both.
MAX_PAGE_SPREAD_DEG = 0.06

#: Margin around the localized-page union.
MARGIN_LON_DEG = 0.02
MARGIN_LAT_DEG = 0.015

BOUNDS_FILE = "volume-bounds.json"


def bounds_file(paths: VolumePaths) -> Path:
    return paths.root / BOUNDS_FILE


def persisted_bounds(paths: VolumePaths) -> Bounds | None:
    """A previous run's derivation, replayed free — `volume-constants.json`'s twin."""
    f = bounds_file(paths)
    if not f.exists():
        return None
    try:
        raw = json.loads(f.read_text())
        w, s, e, n = (float(v) for v in raw["bounds"])
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("%s: unreadable, ignoring (will re-derive)", f)
        return None
    if not (w < e and s < n):
        # a garbage-but-parseable box would sail into an empty centerline index
        # and die two stages later, pointing at the wrong culprit
        logger.warning("%s: bounds not ordered west<east, south<north; re-deriving", f)
        return None
    return (w, s, e, n)


def _page_order(page: str) -> tuple[int, int | str]:
    """Numeric page order: p2 before p10, non-numeric plates (ptitl) last."""
    return (0, int(page[1:])) if page[1:].isdigit() else (1, page)


def eligible_pages(paths: VolumePaths) -> list[str]:
    """Manifest pages in NUMERIC order, only those with a small on disk.

    ``manifest_pages`` sorts lexicographically (p10 before p2) and still lists
    map-less plates an older manifest carries (ptitl, pind1 — no small was
    ever written). Sampling from that order is front-heavy and burns sample
    slots on pages that cannot contribute a single street name, silently
    shrinking the sample toward the localization quorum.
    """
    from .annotate_volume import manifest_pages

    return sorted(
        (p for p in manifest_pages(paths) if (paths.sheets / f"{p}_small.jpg").exists()),
        key=_page_order,
    )


def sample_evenly(pages: list[str], k: int = SAMPLE_PAGES) -> list[str]:
    """``k`` pages spread through the given page order, ends included."""
    if len(pages) <= k:
        return list(pages)
    step = (len(pages) - 1) / (k - 1)
    return sorted({pages[round(i * step)] for i in range(k)}, key=_page_order)


def _cell(lon: float, lat: float) -> tuple[int, int]:
    return (math.floor(lon / GRID_DEG), math.floor(lat / GRID_DEG))


def _feature_cells(geometry: dict[str, Any]) -> set[tuple[int, int]]:
    """Grid cells a (Multi)LineString passes through, sampled at half-cell steps."""
    kind = geometry.get("type")
    lines = (
        geometry.get("coordinates", [])
        if kind == "MultiLineString"
        else [geometry.get("coordinates", [])]
        if kind == "LineString"
        else []
    )
    cells: set[tuple[int, int]] = set()
    for line in lines:
        for (x0, y0), (x1, y1) in itertools.pairwise(line):
            steps = max(1, int(max(abs(x1 - x0), abs(y1 - y0)) / (GRID_DEG / 2)))
            for i in range(steps + 1):
                t = i / steps
                cells.add(_cell(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return cells


def _dilate(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """3x3 dilation: names on one sheet may sit a cell apart, never further."""
    return {(i + di, j + dj) for i, j in cells for di in (-1, 0, 1) for dj in (-1, 0, 1)}


def _name_cells(
    features: list[dict[str, Any]],
    aliases: Aliases,
    name_property: str,
    type_property: str,  # noqa: ARG001 — see comment below
) -> dict[str, set[tuple[int, int]]]:
    """Normalized street name -> grid cells it runs through, citywide.

    Deliberately NOT :func:`centerlines.centerline_key`: the PL/CT twin suffix
    distinguishes parallel streets half a block apart, and this localization
    works in kilometre cells where the twins are the same place. Folding them
    together here makes an annotation reading "37TH" or "37TH PL" hit either —
    the right behaviour for a search-area vote, and the index the matcher
    builds later still applies THE key rule unchanged.
    """
    out: dict[str, set[tuple[int, int]]] = {}
    for f in features:
        props = f.get("properties") or {}
        name = props.get(name_property)
        if not name:
            continue
        key = normalize(str(name), aliases)
        if not key:
            continue
        out.setdefault(key, set()).update(_feature_cells(f.get("geometry") or {}))
    return out


def localize_page(
    names: list[str],
    name_cells: dict[str, set[tuple[int, int]]],
    aliases: Aliases,
) -> tuple[Bounds | None, int]:
    """Where one sheet's street names co-occur; ``(None, matched)`` = abstain.

    Votes are DISTINCT normalized names; a cell qualifies when at least
    ``max(MIN_NAMES_PER_PAGE, 60%)`` of the page's matched names run within one dilated cell of
    it. Qualifying cells spread wider than :data:`MAX_PAGE_SPREAD_DEG` mean the combination
    recurs across the city — the page abstains rather than vote for two places at once.
    """
    keys = {k for k in (normalize(n, aliases) for n in names) if k and k in name_cells}
    if len(keys) < MIN_NAMES_PER_PAGE:
        return None, len(keys)
    counts: dict[tuple[int, int], int] = {}
    for k in keys:
        for cell in _dilate(name_cells[k]):
            counts[cell] = counts.get(cell, 0) + 1
    need = max(MIN_NAMES_PER_PAGE, math.ceil(0.6 * len(keys)))
    hits = [c for c, n in counts.items() if n >= need]
    if not hits:
        return None, len(keys)
    is_ = [i for i, _ in hits]
    js = [j for _, j in hits]
    w, e = min(is_) * GRID_DEG, (max(is_) + 1) * GRID_DEG
    s, n = min(js) * GRID_DEG, (max(js) + 1) * GRID_DEG
    if (e - w) > MAX_PAGE_SPREAD_DEG or (n - s) > MAX_PAGE_SPREAD_DEG:
        return None, len(keys)
    return (w, s, e, n), len(keys)


def localize_pages(
    page_names: dict[str, list[str]],
    features: list[dict[str, Any]],
    aliases: Aliases,
    *,
    name_property: str = "street_nam",
    type_property: str = "street_typ",
) -> tuple[Bounds, dict[str, Any]]:
    """Union of the sample's AGREEING page localizations + margin, with evidence.

    Pure and deterministic: annotations in, bounds out. A page whose localization centre sits
    further than :data:`MAX_CENTER_SPREAD_DEG` from the median is an OUTLIER — one misread
    page's names meeting in the wrong part of the city — and is dropped from the union, so one
    liar cannot inflate bounds the honest pages earned. Raises :class:`BoundsError` when fewer
    than :data:`MIN_LOCALIZED_PAGES` pages localize AND agree.
    """
    name_cells = _name_cells(features, aliases, name_property, type_property)
    pages: dict[str, Any] = {}
    boxes: dict[str, Bounds] = {}
    for page, names in sorted(page_names.items()):
        box, matched = localize_page(names, name_cells, aliases)
        pages[page] = {"bbox": list(box) if box else None, "matched_names": matched}
        if box:
            boxes[page] = box
    inliers = dict(boxes)
    if len(boxes) >= MIN_LOCALIZED_PAGES:
        # median center is robust to the minority it exists to catch
        cxs = sorted((b[0] + b[2]) / 2 for b in boxes.values())
        cys = sorted((b[1] + b[3]) / 2 for b in boxes.values())
        cx, cy = cxs[len(cxs) // 2], cys[len(cys) // 2]
        inliers = {
            page: b
            for page, b in boxes.items()
            if abs((b[0] + b[2]) / 2 - cx) <= MAX_CENTER_SPREAD_DEG
            and abs((b[1] + b[3]) / 2 - cy) <= MAX_CENTER_SPREAD_DEG
        }
        for page in boxes.keys() - inliers.keys():
            pages[page]["outlier"] = True
    if len(inliers) < MIN_LOCALIZED_PAGES:
        raise BoundsError(
            f"bounds bootstrap: only {len(inliers)} of {len(page_names)} sampled pages "
            f"localized and agreed (need {MIN_LOCALIZED_PAGES}; {len(boxes)} localized) — "
            "too few street names met in one place, or the localizations scatter across "
            "the city. Declare bounds_bbox in the city TOML instead: the volume's key map "
            "(usually p0) names its boundary streets outright."
        )
    w = min(b[0] for b in inliers.values()) - MARGIN_LON_DEG
    s = min(b[1] for b in inliers.values()) - MARGIN_LAT_DEG
    e = max(b[2] for b in inliers.values()) + MARGIN_LON_DEG
    n = max(b[3] for b in inliers.values()) + MARGIN_LAT_DEG
    return (w, s, e, n), {
        "pages": pages,
        "localized": len(inliers),
        "outliers": sorted(boxes.keys() - inliers.keys()),
    }


def derive_bounds(
    paths: VolumePaths,
    volume: str,
    city: CityConfig,
    vol: VolumeConfig,
    *,
    spend: bool = True,
) -> Bounds:
    """Prep if needed, read the sample, localize, persist.

    ``spend=True`` reads the sample's uncached primaries, logging the exact planned count first,
    and every one is a page the place run was about to read anyway. ``spend=False`` is for a run
    whose flags already promised a capped or zero annotation spend: the bootstrap then localizes
    from cached sample annotations only and refuses honestly when that is too few.

    An OSM-default city cannot bootstrap: its centerline cache is fetched BY bounds, so there is
    nothing citywide to localize against until a human declares a bbox.
    """
    if getattr(city, "centerlines_from_osm", False) or not city.centerlines_path.exists():
        raise BoundsError(
            f"{volume}: bounds bootstrap needs a citywide centerlines file, and "
            f"{city.centerlines_path} is absent (OSM-default cities fetch centerlines "
            "BY bounds). Declare bounds_bbox for this volume."
        )
    from .annotate_volume import ReadIdentity, annotate_volume, plan

    if not paths.manifest.exists():
        from .prep import prep_volume

        logger.info("%s: bounds bootstrap: prep first (no manifest yet)", volume)
        prep_volume(paths.regions, paths.sheets)

    sample = sample_evenly(eligible_pages(paths))
    identity = ReadIdentity(vol.annotation_model, vol.annotation_variant)
    if spend:
        batch = plan(
            paths,
            volume,
            identity=identity,
            pages=sample,
        )
        # the shared formatter (budget.render); no escalation term — the
        # bootstrap only buys primary sample reads
        est = estimate_spend(
            sheets=len(sample),
            cached=len(sample) - len(batch.todo),
            unread=len(batch.todo),
            attempts=batch.attempts,
        )
        logger.info(
            "%s: bounds bootstrap (%s): %d uncached of %d sample sheets to read = "
            "%s — spent now, replayed free by the annotate stage",
            volume,
            vol.annotation_model,
            len(batch.todo),
            len(sample),
            est.render(),
        )
        annotate_volume(
            paths,
            volume,
            identity=identity,
            pages=sample,
        )
    else:
        logger.info(
            "%s: bounds bootstrap: cached sample reads only (--limit/--no-annotate "
            "already promised this run's spend; the bootstrap spends nothing extra)",
            volume,
        )

    page_names: dict[str, list[str]] = {}
    for page in sample:
        f = paths.annotations / f"{page}.json"
        if not f.exists():
            continue
        try:
            streets = json.loads(f.read_text()).get("streets") or []
        except (OSError, ValueError):
            continue
        names = [s.get("name") for s in streets if isinstance(s, dict) and s.get("name")]
        if names:
            page_names[page] = [str(n) for n in names]

    aliases = load_aliases(city.aliases_path(volume))
    features = json.loads(city.centerlines_path.read_text())["features"]
    try:
        bounds, evidence = localize_pages(
            page_names,
            features,
            aliases,
            name_property=city.centerline_name_property,
            type_property=city.centerline_type_property,
        )
    except BoundsError as exc:
        if not spend:
            raise BoundsError(
                f"{exc} Or run once without --limit/--no-annotate: the bootstrap "
                "may only use CACHED reads under those flags, and this volume has "
                "too few."
            ) from exc
        raise
    atomic_write_text(
        bounds_file(paths),
        json.dumps(
            {
                "bounds": list(bounds),
                "derived": {
                    "sampled": sample,
                    "grid_deg": GRID_DEG,
                    "date": time.strftime("%Y-%m-%d"),
                    **evidence,
                },
            },
            indent=2,
        ),
    )
    logger.info(
        "%s: bounds bootstrap: %d/%d pages localized -> %s (persisted to %s)",
        volume,
        evidence["localized"],
        len(page_names),
        [round(b, 4) for b in bounds],
        bounds_file(paths),
    )
    return bounds
