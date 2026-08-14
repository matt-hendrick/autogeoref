"""Volume search bounds: volume-level knowledge only, never per-sheet.

Priority (``run_inputs.resolve_bounds`` owns it):
  1. an explicit bbox;
  2. the footprint of a georeferenced counterpart edition (the 1949-51
     edition of the SAME Sanborn volume number covers the same ground);
  3. the union of named community areas;
  4. none of the above: the run DERIVES bounds from the volume's own sampled
     sheets and persists them in the work tree (``bounds_bootstrap``).

Human pins are NOT a source and must not become one. ``load_ground_truth`` and
``volume_bounds`` live here for the scorer and for display; a box drawn round
hand-placed pins would be those pins deciding which sheets can place at all.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .slugs import page_from_slug

Bounds = tuple[float, float, float, float]

BUFFER_DEG = 0.01


class BoundsError(ValueError):
    """No usable bounds source for a volume."""


def volume_bounds(gt: Mapping[str, dict[str, Any]], buffer_deg: float = BUFFER_DEG) -> Bounds:
    """Union of all ground-truth layer extents + buffer."""
    xs: list[float] = []
    ys: list[float] = []
    for lyr in gt.values():
        e = lyr.get("extent")
        if e:
            xs += [e[0], e[2]]
            ys += [e[1], e[3]]
    if not xs:
        raise BoundsError("ground truth has no layer extents")
    return (min(xs) - buffer_deg, min(ys) - buffer_deg, max(xs) + buffer_deg, max(ys) + buffer_deg)


def mercator_correction_lat(gt: Mapping[str, dict[str, Any]]) -> float:
    """Volume-midpoint latitude for the cos(lat) true-ground-meters correction.

    The single derivation behind every EPSG:3857 -> ground-meters conversion:
    the midpoint latitude of the volume's ground-truth extent union (the
    symmetric buffer cancels in the midpoint). Raises :class:`BoundsError`
    when the ground truth carries no layer extents — an explicit failure,
    never a fallback latitude.
    """
    b = volume_bounds(gt)
    return (b[1] + b[3]) / 2.0


def bbox_with_buffer(bounds: Bounds, buffer_deg: float = BUFFER_DEG) -> Bounds:
    b = bounds
    return (b[0] - buffer_deg, b[1] - buffer_deg, b[2] + buffer_deg, b[3] + buffer_deg)


def counterpart_bounds(
    viewer_manifest: dict[str, Any],
    counterpart_id: str,
    buffer_deg: float = BUFFER_DEG,
) -> Bounds:
    """Footprint of a harvested counterpart volume from a viewer manifest."""
    for v in viewer_manifest["volumes"]:
        if v["id"] == counterpart_id:
            return bbox_with_buffer(tuple(v["bounds"]), buffer_deg)
    raise BoundsError(f"{counterpart_id}: not in viewer manifest")


def community_area_bounds(
    features: list[dict[str, Any]],
    area_names: list[str],
    buffer_deg: float = BUFFER_DEG,
    name_property: str = "community",
) -> Bounds:
    """Union bbox of the named community areas."""
    wanted = {a.strip().upper() for a in area_names if a.strip()}
    xs: list[float] = []
    ys: list[float] = []
    found: set[str] = set()

    def walk(coords: Any) -> None:
        if isinstance(coords[0], int | float):
            xs.append(coords[0])
            ys.append(coords[1])
        else:
            for c in coords:
                walk(c)

    for f in features:
        name = (f["properties"].get(name_property) or "").upper()
        if name in wanted and f.get("geometry"):
            found.add(name)
            walk(f["geometry"]["coordinates"])
    missing = wanted - found
    if missing:
        raise BoundsError(
            f"unknown community area(s) {sorted(missing)} — check spelling "
            f"against the official list (e.g. LINCOLN SQUARE, not Ravenswood)"
        )
    return (min(xs) - buffer_deg, min(ys) - buffer_deg, max(xs) + buffer_deg, max(ys) + buffer_deg)


def load_ground_truth(path: Path, slug_prefix: str | None = None) -> dict[str, dict[str, Any]]:
    """``{page: layer}`` for ground-truth layers with GCPs.

    With ``slug_prefix`` the page is the slug remainder; otherwise it is parsed from the
    trailing ``_p<N>``. An empty file means the corpus was checked and this volume never pinned.
    Layers whose slug carries no page id are DROPPED, and that is load-bearing: where a
    volunteer split a sheet into regions before pinning it, the GCP pixels are in the CROP's
    frame and the export carries no offset back to the page. Do NOT make them parse — a crop's
    pixels are not the page's, and pairing them fabricates a placement that still fits cleanly.
    `autogeoref status` counts them as `+N unusable` so the loss is not silent.
    """
    text = path.read_text()
    if not text.strip():
        return {}
    layers = json.loads(text)
    out: dict[str, dict[str, Any]] = {}
    for lyr in layers:
        slug = lyr.get("slug") or ""
        if not lyr.get("gcps_geojson"):
            continue
        if slug_prefix is not None:
            if slug.startswith(slug_prefix):
                out[slug[len(slug_prefix) :]] = lyr
        else:
            page = page_from_slug(slug)
            if page is not None:
                out[page] = lyr
    return out
