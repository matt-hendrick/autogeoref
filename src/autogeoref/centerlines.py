"""Modern street-centerline index: normalized name -> merged geometry.

Intersections between two named streets are computed purely geometrically
(``unary_union`` per name, then ``intersection``); there is no reliance on
any node-id graph in the source data. Only the ``street_nam`` and
``street_typ`` properties are read, so any centerline GeoJSON with those two
fields (or a caller-supplied property mapping) works.

Numbered PLACE/COURT segments are indexed under distinct keys
(``"31ST PL"``), mirroring :func:`autogeoref.names.normalize` — 31st St and
31st Pl are different parallel streets half a block apart.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shapely.geometry import box, shape
from shapely.ops import unary_union

from .names import _NUMERIC_ORDINAL, Aliases, normalize

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

Bounds = tuple[float, float, float, float]


def centerline_key(
    props: Mapping[str, Any],
    aliases: Aliases | None = None,
    name_property: str = "street_nam",
    type_property: str = "street_typ",
) -> str | None:
    """Normalized index key for one centerline feature; ``None`` when unnamed.

    THE one implementation of the key rule every consumer must share (this index and the
    verified-accept segment builder): the normalized name plus the numbered PLACE/COURT twin
    suffix — ``street_nam "W 37TH"`` with ``street_typ "PL"`` keys as ``"37TH PL"``, a different
    parallel street half a block from ``"37TH"``. A hand-synced copy drifting in one consumer
    silently mis-keys it against the others — an invisible regression.

    Being a twin is decided BEFORE aliases, as ``normalize`` decides it on the read side.
    """
    name = props.get(name_property)
    if not name:
        return None
    typ = str(props.get(type_property) or "").upper()
    twin = typ in {"PL", "CT"}
    bare = normalize(str(name))
    if twin and _NUMERIC_ORDINAL.match(bare):
        return f"{bare} {typ}"
    key = normalize(str(name), aliases)
    if twin and _NUMERIC_ORDINAL.match(key):
        key = f"{key} {typ}"
    return key


def _bbox_disjoint(geometry: dict[str, Any], bounds: Bounds) -> bool:
    """True when the geometry's raw-coordinate bbox cannot touch ``bounds``.

    Cheap prefilter run before shapely geometry construction: bbox-disjoint
    implies geometrically disjoint, so a skipped feature could never have
    passed the exact ``intersects`` test. Bbox-overlapping features (and
    geometry types this doesn't know) fall through to the exact test, so
    the index contents are unchanged.
    """
    gtype = geometry.get("type")
    if gtype == "LineString":
        parts = (geometry["coordinates"],)
    elif gtype == "MultiLineString":
        parts = geometry["coordinates"]
    else:
        return False
    minx, miny, maxx, maxy = bounds
    gminx = gminy = float("inf")
    gmaxx = gmaxy = float("-inf")
    for part in parts:
        for c in part:
            x, y = c[0], c[1]
            if x < gminx:
                gminx = x
            if x > gmaxx:
                gmaxx = x
            if y < gminy:
                gminy = y
            if y > gmaxy:
                gmaxy = y
    if gmaxx == float("-inf"):
        return False
    return gmaxx < minx or gminx > maxx or gmaxy < miny or gminy > maxy


class CenterlineIndex:
    """Normalized street name -> centerline segments, with cached merged geometry."""

    def __init__(
        self,
        features: list[dict[str, Any]],
        aliases: Aliases | None = None,
        bounds_4326: Bounds | None = None,
        name_property: str = "street_nam",
        type_property: str = "street_typ",
    ) -> None:
        self.aliases = dict(aliases or {})
        # kept so a consumer of ``by_name`` can read the raw names back out
        # without being handed the city config a second time
        self.name_property = name_property
        self.type_property = type_property
        self.by_name: dict[str, list[dict[str, Any]]] = {}
        self._merged_cache: dict[str, BaseGeometry | None] = {}
        self._intersection_cache: dict[tuple[str, str], list[tuple[float, float]]] = {}
        # Escalation workers share this index. Cache publication stays
        # single-writer, while provider calls remain outside this boundary.
        self._cache_lock = threading.RLock()
        clip = box(*bounds_4326) if bounds_4326 else None
        for f in features:
            props = f["properties"]
            if not props.get(name_property) or f["geometry"] is None:
                continue
            if bounds_4326 is not None and _bbox_disjoint(f["geometry"], bounds_4326):
                continue
            geom = shape(f["geometry"])
            if clip is not None and not geom.intersects(clip):
                continue
            key = centerline_key(props, self.aliases, name_property, type_property)
            if key is None:  # unreachable after the name check; keeps types honest
                continue
            self.by_name.setdefault(key, []).append({"geom": geom, "props": props})

    @classmethod
    def from_geojson(
        cls,
        path: Path,
        aliases: Aliases | None = None,
        bounds_4326: Bounds | None = None,
        name_property: str = "street_nam",
        type_property: str = "street_typ",
    ) -> CenterlineIndex:
        data = json.loads(path.read_text())
        return cls(
            data["features"],
            aliases=aliases,
            bounds_4326=bounds_4326,
            name_property=name_property,
            type_property=type_property,
        )

    def merged(self, key: str) -> BaseGeometry | None:
        """Union of all segments for a normalized name (cached)."""
        with self._cache_lock:
            if key not in self._merged_cache:
                segs = self.by_name.get(key)
                self._merged_cache[key] = unary_union([s["geom"] for s in segs]) if segs else None
            return self._merged_cache[key]

    def intersections(self, key_a: str, key_b: str) -> list[tuple[float, float]]:
        """All point intersections (4326 lng/lat) between two named streets.

        Collinear overlaps and other non-point intersection geometries are
        dropped — a shared segment is not a usable anchor. Results are
        memoized per EXACT (key_a, key_b) orientation: operand order can
        change the point ordering GEOS returns, and candidate order feeds
        the deterministic RANSAC sampling — an orientation-collapsing cache
        would subtly change results.
        """
        cache_key = (key_a, key_b)
        with self._cache_lock:
            cached = self._intersection_cache.get(cache_key)
            if cached is not None:
                return cached
            ga, gb = self.merged(key_a), self.merged(key_b)
            if ga is None or gb is None:
                self._intersection_cache[cache_key] = []
                return []
            inter = ga.intersection(gb)
            if inter.is_empty:
                pts = []
            elif inter.geom_type == "Point":
                pts = [inter]
            elif inter.geom_type == "MultiPoint":
                pts = list(inter.geoms)
            else:  # collinear overlaps etc.
                pts = []
            out = [(p.x, p.y) for p in pts]
            self._intersection_cache[cache_key] = out
            return out
