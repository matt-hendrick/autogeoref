"""Experimental waterway-crossing candidates for the water-gazetteer study.

No pipeline stage imports this: it is a harness library, and living here rather
than in the package is what says so. A candidate exists only when a
hand-reviewed gazetteer maps an observed water label to an exact named OSM
waterway group. Lake and shoreline polygons are not represented.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shapely.geometry import LineString
from shapely.ops import unary_union

from autogeoref.matching import Candidate, label_axis
from autogeoref.names import Aliases, normalize

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

    from autogeoref.centerlines import CenterlineIndex

_NON_ALNUM = re.compile(r"[^A-Z0-9 ]")
_WS = re.compile(r"\s+")

Gazetteer = dict[str, tuple[str, ...]]


def normalize_water_name(name: str) -> str:
    """Normalize water names without applying street aliases or directions."""
    return _WS.sub(" ", _NON_ALNUM.sub(" ", name.upper())).strip()


class WaterIndex:
    """Named, linear OSM waterways grouped by exact normalized name."""

    def __init__(self, data: dict[str, Any], gazetteer: Gazetteer) -> None:
        lines: dict[str, list[BaseGeometry]] = {}
        for way in data.get("elements", []):
            points = way.get("geometry")
            tags = way.get("tags") or {}
            name = tags.get("name")
            if way.get("type") != "way" or not name or not points or len(points) < 2:
                continue
            key = normalize_water_name(name)
            line = LineString([(point["lon"], point["lat"]) for point in points])
            lines.setdefault(key, []).append(line)
        self.groups = {key: unary_union(geometries) for key, geometries in lines.items()}
        self.gazetteer = gazetteer

    @classmethod
    def from_json(cls, path: Path, gazetteer: Gazetteer) -> WaterIndex:
        return cls(json.loads(path.read_text()), gazetteer)


def water_crossing_candidates(
    annotation: dict[str, Any],
    water_index: WaterIndex,
    centerline_index: CenterlineIndex,
    aliases: Aliases | None,
    scale: float,
) -> list[Candidate]:
    """Return gazetteer-bound waterway x street rescue candidates.

    A water label supplies an explicit axis because curved waterways have no
    reliable horizontal/vertical inference. The candidate's first key names the
    physical OSM group, so water + street evidence is disjoint from a
    street-pair anchor without making two crossings on one waterway independent.
    """
    aliases = aliases if aliases is not None else centerline_index.aliases
    candidates: list[Candidate] = []
    for water in annotation.get("water_labels") or []:
        groups = water_index.gazetteer.get(normalize_water_name(water["name"]))
        if not groups:
            continue
        water_axis = label_axis(water)
        for street in annotation.get("streets") or []:
            street_geometry = centerline_index.merged(normalize(street["name"], aliases))
            if street_geometry is None:
                continue
            pixel = water_axis.intersection(label_axis(street))
            if pixel.is_empty or pixel.geom_type != "Point":
                continue
            for group in groups:
                water_geometry = water_index.groups.get(group)
                if water_geometry is None:
                    continue
                crossing = water_geometry.intersection(street_geometry)
                points = (
                    [crossing]
                    if crossing.geom_type == "Point"
                    else list(crossing.geoms)
                    if crossing.geom_type == "MultiPoint"
                    else []
                )
                candidates.extend(
                    Candidate(
                        pixel=(pixel.x / scale, pixel.y / scale),
                        world4326=(point.x, point.y),
                        streets=(f"WTR {group}", street["name"]),
                    )
                    for point in points
                )
    return candidates
