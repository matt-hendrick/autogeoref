"""Rail-crossing evidence channel for rescue anchors.

Candidates combine annotated rail and street labels with reference rail and
centerline intersections. An explicit gazetteer binds printed rail names to
reference groups; unbound labels produce no candidates. Rail anchors are used
only by translation rescue, never as strict-gate evidence.

Rail names use :func:`normalize_rail_name`, not street normalization or
volume aliases. Candidate disjointness keys on the physical rail group,
prefixed with ``"RR "``, rather than the printed label.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shapely.geometry import LineString, shape
from shapely.ops import unary_union

from .matching import Candidate, label_axis
from .names import Aliases, normalize

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

    from .centerlines import CenterlineIndex

logger = logging.getLogger(__name__)

#: Group key for reference rail geometry carrying neither a name nor an
#: operator tag — such trackage still anchors, it just all shares one name
#: (so a cluster riding only anonymous rail stays provisional).
CATCH_ALL_GROUP = "RAIL"

_PARENTHETICAL = re.compile(r"\([^)]*\)")
_NON_ALNUM = re.compile(r"[^A-Z0-9 ]")
_WS = re.compile(r"\s+")

#: label-normalized-name -> tuple of group keys the label may pair with.
Gazetteer = dict[str, tuple[str, ...]]


def normalize_rail_name(name: str) -> str:
    """Railroad-name normalization — its own lane, never street ``normalize``.

    Uppercase; punctuation replaced by spaces and whitespace collapsed, so
    dotted initialisms canonicalize consistently while parenthetical division
    qualifiers remain available to the gazetteer. No suffix or direction
    stripping and no street aliases (see the module docstring).
    """
    n = name.upper()
    n = _NON_ALNUM.sub(" ", n)
    return _WS.sub(" ", n).strip()


def load_rail_gazetteer(path: Path) -> Gazetteer:
    """Load a gazetteer file's ``bindings`` table, re-normalizing both sides.

    Keys and group values pass through :func:`normalize_rail_name` so a
    hand-edited row keeps working whether it was written raw or normalized.
    Every other top-level key (``unbound``, notes) is documentation and
    ignored here.
    """
    data = json.loads(path.read_text())
    return {
        normalize_rail_name(label): tuple(normalize_rail_name(g) for g in groups)
        for label, groups in data.get("bindings", {}).items()
    }


class RailIndex:
    """Reference rail geometry (4326) grouped by physical railroad.

    Accepts either an Overpass JSON response or a GeoJSON FeatureCollection, mirroring
    ``CenterlineIndex``'s BYO-reference contract. Ways and features are grouped by normalized
    ``name`` tag, falling back to ``operator`` and then a single catch-all group, and ``groups``
    maps group key to merged shapely geometry. Insertion order follows the source file, so
    downstream candidate order is deterministic. ``gazetteer`` binds printed era labels to group
    keys; without one, :func:`rail_crossing_candidates` yields nothing.
    """

    def __init__(self, data: dict[str, Any], gazetteer: Gazetteer | None = None) -> None:
        self.gazetteer = gazetteer
        lines: dict[str, list[BaseGeometry]] = {}

        def add(raw_name: str | None, geom: BaseGeometry) -> None:
            key = normalize_rail_name(raw_name) if raw_name else ""
            lines.setdefault(key or CATCH_ALL_GROUP, []).append(geom)

        if "elements" in data:  # Overpass JSON
            for way in data["elements"]:
                pts = way.get("geometry")
                if not pts or len(pts) < 2:
                    continue
                tags = way.get("tags") or {}
                add(
                    tags.get("name") or tags.get("operator"),
                    LineString([(p["lon"], p["lat"]) for p in pts]),
                )
        else:  # GeoJSON FeatureCollection
            for feat in data.get("features", []):
                if feat.get("geometry") is None:
                    continue
                props = feat.get("properties") or {}
                add(props.get("name") or props.get("operator"), shape(feat["geometry"]))

        self.groups: dict[str, BaseGeometry] = {
            key: unary_union(geoms) for key, geoms in lines.items()
        }

    @classmethod
    def from_json(cls, path: Path, gazetteer_path: Path | None = None) -> RailIndex:
        gaz = load_rail_gazetteer(gazetteer_path) if gazetteer_path is not None else None
        return cls(json.loads(path.read_text()), gazetteer=gaz)

    def merged(self, key: str) -> BaseGeometry | None:
        """Merged geometry for a group key, or ``None`` (CenterlineIndex-style)."""
        return self.groups.get(key)


def rail_crossing_candidates(
    annotation: dict[str, Any],
    rail_index: RailIndex,
    centerline_index: CenterlineIndex,
    aliases: Aliases | None,
    scale: float,
) -> list[Candidate]:
    """Rail-label x street-label crossings as rescue-weight ``Candidate``s.

    For each rail label (orientation guessed from bbox aspect) and each street label: the pixel
    point is the intersection of the two label axes, and the world points are the crossings of
    each BOUND rail group's merged geometry with the street's merged centerline. A label pairs
    only with the groups its gazetteer row names, so an unbound label — or an index with no
    gazetteer — yields nothing. The candidate's street pair names the PHYSICAL railroad group,
    not the printed label. ``scale`` is the small-frame/full-res ratio from the manifest.
    """
    rails = annotation.get("rail_labels") or []
    if not rails:
        return []
    gazetteer = rail_index.gazetteer
    if gazetteer is None:
        logger.warning(
            "rail index has no gazetteer — %d rail label(s) unbound, no candidates",
            len(rails),
        )
        return []
    streets = annotation.get("streets") or []
    aliases = aliases if aliases is not None else centerline_index.aliases
    cands: list[Candidate] = []
    for rail in rails:
        bound = gazetteer.get(normalize_rail_name(rail["name"]))
        if bound is None:
            # Retry without a parenthetical qualifier when no exact binding exists.
            fallback = normalize_rail_name(_PARENTHETICAL.sub(" ", rail["name"]))
            bound = gazetteer.get(fallback)
        if not bound:
            logger.info("rail label %r unbound in gazetteer — no candidates", rail["name"])
            continue
        x0, y0, x1, y1 = rail["bbox"]
        orientation = "horizontal" if (x1 - x0) >= (y1 - y0) else "vertical"
        rail_axis = label_axis({"bbox": rail["bbox"], "orientation": orientation})
        for s in streets:
            merged = centerline_index.merged(normalize(s["name"], aliases))
            if merged is None:
                continue
            pix = rail_axis.intersection(label_axis(s))
            if pix.is_empty or pix.geom_type != "Point":
                continue
            for group_key in bound:
                rail_geom = rail_index.groups.get(group_key)
                if rail_geom is None:
                    continue
                crossing = rail_geom.intersection(merged)
                if crossing.is_empty:
                    continue
                pts = (
                    [crossing]
                    if crossing.geom_type == "Point"
                    else list(crossing.geoms)
                    if crossing.geom_type == "MultiPoint"
                    else []
                )
                if not pts:
                    continue
                # printed-label -> physical-group pairing, logged for audit
                logger.info(
                    "rail label %r -> group %r x %s: %d crossing(s)",
                    rail["name"],
                    group_key,
                    s["name"],
                    len(pts),
                )
                cands.extend(
                    Candidate(
                        pixel=(pix.x / scale, pix.y / scale),
                        world4326=(p.x, p.y),
                        streets=(f"RR {group_key}", s["name"]),
                    )
                    for p in pts
                )
    return cands
