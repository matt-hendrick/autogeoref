"""A sheet's own drawn rotation, estimated from its cardinal street labels.

Placement-free: it reads the labels the annotation calls horizontal or vertical,
looks up each street's principal bearing in the centerline index, and takes the
mode. It never reads a placement, a fit or a result record.

The estimate is an AXIS — street bearings have no direction — so it fixes a
rotation only mod 180. A caller that needs the full turn must try both ends.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, Any

from .affine import TO_3857
from .names import normalize

if TYPE_CHECKING:
    from collections.abc import Mapping

    from shapely.geometry.base import BaseGeometry

    from .centerlines import Bounds, CenterlineIndex

#: Fewest agreeing cardinal votes before an estimate is trusted.
MIN_VOTES = 3
#: A vote further than this from the consensus does not count as agreeing.
VOTE_TOL_DEG = 15.0
#: A street whose segment bearings scatter more than this has no single axis.
SPREAD_MAX_DEG = 15.0


def _segments(geom: BaseGeometry) -> list[tuple[Any, Any]]:
    from shapely.geometry import LineString, MultiLineString

    if isinstance(geom, LineString):
        lines: list[Any] = [geom]
    elif isinstance(geom, MultiLineString):
        lines = list(geom.geoms)
    else:
        lines = [g for g in getattr(geom, "geoms", []) if isinstance(g, LineString)]
    out: list[tuple[Any, Any]] = []
    for line in lines:
        cs = list(line.coords)
        out.extend((a[:2], b[:2]) for a, b in itertools.pairwise(cs))
    return out


def _principal_bearing(geom: BaseGeometry) -> float | None:
    """Length-weighted principal bearing in 3857 degrees, or None if it scatters.

    Bearings are doubled before averaging so a segment and its reverse agree;
    the resultant length gives the scatter that :data:`SPREAD_MAX_DEG` bounds.
    """
    sx = sy = total = 0.0
    for (ax, ay), (bx, by) in _segments(geom):
        x0, y0 = TO_3857.transform(ax, ay)
        x1, y1 = TO_3857.transform(bx, by)
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        theta = 2 * math.atan2(dy, dx)
        sx += length * math.cos(theta)
        sy += length * math.sin(theta)
        total += length
    if total <= 0:
        return None
    resultant = math.hypot(sx, sy) / total
    if resultant <= 0:
        return None
    spread = math.degrees(math.sqrt(max(0.0, -2 * math.log(resultant)))) / 2
    if spread > SPREAD_MAX_DEG:
        return None
    return math.degrees(math.atan2(sy, sx) / 2)


def _consensus_mod180(angles: list[float]) -> float:
    best, best_score = angles[0], None
    for cand in angles:
        inliers = [a for a in angles if abs(((a - cand + 90) % 180) - 90) <= VOTE_TOL_DEG]
        dev = sum(abs(((a - cand + 90) % 180) - 90) for a in inliers)
        score = (len(inliers), -dev)
        if best_score is None or score > best_score:
            best, best_score = cand, score
    return best


class OwnGridEstimator:
    """Per-volume estimator of a sheet's own drawn rotation. Build one, reuse it.

    Caches each street's principal bearing, which is the expensive part and is
    shared across every sheet in the volume.
    """

    def __init__(self, index: CenterlineIndex, bounds: Bounds | None = None) -> None:
        self._index = index
        self._bounds = bounds
        self._axis: dict[str, float | None] = {}
        self._unique: dict[str, bool] = {}

    def _is_unique(self, key: str) -> bool:
        """True when every segment under this key carries the same street name.

        A key several distinct streets normalize onto has no single axis, and a
        label naming it would vote for a bearing no street on the page has.
        """
        if key not in self._unique:
            names = {
                f"{s['props'].get(self._index.name_property) or ''} "
                f"{s['props'].get(self._index.type_property) or ''}".strip().upper()
                for s in self._index.by_name.get(key, ())
            }
            self._unique[key] = len(names) == 1
        return self._unique[key]

    def street_axis_deg(self, key: str) -> float | None:
        """Principal bearing of one normalized street key, degrees, or None."""
        if key not in self._axis:
            geom = self._index.merged(key)
            if geom is not None and self._bounds is not None:
                from shapely.geometry import box

                geom = geom.intersection(box(*self._bounds))
            self._axis[key] = None if geom is None or geom.is_empty else _principal_bearing(geom)
        return self._axis[key]

    def estimate(self, annotation: Mapping[str, Any]) -> float | None:
        """The sheet's own rotation mod 180, or None when the labels do not agree.

        Under a pinned linear part at ``theta`` a pixel-horizontal direction has
        world bearing ``theta`` and a pixel-vertical one ``theta - 90``, so each
        cardinal label whose street resolves votes for one ``theta``.
        """
        votes: list[float] = []
        for st in annotation.get("streets", ()):
            orientation = st.get("orientation")
            if orientation not in ("horizontal", "vertical"):
                continue
            key = normalize(st["name"], self._index.aliases)
            if not key or not self._is_unique(key):
                continue
            beta = self.street_axis_deg(key)
            if beta is None:
                continue
            votes.append((beta if orientation == "horizontal" else beta + 90.0) % 180.0)
        if len(votes) < MIN_VOTES:
            return None
        theta = _consensus_mod180(votes)
        inliers = [a for a in votes if abs(((a - theta + 90) % 180) - 90) <= VOTE_TOL_DEG]
        if len(inliers) < MIN_VOTES or len(inliers) / len(votes) < 0.5:
            return None
        s = sum(math.sin(math.radians(2 * a)) for a in inliers)
        c = sum(math.cos(math.radians(2 * a)) for a in inliers)
        return math.degrees(math.atan2(s, c)) / 2
