"""Shared GeoJSON geometry helpers: the one centerline-clip rule.

Two clip implementations used to disagree about a known bug (first-vertex vs
any-vertex, ``); every
consumer now shares this one.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def iter_vertices(coords: Any) -> Iterator[tuple[float, float]]:
    """Yield every ``(x, y)`` position in a GeoJSON coordinates array."""
    if not isinstance(coords, list) or not coords:
        return
    if isinstance(coords[0], (int, float)):
        if len(coords) >= 2:
            yield float(coords[0]), float(coords[1])
        return
    for sub in coords:
        yield from iter_vertices(sub)


def clip_features_4326(
    features: list[dict[str, Any]],
    bounds_4326: tuple[float, float, float, float],
    margin_deg: float = 0.02,
) -> list[dict[str, Any]]:
    """Clip GeoJSON features to volume bounds: ANY vertex inside keeps a feature.

    The stage contracts want caller-clipped features (the per-street segment
    index is built over whatever arrives; far-away same-named segments would
    otherwise become votable), and a ~2 km margin is exact enough for
    block-length centerline segments. Any-vertex is the rule because testing
    only the first vertex silently dropped long streets that cross the box
    but happen to START outside it, thinning the layer exactly at volume
    edges.
    """
    minx, miny, maxx, maxy = bounds_4326
    out: list[dict[str, Any]] = []
    for f in features:
        geometry = f.get("geometry") or {}
        if any(
            minx - margin_deg <= x <= maxx + margin_deg
            and miny - margin_deg <= y <= maxy + margin_deg
            for x, y in iter_vertices(geometry.get("coordinates"))
        ):
            out.append(f)
    return out
