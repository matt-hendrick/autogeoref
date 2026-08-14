"""The inputs a run resolves before any stage: its bounds and its street index.

Neither is a stage — no file target, nothing idempotent to resume — and neither
calls the other. The stages themselves are in ``stages/``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .bounds import community_area_bounds, counterpart_bounds
from .centerlines import CenterlineIndex
from .config.model import ConfigError
from .names import load_aliases

if TYPE_CHECKING:
    from pathlib import Path

    from .config.model import CityConfig, VolumeConfig


class NoBoundsSourceError(ConfigError):
    """No bounds source is declared; the caller may bootstrap one."""


def resolve_bounds(
    city: CityConfig,
    vol: VolumeConfig,
    viewer_manifest_path: Path | None = None,
) -> tuple[float, float, float, float]:
    """Bounds priority: bbox > counterpart > community areas.

    Human pins are NOT a source. Bounds is a hard membership filter — a street
    outside the box is a candidate for no sheet at all — so drawing it from
    hand-placed pins is those pins deciding a placement.

    A volume with NONE of those raises :class:`NoBoundsSourceError`, which
    ``cli._cmd_run`` answers with the bootstrap (``bounds_bootstrap``)."""
    if vol.bounds_bbox is not None:
        return vol.bounds_bbox
    if vol.bounds_from_counterpart:
        if viewer_manifest_path is None or not viewer_manifest_path.exists():
            raise ConfigError(
                f"{vol.identifier}: bounds_from={vol.bounds_from_counterpart} "
                f"needs a viewer manifest"
            )
        manifest = json.loads(viewer_manifest_path.read_text())
        return counterpart_bounds(manifest, vol.bounds_from_counterpart)
    if vol.bounds_areas:
        if city.community_areas_path is None:
            raise ConfigError(f"{vol.identifier}: bounds_areas needs city community_areas")
        features = json.loads(city.community_areas_path.read_text())["features"]
        return community_area_bounds(features, list(vol.bounds_areas))
    raise NoBoundsSourceError(
        f"{vol.identifier}: no bounds source (bounds_bbox / bounds_from / bounds_areas)"
    )


def build_index(
    city: CityConfig,
    vol: VolumeConfig,
    bounds: tuple[float, float, float, float],
    features: list[dict[str, Any]] | None = None,
) -> CenterlineIndex:
    """Volume-bounded centerline index; ``features`` skips re-parsing the file.

    The citywide GeoJSON runs to tens of MB and several pipeline stages need
    its features — a caller that already parsed it passes the feature list so
    one run never parses the same file twice.
    """
    aliases = load_aliases(city.aliases_path(vol.identifier))
    if features is not None:
        return CenterlineIndex(
            features,
            aliases=aliases,
            bounds_4326=bounds,
            name_property=city.centerline_name_property,
            type_property=city.centerline_type_property,
        )
    return CenterlineIndex.from_geojson(
        city.centerlines_path,
        aliases=aliases,
        bounds_4326=bounds,
        name_property=city.centerline_name_property,
        type_property=city.centerline_type_property,
    )
