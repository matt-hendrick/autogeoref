"""The warp stage: committed sheets to COGs, through GDAL."""

from __future__ import annotations

# `..warp` below is the top-level module, not this one.
import json
import logging
from typing import TYPE_CHECKING, Any

from ..errors import PipelineError
from ..paths import regions_by_page, write_if_changed
from .layers import committed_layers

if TYPE_CHECKING:
    from ..paths import VolumePaths

logger = logging.getLogger(__name__)


def stage_warp(
    paths: VolumePaths,
    volume: str,
    *,
    timeout_s: float = 600.0,
    force: bool = False,
) -> dict[str, Any]:
    """Warp every committed sheet with a full-res image to a COG."""
    from ..warp import gcps_from_feature_collection, warp_sheet

    images = regions_by_page(paths.regions)
    committed = committed_layers(paths, volume)
    warped: dict[str, str] = {}
    skipped: list[str] = []
    for page, slug, record in committed:
        image = images.get(page)
        if image is None:
            skipped.append(page)
            continue
        gcps = gcps_from_feature_collection(record["gcps_geojson"])
        warp_sheet(image, gcps, paths.warped, slug=slug, timeout_s=timeout_s, force=force)
        warped[slug] = (paths.warped / f"{slug}.tif.gcps.json").read_text().strip()
    if not warped:
        raise PipelineError(
            f"warp: none of the {len(committed)} committed sheets has a "
            f"full-res image under {paths.regions}"
        )
    if skipped:
        logger.warning(
            "warp: %d committed sheets have no full-res image (pages %s)",
            len(skipped),
            ", ".join(skipped),
        )
    summary = {"warped": warped, "skipped_no_image": skipped}
    write_if_changed(paths.warped / "warp-summary.json", json.dumps(summary, indent=2))
    return summary
