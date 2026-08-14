"""The mosaic stage: pack the warped, masked sheets into one raster."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from ..errors import PipelineError
from ..paths import atomic_output_path, regions_by_page, write_if_changed
from ..slugs import DuplicateCoverage, mosaic_paint_order, overview_slug

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

    from ..paths import VolumePaths

logger = logging.getLogger(__name__)


def stage_mosaic(
    paths: VolumePaths, *, overview_pages: Collection[str] = (), timeout_s: int = 3600
) -> Path:
    """Composite current warped layers into ``mosaic.tif``, in paint order.

    Declared overview sheets do NOT join the detail mosaic: their district-scale
    paint is far coarser and looser, and compositing it underneath presents it as
    detail coverage. They composite into ``mosaic-overview.tif``, which
    :func:`stage_tiles` packs as a separate archive nothing serves, kept so the
    paint survives without a re-warp. A volume whose committed sheets are ALL
    overview pages keeps its single ``mosaic.tif``, and that one IS served.
    """
    from ..tiles import cutline_vrt, mosaic_gtiff

    summary_path = paths.warped / "warp-summary.json"
    if not summary_path.is_file():
        raise PipelineError(f"mosaic: {summary_path} missing — run the warp stage first")
    # the same inventory the mask stage resolves twins from, so the two stages
    # cannot classify one sheet differently. An empty one is not fatal here —
    # the mask stage would already have refused — but it silently demotes every
    # skeleton twin to a regular sheet, so say so.
    pages = regions_by_page(paths.regions)
    if not pages:
        logger.warning("mosaic: no sheet scans under %s; skeleton twins unresolved", paths.regions)
    duplicates = DuplicateCoverage.resolve(pages, overview_pages)
    slugs = mosaic_paint_order(list(json.loads(summary_path.read_text())["warped"]), duplicates)
    cogs = [paths.warped / f"{slug}.tif" for slug in slugs]
    missing = [cog.name for cog in cogs if not cog.is_file()]
    if missing:
        raise PipelineError(f"mosaic: warp summary lists missing COGs: {missing}")
    if not cogs:
        raise PipelineError(f"mosaic: no warped COGs recorded in {summary_path}")
    parts_dir = paths.root / "mosaic-parts"
    all_parts: list[Path] = []
    parts: list[Path] = []
    overview_parts: list[Path] = []
    for slug, cog in zip(slugs, cogs, strict=True):
        cutline = paths.masks / f"{cog.stem}.geojson"
        part = parts_dir / f"{cog.stem}.vrt"
        with atomic_output_path(part, publish=False) as temporary:
            cutline_vrt(
                cog.resolve(),
                cutline.resolve() if cutline.is_file() else None,
                temporary,
                timeout_s=timeout_s,
            )
            write_if_changed(part, temporary.read_text())
        all_parts.append(part.resolve())
        (overview_parts if overview_slug(slug, duplicates) else parts).append(part.resolve())
    if not parts:
        # overview-only volume: the archive is the overview
        parts, overview_parts = overview_parts, []
    # the flat ordered list is an on-disk contract (decomposition tooling reads
    # it) and is kept as it was
    parts_manifest = write_if_changed(
        parts_dir / "parts.json", json.dumps([part.name for part in all_parts], indent=2)
    )
    # the detail/overview partition is CONFIG, invisible to every file mtime
    # above: without persisting it, changing a declaration on an already-baked
    # volume would leave mosaic.tif "fresh" with the WRONG paint in it. Both
    # composites read this as an input, so a declaration change rebuilds them,
    # and that in turn makes the tile stage stale.
    partition_manifest = write_if_changed(
        parts_dir / "partition.json",
        json.dumps(
            {
                "detail": [part.name for part in parts],
                "overview": [part.name for part in overview_parts],
            },
            indent=2,
        ),
    )

    def compose(out: Path, out_parts: list[Path]) -> None:
        if out.is_file():
            newest_input = max(
                p.stat().st_mtime for p in [*out_parts, *cogs, parts_manifest, partition_manifest]
            )
            if out.stat().st_mtime >= newest_input:
                logger.info("%s up to date, skipping", out.name)
                return
        with atomic_output_path(out) as temporary_mosaic:
            mosaic_gtiff(out_parts, temporary_mosaic, timeout_s=timeout_s)

    mosaic = paths.root / "mosaic.tif"
    overview_mosaic = paths.root / "mosaic-overview.tif"
    compose(mosaic, parts)
    if overview_parts:
        compose(overview_mosaic, overview_parts)
    else:
        # a declaration change must not leave yesterday's underlay behind
        overview_mosaic.unlink(missing_ok=True)
    return mosaic
