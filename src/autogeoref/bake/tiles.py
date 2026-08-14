"""The tile stage: pack the mosaic as PMTiles, and record the zoom it was built at."""

from __future__ import annotations

# `..tiles` below is the top-level module, not this one.
import json
import shutil
from typing import TYPE_CHECKING

from ..paths import write_if_changed

if TYPE_CHECKING:
    from pathlib import Path

    from ..paths import VolumePaths


def write_tile_params(
    paths: VolumePaths, *, min_zoom: int | None = None, max_zoom: int | None = None
) -> Path:
    """Persist the effective tile zoom range; mtime changes only with content."""
    from ..tiles import DEFAULT_MAX_ZOOM, DEFAULT_MIN_ZOOM

    params = {
        "min_zoom": DEFAULT_MIN_ZOOM if min_zoom is None else min_zoom,
        "max_zoom": DEFAULT_MAX_ZOOM if max_zoom is None else max_zoom,
    }
    return write_if_changed(paths.root / "tiles-params.json", json.dumps(params, indent=2))


def stage_tiles(
    paths: VolumePaths,
    volume: str,
    *,
    min_zoom: int | None = None,
    max_zoom: int | None = None,
    processes: int = 4,
    timeout_s: int = 3600,
) -> Path:
    """Render the mosaic(s) to XYZ WEBP tiles and pack ``<volume>.pmtiles``.

    When the mosaic stage separated the volume's declared overview paint into
    ``mosaic-overview.tif``, that renders and packs as
    ``<volume>-overview.pmtiles`` beside the detail archive. Nothing serves that
    companion — publish leaves it here — so it is a local artifact only. A stale
    one from a withdrawn declaration is removed.
    """
    from ..tiles import DEFAULT_MAX_ZOOM, DEFAULT_MIN_ZOOM, pack_pmtiles, render_xyz_tiles

    def render(mosaic: Path, tiles_dir: Path, out_pmtiles: Path) -> Path:
        if tiles_dir.exists():
            shutil.rmtree(tiles_dir)
        render_xyz_tiles(
            mosaic,
            tiles_dir,
            min_zoom=DEFAULT_MIN_ZOOM if min_zoom is None else min_zoom,
            max_zoom=DEFAULT_MAX_ZOOM if max_zoom is None else max_zoom,
            processes=processes,
            timeout_s=timeout_s,
        )
        return pack_pmtiles(tiles_dir, out_pmtiles)

    overview_mosaic = paths.root / "mosaic-overview.tif"
    overview_pmtiles = paths.root / f"{volume}-overview.pmtiles"
    if overview_mosaic.is_file():
        render(overview_mosaic, paths.root / "tiles-overview", overview_pmtiles)
    else:
        overview_pmtiles.unlink(missing_ok=True)
        overview_tiles = paths.root / "tiles-overview"
        if overview_tiles.exists():
            shutil.rmtree(overview_tiles)
    return render(paths.root / "mosaic.tif", paths.root / "tiles", paths.root / f"{volume}.pmtiles")
