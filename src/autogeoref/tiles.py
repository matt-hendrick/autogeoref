"""Mosaic + tiling: warped COGs -> mosaic GTiff -> XYZ tiles -> PMTiles.

Static output by design: PMTiles + a MapLibre viewer + a manifest JSON is
the whole serving story (object storage + CDN; no tile server).

Uses the GDAL CLI tools (`gdalwarp`, `gdal2tiles.py --xyz
--tiledriver=WEBP`) and the ``pmtiles`` writer. WEBP-with-alpha XYZ tiles
rendered in parallel then packed was measured 4-6x faster than serial
MBTILES+gdaladdo in the original pipeline.
"""

from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path

from .paths import atomic_output_path

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 3600
#: Sanborn detail is legible at these web-mercator zooms.
DEFAULT_MIN_ZOOM = 12
DEFAULT_MAX_ZOOM = 20


class TilingError(RuntimeError):
    """A GDAL tiling subprocess failed; message carries stderr."""


def _run(cmd: list[str], timeout_s: int = DEFAULT_TIMEOUT_S) -> None:
    logger.debug("run: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    if proc.returncode != 0:
        raise TilingError(f"{cmd[0]} failed ({proc.returncode}): {proc.stderr[-2000:]}")


def cutline_vrt(
    src: Path, cutline_geojson: Path | None, out_vrt: Path, timeout_s: int = 600
) -> Path:
    """Warped VRT of ``src`` with an explicit ALPHA band (+ optional cutline).

    ``-dstalpha`` is load-bearing: the layer COGs are JPEG-compressed, so their
    transparency lives in a PER_DATASET internal mask that a plain warp VRT
    silently drops, and the mosaic then renders nodata as opaque black.
    Converting mask -> alpha here gives every part a uniform RGBA shape.
    """
    out_vrt.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["gdalwarp", "-overwrite", "-of", "VRT", "-dstalpha"]
    if cutline_geojson is not None:
        cmd += ["-cutline", str(cutline_geojson), "-crop_to_cutline"]
    cmd += [str(src), str(out_vrt)]
    _run(cmd, timeout_s=timeout_s)
    return out_vrt


def mosaic_gtiff(part_paths: list[Path], out_tif: Path, timeout_s: int = 3600) -> Path:
    """Composite RGBA parts into one mosaic GTiff via multi-source gdalwarp.

    The production mosaic semantics: gdalwarp composites sources in order, and
    a source's transparent pixels never overwrite the sheet beneath.
    ``gdalbuildvrt`` is NOT a compositor — it picks the last source covering a
    pixel per band, so a later sheet's transparent margin erases earlier ones.
    """
    if not part_paths:
        raise TilingError("no parts to mosaic")
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "gdalwarp",
            "-overwrite",
            "-of",
            "GTiff",
            "-co",
            "TILED=YES",
            "-co",
            "COMPRESS=DEFLATE",
            # Forced, not IF_NEEDED: GDAL cannot predict a compressed size, so
            # it commits to 4-byte offsets and hits the 4 GB TIFF ceiling only
            # at write time, after every part has been warped.
            "-co",
            "BIGTIFF=YES",
            *[str(p) for p in part_paths],
            str(out_tif),
        ],
        timeout_s=timeout_s,
    )
    return out_tif


def render_xyz_tiles(
    mosaic: Path,
    tiles_dir: Path,
    min_zoom: int = DEFAULT_MIN_ZOOM,
    max_zoom: int = DEFAULT_MAX_ZOOM,
    processes: int = 4,
    webp: bool = True,
    webp_lossless: bool = True,
    resampling: str = "average",
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> Path:
    """Render an XYZ tile tree with gdal2tiles (WEBP with alpha by default).

    ``webp_lossless`` defaults to the round-trip-safe behaviour; benchmark lossy
    WEBP before choosing it. ``resampling`` defaults to ``average`` because
    nearest-neighbour turns Sanborn linework into pepper noise at overview
    zooms: each overview pixel picks one arbitrary source pixel, so thin lines
    and lettering randomly survive or vanish.
    """
    tiles_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gdal2tiles.py",
        "--xyz",
        "-z",
        f"{min_zoom}-{max_zoom}",
        "-w",
        "none",
        "-r",
        resampling,
        f"--processes={processes}",
    ]
    if webp:
        cmd += ["--tiledriver=WEBP"]
        if webp_lossless:
            cmd += ["--webp-lossless"]
    cmd += [str(mosaic), str(tiles_dir)]
    _run(cmd, timeout_s=timeout_s)
    return tiles_dir


def pack_pmtiles(tiles_dir: Path, out_pmtiles: Path, tile_ext: str = "webp") -> Path:
    """Pack an XYZ tile tree into a single PMTiles archive.

    Tile BYTES are streamed at write time (only ``(tileid, path)`` pairs are
    held in memory) so a citywide era archive does not need the whole tile
    tree in RAM.
    """
    from pmtiles.tile import Compression, TileType, zxy_to_tileid
    from pmtiles.writer import Writer

    entries: list[tuple[int, Path]] = []
    zooms: list[int] = []
    xs_by_zoom: dict[int, list[int]] = {}
    ys_by_zoom: dict[int, list[int]] = {}
    for z_dir in sorted(tiles_dir.iterdir()):
        if not z_dir.is_dir() or not z_dir.name.isdigit():
            continue
        z = int(z_dir.name)
        for x_dir in sorted(z_dir.iterdir()):
            if not x_dir.is_dir():
                continue
            x = int(x_dir.name)
            for tile in sorted(x_dir.glob(f"*.{tile_ext}")):
                y = int(tile.stem)
                entries.append((zxy_to_tileid(z, x, y), tile))
                zooms.append(z)
                xs_by_zoom.setdefault(z, []).append(x)
                ys_by_zoom.setdefault(z, []).append(y)
    if not entries:
        raise TilingError(f"no .{tile_ext} tiles under {tiles_dir}")
    entries.sort(key=lambda e: e[0])

    min_z, max_z = min(zooms), max(zooms)
    # geographic bounds from the tile range at max zoom (XYZ slippy-map math)
    n = 2**max_z
    x0, x1 = min(xs_by_zoom[max_z]), max(xs_by_zoom[max_z]) + 1
    y0, y1 = min(ys_by_zoom[max_z]), max(ys_by_zoom[max_z]) + 1
    min_lon = x0 / n * 360.0 - 180.0
    max_lon = x1 / n * 360.0 - 180.0
    max_lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y0 / n))))
    min_lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y1 / n))))

    with atomic_output_path(out_pmtiles) as temporary, temporary.open("wb") as f:
        writer = Writer(f)
        for tileid, tile_path in entries:
            writer.write_tile(tileid, tile_path.read_bytes())
        writer.finalize(
            {
                "tile_type": TileType.WEBP if tile_ext == "webp" else TileType.PNG,
                "tile_compression": Compression.NONE,
                "min_zoom": min_z,
                "max_zoom": max_z,
                "min_lon_e7": int(min_lon * 1e7),
                "min_lat_e7": int(min_lat * 1e7),
                "max_lon_e7": int(max_lon * 1e7),
                "max_lat_e7": int(max_lat * 1e7),
                "center_zoom": (min_z + max_z) // 2,
                "center_lon_e7": int((min_lon + max_lon) / 2 * 1e7),
                "center_lat_e7": int((min_lat + max_lat) / 2 * 1e7),
            },
            {"generator": "autogeoref"},
        )
    logger.info("packed %d tiles (z%d-z%d) -> %s", len(entries), min_z, max_z, out_pmtiles)
    return out_pmtiles
