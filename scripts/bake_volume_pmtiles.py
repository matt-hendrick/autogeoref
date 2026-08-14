"""Bake one volume mosaic into an alpha-correct WEBP PMTiles archive.

The serving-side re-bake tool. Every archive goes through `autogeoref.tiles`'s
measured-correct chain — `gdalwarp -of VRT -dstalpha` (a JPEG-compressed mosaic
keeps transparency in an internal mask a plain VRT drops, and without the alpha
band sheet-edge nodata renders as an opaque black collar) → `gdal2tiles --xyz
--tiledriver=WEBP` → PMTiles. POLICY: never produce another JPEG-tile artifact.

    uv run python scripts/bake_volume_pmtiles.py <mosaic.tif> <out.pmtiles>

The tile tree renders under work/bake/<name>/ and is removed after a successful
pack; the output lands atomically, so the viewer never range-reads a
half-written archive.
"""

import argparse
import shutil
import sys
from pathlib import Path

from autogeoref.tiles import cutline_vrt, pack_pmtiles, render_xyz_tiles

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mosaic", type=Path)
    ap.add_argument("out_pmtiles", type=Path)
    ap.add_argument("--min-zoom", type=int, default=12)
    ap.add_argument("--max-zoom", type=int, default=21)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument(
        "--lossless",
        action="store_true",
        help="lossless WEBP (default lossy: the production 4-6x speed/size "
        "numbers were measured on lossy, and scanned paper compresses well)",
    )
    ap.add_argument(
        "--resampling",
        default="average",
        help="gdal2tiles resampling; 'average' keeps linework readable at "
        "overview zooms (nearest = pepper noise), 'antialias' is the "
        "slower/crisper alternative",
    )
    args = ap.parse_args()

    name = args.out_pmtiles.stem
    scratch = ROOT / "work" / "bake" / name
    scratch.mkdir(parents=True, exist_ok=True)
    tiles_dir = scratch / "tiles"

    vrt = cutline_vrt(args.mosaic, None, scratch / f"{name}.vrt")
    render_xyz_tiles(
        vrt,
        tiles_dir,
        min_zoom=args.min_zoom,
        max_zoom=args.max_zoom,
        processes=args.processes,
        webp=True,
        webp_lossless=args.lossless,
        resampling=args.resampling,
        timeout_s=6 * 3600,
    )
    tmp = args.out_pmtiles.with_suffix(".pmtiles.tmp")
    pack_pmtiles(tiles_dir, tmp)
    args.out_pmtiles.parent.mkdir(parents=True, exist_ok=True)
    tmp.replace(args.out_pmtiles)
    shutil.rmtree(scratch)
    print(f"baked {args.out_pmtiles} (z{args.min_zoom}-{args.max_zoom}, webp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
