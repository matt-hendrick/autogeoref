"""Served-archive bounds probe (the PMTiles header)."""

from __future__ import annotations

from pathlib import Path


def pmtiles_bounds(path: Path) -> list[float]:
    """WGS84 bounds from a PMTiles archive header (no GDAL needed)."""
    from pmtiles.reader import MmapSource, Reader  # optional [tiles] extra

    with path.open("rb") as f:
        header = Reader(MmapSource(f)).header()
    return [
        header["min_lon_e7"] / 1e7,
        header["min_lat_e7"] / 1e7,
        header["max_lon_e7"] / 1e7,
        header["max_lat_e7"] / 1e7,
    ]
