#!/usr/bin/env python
"""Find committed sheets whose placement model MIRRORS the scan.

A sheet's poly1 model maps page pixels to ground. Page y grows DOWNWARD while
EPSG:3857 y grows upward, so an upright placement has a NEGATIVE determinant
for the 2x2 linear part; a POSITIVE one is a reflection. The warped sheet then
reads backwards, and gdalwarp fills its rotated frame's corners with opaque
black that the mosaic paints wherever the sheet's mask reaches them.

Nothing in the accept path tests handedness — RANSAC scores residuals, and a
reflected model can fit a symmetric junction set as well as the upright one, so
a mirrored sheet reaches the mosaic as a normal accept.
READ-ONLY: reads ``work/<vid>/results/`` only.

    uv run python scripts/audit_reflected_placements.py [--volume <vid>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autogeoref.affine import AffineMatrix, fit_affine, gcps_from_geojson
from autogeoref.placement_records import pinned_orientation


def determinant(matrix: AffineMatrix) -> float:
    """det of the 2x2 linear part of the ``[X, Y] = M @ [1, px, py]`` model."""
    return float(matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1])


def audit(volume: str, work: Path) -> tuple[int, list[tuple[str, float, dict[str, Any]]]]:
    results = work / volume / "results"
    if not results.is_dir():
        return 0, []
    committed = 0
    mirrored: list[tuple[str, float, dict[str, Any]]] = []
    for path in sorted(results.glob("*.json")):
        record = json.loads(path.read_text())
        if "REJECT" in str(record.get("status", "")).upper():
            continue
        try:
            matrix = fit_affine(gcps_from_geojson(record["gcps_geojson"]))
        except Exception:
            continue
        committed += 1
        det = determinant(matrix)
        if det > 0:
            mirrored.append((path.stem, det, record))
    return committed, mirrored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", action="append", default=[])
    parser.add_argument("--work", type=Path, default=Path("work"))
    args = parser.parse_args()

    volumes = args.volume or sorted(
        p.parent.name for p in args.work.glob("*/results") if p.is_dir()
    )
    total_committed = total_mirrored = 0
    for volume in volumes:
        committed, mirrored = audit(volume, args.work)
        total_committed += committed
        total_mirrored += len(mirrored)
        if not mirrored:
            continue
        print(f"{volume}: {len(mirrored)} of {committed} committed sheets are MIRRORED")
        for page, det, record in mirrored:
            print(
                f"    {page:>6}  det={det:+.3e}  n_inliers={record.get('n_inliers')}"
                f"  pinned={pinned_orientation(record)}"
                f"  status={record.get('status')}"
            )
    print(
        f"\n{total_mirrored} mirrored of {total_committed} committed, over {len(volumes)} volumes"
    )


if __name__ == "__main__":
    main()
