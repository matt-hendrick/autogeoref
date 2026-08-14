"""Fetch one cached named linear-waterway extract for a bounded experiment.

The waterway experiment excludes lake and shoreline polygons. This script uses
the repository's controlled Overpass client, stores the raw response, and
refuses to overwrite it so later calibration replays the same reference.

    uv run python scripts/fetch_osm_waterways.py chicago \
        --bbox -87.95 41.62 -87.50 42.05 \
        --out fixtures/reference/water-chicago-overpass.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autogeoref.osm import fetch_overpass

ROOT = Path(__file__).resolve().parent.parent


def waterway_query(bounds: tuple[float, float, float, float], timeout_s: int = 300) -> str:
    """Return the one-query, named-linear-waterway Overpass request."""
    minx, miny, maxx, maxy = bounds
    return (
        f"[out:json][timeout:{timeout_s}];\n"
        f'way["waterway"]["name"]({miny},{minx},{maxy},{maxx});\n'
        "out geom;\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    out = args.out or ROOT / "cache" / f"water-{args.slug}-overpass.json"
    if out.exists():
        raise SystemExit(f"{out} already exists; cached reference is immutable")

    bbox = tuple(args.bbox)
    query = waterway_query(bbox, args.timeout)
    print(f"Overpass query:\n{query}")
    data = fetch_overpass(query, timeout_s=args.timeout + 60)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data))
    ways = [element for element in data.get("elements", []) if element.get("type") == "way"]
    names: set[str] = set()
    for way in ways:
        name = (way.get("tags") or {}).get("name")
        if isinstance(name, str) and name:
            names.add(name)
    print(f"{len(ways)} named linear waterways: {', '.join(sorted(names))}")
    print(f"raw: {out}")


if __name__ == "__main__":
    main()
