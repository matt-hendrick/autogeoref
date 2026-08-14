"""One-shot OSM rail-reference prefetch for a city bbox (rail.RailIndex input).

The rail-crossing channel needs modern railway geometry to cross against street
centerlines. This fetches it in ONE polite Overpass query and caches it forever,
same conduct as `fetch_osm_centerlines.py`, refusing to overwrite an existing
output.

By default only the classes the sheets draw as steam railroad are requested,
and service trackage stays IN: yards and spurs are exactly what an industrial
volume crosses. Where a city's era labels name surviving rapid-transit
alignments, add them via `--classes`. The raw Overpass response is written
as-is; `RailIndex` reads that shape directly.
"""

import argparse
import json
import sys
from pathlib import Path

from autogeoref.osm import fetch_overpass

ROOT = Path(__file__).resolve().parent.parent

RAIL_CLASSES = ("rail", "narrow_gauge")


def rail_query(
    bbox: tuple[float, float, float, float],
    timeout_s: int = 300,
    classes_seq: tuple[str, ...] = RAIL_CLASSES,
) -> str:
    minx, miny, maxx, maxy = bbox
    classes = "|".join(classes_seq)
    return (
        f"[out:json][timeout:{timeout_s}];\n"
        f'way["railway"~"^({classes})$"]({miny},{minx},{maxy},{maxx});\n'
        "out geom;\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slug")
    ap.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument(
        "--classes",
        nargs="+",
        default=list(RAIL_CLASSES),
        metavar="RAILWAY",
        help="OSM railway= classes to request (default: %(default)s)",
    )
    args = ap.parse_args()

    out_path = args.out or (ROOT / "cache" / f"rail-{args.slug}-overpass.json")
    if out_path.exists():
        sys.exit(f"{out_path} already exists — cached forever, refusing to re-fetch.")

    bbox = (args.bbox[0], args.bbox[1], args.bbox[2], args.bbox[3])
    query = rail_query(bbox, timeout_s=args.timeout, classes_seq=tuple(args.classes))
    print(f"Overpass query:\n{query}")
    data = fetch_overpass(query, timeout_s=args.timeout + 60)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data))

    ways = [e for e in data.get("elements", []) if e.get("type") == "way"]
    named = {
        (e.get("tags") or {}).get("name") or (e.get("tags") or {}).get("operator") for e in ways
    }
    named.discard(None)
    print(f"{len(ways)} rail ways -> {len(named)} distinct name/operator groups")
    print(f"raw: {out_path}")


if __name__ == "__main__":
    main()
