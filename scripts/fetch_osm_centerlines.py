"""One-shot OSM centerline prefetch for a city bbox.

`autogeoref run` fetches and caches OSM centerlines automatically for any city
TOML WITHOUT a `centerlines` key, so this is for MANUAL prefetching: warming
the cache before a batch, fetching on a network-connected machine for an
offline one, or producing a bring-your-own file. It writes the same cache
format `autogeoref.osm.ensure_city_centerlines` maintains, so a prefetched file
is recognized as covering its bbox and never re-fetched.

Conduct, enforced by `autogeoref.osm.fetch_overpass`: GET only, https, one
host, honest User-Agent, one query per invocation. Cached forever — if the
output exists the script refuses to run, because re-fetching is a deliberate
human act.
"""

import argparse
import json
import sys
from pathlib import Path

from autogeoref.osm import centerline_geojson, fetch_overpass, overpass_query

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slug", help="output name: cache/osm-centerlines-<slug>.geojson")
    ap.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default cache/osm-centerlines-<slug>.geojson; for an "
        "OSM-default city, point this at the config's expected cache file)",
    )
    ap.add_argument("--timeout", type=int, default=300, help="Overpass server timeout (s)")
    args = ap.parse_args()

    out_path = args.out
    if out_path is None:
        cache = ROOT / "cache"
        cache.mkdir(exist_ok=True)
        out_path = cache / f"osm-centerlines-{args.slug}.geojson"
    raw_path = out_path.with_name(out_path.stem + "-overpass.json")
    for p in (raw_path, out_path):
        if p.exists():
            sys.exit(
                f"{p} already exists — cached forever, refusing to re-fetch. "
                "Delete both output files first if a re-fetch is truly intended."
            )

    bbox = tuple(args.bbox)
    query = overpass_query(bbox, timeout_s=args.timeout)
    print(f"Overpass query:\n{query}")
    data = fetch_overpass(query, timeout_s=args.timeout + 60)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(data))

    fc = centerline_geojson(data)  # carries schema_version, so the cache is recognized
    fc["fetched_bboxes"] = [list(bbox)]  # ensure_city_centerlines coverage record
    out_path.write_text(json.dumps(fc))
    n_ways = sum(1 for el in data.get("elements", []) if el.get("type") == "way")
    names = {f["properties"]["street_nam"] for f in fc["features"]}
    nodes = {f["properties"][k] for f in fc["features"] for k in ("fnode_id", "tnode_id")}
    print(
        f"{n_ways} ways fetched -> {len(fc['features'])} features, "
        f"{len(names)} distinct names, {len(nodes)} distinct nodes"
    )
    print(f"raw:     {raw_path}")
    print(f"geojson: {out_path}")


if __name__ == "__main__":
    main()
