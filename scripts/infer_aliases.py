"""Suggest historical->modern aliases for unmatched labels on failed sheets.

Volume-level knowledge only: the volume scan scale and north-up orientation.
For each REJECTED page, every matched intersection candidate is tried as a
translation-only anchor; each unmatched vertical label's axis projects to a
longitude and each horizontal to a latitude, and the nearest modern street at
that coordinate is suggested. The anchor yielding the most and closest
suggestions wins.

Suggestions are CURATION INPUT, not adopted automatically — except via
`--adopt`, which writes only suggestions recurring on two or more sheets within
a tight distance into `<volume-root>/aliases-inferred.json`. Review the
printout either way.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from shapely.geometry import LineString

from autogeoref.affine import TO_3857, TO_4326
from autogeoref.bounds import load_ground_truth, volume_bounds
from autogeoref.centerlines import CenterlineIndex
from autogeoref.matching import candidate_gcps
from autogeoref.names import normalize

ROOT = Path(__file__).resolve().parent.parent

#: A suggestion is usable when the projected label sits within this of a
#: modern street (the origin's threshold; ~half a short block).
MAX_SUGGEST_M = 60.0
#: --adopt keeps only suggestions recurring on >= 2 sheets at <= this.
ADOPT_MAX_M = 30.0
ADOPT_MIN_SHEETS = 2


def street_position(
    index: CenterlineIndex,
    key: str,
    kind: str,
    ref_lat: float,
    ref_lng: float,
    bounds: tuple[float, float, float, float],
) -> float | None:
    """Longitude of a N-S street at ref_lat (kind='v') / latitude at ref_lng."""
    merged = index.merged(key)
    if merged is None:
        return None
    if kind == "v":
        cut = LineString([(bounds[0] - 0.05, ref_lat), (bounds[2] + 0.05, ref_lat)])
    else:
        cut = LineString([(ref_lng, bounds[1] - 0.05), (ref_lng, bounds[3] + 0.05)])
    inter = merged.intersection(cut)
    if inter.is_empty:
        return None
    c = inter.centroid
    return float(c.x if kind == "v" else c.y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("volume_root", type=Path)
    ap.add_argument("--slug-prefix", required=True)
    ap.add_argument("--adopt", action="store_true")
    args = ap.parse_args()
    root = args.volume_root
    vid = root.name

    gt = load_ground_truth(root / f"api-layers-{vid}.json", slug_prefix=args.slug_prefix)
    bounds = volume_bounds(gt)
    index = CenterlineIndex.from_geojson(
        ROOT / "fixtures" / "reference" / "street_center_lines.geojson",
        aliases={},
        bounds_4326=bounds,
    )
    constants = json.loads((root / "volume-constants.json").read_text())
    scale_m = float(constants["scale_m_per_px"])
    manifest = json.loads((root / "sheets" / "manifest.json").read_text())
    # degree->meter factors at the volume's mid-latitude
    mid_lat = (bounds[1] + bounds[3]) / 2
    import math

    m_per_deg_lng = 111320.0 * math.cos(math.radians(mid_lat))
    m_per_deg_lat = 110900.0

    # miss summary first (the origin's miss_summary.py, folded in): unmatched
    # normalized names by frequency across rejected sheets — the curation
    # priority list
    miss_freq: dict[str, int] = defaultdict(int)
    for rp in sorted((root / "results").glob("p*.json")):
        r = json.loads(rp.read_text())
        if not str(r.get("status", "")).startswith("REJECTED"):
            continue
        ann_path = root / "annotations" / f"p{r['page']}.json"
        if not ann_path.exists():
            continue
        for st in json.loads(ann_path.read_text()).get("streets") or []:
            key = normalize(st["name"], {})
            if index.merged(key) is None:
                miss_freq[key] += 1
    if miss_freq:
        print("unmatched normalized names across rejected sheets (by frequency):")
        for name, n in sorted(miss_freq.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>3}  {name}")
        print()

    tally: dict[tuple[str, str], list[float]] = defaultdict(list)
    for rp in sorted((root / "results").glob("p*.json")):
        r = json.loads(rp.read_text())
        if not str(r.get("status", "")).startswith("REJECTED"):
            continue
        page = str(r["page"])
        ann_path = root / "annotations" / f"p{page}.json"
        info = manifest.get(f"p{page}")
        if not ann_path.exists() or info is None:
            continue
        ann = json.loads(ann_path.read_text())
        s = float(ann.get("scale", info["scale"]))
        cands = candidate_gcps(ann, index, s, {})
        if not cands:
            continue
        unmatched = [
            st for st in ann.get("streets") or [] if index.merged(normalize(st["name"], {})) is None
        ]
        if not unmatched:
            continue

        best: tuple[tuple[int, float], tuple[str, str], list[tuple[str, str, float]]] | None = None
        for anchor in cands:
            ax_px, ay_px = anchor.pixel
            ax_w, ay_w = TO_3857.transform(*anchor.world4326)
            suggestions: list[tuple[str, str, float]] = []
            total_err = 0.0
            for st in unmatched:
                x0, y0, x1, y1 = st["bbox"]
                cx, cy = (x0 + x1) / 2 / s, (y0 + y1) / 2 / s
                lng, lat = TO_4326.transform(
                    ax_w + (cx - ax_px) * scale_m, ay_w - (cy - ay_px) * scale_m
                )
                kind = "v" if st.get("orientation") == "vertical" else "h"
                best_name, best_d = None, 1e9
                for key in index.by_name:
                    pos = street_position(index, key, kind, lat, lng, bounds)
                    if pos is None:
                        continue
                    d = abs(pos - (lng if kind == "v" else lat)) * (
                        m_per_deg_lng if kind == "v" else m_per_deg_lat
                    )
                    if d < best_d:
                        best_name, best_d = key, d
                if best_name and best_d < MAX_SUGGEST_M:
                    suggestions.append((st["name"], best_name, round(best_d, 1)))
                    total_err += best_d
                else:
                    total_err += 200.0
            score = (len(suggestions), -total_err)
            if best is None or score > best[0]:
                best = (score, anchor.streets, suggestions)
        if best is None:
            continue
        _score, anchor_used, suggestions = best
        if suggestions:
            print(f"p{page}: anchor {anchor_used}")
            for orig, sug, d in suggestions:
                print(f"    {orig!r} -> {sug!r} ({d} m)")
                tally[(normalize(orig, {}), sug)].append(d)

    if args.adopt:
        adopted = {
            old: new
            for (old, new), dists in sorted(tally.items())
            if len(dists) >= ADOPT_MIN_SHEETS and max(dists) <= ADOPT_MAX_M and old != new
        }
        out = root / "aliases-inferred.json"
        out.write_text(
            json.dumps(
                {
                    "_comment": (
                        f"GEOMETRICALLY INFERRED aliases for {vid} "
                        f"(scripts/infer_aliases.py port of the origin tool): only "
                        f"suggestions recurring on >={ADOPT_MIN_SHEETS} sheets at "
                        f"<={ADOPT_MAX_M:.0f} m. Volume-scoped; owner review advised."
                    ),
                    **adopted,
                },
                indent=1,
            )
        )
        print(f"\nadopted {len(adopted)} aliases -> {out}")
        for old, new in adopted.items():
            print(f"  {old!r} -> {new!r}")


if __name__ == "__main__":
    main()
