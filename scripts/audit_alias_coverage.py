"""Corpus sweep for historic street-name alias gaps.

For every volume under ``work/`` with cached annotations, rebuild the
volume-bounded centerline index the way ``autogeoref run`` does, alias file
and all, then measure two shares: annotated street reads whose normalized key
is in the index, and result records with no candidates. A volume is flagged
below the shared match bar or at or above the shared zero-candidate bar —
bars and counting alike are the shipped tripwire's, so a sweep and a run agree.

Unlike the per-run report note, the sweep floors nothing on sample size. Zero
model calls, zero network; writes nothing under ``work/<volume>/``.

    uv run python scripts/audit_alias_coverage.py --out work/scratch/sweep.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from autogeoref.bounds_bootstrap import persisted_bounds
from autogeoref.config.load import load_city_config
from autogeoref.name_match import (
    HIGH_ZERO_CANDIDATE_SHARE,
    LOW_MATCH_RATE,
    count_name_matches,
)
from autogeoref.paths import VolumePaths
from autogeoref.run_inputs import NoBoundsSourceError, build_index, resolve_bounds
from autogeoref.sheet_inputs import load_sheet_inputs
from autogeoref.viewer.layout import city_manifest
from autogeoref.volume import status_ok

# the counting and the bars belong to the shipped tripwire (``name_match``),
# not to this script: a corpus number and a run-time number that are measured
# or flagged differently cannot be compared, which is the whole point of both
MATCH_FLAG = LOW_MATCH_RATE
ZERO_SHARE_FLAG = HIGH_ZERO_CANDIDATE_SHARE
TOP_UNMATCHED = 12


def audit_volume(
    work: Path,
    city_path: Path,
    volume: str,
    features: list[dict[str, Any]],
    viewer_manifest: Path | None,
) -> dict[str, Any]:
    """Name-match and candidate-starvation metrics for one volume."""
    city = load_city_config(city_path)
    vol = city.volume(volume)
    paths = VolumePaths(root=work / volume)
    bounds: tuple[float, float, float, float] | None
    try:
        bounds = resolve_bounds(city, vol, viewer_manifest)
    except NoBoundsSourceError:
        # undeclared-bounds volumes ran off the bootstrap's persisted derivation
        bounds = persisted_bounds(paths)
        if bounds is None:
            raise
    index = build_index(city, vol, bounds, features=features)

    counts = count_name_matches(load_sheet_inputs(paths), index)

    zero_candidate_pages: list[str] = []
    results = ok = 0
    for path in sorted(paths.results.glob("p*.json")):
        record = json.loads(path.read_text())
        results += 1
        if record.get("n_candidates") == 0:
            zero_candidate_pages.append(record.get("page", path.stem.removeprefix("p")))
        if status_ok(str(record.get("status", ""))):
            ok += 1

    match_rate = counts.match_rate
    zero_share = len(zero_candidate_pages) / results if results else None
    return {
        "volume": volume,
        "alias_file": city.aliases_path(vol.identifier).exists(),
        **counts.document(),
        "results": results,
        "zero_candidate_pages": len(zero_candidate_pages),
        "zero_candidate_share": round(zero_share, 4) if zero_share is not None else None,
        "ok": ok,
        "flagged": bool(
            (match_rate is not None and match_rate < MATCH_FLAG)
            or (zero_share is not None and zero_share >= ZERO_SHARE_FLAG)
        ),
        "top_unmatched": counts.unmatched.most_common(TOP_UNMATCHED),
    }


def annotated_volumes(work: Path) -> list[str]:
    """Volumes with a sheet manifest and at least one loadable bare read."""
    out = []
    for d in sorted(work.iterdir()):
        paths = VolumePaths(root=d)
        if not paths.manifest.is_file():
            continue
        # bare active reads are p<N>.json; cache/marker siblings all dot the stem
        if any("." not in p.stem for p in paths.annotations.glob("p*.json")):
            out.append(d.name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--city", type=Path, default=Path("configs/chicago/chicago.toml"))
    ap.add_argument(
        "--viewer-manifest",
        type=Path,
        default=None,
        help="default: viewer/<city-slug>/manifest.json",
    )
    ap.add_argument("--volumes", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # cache siblings (p<N>.v2.<model>.json, p<N>.escalated.*) have no manifest
    # entry by design; the loader's per-file warnings would drown the table
    logging.getLogger("autogeoref.sheet_inputs").setLevel(logging.ERROR)

    city = load_city_config(args.city)
    features = json.loads(city.centerlines_path.read_text())["features"]
    declared = args.viewer_manifest or city_manifest(city.name)
    viewer_manifest = declared if declared.is_file() else None

    volumes = args.volumes or annotated_volumes(args.work)
    rows = []
    for volume in volumes:
        row = audit_volume(args.work, args.city, volume, features, viewer_manifest)
        rows.append(row)
        rate = f"{row['match_rate']:.0%}" if row["match_rate"] is not None else "n/a"
        print(
            f"{volume}: {rate} of {row['reads']} reads matched, "
            f"{row['zero_candidate_pages']}/{row['results']} zero-candidate pages, "
            f"{row['ok']} ok, aliases={'yes' if row['alias_file'] else 'no'}"
            + (" FLAGGED" if row["flagged"] else "")
        )

    flagged = [r["volume"] for r in rows if r["flagged"]]
    print(f"\nflagged ({len(flagged)}): {', '.join(flagged) or 'none'}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"volumes": rows}, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
