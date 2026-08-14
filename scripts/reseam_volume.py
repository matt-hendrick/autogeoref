#!/usr/bin/env python
"""Re-run the seam solve over a volume's committed records, in place.

The re-bake's first step, and the one a bare ``--warp-only`` bake does not
cover: rewriting ``gcps_geojson`` moves the per-sheet fit that seam ties are
measured against, so the joint solve must re-run before the warp.
``stages.seam`` is idempotent and convergent — it solves in the
PRE-seam frame and applies only the delta against what is recorded — so this
moves each sheet by exactly the correction the new records earn, and a repeat
run applies zero.

Ground truth is auto-resolved from ``fixtures/ground-truth``. Takes the volume
lock; refuses while a run or bake owns the tree. Follow with the back half and
``autogeoref publish``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autogeoref.config.load import load_city_config
from autogeoref.paths import VolumePaths, volume_lock
from autogeoref.score_pass import score_volume
from autogeoref.stages.seam import stage_seam


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("volume")
    ap.add_argument("--city", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="directory of volunteer exports to SCORE against afterwards "
        "(default: <repo>/fixtures/ground-truth). The solve itself never reads it",
    )
    args = ap.parse_args()

    vol = load_city_config(args.city).volume(args.volume)
    paths = VolumePaths(root=args.work / args.volume)
    # anchored to the repo root, not the cwd — a cwd-relative default would
    # silently score against nothing when run from elsewhere
    gt_dir = args.ground_truth or Path(__file__).resolve().parents[1] / "fixtures" / "ground-truth"

    with volume_lock(paths, "reseam"):
        record = stage_seam(paths, overview_pages=vol.overview_pages)
        # the solve moved sheets, so every score on disk now describes a
        # placement that no longer exists; re-grade before anyone reads them
        scored = score_volume(paths, args.volume, [gt_dir])
    nonzero = {k: v for k, v in (record.get("deltas") or {}).items() if any(v)}
    print(f"{args.volume}: seam re-solved")
    print(json.dumps(record, indent=2, default=str)[:2000])
    if nonzero:
        print(f"sheets with a nonzero recorded total delta: {len(nonzero)}")
    seam = scored.get("seam")
    if seam:
        print(
            f"vs human pins: median {seam['gt_median_before_m']:.3f} m -> "
            f"{seam['gt_median_after_m']:.3f} m ({seam['verdict']})"
        )
    else:
        print(f"{args.volume}: no pinned pages in {gt_dir}; the solve is ungraded")


if __name__ == "__main__":
    main()
