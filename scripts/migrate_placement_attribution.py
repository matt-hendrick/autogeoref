#!/usr/bin/env python
"""One-off migration: name the placer on recorded GCPs, drop the dead score key.

Two pieces of inherited residue that reach the published export tree:
- a machine-placed GCP is attributed to ``admin``, which names no one.
  ``matching.gcps_geojson_from`` now writes ``matching.AUTO_PLACED_BY``, so only
  earlier records need this. Any other username is somebody's; untouched.
- ``rmse_vs_human_m`` is a score against volunteer pins, written by nothing
  today; the sidecar is its home, so here it is deleted rather than moved.

Placement values are never touched, and each record keeps its modification time
because ``status`` reads serve freshness off it. Idempotent; dry-run by default.
It takes one work root's own volumes and NAMES any found a level deeper rather
than sweeping them in: a fixture tree and a frozen snapshot are both records a
golden test asserts against, and tidying one would delete its measurement.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from autogeoref.matching import AUTO_PLACED_BY
from autogeoref.paths import VolumeBusyError, VolumePaths, volume_lock, write_result

#: The inherited username this pass replaces. Any other value is somebody's.
INHERITED_USERNAME = "admin"
#: Score key deleted outright — the scoring sidecar is its only home.
DEAD_SCORE_KEY = "rmse_vs_human_m"
#: Volumes sit directly under the work root. A city with a work root of its own,
#: and every frozen measurement snapshot, sits one level further down — those are
#: REPORTED, never swept in: a snapshot is recorded evidence, and rewriting one
#: to tidy a username would edit the measurement it exists to hold.
NESTED_GLOB = "*/*/results"


def migrate_record(record: dict[str, Any]) -> tuple[int, bool]:
    """Rewrite one record in place; returns (points renamed, score key dropped)."""
    dropped = record.pop(DEAD_SCORE_KEY, None) is not None
    renamed = 0
    for feature in (record.get("gcps_geojson") or {}).get("features") or ():
        properties = feature.get("properties")
        if isinstance(properties, dict) and properties.get("username") == INHERITED_USERNAME:
            properties["username"] = AUTO_PLACED_BY
            renamed += 1
    return renamed, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--volume", action="append", default=[])
    ap.add_argument("--apply", action="store_true", help="rewrite records (default: dry-run)")
    args = ap.parse_args()

    found = {p.parent for p in args.work.glob("*/results") if p.is_dir()}
    roots = sorted(v for v in found if not args.volume or v.name in args.volume)
    nested = sorted(p.parent for p in args.work.glob(NESTED_GLOB) if p.is_dir())
    records_touched = 0
    points_renamed = 0
    scores_dropped = 0
    volumes_touched: list[str] = []
    busy: list[str] = []

    def process_volume(root: Path) -> None:
        nonlocal records_touched, points_renamed, scores_dropped
        before = records_touched
        for path in sorted((root / "results").glob("p*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            renamed, dropped = migrate_record(record)
            if not renamed and not dropped:
                continue
            records_touched += 1
            points_renamed += renamed
            scores_dropped += dropped
            if args.apply:
                # `status` reads serve freshness off this mtime, and no placement
                # moved here: bumping it would report every served volume stale
                # and buy nothing but a corpus of no-op re-bakes.
                stat = path.stat()
                write_result(path, record)
                os.utime(path, (stat.st_atime, stat.st_mtime))
        if records_touched > before:
            volumes_touched.append(root.name)

    for root in roots:
        if args.apply:
            # the same per-volume exclusion every mutating entry point takes;
            # a volume mid-bake is skipped and named, never raced
            try:
                with volume_lock(VolumePaths(root=root), "migrate-attribution"):
                    process_volume(root)
            except VolumeBusyError as e:
                busy.append(f"{root.name} (held by {e.holder or 'unknown'})")
        else:
            process_volume(root)

    mode = "APPLIED" if args.apply else "dry-run"
    print(f"{mode}: {records_touched} records in {len(volumes_touched)} of {len(roots)} volumes")
    print(f"  GCP points renamed {INHERITED_USERNAME!r} -> {AUTO_PLACED_BY!r}: {points_renamed}")
    print(f"  {DEAD_SCORE_KEY} keys dropped: {scores_dropped}")
    if busy:
        print(f"  VOLUME BUSY — re-run for these ({len(busy)}): {', '.join(busy)}")
    if nested:
        # Named rather than migrated, and named rather than passed over: a pass
        # that reports "done" while part of the corpus sat one directory deeper
        # is the failure this print exists to prevent.
        print(f"  NOT PROCESSED — a work root of their own ({len(nested)}), pass --work to each:")
        for root in nested:
            print(f"    {root}")


if __name__ == "__main__":
    main()
