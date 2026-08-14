"""Freeze the three cross-era golden testbeds into the fixture tree.

Copies each validated volume's state from work/goldens/<vid>/ into
fixtures/<vid>/ (annotations INCLUDING the legacy v2 sidecars and
escalated-style caches, results with the recorded verified statuses, sheets
smalls + manifest, subset.json, seam_deltas.json), the volunteer GT into
fixtures/ground-truth/, and any inferred alias table into the city's
configs/<city>/aliases/ (tracked, not pinned). Failure markers
(*.failed*.json) are the RETRY LEDGER, not evidence — they stay behind.

After this, run scripts/make_fixture_manifest.py and commit the manifest;
from that moment the trees are read-only (FIXTURES.md protocol).

    .venv/bin/python scripts/freeze_new_goldens.py
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "work" / "goldens"
FIX = ROOT / "fixtures"
#: alias tables are tracked beside the city config, not pinned in the manifest
ALIASES = ROOT / "configs" / "chicago" / "aliases"

VOLUMES = ["sanborn01790_041", "sanborn01790_089", "sanborn01790_130"]


def main() -> None:
    for vid in VOLUMES:
        src = SRC / vid
        dst = FIX / vid
        if dst.exists():
            print(f"{vid}: fixture tree already exists — refusing to overwrite", file=sys.stderr)
            continue
        (dst / "annotations").mkdir(parents=True)
        n_ann = 0
        for f in sorted((src / "annotations").glob("*.json")):
            if ".failed" in f.name:
                continue
            shutil.copy2(f, dst / "annotations" / f.name)
            n_ann += 1
        shutil.copytree(src / "sheets", dst / "sheets")
        shutil.copytree(src / "results", dst / "results")
        for extra in ("subset.json", "seam_deltas.json", "volume-constants.json"):
            if (src / extra).exists():
                shutil.copy2(src / extra, dst / extra)
        gt_dst = FIX / "ground-truth" / f"api-layers-{vid}.json"
        if not gt_dst.exists():
            shutil.copy2(src / f"api-layers-{vid}.json", gt_dst)
        inferred = src / "aliases-inferred.json"
        alias_dst = ALIASES / f"aliases-{vid}.json"
        if inferred.exists() and not alias_dst.exists():
            # only freeze non-empty tables; a volume can adopt zero aliases
            import json

            table = {
                k: v for k, v in json.loads(inferred.read_text()).items() if not k.startswith("_")
            }
            if table:
                shutil.copy2(inferred, alias_dst)
        n_results = len(list((dst / "results").glob("p*.json")))
        n_smalls = len(list((dst / "sheets").glob("p*_small.jpg")))
        print(f"{vid}: froze {n_ann} annotation files, {n_results} results, {n_smalls} smalls")


if __name__ == "__main__":
    main()
