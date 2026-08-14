"""Flip-set measurement: raw-label vs normalized-label disjoint rule.

The normalized-label semantics this measured were adopted as the production
rule (``rescue.has_disjoint_pair`` normalizes internally); the "raw" side is
computed inline here, matching the pre-switch rule verbatim.

The original rule compared anchors on RAW annotation labels, so case and
punctuation variance of ONE street could fake disjointness. Normalization only
merges labels, never splits, so every recorded revocation must stay revoked and
the only movable outcomes are recorded rescues that flip to revoked.

Read-only over ``fixtures/``, no model calls. Exits 1 on any regression.

    uv run python scripts/measure_disjoint_flipset.py
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import RawDescriptionHelpFormatter
from pathlib import Path

from autogeoref.names import load_aliases
from autogeoref.rescue import has_disjoint_pair
from autogeoref.volume import REVOKED_PREFIX, STATUS_CORROBORATED

ROOT = Path(__file__).resolve().parents[1]


def raw_disjoint(anchor_streets: list[tuple[str, ...]]) -> bool:
    """The PRE-SWITCH production rule: raw-label set comparison, verbatim."""
    sets = [frozenset(s) for s in anchor_streets]
    return any(not (sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets)))


def main() -> int:
    # argparse for --help alone: without it, `--help` silently runs the whole
    # sweep and prints a measurement a reader can mistake for the answer.
    argparse.ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    ).parse_args()
    fixtures = ROOT / "fixtures"
    aliases_dir = ROOT / "configs" / "chicago" / "aliases"
    total_rescued = total_flip = total_revoked = regressions = 0
    for vol_dir in sorted(fixtures.glob("sanborn*")):
        results = vol_dir / "results"
        if not results.is_dir():
            continue
        vol = vol_dir.name
        aliases = load_aliases(aliases_dir / f"aliases-{vol}.json")
        flips: list[str] = []
        n_rescued = n_revoked = 0
        for rp in sorted(results.glob("p*.json")):
            r = json.loads(rp.read_text())
            anchors = r.get("rescue_anchors")
            if not anchors:
                continue
            raw = [tuple(a) for a in anchors]
            status = str(r.get("status", ""))
            if status.startswith(REVOKED_PREFIX) or status == STATUS_CORROBORATED:
                n_revoked += 1
                # invariant: normalization can only merge -> stays non-disjoint
                if not raw_disjoint(raw) and has_disjoint_pair(raw, aliases):
                    regressions += 1
                    print(f"  REGRESSION {vol} p{r['page']}: revoked set became disjoint?!")
            elif status == "OK (rescued)":
                n_rescued += 1
                if raw_disjoint(raw) and not has_disjoint_pair(raw, aliases):
                    flips.append(str(r["page"]))
        if n_rescued or n_revoked:
            print(
                f"{vol}: {n_rescued} recorded rescues, {n_revoked} revoked/corroborated; "
                f"flip set (rescued that would revoke under normalized labels): "
                f"{len(flips)}{' -> p' + ', p'.join(flips) if flips else ''}"
            )
        total_rescued += n_rescued
        total_flip += len(flips)
        total_revoked += n_revoked
    print(
        f"\nTOTAL: {total_flip} of {total_rescued} recorded rescues flip to revoked "
        f"under normalized-label semantics; {regressions} of {total_revoked} recorded "
        f"revocations regress (must be 0)"
    )
    return 0 if regressions == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
