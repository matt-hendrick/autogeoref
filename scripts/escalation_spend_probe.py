"""How many model calls would a re-place actually spend in ``stage_escalate``?

``autogeoref run --dry-run`` cannot answer this: it plans the annotate stage
only, so its figure is not the run's cost. Two answers, because they differ:
- ELIGIBLE NOW — pages passing ``escalate_volume``'s own filters today (plain
  rejected, not rescue-revoked, a usable small, enough drawn junctions) whose
  ladder tiers hold no cached reading. This is the expected bill.
- WORST CASE — the same gate with the status filter dropped, because a
  re-place can demote an accepted page, which has no escalation cache.

The gate and the per-tier cache check reuse the stage's own helpers, so this
cannot drift from it without the import breaking. Failure markers count as
cache hits. The junction gate reads the IMAGE only, so a zero here survives an
alias change. Zero model calls, zero network; needs a populated ``work/``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from autogeoref.annotate.providers import canonical_model, model_cache_key, prior_variant_cache_key
from autogeoref.config.load import load_city_config
from autogeoref.escalate import MIN_JUNCTIONS_TO_ESCALATE
from autogeoref.junction_snap import JunctionSnapError, extract_junctions
from autogeoref.paths import VolumePaths, iter_results, small_sheet_entry
from autogeoref.volume import REJECTED_PREFIX, REVOKED_PREFIX, STATUS_OK

SPENDS = "WOULD SPEND"


def tier_state(paths: VolumePaths, page: str, model: str, variant: str | None) -> str:
    """What spares this tier a call for this page, or ``SPENDS``.

    Mirrors ``escalate_page``'s lookup order: this variant's reading, this
    variant's failure marker, then the prior variant's reading/marker.
    """
    key = model_cache_key(canonical_model(model), variant)
    if (paths.annotations / f"p{page}.escalated.{key}.json").exists():
        return "reading"
    if (paths.annotations / f"p{page}.escalated.{key}.failed.json").exists():
        return "marker"
    prior = prior_variant_cache_key(canonical_model(model), variant)
    if prior is not None:
        if (paths.annotations / f"p{page}.escalated.{prior}.json").exists():
            return "reading(prior-variant)"
        if (paths.annotations / f"p{page}.escalated.{prior}.failed.json").exists():
            return "marker(prior-variant)"
    return SPENDS


def gate_pages(paths: VolumePaths) -> tuple[list[tuple[str, str]], Counter[str]]:
    """Every page passing the junction gate, tagged with its current status bucket."""
    manifest = json.loads(paths.manifest.read_text())
    passed: list[tuple[str, str]] = []
    skipped: Counter[str] = Counter()
    for page, record, _rp in iter_results(paths):
        status = str(record.get("status", ""))
        entry = small_sheet_entry(paths, manifest, page, stage="escalate", skip_rotated=False)
        if entry is None:
            skipped["no usable small image"] += 1
            continue
        try:
            n_junc = extract_junctions(entry[1]).n_junctions
        except JunctionSnapError:
            skipped["junction extraction errored"] += 1
            continue
        if n_junc < MIN_JUNCTIONS_TO_ESCALATE:
            skipped[f"under the {MIN_JUNCTIONS_TO_ESCALATE}-junction gate"] += 1
            continue
        if status.startswith(REVOKED_PREFIX):
            bucket = "revoked"
        elif status.startswith(REJECTED_PREFIX):
            bucket = "rejected"
        elif status.startswith(STATUS_OK):
            bucket = "ok"
        else:
            bucket = "other"
        passed.append((page, bucket))
    return passed, skipped


def probe(volume: str, city_path: Path, work: Path) -> int:
    city = load_city_config(city_path)
    tiers = city.volume(volume).escalation_tiers()
    paths = VolumePaths(root=work / volume)
    if not paths.results.is_dir():
        print(f"{volume}: no results/ on disk — nothing to size")
        return 0

    passed, skipped = gate_pages(paths)
    print(f"\n=== {volume} ===")
    print(f"  ladder: {' -> '.join(f'{m}/{v}' for m, v in tiers) or '(none configured)'}")
    if not tiers:
        print("  no ladder: stage_escalate cannot spend here at all")
        return 0
    for reason, n in sorted(skipped.items()):
        print(f"  skipped, {reason}: {n}")
    print(f"  passing the gate: {len(passed)}  by status {dict(Counter(b for _p, b in passed))}")

    worst = 0
    for scope, wanted in (("ELIGIBLE NOW", {"rejected"}), ("WORST CASE (status-blind)", None)):
        pages = [p for p, b in passed if wanted is None or b in wanted]
        print(f"  -- {scope}: {len(pages)} page(s) --")
        total = 0
        for model, variant in tiers:
            states = Counter(tier_state(paths, p, model, variant) for p in pages)
            spend = states.get(SPENDS, 0)
            total += spend
            print(f"       {model}/{variant}: would spend {spend}   ({dict(states)})")
        print(
            f"       calls, all tiers: {total}   "
            "(a later tier only runs if an earlier one fails to flip the page)"
        )
        if wanted is None:
            worst = total
    return worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("volumes", nargs="+")
    ap.add_argument("--city", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("work"))
    args = ap.parse_args()
    worst = {v: probe(v, args.city, args.work) for v in args.volumes}
    print("\n=== worst-case call ceiling, per volume ===")
    for v, n in worst.items():
        print(f"  {v}: {n}")
    print(f"  total: {sum(worst.values())}")


if __name__ == "__main__":
    main()
