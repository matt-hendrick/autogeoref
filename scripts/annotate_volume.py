"""Batch v2 annotation for one volume: smalls -> work/<vid>/annotations/p<N>.json.

SPENDS BUDGET: one model call per uncached page; --dry-run prices it and spends nothing.

`autogeoref run` DOES this now, as its `annotate` stage — this script is the same
code (`autogeoref.annotate_volume`), reachable on its own when you want to
annotate a volume without running the pipeline over it, or to top up one page.
There is no second annotator: the caching, the resume, the failure markers and
the budget rules all live in the module, and the module's docstring states them.

    nice uv run python scripts/annotate_volume.py <volume> \
        --work work [--jobs 5] [--pages p1,p2] [--limit 10] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from autogeoref.annotate.failures import BudgetLimitError
from autogeoref.annotate.providers import DEFAULT_MODEL
from autogeoref.annotate_volume import ReadIdentity, annotate_volume, plan
from autogeoref.paths import VolumePaths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("volume")
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--variant", help="provider reasoning effort (Codex, OpenCode and the OpenAI API only)"
    )
    ap.add_argument(
        "--prompt",
        help="named annotation prompt; default is the frozen one. Part of the cache identity, "
        "so a second prompt re-reads rather than replaying the first prompt's answers",
    )
    ap.add_argument("--pages", help="comma-separated page ids (p12,p14S); default all")
    ap.add_argument("--limit", type=int, help="stop after N model calls (budget control)")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=1, help="pages annotated concurrently")
    ap.add_argument("--dry-run", action="store_true", help="print the call count, spend nothing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    paths = VolumePaths(root=args.work / args.volume)
    pages = (
        [p if p.startswith("p") else f"p{p}" for p in args.pages.split(",")] if args.pages else None
    )
    if args.dry_run:
        print(
            plan(
                paths,
                args.volume,
                identity=ReadIdentity(args.model, args.variant, args.prompt),
                pages=pages,
                limit=args.limit,
            ).summary()
        )
        return 0
    result = annotate_volume(
        paths,
        args.volume,
        identity=ReadIdentity(args.model, args.variant, args.prompt),
        pages=pages,
        limit=args.limit,
        attempts=args.attempts,
        jobs=args.jobs,
    )
    print(f"{args.volume}: annotated {result.annotated}/{result.plan.calls}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BudgetLimitError as exc:  # terminal: every further call is a doomed spend
        print(f"BUDGET LIMIT — batch stopped, cached pages kept: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
