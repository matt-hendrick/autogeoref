"""Do the pipeline and the human pins describe a sheet of the same SIZE?

The pin grader fits an affine to each side and reports the displacement between
them. Where the two sides imply different sheet extents, that number folds a
scale disagreement into its metres and is not a displacement. This reports the
ratio per page, the off-band pages with their counts, and the tail figures with
and without them. Read-only: no model call, no network, and nothing is written
unless `--json` asks for it. `score_extent.py` beside it does the measuring.

Usage:
    uv run python scripts/audit_score_extent.py --corpus --bootstrap
    uv run python scripts/audit_score_extent.py --corpus --pages --json out.json
    uv run python scripts/audit_score_extent.py <volume> \\
        --work work --ground-truth fixtures/ground-truth
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from score_extent import (
    BAND,
    DATA_ROOT,
    SWEEP_BANDS,
    WORK,
    PageExtent,
    attribute_sides,
    city_specs,
    declared_volumes,
    scoreable_volumes,
    unclaimed_volumes,
    volume_extents,
)

from autogeoref.scoring import GT_COMMIT_RMSE_M

logger = logging.getLogger("audit_score_extent")


@dataclass
class Tail:
    """The published tail figures over one population."""

    n: int = 0
    median: float | None = None
    p90: float | None = None
    worst: float | None = None
    over_gate: int = 0
    errors: list[float] = field(default_factory=list)

    @classmethod
    def over(cls, pages: list[PageExtent]) -> Tail:
        errors = sorted(p.rmse_m for p in pages)
        tail = cls(n=len(errors), errors=errors)
        if errors:
            tail.median = float(statistics.median(errors))
            tail.p90 = float(percentile(errors, 90.0))
            tail.worst = errors[-1]
            tail.over_gate = sum(1 for e in errors if e > GT_COMMIT_RMSE_M)
        return tail

    def line(self) -> str:
        if not self.n:
            return "n=0"
        share = 100.0 * self.over_gate / self.n
        return (
            f"n={self.n:4d}  median={self.median:6.2f}  p90={self.p90:6.2f}  "
            f"max={self.worst:7.2f}  >{GT_COMMIT_RMSE_M:.0f}m={self.over_gate:3d} ({share:4.1f}%)"
        )


def percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile over an already-sorted list."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct / 100.0
    low = math.floor(pos)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low)


#: Bootstrap resamples behind a reported interval.
RESAMPLES = 10_000
#: Fixed so the interval is reproducible; the clock here is not monotonic.
SEED = 20260807


def volume_bootstrap(
    pages: list[PageExtent], band: tuple[float, float] | None
) -> tuple[float, float, float]:
    """Past-gate share and its 95% interval, resampling WHOLE VOLUMES.

    Sheets inside a volume share a fit, a scale and a reference extent, so a
    by-sheet interval is far too narrow. Returns `(share, low, high)` in per
    cent over the pages in ``band``, or over all of them when it is None.
    """
    population = [p for p in pages if band is None or p.in_band(band)]
    groups: dict[str, list[PageExtent]] = {}
    for page in population:
        groups.setdefault(page.volume, []).append(page)
    names = sorted(groups)
    rng = random.Random(SEED)
    shares: list[float] = []
    for _ in range(RESAMPLES):
        draw = [p for _ in names for p in groups[rng.choice(names)]]
        shares.append(100.0 * sum(1 for p in draw if p.rmse_m > GT_COMMIT_RMSE_M) / len(draw))
    shares.sort()
    point = 100.0 * sum(1 for p in population if p.rmse_m > GT_COMMIT_RMSE_M) / len(population)
    return point, shares[int(0.025 * RESAMPLES)], shares[int(0.975 * RESAMPLES)]


def report(
    pages: list[PageExtent],
    band: tuple[float, float],
    show_pages: bool,
    bootstrap: bool = False,
) -> None:
    """Print the per-volume table, the corpus tails, and the band sweep."""
    kept = [p for p in pages if p.in_band(band)]
    dropped = [p for p in pages if not p.in_band(band)]
    volumes = sorted({p.volume for p in pages})

    print(
        f"\nband {band[0]:.2f}-{band[1]:.2f}   scored pages {len(pages)}   off-band {len(dropped)}"
    )
    print(f"\n{'volume':<22} {'scored':>6} {'off':>4}  {'kept tail':<58}")
    for volume in volumes:
        mine = [p for p in pages if p.volume == volume]
        off = [p for p in mine if not p.in_band(band)]
        print(
            f"{volume:<22} {len(mine):>6} {len(off):>4}  "
            f"{Tail.over([p for p in mine if p.in_band(band)]).line()}"
        )
    print(f"\n{'ALL, no filter':<22} {'':>6} {'':>4}  {Tail.over(pages).line()}")
    print(f"{'ALL, in band':<22} {'':>6} {'':>4}  {Tail.over(kept).line()}")
    print(f"{'ALL, off band':<22} {'':>6} {'':>4}  {Tail.over(dropped).line()}")

    print("\nband sweep (kept pages / over gate / worst kept):")
    for low, high in SWEEP_BANDS:
        inside = [p for p in pages if p.in_band((low, high))]
        tail = Tail.over(inside)
        share = 100.0 * tail.over_gate / tail.n if tail.n else 0.0
        worst = f"{tail.worst:.1f}" if tail.worst is not None else "-"
        print(
            f"  {low:.2f}-{high:.2f}   kept {tail.n:4d} of {len(pages):4d}   "
            f"over gate {tail.over_gate:3d} ({share:4.1f}%)   worst {worst:>8} m"
        )

    if bootstrap:
        print(f"\npast-gate share, 95% interval over {RESAMPLES} whole-volume resamples:")
        # per city as well as over everything: the published headline is one
        # city's, and an interval quoted for it has to come from its own pages
        groups: list[tuple[str, list[PageExtent]]] = [("corpus", pages)]
        groups += [
            (city, [p for p in pages if p.city == city])
            for city in sorted({p.city for p in pages if p.city})
        ]
        for name, population in groups:
            # one volume resamples to itself, so its interval has zero width and
            # is not one; say so rather than print a spuriously tight number
            single = len({p.volume for p in population}) < 2
            for label, window in (("no filter", None), ("in band", band)):
                point, low, high = volume_bootstrap(population, window)
                interval = "one volume, no interval" if single else f"95% CI {low:.1f}-{high:.1f}"
                print(f"  {name:<14} {label:<10} {point:5.2f}%  ({interval})")

    if dropped:
        suspects = Counter(p.suspect for p in dropped)
        print(f"\noff-band pages, by ratio  (suspect side: {dict(suspects)}):")
        print(
            f"  {'volume':<22} {'page':>5} {'ratio':>7} {'rmse':>7} {'pins':>4} "
            f"{'pipeZ':>6} {'humanZ':>7} {'suspect':>10}  status"
        )
        for p in sorted(dropped, key=lambda p: p.ratio):
            print(
                f"  {p.volume:<22} {p.page:>5} {p.ratio:>7.3f} {p.rmse_m:>7.2f} "
                f"{p.pins:>4} {p.pipeline_z:>6.3f} {p.human_z:>7.3f} {p.suspect:>10}  {p.status}"
            )
    if show_pages:
        print("\nevery scored page:")
        print(
            f"  {'volume':<22} {'page':>5} {'ratio_w':>8} {'ratio_h':>8} {'rmse':>7} "
            f"{'pipe_w':>7} {'human_w':>7} {'pipe_h':>7} {'human_h':>7} {'pins':>4}"
        )
        for p in pages:
            print(
                f"  {p.volume:<22} {p.page:>5} {p.ratio_w:>8.3f} {p.ratio_h:>8.3f} "
                f"{p.rmse_m:>7.2f} {p.pipeline_w_m:>7.1f} {p.human_w_m:>7.1f} "
                f"{p.pipeline_h_m:>7.1f} {p.human_h_m:>7.1f} {p.pins:>4}"
            )


def collect(args: argparse.Namespace) -> list[PageExtent]:
    """Measure every requested volume, printing the volume set before it runs."""
    pages: list[PageExtent] = []
    unmeasurable: list[str] = []
    if args.corpus:
        specs = city_specs()
        for spec in specs:
            volumes = scoreable_volumes(spec)
            print(f"{spec.city}: work={spec.work}  config={spec.config.name}")
            print(f"  ground truth: {', '.join(str(d) for d in spec.ground_truth)}")
            print(
                f"  {len(volumes)} scoreable of {len(declared_volumes(spec))} declared: "
                f"{', '.join(volumes) or '(none pinned)'}"
            )
            for volume in volumes:
                measured = volume_extents(volume, spec.work, list(spec.ground_truth), unmeasurable)
                for page in measured:
                    page.city = spec.city
                pages.extend(measured)
        unclaimed = unclaimed_volumes(specs)
        if unclaimed:
            print(f"  SCOREABLE BUT DECLARED BY NO CONFIG: {', '.join(unclaimed)}")
    else:
        roots = [Path(d) for d in args.ground_truth] or [DATA_ROOT / "fixtures/ground-truth"]
        print(f"work={args.work}  ground truth: {', '.join(str(d) for d in roots)}")
        print(f"  {len(args.volumes)} requested: {', '.join(args.volumes)}")
        for volume in args.volumes:
            pages.extend(volume_extents(volume, Path(args.work), roots, unmeasurable))
    print(f"  scored by the grader but not measurable here: {len(unmeasurable)}", end="")
    print(f" ({', '.join(unmeasurable)})" if unmeasurable else "")
    return pages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("volumes", nargs="*", help="volume ids; omit with --corpus")
    parser.add_argument("--corpus", action="store_true", help="every scoreable volume, both cities")
    parser.add_argument("--work", default=str(WORK), help="work root for named volumes")
    parser.add_argument(
        "--ground-truth", action="append", default=[], help="pin corpus; repeatable, first wins"
    )
    parser.add_argument("--band", nargs=2, type=float, metavar=("LOW", "HIGH"), default=list(BAND))
    parser.add_argument("--pages", action="store_true", help="print every scored page")
    parser.add_argument(
        "--bootstrap", action="store_true", help="95%% interval over whole-volume resamples"
    )
    parser.add_argument("--json", help="write the per-page measurements here")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if not args.corpus and not args.volumes:
        parser.error("name at least one volume, or pass --corpus")

    pages = collect(args)
    if pages:
        attribute_sides(pages)
    if not pages:
        print("nothing scoreable was found")
        return 1
    band = (float(args.band[0]), float(args.band[1]))
    report(pages, band, args.pages, args.bootstrap)
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "band": list(band),
                    "gate_m": GT_COMMIT_RMSE_M,
                    "pages": [p.as_record() for p in pages],
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
