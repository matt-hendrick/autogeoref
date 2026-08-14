"""Freeze and report the zero-spend waterway incidence survey.

The survey deliberately reviews sheet images, not existing annotations: v2 did
not request water labels, so its omissions are not negative observations. The
sample is stratified to expose the industrial and downtown material where the
channel could plausibly rescue a flagged sheet, while retaining an independent
remainder stratum. Every row retains its result status and review evidence.
``freeze`` refuses to overwrite an existing sample; ``report`` fails on any
unreviewed row and on an eligible row without named non-lake water.

Zero model calls, zero network.

    uv run python scripts/water_incidence_survey.py freeze
    uv run python scripts/water_incidence_survey.py report
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("AUTOGEOREF_WORK", str(ROOT / "work")))
SAMPLE = Path(__file__).with_name("water_incidence_sample.json")
SEED = 20260718

STRATA: tuple[tuple[str, frozenset[str], int], ...] = (
    ("stockyards", frozenset({"sanborn01790_015"}), 10),
    ("river-cbd", frozenset({"sanborn01790_017", "sanborn01790_018"}), 10),
    ("remainder", frozenset(), 10),
)


def _volume_dirs(work: Path) -> list[Path]:
    return sorted(work.glob("sanborn01790_*")) + sorted(work.glob("goldens/sanborn01790_*"))


def flagged_pool(work: Path) -> list[dict[str, Any]]:
    """Return current flagged sheets with images, excluding unaddressable pages."""
    pool = []
    for volume in _volume_dirs(work):
        for result_path in sorted((volume / "results").glob("p*.json")):
            image = volume / "sheets" / f"{result_path.stem}_small.jpg"
            if not image.exists():
                continue
            status = str(json.loads(result_path.read_text())["status"])
            if status.startswith("OK"):
                continue
            pool.append(
                {
                    "volume": volume.name,
                    "golden": volume.parent.name == "goldens",
                    "page": result_path.stem,
                    "status": status,
                    "image": str(image.relative_to(work)),
                }
            )
    return pool


def draw_sample(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Draw the pre-registered stratified sample reproducibly."""
    selected: list[dict[str, Any]] = []
    all_focused = frozenset().union(*(volumes for _name, volumes, _n in STRATA[:-1]))
    for index, (name, volumes, count) in enumerate(STRATA):
        candidates = [
            sheet
            for sheet in pool
            if (sheet["volume"] in volumes if volumes else sheet["volume"] not in all_focused)
        ]
        if len(candidates) < count:
            raise ValueError(f"{name}: need {count} flagged sheets, found {len(candidates)}")
        rng = random.Random(SEED + index)
        selected.extend(
            {
                **sheet,
                "stratum": name,
                "waterway_label": None,
                "feature_class": None,
                "crossing_streets": [],
                "eligible": False,
                "review_note": "",
            }
            for sheet in rng.sample(candidates, count)
        )
    return selected


def _load_sample(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: sample must be a JSON list")
    return data


def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate completed reviews and aggregate water-crossing test cases."""
    missing = [row["volume"] + ":" + row["page"] for row in rows if not row["review_note"]]
    if missing:
        raise ValueError(f"unreviewed sample rows: {', '.join(missing)}")
    for row in rows:
        label = row["waterway_label"]
        feature_class = row["feature_class"]
        eligible = row["eligible"]
        if label is not None and (not isinstance(label, str) or not label.strip()):
            raise ValueError(
                f"{row['volume']} {row['page']}: waterway_label must be a nonempty string or null"
            )
        if feature_class not in {None, "river", "canal", "lake"}:
            raise ValueError(
                f"{row['volume']} {row['page']}: unsupported feature_class {feature_class!r}"
            )
        if eligible and (
            not label or feature_class in {None, "lake"} or not row["crossing_streets"]
        ):
            raise ValueError(
                f"{row['volume']} {row['page']}: eligible requires named non-lake water "
                "and a crossing street"
            )

    by_stratum = {}
    for stratum in sorted({str(row["stratum"]) for row in rows}):
        members = [row for row in rows if row["stratum"] == stratum]
        by_stratum[stratum] = {
            "reviewed": len(members),
            "named_waterways": sum(bool(row["waterway_label"]) for row in members),
            "eligible": sum(bool(row["eligible"]) for row in members),
        }
    labels = Counter(str(row["waterway_label"]) for row in rows if row["waterway_label"])
    eligible = [row for row in rows if row["eligible"]]
    return {
        "sample_size": len(rows),
        "eligible_flagged_sheets": len(eligible),
        "direct_test_ready": bool(eligible),
        "by_stratum": by_stratum,
        "label_counts": dict(sorted(labels.items())),
        "eligible_sheets": [f"{row['volume']}:{row['page']}" for row in eligible],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "report"))
    parser.add_argument("--sample", type=Path, default=SAMPLE)
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()

    if args.command == "freeze":
        if args.sample.exists():
            raise SystemExit(f"{args.sample} already exists; the sample is frozen")
        rows = draw_sample(flagged_pool(args.work))
        args.sample.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"froze {len(rows)} flagged sheets at {args.sample}")
        return

    print(json.dumps(report(_load_sample(args.sample)), indent=2))


if __name__ == "__main__":
    main()
