"""Render the illustrated pipeline walkthrough.

Zero model spend and zero network. It reads a placed volume out of ``work/``,
calls the pipeline's own functions on it, and writes three things: the rendered
plates and the JSON the stepper reads, both under ``--out``, and the markdown
page at ``--doc``. All three are tracked.

Its INPUT is gitignored and its OUTPUT is tracked, on purpose: a cold clone has
no ``work/`` tree and could not run this, so the rendered plates are committed
the way ``FIXTURE-SHA256SUMS`` is. Do not "fix" that by gitignoring the output.

    uv run python scripts/walkthrough/make_walkthrough_assets.py --work work
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import walkthrough_agreement as agreement
import walkthrough_candidates as candidates
import walkthrough_channels as channels
import walkthrough_gate_plates as gate_plates
import walkthrough_markdown as markdown
import walkthrough_outcome as outcome
import walkthrough_reading as reading
import walkthrough_rescue as rescue
import walkthrough_serving as serving
import walkthrough_sources as sources
import walkthrough_volume as book
from walkthrough_page import Emitter, Panel, write_panels

#: The checkout root — this file sits two directories below it.
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "viewer" / "walkthrough"
DEFAULT_DOC = ROOT / "docs" / "HOW-IT-WORKS.md"
EXEMPLARS = Path(__file__).resolve().with_name("walkthrough_exemplars.json")

#: Terms the page offers as a small glossary. Plain words, stated once.
GLOSSARY = (
    ("sheet", "one page of a bound atlas, and the unit everything here works on"),
    ("volume", "one bound atlas: a few hundred sheets printed at the same scale"),
    (
        "corner",
        "a street junction the pipeline believes it has found on both the scan and a modern map",
    ),
    ("placement", "the single slide, turn and stretch that puts a whole sheet on the ground"),
    ("flagged", "a sheet the pipeline refused to place, and says so about"),
    ("gate", "one of the seven checks a placement clears before it can be published"),
    ("strict accept", "a sheet that cleared all seven gates on its first fit"),
    (
        "rescue",
        "a second attempt at a sheet the gates refused, using what the rest of the volume knows",
    ),
    (
        "revoked",
        "a rescued placement held back as a proposal because all its corners sat on one street",
    ),
    ("corroboration", "agreement with sheets already placed, measured where they overlap"),
    (
        "verified accept",
        "the vote that can commit a revoked sheet on evidence the matcher never used",
    ),
)


def build(
    volume: sources.Volume,
    escalated: Callable[[], sources.Volume],
    ex: dict[str, Any],
    work: Path,
    out: Emitter,
    wanted: set[int],
) -> list[Panel]:
    """Every panel, in order. ``wanted`` renders a subset without paying for the rest."""
    funnel = sources.funnel(volume)
    makers: tuple[Callable[[], Panel], ...] = (
        lambda: reading.panel_problem(volume, ex, out),
        lambda: reading.panel_prep(volume, ex, out),
        lambda: reading.panel_reading(volume, ex, out),
        lambda: reading.panel_escalate(escalated(), ex, out),
        lambda: book.panel_volume(volume, ex, out),
        lambda: candidates.panel_axes(volume, ex, out),
        lambda: candidates.panel_candidates(volume, ex, out),
        lambda: candidates.panel_world(volume, ex, out),
        lambda: agreement.panel_fit(volume, ex, out),
        lambda: gate_plates.panel_gates(volume, ex, out),
        lambda: rescue.panel_rescue(volume, ex, out),
        lambda: rescue.panel_revoked(volume, ex, out),
        lambda: rescue.panel_corroborate(volume, ex, out),
        lambda: channels.panel_independent(volume, ex, out),
        lambda: channels.panel_verified(volume, ex, out),
        lambda: serving.panel_seam(volume, ex, out),
        lambda: serving.panel_warp(volume, ex, out),
        lambda: serving.panel_masks(volume, ex, out),
        lambda: serving.panel_mosaic(volume, ex, out),
        lambda: outcome.panel_funnel(work, ex, funnel, out),
        lambda: outcome.panel_score(work, ex, out),
    )
    return [make() for number, make in enumerate(makers, start=1) if not wanted or number in wanted]


def running_funnel(volume: sources.Volume) -> dict[str, Any]:
    """The counter the page carries across every panel."""
    funnel = sources.funnel(volume)
    return {
        "total": funnel.total,
        "stages": {
            stage: {"placed": placed, "provisional": provisional, "flagged": flagged}
            for stage, (placed, provisional, flagged) in funnel.by_stage.items()
        },
        "order": ["match", "rescue", "corroborate", "verified-accept"],
        "labels": {
            "match": "after the strict fit",
            "rescue": "after the slide",
            "corroborate": "after the neighbours",
            "verified-accept": "after the vote",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("work"), help="the volume work tree")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where the plates land")
    parser.add_argument("--panels", type=str, default="", help="render only these numbers")
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC, help="the markdown page")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    ex = sources.exemplars(EXEMPLARS)
    city_toml = ROOT / ex["city_toml"]
    volume = sources.load(args.work, city_toml, ex["volume"])
    # panel 4 alone comes from a second atlas, named on its own plate; loaded
    # only if that panel is being drawn, since an index build is not cheap
    escalated = cache(lambda: sources.load(args.work, city_toml, ex["escalated_volume"]))
    out = Emitter(args.out)
    wanted = {int(n) for n in args.panels.split(",") if n.strip()}
    panels = build(volume, escalated, ex, args.work, out, wanted)
    if wanted:
        print(f"rendered {len(panels)} of 21 panels into {args.out}")
        return 0
    meta = {
        "volume": volume.identifier,
        "title": ex["title"],
        "sheets": len(volume.results),
        "scan_credit": ex["scan_credit"],
        "centerline_credit": ex["centerline_credit"],
    }
    write_panels(
        args.out / "panels.json",
        meta=meta,
        panels=panels,
        funnel=running_funnel(volume),
        glossary=list(GLOSSARY),
    )
    markdown.write_markdown(
        args.doc, meta=meta, panels=panels, glossary=list(GLOSSARY), out=args.out
    )
    total = sum(len(p.states) for p in panels)
    size = sum(f.stat().st_size for f in args.out.glob("*.jpg"))
    print(f"{len(panels)} panels, {total} plates, {size / 1e6:.1f} MB into {args.out}")
    print(f"markdown page written to {args.doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
