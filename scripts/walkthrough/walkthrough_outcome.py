"""Panels 20 and 21: where the sheets ended up, and how they were marked."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import walkthrough_charts as charts
import walkthrough_theme as theme
from walkthrough_page import Emitter, Panel, State

if TYPE_CHECKING:
    from walkthrough_sources import Funnel

Box = tuple[float, float, float, float]

#: A volume with fewer results than this is a novelty item, not a book.
CORPUS_FLOOR = 25


@dataclass(frozen=True)
class Corpus:
    """How the flagged share varies across every volume placed on this machine."""

    volumes: int
    shares: list[float]

    @property
    def range_text(self) -> str:
        return f"{min(self.shares):.0%} to {max(self.shares):.0%}"

    @property
    def median_text(self) -> str:
        return f"{sorted(self.shares)[len(self.shares) // 2]:.0%} flagged"

    @property
    def sentence(self) -> str:
        return (
            f"across the {self.volumes} volumes placed here the flagged share runs "
            f"{self.range_text}, middle volume {self.median_text}"
        )


def survey(work: Path) -> Corpus:
    """The flagged share of every placed volume on disk, measured not guessed."""
    shares: list[float] = []
    for results in sorted(work.glob("*/results")):
        statuses = []
        for path in results.glob("p*.json"):
            try:
                statuses.append(str(json.loads(path.read_text()).get("status", "")))
            except (OSError, ValueError):
                continue
        if len(statuses) < CORPUS_FLOOR:
            continue
        shares.append(sum(1 for s in statuses if not s.startswith("OK")) / len(statuses))
    return Corpus(volumes=len(shares), shares=shares)


STATUS_NOTES = (
    ("OK", "the strict fit was clean"),
    ("OK (rescued)", "placed by a slide at the book's pinned scale and turn"),
    ("OK (rescued, neighbor-corroborated)", "put back by corners shared with placed neighbours"),
    ("OK (verified: junction+addresses)", "put back by two independent checks"),
    ("REJECTED (rescue revoked: anchors share one street)", "a proposal nothing could confirm"),
    ("REJECTED (no valid RANSAC model)", "no placement the checks would accept"),
)


def panel_funnel(work: Path, _ex: dict[str, Any], funnel: Funnel, out: Emitter) -> Panel:
    """20 - where every sheet in the volume ended up."""
    corpus = survey(work)
    plate = theme.plate(20, "Where the sheets ended up", kicker="the funnel")
    left, top, right, bottom = plate.box
    y = theme.paragraph(
        plate.draw,
        (left, top),
        "Every sheet in the volume, after each stage that can accept one. Bars run "
        "left to right: placed, held as a proposal, and flagged. A flagged sheet "
        "is not a failure of the run - it is the run declining to publish a "
        "placement it cannot support.",
        right - left,
        size=theme.BODY,
    )
    steps = [
        charts.Step(label, *funnel.by_stage[key])
        for key, label in (
            ("match", "after the strict fit"),
            ("rescue", "after the slide"),
            ("corroborate", "after the neighbours"),
            ("verified-accept", "after the vote"),
        )
    ]
    charts.funnel_chart(plate.draw, (left, y + 24, right, y + 300), steps, funnel.total)
    ry = y + 336
    theme.caps(plate.draw, (left, ry), "what a sheet's record actually says", size=theme.TINY)
    ry += 32
    for status, note in STATUS_NOTES:
        colour = theme.KEEP if status.startswith("OK") else theme.DROP
        theme.text(plate.draw, (left, ry), status, size=theme.SMALL, fill=colour, bold=True)
        theme.text(plate.draw, (left + 640, ry), note, size=theme.SMALL, fill=theme.INK_SOFT)
        ry += 34
    placed, _prov, flagged = funnel.by_stage["verified-accept"]
    theme.text(
        plate.draw,
        (right, bottom - 76),
        f"{placed} placed, {flagged} flagged, of {funnel.total}",
        size=theme.LEAD,
        fill=theme.INK,
        anchor="ra",
        bold=True,
    )
    theme.text(
        plate.draw,
        (right, bottom - 30),
        corpus.sentence,
        size=theme.SMALL,
        fill=theme.INK_SOFT,
        anchor="ra",
    )
    panel = Panel(
        number=20,
        slug="the-funnel",
        act="VI. Grading, afterwards",
        title="Where the sheets ended up",
        dek="The whole volume, stage by stage, including the sheets nothing could place.",
        caption=(
            "The pipeline does not place every sheet. It places every sheet the "
            "evidence supports and flags the rest. How large "
            "that rest is varies enormously from one volume to the next, and the "
            "spread is itself the honest picture: a book of ordinary street grids "
            "gives up almost everything, while a book of rail yards or a world's "
            "fair site can give up nothing at all. Refusing is the design working. "
            "A wrong sheet published quietly is worse than a missing one, because a "
            "reader has no way to tell it is wrong. Each status below is the "
            "literal text written into that sheet's record, so an operator reading "
            "the files sees the same vocabulary as a reader of this page."
        ),
        figures=[
            ("Sheets", str(funnel.total)),
            ("Placed", str(placed)),
            ("Flagged", str(flagged)),
            ("Flagged share", f"{flagged / funnel.total:.0%}"),
            ("Volumes measured", str(corpus.volumes)),
            ("Flagged share across them", corpus.range_text),
            ("Middle volume", corpus.median_text),
            # the literal text written into a sheet's record, so a reader of
            # this page and an operator reading the files share one vocabulary
            *((note.capitalize(), status) for status, note in STATUS_NOTES),
        ],
        stage="report",
    )
    state = State("main", "The funnel", alt="Placed, proposed and flagged counts after each stage.")
    panel.states = [state]
    out.save(panel, state, plate.image)
    return panel


def panel_score(work: Path, ex: dict[str, Any], out: Emitter) -> Panel:
    """21 - graded afterwards against work no run stage was allowed to read."""
    identifier = ex["scored_volume"]
    root = work / identifier
    scores = json.loads((root / "results-scores.json").read_text())
    values = [
        float(entry["rmse_vs_human_m"])
        for entry in scores["pages"].values()
        if entry.get("rmse_vs_human_m") is not None
    ]
    gate = float(scores.get("gate_m", 15.0))
    ordered = sorted(values)
    median = ordered[len(ordered) // 2]
    within = sum(1 for v in values if v <= gate)
    plate = theme.plate(
        21, "Marked afterwards, never during", kicker=f"scored: {ex['scored_title']}"
    )
    left, top, right, bottom = plate.box
    y = theme.paragraph(
        plate.draw,
        (left, top),
        "Volunteers have hand-placed some of these atlases over the years. Where "
        "that work exists it is used as a ruler and nothing else: a separate "
        "command compares finished placements against it and writes the answer to "
        "a file beside the results. No stage of a run reads it, and it cannot "
        "move a sheet.",
        right - left,
        size=theme.BODY,
    )
    theme.text(
        plate.draw,
        (left, y + 8),
        f"The volume this walkthrough follows, {ex['title']}, has no hand "
        f"placements at all. These figures are {ex['scored_title']}, which does.",
        size=theme.SMALL,
        fill=theme.SHEET,
    )
    y += 40
    charts.histogram(
        plate.draw,
        (left, y + 40, right, y + 400),
        values,
        unit="metres from where a person placed the same sheet",
        mark=gate,
        mark_label=f"{gate:g} m",
    )
    theme.card(plate.draw, (left, y + 430, right, bottom))
    for i, (value, label) in enumerate(
        (
            (f"{median:.2g} m", "typical distance from the hand placement"),
            (f"{within} of {len(values)}", f"scored sheets within {gate:g} m"),
        )
    ):
        theme.text(
            plate.draw, (left + 30, y + 456 + i * 84), value, size=theme.TITLE, fill=theme.WORLD
        )
        theme.text(plate.draw, (left + 400, y + 472 + i * 84), label, size=theme.BODY)
    theme.paragraph(
        plate.draw,
        (left + 30, bottom - 66),
        "Nothing a person placed decided where any of these sheets landed. That is "
        "what makes the comparison worth anything.",
        right - left - 60,
        size=theme.BODY,
        fill=theme.SHEET,
    )
    panel = Panel(
        number=21,
        slug="the-score",
        act="VI. Grading, afterwards",
        title="Marked afterwards, never during",
        dek="Hand placements are the ruler, and no run stage is allowed to read one.",
        caption=(
            "There is an obvious temptation with a system like this: if someone "
            "has already placed a sheet by hand, use it. Doing so would destroy "
            "the only honest measurement available, because a pipeline that has "
            "seen the answer cannot be tested against it. So hand placements are "
            "kept strictly outside every run. They are used by one separate "
            "command, afterwards, which writes its comparison to a file beside the "
            "results and never onto a sheet's record. The volume this walkthrough "
            "follows has no hand placements at all; the figures here come from one "
            "that does. The yardstick exists only where a volunteer happened to "
            "place a volume by hand, which is a small part of the corpus and "
            "never the part a run reads."
        ),
        figures=[
            ("Volume graded", ex["scored_title"]),
            ("Sheets scored", str(len(values))),
            ("Typical distance", f"{median:.3g} m"),
            ("Within the gate", f"{within} of {len(values)}"),
        ],
        stage="score",
    )
    state = State("main", "The score", alt="How far placements fall from hand-placed ones.")
    panel.states = [state]
    out.save(panel, state, plate.image)
    return panel
