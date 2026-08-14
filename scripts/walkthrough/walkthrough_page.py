"""The shape of what the generator emits: panels, their states, and panels.json.

The page is a stepper over these records. Every number a reader sees is written
here by the generator; nothing on the page computes one, so a threshold cannot
acquire a second, drifting copy in JavaScript.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

#: Rendered plates are JPEG at this quality: a scan is a photograph, and the
#: overlays are drawn heavy enough to survive it.
JPEG_QUALITY = 80


@dataclass
class State:
    """One rendered view of a panel. A panel with several is a before/after."""

    key: str
    label: str
    file: str = ""
    alt: str = ""


@dataclass
class Panel:
    """One step of the walkthrough."""

    number: int
    slug: str
    act: str
    title: str
    dek: str
    caption: str
    states: list[State] = field(default_factory=list)
    figures: list[tuple[str, str]] = field(default_factory=list)
    stage: str = ""
    note: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "slug": self.slug,
            "act": self.act,
            "title": self.title,
            "dek": self.dek,
            "caption": self.caption,
            "stage": self.stage,
            "note": self.note,
            "figures": [{"label": label, "value": value} for label, value in self.figures],
            "states": [
                {"key": s.key, "label": s.label, "file": s.file, "alt": s.alt} for s in self.states
            ],
        }


@dataclass
class Emitter:
    """Where plates land, and what they are called."""

    out: Path

    def save(self, panel: Panel, state: State, image: Image.Image) -> None:
        name = f"panel-{panel.number:02d}-{panel.slug}"
        if state.key != "main":
            name += f"-{state.key}"
        state.file = f"{name}.jpg"
        self.out.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(self.out / state.file, quality=JPEG_QUALITY, optimize=True)


def carried_tally(panels: list[Panel], funnel: dict[str, Any]) -> list[str]:
    """For each panel, the funnel stage its counter should show.

    Most panels are not themselves a funnel stage: the seam solve and the whole
    back half run on sheets the funnel already resolved, and one verification
    panel sits between two stages. Each shows the last stage that HAD resolved,
    so the counter carries instead of resetting to nothing-decided-yet — which
    on a back-half panel would be false. Panels before the first stage carry
    "", the only place the idle reading is true.
    """
    stages = funnel.get("stages", {})
    carried, seen = [], ""
    for panel in panels:
        if panel.stage in stages:
            seen = panel.stage
        carried.append(seen)
    return carried


def write_panels(
    path: Path,
    *,
    meta: dict[str, Any],
    panels: list[Panel],
    funnel: dict[str, Any],
    glossary: list[tuple[str, str]],
) -> None:
    """Write the one file the page reads."""
    tallies = carried_tally(panels, funnel)
    payload = {
        "meta": meta,
        "funnel": funnel,
        "glossary": [{"term": term, "gloss": gloss} for term, gloss in glossary],
        "panels": [
            {**panel.payload(), "tally": tally}
            for panel, tally in zip(panels, tallies, strict=True)
        ],
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
