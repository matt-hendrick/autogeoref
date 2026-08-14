"""Act II plates: from a reading to a fit, and the gates that judge it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import walkthrough_charts as charts
import walkthrough_theme as theme
from PIL import Image
from walkthrough_candidates import window_text
from walkthrough_page import Emitter, Panel, State

from autogeoref.affine import model_rotation_deg, model_scales
from autogeoref.matching import (
    FitGates,
    candidate_gcps,
    fold_quadrant_deg,
    ransac_affine,
)

if TYPE_CHECKING:
    from walkthrough_sources import Volume

Box = tuple[float, float, float, float]


def _pass_one(volume: Volume) -> tuple[list[float], list[float]]:
    """Every sheet's own unconstrained scale and rotation, as pass 1 measures them."""
    scales: list[float] = []
    rotations: list[float] = []
    for sheet in volume.sheets.values():
        cands = candidate_gcps(sheet.annotation, volume.index, sheet.scale, volume.index.aliases)
        model, _ = ransac_affine(cands, sheet.full_size, gates=FitGates(loo_spread=False))
        if model is not None:
            sx, sy = model_scales(model)
            scales.append((sx + sy) / 2)
            rotations.append(fold_quadrant_deg(model_rotation_deg(model)))
    return scales, rotations


def panel_volume(volume: Volume, _ex: dict[str, Any], out: Emitter) -> Panel:
    """5 - the book sets the window; the page is judged against it."""
    scales, rotations = _pass_one(volume)
    plate = theme.plate(5, "The book decides, not the page", kicker="volume constants")
    left, top, right, bottom = plate.box
    y = theme.paragraph(
        plate.draw,
        (left, top),
        "Every sheet in a bound atlas was printed at one scale and bound the same "
        "way up. So the two numbers that matter most - how many metres one scanned "
        "pixel covers, and how far the page is turned from north - belong to the "
        "whole book, not to any one page. They are measured across all its sheets, "
        "and each page is then judged against the result.",
        right - left,
        size=theme.BODY,
    )
    y += 16
    strip = _thumb_strip(volume, int(right - left), 116)
    plate.image.paste(strip, (int(left), int(y)))
    plate.draw.rectangle((left, y, right, y + strip.height), outline=theme.PLATE_EDGE, width=1)
    y += strip.height + 6
    theme.text(
        plate.draw,
        (left, y),
        f"{len(volume.sheets)} sheets measured together; "
        f"{len(rotations)} of them produced a usable first-pass fit",
        size=theme.SMALL,
        fill=theme.INK_SOFT,
    )
    smed = sorted(scales)[len(scales) // 2]
    rmed = sorted(rotations)[len(rotations) // 2]
    charts.strip_plot(
        plate.draw,
        (left, y + 40, right, bottom),
        [
            charts.Strip(
                "metres per pixel",
                scales,
                volume.constraints.scale_range or (0, 0),
                round(smed, 5),
                "",
            ),
            charts.Strip(
                "degrees off north",
                rotations,
                volume.constraints.rot_range_deg or (0, 0),
                round(rmed, 3),
                "deg",
            ),
        ],
    )
    panel = Panel(
        number=5,
        slug="the-volume",
        act="II. From a reading to a fit",
        title="The book decides, not the page",
        dek="Scale and orientation are measured across the whole volume, then applied per sheet.",
        caption=(
            "This is the idea most often missed about the system. A single sheet "
            "cannot be checked on its own: any three corners can be joined by some "
            "stretch of the page, and a wrong stretch looks as convincing as a "
            "right one. A bound volume can be checked, because its sheets were "
            "printed at one scale and bound the same way up. So a first pass fits "
            "every sheet with no constraints at all and takes the middle value of "
            "what comes back. Every sheet is then refitted against a narrow window, "
            "and a fit landing outside it is refused however neat it looks by "
            "itself. A volume whose scale and turn are already known can declare "
            "them instead, and this one does, so the marks below read as a check: "
            "the measurement and the declaration land in the same place."
        ),
        figures=[
            ("Measured scale", f"{smed:.5g} m per pixel"),
            ("Measured rotation", f"{rmed:.4g} deg"),
            ("Scale window", window_text(volume.constraints.scale_range, "{:.5g}")),
            ("Rotation window", window_text(volume.constraints.rot_range_deg, "{:.4g}") + " deg"),
        ],
        stage="match",
    )
    state = State(
        "main", "The book", alt="Per-sheet measurements against the window the volume derived."
    )
    panel.states = [state]
    out.save(panel, state, plate.image)
    return panel


def _thumb_strip(volume: Volume, width: int, height: int) -> Image.Image:
    """A row of the volume's own sheets, filling ``width`` exactly."""
    pages = sorted(volume.sheets)
    step = max(len(pages) // 22, 1)
    chosen = pages[::step][:22]
    strip = Image.new("RGB", (width, height), theme.WASH)
    cell = width / len(chosen)
    for i, page in enumerate(chosen):
        small = volume.small(page)
        scaled = small.resize(
            (max(int(small.width * height / small.height), 1), height), Image.Resampling.LANCZOS
        )
        piece = scaled.crop((0, 0, max(int(cell) + 1, 1), height))
        strip.paste(piece, (int(i * cell), 0))
    return strip
