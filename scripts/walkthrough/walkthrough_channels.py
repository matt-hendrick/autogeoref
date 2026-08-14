"""Panels 14 and 15: evidence the matcher never looked at, and the vote."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import walkthrough_charts as charts
import walkthrough_draw as sketch
import walkthrough_geo as geo
import walkthrough_theme as theme
from PIL import Image, ImageDraw
from walkthrough_page import Emitter, Panel, State

from autogeoref.affine import TO_3857, fit_affine_checked, gcps_from_geojson
from autogeoref.verified_accept import CHANNELS, MIN_CHANNELS

if TYPE_CHECKING:
    from typing import Any

    from walkthrough_sources import Volume

Box = tuple[float, float, float, float]


def panel_independent(volume: Volume, ex: dict[str, Any], out: Emitter) -> Panel:
    """14 - two kinds of evidence the matcher never looked at."""
    page = ex["pages"]["verified"]
    record = volume.results[page]
    snap = record.get("junction_snap") or {}
    detail = ((record.get("verified_accept") or {}).get("addresses")) or {}
    numerals = detail.get("numerals") or []
    plate = theme.plate(14, "Evidence the matcher never saw", kicker="independent channels")
    left, top, right, bottom = plate.box
    half = left + (right - left) * 0.48
    _junction_figure(volume, plate, (left, top + 46, half - 30, bottom - 30), page, record)
    theme.caps(plate.draw, (left, top), "drawn crossings", size=theme.TINY, fill=theme.SHEET)
    theme.text(
        plate.draw,
        (left, bottom - 22),
        f"{snap.get('n_junctions', 0)} crossings found in the ink; "
        f"best agreement {snap.get('best_offset_m', 0):g} m from the placement",
        size=theme.SMALL,
        fill=theme.INK_SOFT,
    )
    theme.caps(plate.draw, (half + 30, top), "house numbers", size=theme.TINY, fill=theme.SHEET)
    side = 430.0
    _numeral_figure(plate, (half + 30, top + 46, half + 30 + side, top + 46 + side), detail)
    cl = half + 30
    y = theme.paragraph(
        plate.draw,
        (cl, top + 66 + side),
        "Every house number two different readers agreed on is put on the ground "
        "by this sheet's own placement, then compared with the numbers a modern "
        "address grid expects on the block it landed on.",
        right - cl,
        size=theme.BODY,
    )
    theme.card(plate.draw, (cl, y + 16, right, y + 132))
    theme.text(
        plate.draw,
        (cl + 24, y + 34),
        f"{detail.get('in_block', 0)} of {detail.get('votable', 0)}",
        size=theme.TITLE,
        fill=theme.KEEP,
    )
    theme.paragraph(
        plate.draw,
        (cl + 250, y + 48),
        "checkable numbers land on the block they belong to",
        right - cl - 270,
        size=theme.BODY,
    )
    theme.text(
        plate.draw,
        (cl, bottom - 18),
        f"{len(numerals)} dots; several sit on top of each other",
        size=theme.TINY,
        fill=theme.INK_SOFT,
    )
    panel = Panel(
        number=14,
        slug="independent-evidence",
        act="III. Second chances",
        title="Evidence the matcher never saw",
        dek="Drawn street crossings and printed house numbers, checked against the placement.",
        caption=(
            "The matcher only ever looked at street names. Two other things on the "
            "same page can be checked against the placement it proposed, and "
            "neither had any part in producing it. The first is the ink itself: "
            "where the drawn streets cross, compared with where a modern map says "
            "those crossings are. The second is the house numbers printed along "
            "every block, compared with the numbers a modern address grid expects "
            "there. Both are simply asked whether the proposed placement makes "
            "sense; neither can move it."
        ),
        figures=[
            ("Sheet", f"page {page}"),
            ("Drawn crossings found", str(snap.get("n_junctions", 0))),
            ("Best agreement", f"{snap.get('best_offset_m', 0):g} m"),
            ("House numbers two readers agreed on", str(detail.get("consensus_numerals", 0))),
            ("Of those, on the right block", str(detail.get("in_block", 0))),
        ],
        stage="junction-verify",
    )
    state = State(
        "main",
        "Two channels",
        alt="Drawn crossings and house numbers checked against the placement.",
    )
    panel.states = [state]
    out.save(panel, state, plate.image)
    return panel


def _junction_figure(
    volume: Volume, plate: theme.Plate, box: Box, page: str, record: dict[str, Any]
) -> None:
    """Modern centrelines drawn back onto the scan through the sheet's own placement."""
    small = volume.small(page)
    view = sketch.place_sheet(plate, small, box, volume.scale(page))
    coef = fit_affine_checked(gcps_from_geojson(record.get("gcps_geojson") or {}))
    if coef is None:
        return
    matrix = np.array(
        [[coef[0][1], coef[0][2], coef[0][0]], [coef[1][1], coef[1][2], coef[1][0]], [0, 0, 1]]
    )
    inverse = np.linalg.inv(matrix)
    overlay = Image.new(
        "RGBA",
        (int(view.rect[2] - view.rect[0]) + 1, int(view.rect[3] - view.rect[1]) + 1),
        (0, 0, 0, 0),
    )
    pen = ImageDraw.Draw(overlay)
    for feature in volume.features:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        for ring in geo.rings(geometry):
            pts = []
            for lng, lat, *_ in ring:
                x, y = TO_3857.transform(lng, lat)
                px = inverse[0][0] * x + inverse[0][1] * y + inverse[0][2]
                py = inverse[1][0] * x + inverse[1][1] * y + inverse[1][2]
                at = view.full(px, py)
                pts.append((at[0] - view.rect[0], at[1] - view.rect[1]))
            if len(pts) > 1:
                pen.line([c for p in pts for c in p], fill=(*theme.WORLD, 200), width=3)
    frame = (int(view.rect[0]), int(view.rect[1]), int(view.rect[2]), int(view.rect[3]))
    base = plate.image.crop(frame).convert("RGBA")
    base.alpha_composite(overlay.crop((0, 0, base.width, base.height)))
    plate.image.paste(base.convert("RGB"), (int(view.rect[0]), int(view.rect[1])))
    plate.draw.rectangle(view.rect, outline=theme.PLATE_EDGE, width=1)


def _numeral_figure(plate: theme.Plate, box: Box, detail: dict[str, Any]) -> None:
    """Every agreed house number, by distance from its block and from its number."""
    left, top, right, bottom = box
    theme.card(plate.draw, box, fill=(250, 246, 236))
    numerals = detail.get("numerals") or []
    perp_tol = float(detail.get("perp_tol_m") or 50.0)
    addr_tol = float(detail.get("addr_tol") or 75.0)
    if not numerals:
        return
    # scaled to the readings, not to the limits: on a well-placed sheet every
    # number lands so far inside them that a limit-scaled plot is one dot in a
    # corner, and how tightly the readings cluster is the thing worth seeing
    max_perp = max(n["perp_m"] for n in numerals) * 1.4 or 1.0
    max_addr = max(max(n["addr_diff"] for n in numerals) * 1.4, 1.0)
    plot = (left + 100, top + 52, right - 40, bottom - 116)
    theme.text(
        plate.draw,
        (left + 16, top + 16),
        "how far off its own street the number lands",
        size=theme.TINY,
        fill=theme.INK_SOFT,
    )
    for numeral in numerals:
        x = plot[0] + (plot[2] - plot[0]) * min(numeral["addr_diff"] / max_addr, 1.0)
        y = plot[3] - (plot[3] - plot[1]) * min(numeral["perp_m"] / max_perp, 1.0)
        inside = numeral["perp_m"] <= perp_tol and numeral["addr_diff"] <= addr_tol
        sketch.ring(
            plate.draw, (x, y), fill=theme.KEEP if inside else theme.DROP, radius=9, solid=inside
        )
    plate.draw.line((plot[0], plot[3], plot[2], plot[3]), fill=theme.RULE, width=1)
    plate.draw.line((plot[0], plot[1], plot[0], plot[3]), fill=theme.RULE, width=1)
    theme.text(plate.draw, (plot[0], plot[3] + 10), "0", size=theme.TINY, fill=theme.INK_SOFT)
    theme.text(
        plate.draw,
        (plot[2], plot[3] + 10),
        f"{max_addr:.0f} house numbers out",
        size=theme.TINY,
        fill=theme.INK_SOFT,
        anchor="ra",
    )
    theme.text(
        plate.draw,
        (plot[0] - 12, plot[1]),
        f"{max_perp:.0f} m",
        size=theme.TINY,
        fill=theme.INK_SOFT,
        anchor="ra",
    )
    # the limits are stated, and drawn only where they actually fall on the plot
    for value, limit, vertical in ((max_perp, perp_tol, False), (max_addr, addr_tol, True)):
        if limit > value:
            continue
        at = limit / value
        if vertical:
            x = plot[0] + (plot[2] - plot[0]) * at
            plate.draw.line((x, plot[1], x, plot[3]), fill=theme.KEEP, width=2)
        else:
            y = plot[3] - (plot[3] - plot[1]) * at
            plate.draw.line((plot[0], y, plot[2], y), fill=theme.KEEP, width=2)
    beyond = perp_tol > max_perp and addr_tol > max_addr
    theme.paragraph(
        plate.draw,
        (left + 16, bottom - 76),
        f"A number counts when it lands within {perp_tol:g} m of its street and "
        f"{addr_tol:g} house numbers of where the grid puts it"
        + (" - both further out than this plot reaches." if beyond else "."),
        right - left - 32,
        size=theme.TINY,
        fill=theme.KEEP,
    )


def panel_verified(volume: Volume, ex: dict[str, Any], out: Emitter) -> Panel:
    """15 - two independent voices, and no objection."""
    page = ex["pages"]["verified"]
    record = volume.results[page]
    block = record.get("verified_accept") or {}
    votes = block.get("votes") or {}
    detail = block.get("addresses") or {}
    snap = record.get("junction_snap") or {}
    plate = theme.plate(15, "Two voices, and nothing against", kicker="verified accept")
    left, top, right, bottom = plate.box
    y = theme.paragraph(
        plate.draw,
        (left, top),
        "A sheet held aside is offered to three checks that work in different "
        "ways and share no evidence. Each may say yes once, or say nothing. One "
        "of them - the house numbers - may also object outright. Two yeses and no "
        "objection puts the sheet back.",
        right - left,
        size=theme.BODY,
    )
    rows = [
        (
            "the neighbours",
            _vote_note(votes.get("corroboration"), "shared corners with sheets already placed"),
            _vote_word(votes.get("corroboration")),
        ),
        (
            "drawn crossings",
            _vote_note(
                votes.get("junction"),
                f"{snap.get('n_junctions', 0)} crossings in the ink, best agreement "
                f"{snap.get('best_offset_m', 0):g} m",
            ),
            _vote_word(votes.get("junction")),
        ),
        (
            "house numbers",
            _vote_note(
                votes.get("addresses"),
                f"{detail.get('in_block', 0)} of {detail.get('consensus_numerals', 0)} agreed "
                f"numbers land on the block they belong to",
            ),
            _vote_word(votes.get("addresses")),
        ),
    ]
    charts.vote_table(
        plate.draw,
        (left, y + 40, right, bottom),
        rows,
        verdict=str(record.get("status")),
    )
    panel = Panel(
        number=15,
        slug="verified-accept",
        act="III. Second chances",
        title="Two voices, and nothing against",
        dek="Three independent checks; two agreeing and none objecting puts a sheet back.",
        caption=(
            "The last of the second chances is a vote, and its value is entirely "
            "in the independence of the voters. The neighbours check knows about "
            "corners shared with other sheets. The crossings check knows about ink "
            "on this page. The house-number check knows about printed numbers and "
            "a modern address grid. None of them uses the street names the "
            "placement was built from, so agreement between two of them is not the "
            "same reasoning counted twice. One voice is never enough, and the "
            "house-number check is the only one allowed to object. This vote is "
            "called verified accept, and if the house numbers object, "
            "the sheet stays flagged however many others agreed."
        ),
        figures=[
            ("Sheet", f"page {page}"),
            ("Checks available", str(len(CHANNELS))),
            ("Agreement needed", str(MIN_CHANNELS)),
            ("Result", str(record.get("status"))),
        ],
        stage="verified-accept",
    )
    state = State("main", "The vote", alt="The three checks and how each voted.")
    panel.states = [state]
    out.save(panel, state, plate.image)
    return panel


def _vote_word(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "silent"


def _vote_note(value: Any, note: str) -> str:
    return note if value is not None else "had nothing to say about this sheet"
