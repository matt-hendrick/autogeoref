"""Panels 8 and 9, and the world-space figures the gate plates share.

The world side of a corner guess and the fit that settles it, plus the two
figure shapes panel 10 reuses: a page placed by a model, and the constellation
of corners a placement rests on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import walkthrough_draw as sketch
import walkthrough_geo as geo
import walkthrough_theme as theme
from walkthrough_page import Emitter, Panel, State
from walkthrough_reading import column_head, split
from walkthrough_sources import FidelityError

from autogeoref.affine import TO_3857
from autogeoref.centerlines import CenterlineIndex
from autogeoref.matching import candidate_gcps

if TYPE_CHECKING:
    from walkthrough_sources import Volume

Box = tuple[float, float, float, float]


def window_text(window: tuple[float, float] | None, form: str) -> str:
    if window is None:
        return "none"
    return f"{form.format(window[0])} to {form.format(window[1])}"


def panel_axes(volume: Volume, ex: dict[str, Any], out: Emitter) -> Panel:
    """6 - a name points along the street it names."""
    page = ex["pages"]["hero"]
    streets = volume.annotation(page)["streets"]
    plate = theme.plate(6, "A name points along its street", kicker="label axes")
    small = volume.small(page)
    sheet_box, col = split(plate, 0.44)
    view = sketch.place_sheet(plate, small, sheet_box, volume.scale(page))
    for street in streets:
        bbox = tuple(street["bbox"])
        sketch.clipped_axis(view, plate.draw, bbox, _direction(street))
        sketch.label_box(view, plate.draw, bbox)
    left, _, right, bottom = col
    y = column_head(
        plate,
        col,
        "label axes",
        "A street name is printed along the street it names. That is a strong "
        "hint about geometry, not just about words: the line the text sits on "
        "runs down the middle of the roadway. So each name is extended into a "
        "line across the whole page, in the direction the reader said the text "
        "was written.",
    )
    sketch.legend(
        plate.draw,
        (left, y + 12),
        [
            sketch.Key("box", theme.SHEET, "where the name was printed"),
            sketch.Key("line", theme.SHEET, "the line the name lies along"),
        ],
    )
    theme.paragraph(
        plate.draw,
        (left, bottom - 130),
        "Nothing here has looked at the drawn streets themselves. A line is "
        "produced from a box and a direction, and that is enough, because two of "
        "these lines crossing is a street corner.",
        right - left,
        size=theme.SMALL,
        fill=theme.INK_SOFT,
    )
    panel = Panel(
        number=6,
        slug="label-axes",
        act="II. From a reading to a fit",
        title="A name points along its street",
        dek="Each name becomes a line across the page, in the direction it was printed.",
        caption=(
            "Street names on an atlas sheet are set along the roadway they name, "
            "so the line the lettering sits on is a good stand-in for the street "
            "itself. Extending each name into a full-width line turns a handful of "
            "words into a handful of lines, and lines can be crossed. This is why "
            "the reading step asks for a direction as well as a name: a name "
            "printed across the sheet and the same name printed down it produce "
            "completely different lines."
        ),
        figures=[("Names on this sheet", str(len(streets)))],
        stage="match",
    )
    state = State("main", "Axes", alt="Each street name extended into a line across the sheet.")
    panel.states = [state]
    out.save(panel, state, plate.image)
    return panel


def _direction(street: dict[str, Any]) -> tuple[float, float]:
    if street["orientation"] == "horizontal":
        return (1.0, 0.0)
    if street["orientation"] == "vertical":
        return (0.0, 1.0)
    dx, dy = street.get("direction", [1, 0])
    return (float(dx), float(dy))


def panel_candidates(volume: Volume, ex: dict[str, Any], out: Emitter) -> Panel:
    """7 - crossings on the page, and why there are more of them than corners."""
    page = ex["pages"]["hero"]
    fit = volume.match(page)
    pixels = sorted({(round(c.pixel[0]), round(c.pixel[1])) for c in fit.candidates})
    plate = theme.plate(7, "Where two names cross", kicker="candidates")
    small = volume.small(page)
    sheet_box, col = split(plate, 0.44)
    view = sketch.place_sheet(plate, small, sheet_box, volume.scale(page))
    for street in volume.annotation(page)["streets"]:
        sketch.clipped_axis(view, plate.draw, tuple(street["bbox"]), _direction(street))
    for px, py in pixels:
        at = view.full(px, py)
        if view.holds(at):
            sketch.ring(plate.draw, at, fill=theme.SHEET, radius=13, solid=False)
    left, _, right, _bottom = col
    y = column_head(
        plate,
        col,
        "candidates",
        "Where two of those lines cross, the sheet is claiming a street corner. "
        "Each crossing is a guess to be checked, not a fact: the pipeline is "
        "about to go and ask a modern street map where that same pair of streets "
        "meets today.",
    )
    theme.card(plate.draw, (left, y + 8, right, y + 200))
    rows = [
        (f"{len(volume.annotation(page)['streets'])}", "names read off the sheet"),
        (f"{len(pixels)}", "places where two of their lines cross"),
        (f"{len(fit.candidates)}", "corner guesses to check"),
    ]
    ry = y + 30
    for value, label in rows:
        theme.text(plate.draw, (left + 26, ry - 8), value, size=theme.TITLE, fill=theme.SHEET)
        theme.text(plate.draw, (left + 120, ry + 4), label, size=theme.BODY)
        ry += 56
    theme.paragraph(
        plate.draw,
        (left, y + 222),
        _multiplicity(len(pixels), len(fit.candidates)),
        right - left,
        size=theme.BODY,
    )
    panel = Panel(
        number=7,
        slug="candidates",
        act="II. From a reading to a fit",
        title="Where two names cross",
        dek="Every crossing on the page becomes a guess to check, and can become more than one.",
        caption=(
            "Two crossed lines on the page are the sheet saying that these streets meet "
            "here`. That claim is looked up on a modern street map, and the answer "
            "is not always a single point. Streets jog around a park and pick up "
            "again; a name can belong to two separate stretches of road. Each "
            "possible modern location becomes its own guess, so the guesses can "
            f"outnumber the crossings: {_count_note(len(pixels), len(fit.candidates))}. "
            "Nothing is chosen yet. The next step is what decides which of them can "
            "be true at the same time."
        ),
        figures=[
            ("Crossings on the page", str(len(pixels))),
            ("Corner guesses", str(len(fit.candidates))),
        ],
        stage="match",
    )
    state = State("main", "Crossings", alt="Every crossing of two label lines, ringed.")
    panel.states = [state]
    out.save(panel, state, plate.image)
    return panel


def panel_world(volume: Volume, ex: dict[str, Any], out: Emitter) -> Panel:
    """8 - the modern map, and the rename table that reaches it."""
    page = ex["pages"]["hero"]
    fit = volume.match(page)
    bare = CenterlineIndex(
        volume.features,
        aliases={},
        bounds_4326=volume.bounds,
        name_property=volume.city.centerline_name_property,
        type_property=volume.city.centerline_type_property,
    )
    sheet = volume.sheets[page]
    without = candidate_gcps(sheet.annotation, bare, sheet.scale, {})
    printed = [s["name"] for s in sheet.annotation["streets"]]
    acting = _acting(printed, volume.index.aliases)
    _assert_acting_is_the_difference(volume, sheet, acting, len(fit.candidates))
    panel = Panel(
        number=8,
        slug="the-world-side",
        act="II. From a reading to a fit",
        title="The same corner on a modern map",
        dek="Historic names are looked up on today's street centrelines, after renames.",
        caption=(
            "Every corner guess is now looked up in a modern file of street "
            "centrelines: two street names go in, real-world coordinates come out. "
            "This is where a century gets in the way. Streets are renamed, and a "
            "name printed in 1905 may match nothing at all today, so the volume "
            "carries a small table of historic-to-modern names. It is a plain "
            "list, written and checked by hand for each volume, not a guess made "
            "at run time. The table belongs to the whole book, so most of it says "
            "nothing about any one sheet: of its "
            f"{len(volume.index.aliases)} entries only {len(acting)} reaches a "
            f"name printed on this one, {_named(acting)}. That entry is the whole "
            "of the difference "
            "below: the same sheet, the same reading, and fewer corners left to "
            "work with once it is switched off."
        ),
        figures=[
            ("Renames in this volume's table", str(len(volume.index.aliases))),
            ("Of those, used by this sheet", str(len(acting))),
            ("Corner guesses with the table", str(len(fit.candidates))),
            ("Corner guesses without it", str(len(without))),
        ],
        stage="match",
    )
    states: list[State] = []
    for key, label, cands, aliases in (
        ("on", "Renames on", fit.candidates, volume.index.aliases),
        ("off", "Renames off", without, {}),
    ):
        plate = theme.plate(8, "The same corner on a modern map", kicker=label.lower())
        left, top, right, bottom = plate.box
        map_box = (left, top, left + (right - left) * 0.56, bottom)
        world = [TO_3857.transform(*c.world4326) for c in fit.candidates]
        canvas = geo.canvas_for(map_box, geo.bounds_of(world, pad_m=200))
        view = canvas.view
        geo.draw_streets(canvas.draw, view, volume.features, fill=(206, 196, 176), width=2)
        names = {s["name"].upper() for s in sheet.annotation["streets"]}
        geo.draw_streets(
            canvas.draw,
            view,
            volume.features,
            fill=theme.WORLD,
            width=2,
            highlight=_keys(names, aliases),
        )
        for cand in cands:
            sketch.ring(canvas.draw, view.lnglat(*cand.world4326), fill=theme.WORLD, radius=10)
        geo.scale_bar(canvas.draw, view, (26, view.size[1] - 26))
        canvas.commit(plate)
        col = (map_box[2] + 40, top, right, bottom)
        y = column_head(
            plate,
            col,
            label,
            "The modern street map for the same few blocks. A corner guess is "
            "kept only if both of its names can be found on it.",
        )
        cl, _, cr, _ = col
        theme.card(plate.draw, (cl, y + 10, cr, y + 150))
        theme.text(
            plate.draw, (cl + 26, y + 30), str(len(cands)), size=theme.TITLE, fill=theme.WORLD
        )
        theme.paragraph(
            plate.draw,
            (cl + 140, y + 44),
            "corner guesses survive the lookup",
            cr - cl - 160,
            size=theme.BODY,
        )
        live = bool(aliases)
        theme.caps(
            plate.draw,
            (cl, y + 174),
            "the entries this sheet's names reach",
            size=theme.TINY,
            fill=theme.SHEET,
        )
        ty = y + 208
        for old, new in acting:
            theme.text(
                plate.draw, (cl, ty), old, size=theme.SMALL, fill=theme.INK if live else theme.DROP
            )
            theme.text(
                plate.draw,
                (cl + 250, ty),
                f"-> {new}" if live else "(not looked up)",
                size=theme.SMALL,
                fill=theme.KEEP if live else theme.DROP,
            )
            ty += 30
        rest = len(volume.index.aliases) - len(acting)
        theme.paragraph(
            plate.draw,
            (cl, ty + 16),
            f"The volume's table carries {rest} more, for streets this sheet does "
            f"not name. They are looked up for the sheets that do.",
            cr - cl,
            size=theme.TINY,
            fill=theme.INK_SOFT,
        )
        plate.credit(ex["centerline_credit"])
        state = State(key, label, alt=f"Modern centrelines with the volume's renames {key}.")
        states.append(state)
        out.save(panel, state, plate.image)
    panel.states = states
    return panel


def _count_note(crossings: int, guesses: int) -> str:
    """Whether they did outnumber them on this sheet. Never asserts the wrong one."""
    if guesses > crossings:
        return f"here {crossings} crossings gave {guesses}"
    return f"here each of the {crossings} found exactly one"


def _multiplicity(crossings: int, guesses: int) -> str:
    """Panel 7's explanation of the crossing-to-guess count, either way it falls."""
    lead = (
        "There are more guesses than crossings because a pair of street names can"
        if guesses > crossings
        else "A crossing can give more than one guess, because a pair of street names can"
    )
    tail = (
        ""
        if guesses > crossings
        else " On this sheet every crossing found exactly one such place, so the two counts match."
    )
    return (
        f"{lead} meet more than once on the modern map - a street that jogs and "
        f"comes back, or two stretches of road that carry the same name. Every one "
        f"of those possibilities is kept and offered to the fit, which is allowed "
        f"to believe at most one per crossing.{tail}"
    )


def _keys(names: set[str], aliases: dict[str, str]) -> set[str]:
    """The modern spellings a set of printed names reaches."""
    from autogeoref.names import normalize

    return {normalize(name, aliases) for name in names}


def _assert_acting_is_the_difference(
    volume: Volume, sheet: Any, acting: list[tuple[str, str]], full: int
) -> None:
    """The plate says the listed entries are the whole of the difference. Check it.

    Only the entries a sheet's own names reach are shown, so those entries alone
    must reproduce the count the whole table produces.
    """
    index = CenterlineIndex(
        volume.features,
        aliases=dict(acting),
        bounds_4326=volume.bounds,
        name_property=volume.city.centerline_name_property,
        type_property=volume.city.centerline_type_property,
    )
    alone = len(candidate_gcps(sheet.annotation, index, sheet.scale, dict(acting)))
    if alone != full:
        raise FidelityError(
            f"the {len(acting)} entries this sheet reaches give {alone} corner "
            f"guesses, the whole table gives {full}"
        )


def _named(acting: list[tuple[str, str]]) -> str:
    """The acting renames spelled out, so the step carries a checkable example."""
    return ", ".join(f"{old} to {new}" for old, new in acting) or "none of them"


def _acting(printed: list[str], aliases: dict[str, str]) -> list[tuple[str, str]]:
    """The rename entries a sheet's own names reach, in table order.

    The table belongs to the volume, so printing all of it beside one sheet
    says that every entry acted on this lookup, which is rarely true.
    """
    from autogeoref.names import normalize

    reached = {normalize(name, {}) for name in printed}
    return [(old, new) for old, new in sorted(aliases.items()) if old in reached]
