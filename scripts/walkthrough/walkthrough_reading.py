"""Act I plates: a scan arrives, is prepared, is read, and is read again.

Each function renders its plate and returns the panel record the page steps
over. Numbers come from the volume on disk; prose stays conceptual.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import walkthrough_draw as sketch
import walkthrough_theme as theme
from PIL import Image
from walkthrough_page import Emitter, Panel, State
from walkthrough_sources import Escalation, FidelityError

if TYPE_CHECKING:
    from walkthrough_sources import Volume

Box = tuple[float, float, float, float]


def split(plate: theme.Plate, sheet_frac: float = 0.40) -> tuple[Box, Box]:
    """A plate divided into a sheet column and a reading column."""
    left, top, right, bottom = plate.box
    cut = left + (right - left) * sheet_frac
    return (left, top, cut - 34, bottom), (cut + 6, top, right, bottom)


def column_head(plate: theme.Plate, box: Box, kicker: str, body: str) -> float:
    """A kicker and a paragraph at the top of the reading column."""
    left, top, right, _ = box
    theme.caps(plate.draw, (left, top), kicker, size=theme.TINY, fill=theme.SHEET)
    return theme.paragraph(plate.draw, (left, top + 30), body, right - left, size=theme.BODY) + 12


def detail(
    plate: theme.Plate, small: Image.Image, crop: Box, at: Box, label: str, *, zoom_note: str = ""
) -> None:
    """A magnified crop of the scan, filling ``at`` exactly, boxed and captioned."""
    width, height = int(at[2] - at[0]), int(at[3] - at[1])
    cx, cy = (crop[0] + crop[2]) / 2, (crop[1] + crop[3]) / 2
    want = width / height
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    if cw / ch < want:
        cw = ch * want
    else:
        ch = cw / want
    # keep the window on the page: a crop that runs off the edge pads with
    # black, which reads as part of the scan
    cx = min(max(cx, cw / 2), small.width - cw / 2)
    cy = min(max(cy, ch / 2), small.height - ch / 2)
    piece = small.crop(
        (int(cx - cw / 2), int(cy - ch / 2), int(cx + cw / 2), int(cy + ch / 2))
    ).resize((width, height), Image.Resampling.LANCZOS)
    plate.image.paste(piece, (int(at[0]), int(at[1])))
    plate.draw.rectangle(
        (at[0], at[1], at[0] + width, at[1] + height), outline=theme.SHEET, width=2
    )
    theme.text(
        plate.draw, (at[0], at[1] + height + 10), label, size=theme.SMALL, fill=theme.INK, bold=True
    )
    if zoom_note:
        theme.text(
            plate.draw,
            (at[0], at[1] + height + 36),
            zoom_note,
            size=theme.TINY,
            fill=theme.INK_SOFT,
        )


def panel_problem(volume: Volume, ex: dict[str, Any], out: Emitter) -> Panel:
    """1 - a beautiful sheet that says where nothing is."""
    page = ex["pages"]["hero"]
    plate = theme.plate(
        1, "A map of somewhere, and nowhere", kicker=f"Sheet {page} of {ex['title']}"
    )
    small = volume.small(page)
    sheet_box, col = split(plate, 0.36)
    sketch.place_sheet(plate, small, sheet_box, volume.scale(page))
    y = column_head(
        plate,
        col,
        "the problem",
        "Every building on this sheet sits correctly beside its neighbours. The "
        "sheet itself sits nowhere. It carries no latitude and no longitude, so "
        "no computer can lay it over a map of the same blocks today. Supplying "
        "those numbers is called georeferencing, and it is usually hand work: a "
        "person clicks a street corner on the scan, clicks the same corner on a "
        "modern map, and repeats until the sheet holds still.",
    )
    left, _, right, bottom = col
    y = theme.paragraph(
        plate.draw,
        (left, y + 6),
        "The sheet knows its own scale and which way is north. It does not know where it is.",
        right - left,
        size=theme.BODY,
        fill=theme.SHEET,
    )
    width = (right - left - 44) / 3
    height = min(220.0, bottom - y - 66)
    for i, item in enumerate(ex["hero_details"]):
        x = left + i * (width + 22)
        detail(
            plate, small, tuple(item["box"]), (x, y + 34, x + width, y + 34 + height), item["label"]
        )
    plate.credit(ex["scan_credit"])
    panel = _panel_one(page, ex)
    alt = f"A scanned atlas sheet, page {page}, with its compass rose and scale bar called out."
    state = State("main", "The sheet", alt=alt)
    panel.states = [state]
    out.save(panel, state, plate.image)
    return panel


def _panel_one(page: str, ex: dict[str, Any]) -> Panel:
    return Panel(
        number=1,
        slug="the-problem",
        act="I. From a scan to a reading",
        title="A map of somewhere, and nowhere",
        dek="What georeferencing is, and why it is normally done by hand.",
        caption=(
            "This is one sheet of a fire-insurance atlas: a block-by-block survey "
            "of a city, drawn so an underwriter could see what every building was "
            "made of. It is exact about the things it cares about and silent about "
            "the one a computer needs. There is a scale bar and a north arrow, but "
            "no coordinates anywhere on the page, so nothing tells software which "
            "patch of ground the drawing covers. Someone has to say. Doing it by "
            "hand means clicking a street corner on the scan, clicking the same "
            "corner on a modern map, and repeating until the sheet stops sliding "
            "around. That is a few minutes per sheet, and there are thousands of sheets."
        ),
        figures=[("Sheet", f"page {page}"), ("Volume", ex["title"])],
    )


def panel_prep(volume: Volume, ex: dict[str, Any], out: Emitter) -> Panel:
    """2 - the two pixel frames, stated before anything depends on them."""
    page = ex["pages"]["hero"]
    entry = volume.manifest[f"p{page}"]
    scale = volume.scale(page)
    full_w, full_h = entry["full_size"]
    small_w, small_h = entry["small_size"]
    plate = theme.plate(2, "One scan, two sizes", kicker="prep")
    small = volume.small(page)
    sheet_box, col = split(plate, 0.30)
    sketch.place_sheet(plate, small, sheet_box, scale)
    left, _top, right, bottom = col
    y = column_head(
        plate,
        col,
        "prepare",
        f"The first step makes a small copy of the scan, {small_h} pixels tall. "
        f"The street names are read off the small copy, because that is the size a "
        "vision model can take in one look. Everything measured afterwards - "
        "corners, control points, the final stretch onto the map - is measured "
        "on the full-resolution scan.",
    )
    crop = tuple(ex["prep_detail"])
    full = Image.open(volume.paths.regions / f"{volume.identifier}_p{page}.jpg")
    box_w = (right - left - 40) / 2
    box_h = box_w * (crop[3] - crop[1]) / (crop[2] - crop[0])
    detail(
        plate,
        full,
        (crop[0] / scale, crop[1] / scale, crop[2] / scale, crop[3] / scale),
        (left, y + 20, left + box_w, y + 20 + box_h),
        "the full scan",
        zoom_note=f"{full_w} x {full_h} pixels",
    )
    detail(
        plate,
        small,
        crop,
        (left + box_w + 40, y + 20, left + box_w + 40 + box_w, y + 20 + box_h),
        "the small copy",
        zoom_note=f"{small_w} x {small_h} pixels",
    )
    theme.card(plate.draw, (left, bottom - 180, right, bottom))
    theme.text(
        plate.draw, (left + 22, bottom - 162), "One number joins them", size=theme.BODY, bold=True
    )
    theme.text(
        plate.draw, (left + 22, bottom - 126), f"{scale:.10g}", size=theme.TITLE, fill=theme.SHEET
    )
    theme.paragraph(
        plate.draw,
        (left + 22, bottom - 78),
        "small pixels per full pixel. It is written once, in the sheet manifest, "
        "and every later step multiplies by it. Reversed, it draws a picture that "
        "looks entirely reasonable and is wrong.",
        right - left - 44,
        size=theme.SMALL,
        fill=theme.INK_SOFT,
    )
    panel = Panel(
        number=2,
        slug="prep",
        act="I. From a scan to a reading",
        title="One scan, two sizes",
        dek="Names are read on the small copy; everything else is measured on the full one.",
        caption=(
            "Preparation makes a second, smaller copy of each scan and writes down "
            "the ratio between them. The two sizes have different jobs. The small "
            "copy is what a vision model reads, because a model sees a whole image "
            "at once and a 5,000-pixel-wide page would arrive as mush. The "
            "full-resolution scan is what every measurement afterwards uses, "
            "because a corner located to the nearest small pixel is only located "
            f"to within {1 / scale:.0f} full pixels on the ground. Keeping both, "
            f"and keeping the conversion in exactly one file, is what stops the two "
            "frames being mixed up later."
        ),
        figures=[
            ("Full scan", f"{full_w} x {full_h} px"),
            ("Small copy", f"{small_w} x {small_h} px"),
            ("Ratio", f"{scale:.6g}"),
        ],
        stage="prep",
    )
    alt = "The scan beside the same detail at full and reduced resolution."
    state = State("main", "Two frames", alt=alt)
    panel.states = [state]
    out.save(panel, state, plate.image)
    return panel


def panel_reading(volume: Volume, ex: dict[str, Any], out: Emitter) -> Panel:
    """3 - the labels a vision model returned, boxed on the scan."""
    page = ex["pages"]["hero"]
    streets = volume.annotation(page)["streets"]
    plate = theme.plate(3, "Reading the street names", kicker="annotate")
    small = volume.small(page)
    sheet_box, col = split(plate, 0.40)
    view = sketch.place_sheet(plate, small, sheet_box, volume.scale(page))
    for street in streets:
        sketch.label_box(view, plate.draw, tuple(street["bbox"]))
    left, _top, right, bottom = col
    y = column_head(
        plate,
        col,
        "annotate",
        "A vision model is shown the small copy and asked one question: which "
        "street names can you see, where is each one on the page, and which way "
        "does it run? It answers with a box and a direction for each name. That "
        "answer is the only thing the rest of the pipeline reads off the scan.",
    )
    y += 10
    for street in streets:
        sketch.ring(plate.draw, (left + 12, y + 12), fill=theme.SHEET, radius=8)
        theme.text(plate.draw, (left + 34, y), street["name"], size=theme.BODY)
        theme.text(
            plate.draw,
            (right, y),
            street["orientation"],
            size=theme.SMALL,
            fill=theme.INK_SOFT,
            anchor="ra",
        )
        y += 34
    theme.paragraph(
        plate.draw,
        (left, bottom - 96),
        "No buildings, no house numbers, no colours. Names and directions are "
        "everything the placement is built from.",
        right - left,
        size=theme.SMALL,
        fill=theme.INK_SOFT,
    )
    panel = Panel(
        number=3,
        slug="reading",
        act="I. From a scan to a reading",
        title="Reading the street names",
        dek="A vision model returns each name, its box on the page, and its direction.",
        caption=(
            "This is the only step that looks at the drawing. A vision model is "
            "given the small copy and asked for the street names it can see, where "
            "each one sits on the page, and whether it runs across the sheet or "
            "down it. Nothing else on the sheet is read: not the buildings, not "
            "the house numbers, not the colours that tell an underwriter what a "
            "wall is made of. Each reading is kept on disk, so the same reader is "
            "never paid twice for the same sheet, and everything after this point "
            "is arithmetic on those few names."
        ),
        figures=[("Names returned", str(len(streets)))],
        stage="annotate",
    )
    state = State("main", "The reading", alt="Street-name labels boxed on the scan.")
    panel.states = [state]
    out.save(panel, state, plate.image)
    return panel


def _panel_four(ex: dict[str, Any], corners: Escalation, model: str) -> Panel:
    """The record for panel 4, less its plates."""
    page, wrong, right = ex["escalated_page"], ex["escalated_wrong"], ex["escalated_right"]
    return Panel(
        number=4,
        slug="escalate",
        act="I. From a scan to a reading",
        title="When the first read comes back wrong",
        dek="A refused sheet is read again by a stronger model before anyone gives up on it.",
        caption=(
            f"Readers make mistakes, and one bad name can sink a sheet. The heavy "
            f"{right} down one edge of this sheet came back as {wrong}. "
            f"{wrong.title()} is a real Chicago street, but it is miles from these "
            f"blocks, so nothing on the modern map of them answers to it. Every "
            f"corner the pipeline could still find, all {corners.offered} of them, "
            f"was then a crossing of {corners.one_street}, and they fell in a "
            f"straight line down one side of the page. A line will not hold a sheet "
            f"down. Slide the whole placement along it and the corners still match, "
            f"so the checks refuse it: they want corners spread across the sheet, not "
            f"strung along an edge. Refusal is not the end. A sheet that fails is "
            f"offered to a stronger reader, cheapest first, and if a later reading "
            f"passes the same unchanged checks it is kept. Nothing is loosened to let "
            f"it through; the sheet simply gets a better look."
        ),
        figures=[
            ("Sheet", f"page {page}, {ex['escalated_title']}"),
            ("Corners offered", f"{corners.offered}, then {corners.reoffered}"),
            ("Second reader", model),
            ("Corners the fit kept", f"0, then {corners.kept}"),
        ],
        stage="escalate",
        note=(
            "Order note: a run does this AFTER the fit of steps 5 to 10, on the "
            "sheets those steps refused. It is told here because it is about "
            "reading, and because a re-read sheet is put through those same "
            "steps, unchanged, to earn its place. This one sheet comes from "
            "another atlas in the same corpus: the volume the rest of this "
            "walkthrough follows has no sheet whose first read went wrong this "
            "plainly."
        ),
    )


def _checked_names(
    page: str,
    ex: dict[str, Any],
    wrong: set[str],
    second: list[dict[str, Any]],
    corners: Escalation,
) -> tuple[str, str]:
    """The two names the prose says, refused unless the reads still say them.

    The plate crosses out what the index rejects; the sentences beside it name
    the pair by hand. A re-read under a new model can move one and leave the
    other, and the plate would then contradict its own caption.
    """
    misread, actual = ex["escalated_wrong"], ex["escalated_right"]
    if wrong != {misread}:
        raise FidelityError(f"p{page}: the first read's unresolved names are {sorted(wrong)}")
    if actual not in {street["name"] for street in second}:
        raise FidelityError(f"p{page}: the re-read does not return {actual}")
    if not corners.one_street or not corners.narrow:
        raise FidelityError(f"p{page}: the first read's corners no longer form one line")
    return misread, actual


def panel_escalate(volume: Volume, ex: dict[str, Any], out: Emitter) -> Panel:
    """4 - a misread street name, and the second read that fixed it."""
    page = ex["escalated_page"]
    record = volume.results[page]
    corners = volume.escalation(page)
    first = volume.annotation(page)["streets"]
    second = volume.escalated_annotation(page)["streets"]
    wrong = volume.unresolved(first)
    small = volume.small(page)
    misread, actual = _checked_names(page, ex, wrong, second, corners)
    panel = _panel_four(ex, corners, str(record.get("escalated_model", "")))
    notes = (
        f"The first reader returned these names. One is wrong: the street drawn "
        f"down one edge is {actual}, and it came back as {misread} - a real street "
        f"elsewhere in the city, and nothing at all here. Every corner left to match "
        f"with was then a crossing of {corners.one_street}, in a line down one side "
        f"of the page, so the sheet was refused.",
        f"The same sheet, read again by a stronger model. {actual} comes back as it "
        f"is printed, its crossings carry the corners to the other side of the page, "
        f"and the sheet clears every check unchanged.",
    )
    tallies = (
        f"{corners.offered} corners, in a line",
        f"{corners.kept} corners kept, spread across the sheet",
    )
    for key, label, streets, note, colour, read_as, tally in (
        ("first", "First reading", first, notes[0], theme.ALERT, misread, tallies[0]),
        ("second", "Second reading", second, notes[1], theme.KEEP, actual, tallies[1]),
    ):
        plate = theme.plate(
            4,
            "When the first read comes back wrong",
            kicker=f"{label.lower()} - sheet {page}, {ex['escalated_title']}",
        )
        sheet_box, col = split(plate, 0.40)
        view = sketch.place_sheet(plate, small, sheet_box, volume.scale(page))
        for street in streets:
            bad = street["name"] in wrong
            sketch.label_box(
                view, plate.draw, tuple(street["bbox"]), colour=theme.ALERT if bad else theme.SHEET
            )
        left, _top, right, bottom = col
        y = column_head(plate, col, label, note)
        y += 8
        for street in streets:
            bad = street["name"] in wrong
            mark = theme.ALERT if bad else theme.KEEP
            if bad:
                sketch.cross(plate.draw, (left + 12, y + 12), fill=mark, radius=8)
            else:
                sketch.ring(plate.draw, (left + 12, y + 12), fill=mark, radius=8)
            theme.text(plate.draw, (left + 34, y), street["name"], size=theme.BODY, fill=theme.INK)
            if bad:
                theme.text(
                    plate.draw,
                    (right, y),
                    "not in these blocks",
                    size=theme.SMALL,
                    fill=mark,
                    anchor="ra",
                )
            y += 34
        inset = (left, bottom - 300, left + 200, bottom - 60)
        detail(
            plate,
            small,
            tuple(ex["escalated_detail"]),
            inset,
            f"read as {read_as}",
            zoom_note="the same ink, magnified",
        )
        theme.text(
            plate.draw,
            (inset[2] + 26, (inset[1] + inset[3]) / 2 - 16),
            tally,
            size=theme.LEAD,
            fill=colour,
            bold=True,
        )
        state = State(key, label, alt=f"{label} of the same sheet.")
        panel.states.append(state)
        out.save(panel, state, plate.image)
    return panel
