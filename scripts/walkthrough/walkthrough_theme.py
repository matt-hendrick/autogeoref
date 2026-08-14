"""One visual language for every walkthrough plate: palette, type, and frame.

Imported by name from the other ``walkthrough_*`` files, which run from this
directory. Colour is never the only signal — every mark this palette names is
paired with a shape or a word by :mod:`walkthrough_draw`, so the figures survive
a colourblind reader and a monochrome print.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

RGB = tuple[int, int, int]

#: Every plate is this size, so a stepper never reflows between steps.
PLATE = (1600, 1000)
MARGIN = 46

PAPER: RGB = (245, 238, 223)
PLATE_EDGE: RGB = (216, 202, 173)
INK: RGB = (32, 26, 18)
INK_SOFT: RGB = (110, 98, 79)
RULE: RGB = (203, 189, 160)
WASH: RGB = (236, 227, 208)

#: Evidence read off the scan.
SHEET: RGB = (191, 96, 8)
#: The modern street map the scan is matched against.
WORLD: RGB = (23, 87, 158)
#: Survived a gate, or committed.
KEEP: RGB = (12, 105, 78)
#: Considered and dropped.
DROP: RGB = (127, 116, 100)
#: A gate refusing.
ALERT: RGB = (162, 33, 82)
#: Placed, but not yet on its own evidence.
PROV: RGB = (109, 74, 158)


def font(size: int) -> ImageFont.FreeTypeFont:
    """A sized face. Pillow ships it, so a cold clone renders the same plates."""
    face = ImageFont.load_default(size=size)
    assert isinstance(face, ImageFont.FreeTypeFont)  # sized always resolves to one
    return face


#: The type scale. Anything drawn into a plate uses one of these.
TITLE = 40
LEAD = 30
BODY = 25
SMALL = 21
TINY = 18


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    body: str,
    *,
    size: int = BODY,
    fill: RGB = INK,
    bold: bool = False,
    anchor: str = "la",
) -> None:
    """Draw a line of plate text. ``bold`` thickens the stroke, faking a weight."""
    draw.text(
        xy,
        body,
        font=font(size),
        fill=fill,
        anchor=anchor,
        stroke_width=1 if bold else 0,
        stroke_fill=fill if bold else None,
    )


def width_of(body: str, size: int = BODY, *, bold: bool = False) -> float:
    """Advance width of ``body`` at ``size``, matching :func:`text`'s stroke."""
    scratch = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    return scratch.textlength(body, font=font(size)) + (2 if bold else 0)


def caps(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    body: str,
    *,
    size: int = TINY,
    fill: RGB = INK_SOFT,
    tracking: float = 2.6,
) -> float:
    """Letterspaced small caps; returns the x it ended at."""
    x, y = xy
    for char in body.upper():
        draw.text((x, y), char, font=font(size), fill=fill)
        x += draw.textlength(char, font=font(size)) + tracking
    return x


def wrap(body: str, limit: float, size: int = BODY) -> list[str]:
    """Greedy word wrap to a pixel ``limit``."""
    lines: list[str] = []
    line = ""
    for word in body.split():
        trial = f"{line} {word}".strip()
        if line and width_of(trial, size) > limit:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def paragraph(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    body: str,
    limit: float,
    *,
    size: int = BODY,
    fill: RGB = INK,
    leading: float = 1.42,
) -> float:
    """Wrapped prose from ``xy``; returns the y below the last line."""
    x, y = xy
    for line in wrap(body, limit, size):
        text(draw, (x, y), line, size=size, fill=fill)
        y += size * leading
    return y


@dataclass
class Plate:
    """One panel image: a titled sheet of paper with a content box."""

    number: int
    title: str
    image: Image.Image
    draw: ImageDraw.ImageDraw
    #: (left, top, right, bottom) of everything below the header rule.
    box: tuple[int, int, int, int]

    def credit(self, line: str) -> None:
        """A data credit along the bottom edge, inside the content box."""
        text(
            self.draw,
            (self.box[2], PLATE[1] - MARGIN + 8),
            line,
            size=TINY,
            fill=INK_SOFT,
            anchor="ra",
        )


def plate(number: int, title: str, *, kicker: str = "") -> Plate:
    """A blank plate with its header drawn."""
    image = Image.new("RGB", PLATE, PAPER)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, PLATE[0] - 1, PLATE[1] - 1), outline=PLATE_EDGE, width=2)
    top = MARGIN
    end = caps(draw, (MARGIN, top + 4), f"{number:02d}", size=SMALL, fill=SHEET)
    text(draw, (end + 10, top), title, size=LEAD, fill=INK, bold=True)
    if kicker:
        text(draw, (PLATE[0] - MARGIN, top + 6), kicker, size=SMALL, fill=INK_SOFT, anchor="ra")
    rule_y = top + LEAD + 16
    draw.line((MARGIN, rule_y, PLATE[0] - MARGIN, rule_y), fill=RULE, width=1)
    return Plate(
        number=number,
        title=title,
        image=image,
        draw=draw,
        box=(MARGIN, int(rule_y) + 22, PLATE[0] - MARGIN, PLATE[1] - MARGIN - 14),
    )


def card(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    *,
    fill: RGB = WASH,
    outline: RGB = PLATE_EDGE,
) -> None:
    """A tinted sub-panel behind a diagram or a table."""
    draw.rectangle(box, fill=fill, outline=outline, width=1)
