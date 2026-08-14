"""Sheet overlays for the walkthrough plates, and the frame guard behind them.

A scan carries two pixel frames: labels are read on the downsampled small, and
every coordinate the matcher emits is full resolution. :class:`SheetView` is the
only place either is converted for drawing, and it names the two apart rather
than taking a bare number — the reversed conversion draws a plausible, wrong picture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import walkthrough_theme as theme
from PIL import Image, ImageDraw

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class SheetView:
    """A sheet drawn into a plate, plus both pixel frames' mappings onto it."""

    left: float
    top: float
    #: plate pixels per SMALL pixel
    zoom: float
    #: SMALL pixels per FULL-resolution pixel (the manifest's ``scale``)
    scale: float
    small_size: tuple[int, int]

    def small(self, x: float, y: float) -> tuple[float, float]:
        """A small-frame point (an annotation ``bbox``) on the plate."""
        return (self.left + x * self.zoom, self.top + y * self.zoom)

    def full(self, x: float, y: float) -> tuple[float, float]:
        """A full-resolution point (a candidate, a GCP) on the plate."""
        return self.small(x * self.scale, y * self.scale)

    def box(self, b: Box) -> Box:
        x0, y0 = self.small(b[0], b[1])
        x1, y1 = self.small(b[2], b[3])
        return (x0, y0, x1, y1)

    @property
    def rect(self) -> Box:
        w, h = self.small_size
        return (self.left, self.top, self.left + w * self.zoom, self.top + h * self.zoom)

    def holds(self, xy: tuple[float, float], slack: float = 6.0) -> bool:
        """Is a plate point inside the drawn sheet? Marks outside it are noise."""
        x0, y0, x1, y1 = self.rect
        return x0 - slack <= xy[0] <= x1 + slack and y0 - slack <= xy[1] <= y1 + slack


def place_sheet(
    plate: theme.Plate, small: Image.Image, box: Box, scale: float, *, border: bool = True
) -> SheetView:
    """Fit ``small`` inside ``box`` and return its view. ``scale`` is the manifest's.

    Raises when ``scale`` cannot be the small/full ratio, which is the frame
    boundary stated as a precondition instead of trusted.
    """
    if not 0.05 < scale < 1.0:
        raise ValueError(f"implausible small/full scale {scale!r}; the frames are the wrong way up")
    left, top, right, bottom = box
    zoom = min((right - left) / small.width, (bottom - top) / small.height)
    w, h = int(small.width * zoom), int(small.height * zoom)
    x = left + ((right - left) - w) / 2
    y = top + ((bottom - top) - h) / 2
    plate.image.paste(small.resize((w, h), Image.Resampling.LANCZOS), (int(x), int(y)))
    if border:
        plate.draw.rectangle((x, y, x + w, y + h), outline=theme.PLATE_EDGE, width=1)
    return SheetView(left=x, top=y, zoom=zoom, scale=scale, small_size=(small.width, small.height))


def label_box(
    view: SheetView,
    draw: ImageDraw.ImageDraw,
    b: Box,
    name: str = "",
    *,
    colour: tuple[int, int, int] = theme.SHEET,
) -> None:
    """A street label the reader returned, boxed and named."""
    x0, y0, x1, y1 = view.box(b)
    draw.rectangle((x0 - 3, y0 - 3, x1 + 3, y1 + 3), outline=colour, width=3)
    if name:
        tag(draw, (x0 - 3, y0 - 9), name, fill=colour, anchor="ld")


def tag(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    body: str,
    *,
    fill: tuple[int, int, int] = theme.INK,
    anchor: str = "la",
    size: int = theme.SMALL,
) -> None:
    """A short label on a paper chip, so it reads over a busy scan."""
    w = theme.width_of(body, size) + 12
    h = size + 8
    x, y = xy
    if anchor[0] == "r":
        x -= w
    elif anchor[0] == "m":
        x -= w / 2
    if anchor[1] == "d":
        y -= h
    elif anchor[1] == "m":
        y -= h / 2
    draw.rectangle((x, y, x + w, y + h), fill=theme.PAPER, outline=fill, width=1)
    theme.text(draw, (x + 6, y + 3), body, size=size, fill=fill)


def clipped_axis(
    view: SheetView,
    draw: ImageDraw.ImageDraw,
    b: Box,
    direction: tuple[float, float],
    *,
    fill: tuple[int, int, int] = theme.SHEET,
    width: int = 2,
) -> None:
    """The label's text direction, extended to the edges of the drawn sheet."""
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    dx, dy = direction
    norm = math.hypot(dx, dy) or 1.0
    dx, dy = dx / norm, dy / norm
    sw, sh = view.small_size
    # a slab clip per axis, which is all a straight line through the page needs
    ts = []
    for lim, centre, step in ((sw, cx, dx), (sh, cy, dy)):
        if abs(step) < 1e-12:
            continue
        ts.append(sorted(((0 - centre) / step, (lim - centre) / step)))
    if not ts:
        return
    lo = max(pair[0] for pair in ts)
    hi = min(pair[1] for pair in ts)
    if hi <= lo:
        return
    a = view.small(cx + dx * lo, cy + dy * lo)
    z = view.small(cx + dx * hi, cy + dy * hi)
    draw.line((*a, *z), fill=fill, width=width)


def ring(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    *,
    fill: tuple[int, int, int],
    radius: float = 11,
    solid: bool = True,
) -> None:
    """A kept anchor: solid ring with a paper core."""
    x, y = xy
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius), fill=theme.PAPER, outline=fill, width=3
    )
    if solid:
        r = radius - 5
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)


def cross(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    *,
    fill: tuple[int, int, int] = theme.DROP,
    radius: float = 9,
) -> None:
    """A dropped candidate: an X, so the mark reads without its colour."""
    x, y = xy
    draw.line((x - radius, y - radius, x + radius, y + radius), fill=fill, width=3)
    draw.line((x - radius, y + radius, x + radius, y - radius), fill=fill, width=3)


def arrow(
    draw: ImageDraw.ImageDraw,
    a: tuple[float, float],
    b: tuple[float, float],
    *,
    fill: tuple[int, int, int],
    width: int = 3,
    head: float = 12,
) -> None:
    """A straight arrow from ``a`` to ``b``."""
    draw.line((*a, *b), fill=fill, width=width)
    angle = math.atan2(b[1] - a[1], b[0] - a[0])
    for turn in (2.6, -2.6):
        draw.line(
            (
                b[0],
                b[1],
                b[0] + head * math.cos(angle + turn),
                b[1] + head * math.sin(angle + turn),
            ),
            fill=fill,
            width=width,
        )


@dataclass(frozen=True)
class Key:
    """One legend row: a swatch shape, a colour, and what it means."""

    shape: str
    fill: tuple[int, int, int]
    label: str


def legend(draw: ImageDraw.ImageDraw, xy: tuple[float, float], keys: list[Key]) -> float:
    """A stacked key. Returns the y below the last row."""
    x, y = xy
    for key in keys:
        cx, cy = x + 13, y + 13
        if key.shape == "ring":
            ring(draw, (cx, cy), fill=key.fill, radius=10)
        elif key.shape == "hollow":
            ring(draw, (cx, cy), fill=key.fill, radius=10, solid=False)
        elif key.shape == "cross":
            cross(draw, (cx, cy), fill=key.fill)
        elif key.shape == "line":
            draw.line((x, cy, x + 26, cy), fill=key.fill, width=4)
        else:
            draw.rectangle((x + 2, cy - 9, x + 24, cy + 9), fill=key.fill)
        theme.text(draw, (x + 38, y + 2), key.label, size=theme.SMALL, fill=theme.INK)
        y += 34
    return y
