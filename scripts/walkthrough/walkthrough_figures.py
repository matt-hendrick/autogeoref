"""The two figure shapes the fit and the gate plates share.

A page placed in the world by one model, and the constellation of corners a
placement rests on. Both are drawn on their own surface and pasted, so nothing
can spill onto the plate around them.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import walkthrough_draw as sketch
import walkthrough_geo as geo
import walkthrough_theme as theme

if TYPE_CHECKING:
    from walkthrough_sources import Volume

Box = tuple[float, float, float, float]


def sheet_quad(full_size: tuple[float, float], model: Any) -> list[tuple[float, float]]:
    """The page outline placed in the world by ``model`` (3857 metres)."""
    w, h = full_size
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    return [
        (
            model[0][0] + model[0][1] * x + model[0][2] * y,
            model[1][0] + model[1][1] * x + model[1][2] * y,
        )
        for x, y in corners
    ]


def placement_figure(
    plate: theme.Plate,
    box: Box,
    volume: Volume,
    page: str,
    good: list[tuple[float, float]],
    model: Any,
) -> None:
    """The scan as a counterexample would place it, over the accepted footprint.

    Drawing the page rather than its outline is what makes a mirrored placement
    visible at all: reflected and upright cover the same ground, and only the
    lettering says which is which.
    """
    bad = sheet_quad(volume.sheets[page].full_size, model) if model is not None else []
    canvas = geo.canvas_for(box, geo.bounds_of(good + bad, pad_m=120))
    view = canvas.view
    plot = [view.xy(*p) for p in good]
    canvas.draw.line([c for p in plot for c in p] + list(plot[0]), fill=(190, 178, 152), width=5)
    if model is not None:
        canvas.image.alpha_composite(
            geo.warp_sheet(volume.small(page), model, volume.scale(page), view)
        )
        edge = [view.xy(*p) for p in bad]
        canvas.draw.line([c for p in edge for c in p] + list(edge[0]), fill=theme.ALERT, width=4)
    sketch.tag(canvas.draw, (14, 14), "where the counterexample puts the page", fill=theme.ALERT)
    sketch.tag(
        canvas.draw,
        (14, view.size[1] - 14),
        "outline: where the accepted placement puts it",
        fill=theme.INK_SOFT,
        anchor="ld",
    )
    canvas.commit(plate)


def _dotted(
    plate: theme.Plate,
    origin: tuple[float, float],
    zoom: float,
    start: tuple[float, float],
    end: tuple[float, float],
    step: float = 14.0,
) -> None:
    """A dotted segment between two page positions, in plate coordinates."""
    x0, y0 = origin
    ax, ay = x0 + start[0] * zoom, y0 + start[1] * zoom
    bx, by = x0 + end[0] * zoom, y0 + end[1] * zoom
    span = math.hypot(bx - ax, by - ay)
    for i in range(int(span // step) + 1):
        if i % 2:
            continue
        t0, t1 = min(i * step / span, 1.0), min((i + 1) * step / span, 1.0)
        plate.draw.line(
            (
                ax + (bx - ax) * t0,
                ay + (by - ay) * t0,
                ax + (bx - ax) * t1,
                ay + (by - ay) * t1,
            ),
            fill=theme.SHEET,
            width=2,
        )


def anchor_figure(
    plate: theme.Plate,
    box: Box,
    points: list[tuple[float, float]],
    size: tuple[float, float],
    *,
    mark_stray: bool = False,
) -> None:
    """Where the agreeing corners sit on the page, for the three spread checks."""
    left, top, right, bottom = box
    w, h = size
    zoom = min((right - left) / w, (bottom - top) / h)
    x0 = left + ((right - left) - w * zoom) / 2
    y0 = top + ((bottom - top) - h * zoom) / 2
    plate.draw.rectangle(
        (x0, y0, x0 + w * zoom, y0 + h * zoom), fill=(250, 246, 236), outline=theme.PLATE_EDGE
    )
    sketch.tag(plate.draw, (x0 + 12, y0 + 12), "the page", fill=theme.INK_SOFT)
    stray = None
    if len(points) >= 2:
        a, b = max(
            ((p, q) for p in points for q in points), key=lambda pair: math.dist(pair[0], pair[1])
        )
        plate.draw.line(
            (x0 + a[0] * zoom, y0 + a[1] * zoom, x0 + b[0] * zoom, y0 + b[1] * zoom),
            fill=theme.ALERT,
            width=2,
        )
        length = math.dist(a, b) or 1.0
        stray = max(
            points,
            key=lambda p: (
                abs((p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])) / length
            ),
        )
        if mark_stray:
            # a dropped perpendicular, or the odd corner reads as a legend swatch
            # sitting near the frame rather than as a corner off the line
            t = ((stray[0] - a[0]) * (b[0] - a[0]) + (stray[1] - a[1]) * (b[1] - a[1])) / length**2
            foot = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            _dotted(plate, (x0, y0), zoom, stray, foot)
    for px, py in points:
        odd = mark_stray and stray is not None and (px, py) == stray
        sketch.ring(
            plate.draw,
            (x0 + px * zoom, y0 + py * zoom),
            fill=theme.SHEET if odd else theme.KEEP,
            radius=11,
        )
        if odd:
            sketch.tag(
                plate.draw,
                (x0 + px * zoom + 18, y0 + py * zoom),
                "take this one away",
                fill=theme.SHEET,
                anchor="lm",
            )
