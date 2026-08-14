"""The walkthrough's charts: strip plots, a translation cluster, a funnel, a table.

Small, plate-sized figures with one job each. They take numbers already computed
by the pipeline (or by :mod:`walkthrough_sources`) and never compute a pipeline
decision themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import walkthrough_theme as theme

if TYPE_CHECKING:
    from PIL import ImageDraw

Box = tuple[float, float, float, float]
RGB = tuple[int, int, int]


@dataclass(frozen=True)
class Strip:
    """One row of a strip plot: values, an accepted band, and a label."""

    label: str
    values: list[float]
    band: tuple[float, float]
    median: float
    unit: str


#: How much wider than the accepted window a strip's axis runs.
STRIP_REACH = 2.4


def strip_plot(draw: ImageDraw.ImageDraw, box: Box, rows: list[Strip]) -> None:
    """Per-sheet measurements against the window the volume derived from them.

    The axis is scaled to the window, not to the data: one wild sheet would
    otherwise squash every honest one into a single mark. Sheets past the end
    are drawn at the edge and counted there.
    """
    left, top, right, bottom = box
    height = (bottom - top) / len(rows)
    for i, row in enumerate(rows):
        y = top + i * height
        axis_y = y + height * 0.60
        half = (row.band[1] - row.band[0]) / 2 or 1.0
        centre = (row.band[0] + row.band[1]) / 2
        lo, hi = centre - half * STRIP_REACH, centre + half * STRIP_REACH
        x_left, x_right = left + 250, right - 40

        def at(
            value: float, a: float = lo, b: float = hi, p: float = x_left, q: float = x_right
        ) -> float:
            return p + (max(min(value, b), a) - a) / (b - a) * (q - p)

        theme.text(draw, (left, axis_y - 14), row.label, size=theme.BODY, fill=theme.INK, bold=True)
        band = (at(row.band[0]), axis_y - 40, at(row.band[1]), axis_y + 24)
        draw.rectangle(band, fill=(224, 233, 243), outline=theme.WORLD, width=1)
        theme.text(
            draw,
            ((band[0] + band[2]) / 2, band[1] - 8),
            "the window this book allows",
            size=theme.TINY,
            fill=theme.WORLD,
            anchor="md",
        )
        draw.line((x_left, axis_y, x_right, axis_y), fill=theme.RULE, width=1)
        outside = 0
        for value in row.values:
            inside = row.band[0] <= value <= row.band[1]
            outside += not inside
            draw.line(
                (at(value), axis_y - 14, at(value), axis_y + 14),
                fill=theme.KEEP if inside else theme.DROP,
                width=2,
            )
        draw.line(
            (at(row.median), axis_y - 40, at(row.median), axis_y + 24), fill=theme.SHEET, width=4
        )
        theme.text(
            draw,
            (at(row.median), axis_y + 30),
            f"{row.median:g}{(' ' + row.unit) if row.unit else ''}",
            size=theme.SMALL,
            fill=theme.SHEET,
            anchor="ma",
        )
        for edge, anchor in ((row.band[0], "ra"), (row.band[1], "la")):
            theme.text(
                draw,
                (at(edge), axis_y + 30),
                f"{edge:g}",
                size=theme.TINY,
                fill=theme.WORLD,
                anchor=anchor,
            )
        theme.text(
            draw,
            (x_right, axis_y - 40),
            f"{len(row.values) - outside} inside, {outside} outside",
            size=theme.TINY,
            fill=theme.INK_SOFT,
            anchor="rd",
        )


#: How much of the plot the agreement circle takes up.
CLUSTER_REACH = 4.0


def cluster_plot(
    draw: ImageDraw.ImageDraw,
    box: Box,
    offsets: list[tuple[float, float, bool]],
    tol_m: float,
    *,
    centre: tuple[float, float] | None = None,
) -> int:
    """Each corner's implied shift against the tolerance the cluster had to fit.

    Scaled to the tolerance, not to the data: one guess that lands in another
    part of the city would otherwise shrink the cluster to a dot. Guesses past
    the edge are drawn on it and counted in the return value.
    """
    left, top, right, bottom = box
    theme.card(draw, box, fill=(250, 246, 236))
    side = min(right - left, bottom - top)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    ox, oy = centre if centre is not None else (0.0, 0.0)
    ppm = side / (2 * tol_m * CLUSTER_REACH)
    reach = side / 2 - 14
    ring = tol_m * ppm
    draw.ellipse((cx - ring, cy - ring, cx + ring, cy + ring), outline=theme.KEEP, width=3)
    theme.text(
        draw, (cx + ring + 8, cy - ring - 4), f"{tol_m:g} m", size=theme.TINY, fill=theme.KEEP
    )
    for span in (-1, 1):
        draw.line((cx + span * 12, cy, cx + span * 20, cy), fill=theme.RULE, width=1)
        draw.line((cx, cy + span * 12, cx, cy + span * 20), fill=theme.RULE, width=1)
    far = 0
    for dx, dy, agreeing in offsets:
        px, py = (dx - ox) * ppm, -(dy - oy) * ppm
        length = math.hypot(px, py)
        if length > reach:
            far += 1
            px, py = px * reach / length, py * reach / length
        x, y = cx + px, cy + py
        if agreeing:
            draw.ellipse(
                (x - 9, y - 9, x + 9, y + 9), fill=theme.KEEP, outline=theme.PAPER, width=2
            )
        else:
            draw.line((x - 8, y - 8, x + 8, y + 8), fill=theme.DROP, width=3)
            draw.line((x - 8, y + 8, x + 8, y - 8), fill=theme.DROP, width=3)
    return far


@dataclass(frozen=True)
class Step:
    """One bar of the funnel: what a stage left placed, provisional and refused."""

    label: str
    placed: int
    provisional: int
    refused: int


def funnel_chart(draw: ImageDraw.ImageDraw, box: Box, steps: list[Step], total: int) -> None:
    """Where the volume's sheets stand after each acceptance stage."""
    left, top, right, bottom = box
    height = (bottom - top) / len(steps)
    bar_left = left + 320
    bar_right = right - 40
    for i, step in enumerate(steps):
        y = top + i * height + height * 0.16
        bar_h = height * 0.46
        theme.text(
            draw, (left, y + bar_h / 2 - 14), step.label, size=theme.BODY, fill=theme.INK, bold=True
        )
        x = bar_left
        for count, colour, name in (
            (step.placed, theme.KEEP, "placed"),
            (step.provisional, theme.PROV, "provisional"),
            (step.refused, theme.DROP, "flagged"),
        ):
            if not count:
                continue
            w = (bar_right - bar_left) * count / total
            draw.rectangle((x, y, x + w, y + bar_h), fill=colour)
            if w > 54:
                theme.text(
                    draw,
                    (x + w / 2, y + bar_h / 2 - 13),
                    str(count),
                    size=theme.BODY,
                    fill=theme.PAPER,
                    anchor="ma",
                )
            elif w > 6:
                theme.text(
                    draw,
                    (x + w / 2, y + bar_h + 4),
                    str(count),
                    size=theme.TINY,
                    fill=colour,
                    anchor="ma",
                )
            del name
            x += w


def vote_table(
    draw: ImageDraw.ImageDraw,
    box: Box,
    channels: list[tuple[str, str, str]],
    *,
    verdict: str,
) -> None:
    """One page's channel votes: name, what it measured, and how it voted."""
    left, top, right, bottom = box
    row = (bottom - top - 90) / max(len(channels), 1)
    theme.caps(draw, (left, top), "channel", size=theme.TINY)
    theme.caps(draw, (left + 300, top), "what it compared", size=theme.TINY)
    theme.caps(draw, (right - 150, top), "vote", size=theme.TINY)
    draw.line((left, top + 28, right, top + 28), fill=theme.RULE, width=1)
    for i, (name, what, vote) in enumerate(channels):
        y = top + 46 + i * row
        colour = {"yes": theme.KEEP, "no": theme.ALERT}.get(vote, theme.DROP)
        theme.text(draw, (left, y), name, size=theme.BODY, fill=theme.INK, bold=True)
        theme.paragraph(
            draw, (left + 300, y), what, right - left - 480, size=theme.SMALL, fill=theme.INK_SOFT
        )
        mark = {"yes": "YES", "no": "REFUTES"}.get(vote, "silent")
        theme.text(
            draw, (right, y), mark, size=theme.BODY, fill=colour, anchor="ra", bold=vote == "yes"
        )
    draw.line((left, bottom - 52, right, bottom - 52), fill=theme.RULE, width=1)
    theme.text(draw, (left, bottom - 40), verdict, size=theme.LEAD, fill=theme.KEEP, bold=True)


def histogram(
    draw: ImageDraw.ImageDraw,
    box: Box,
    values: list[float],
    *,
    bins: int = 24,
    unit: str,
    mark: float | None = None,
    mark_label: str = "",
) -> None:
    """A distribution, with an optional marked value."""
    left, top, right, bottom = box
    if not values:
        return
    hi = max(values)
    edges = [hi * i / bins for i in range(bins + 1)]
    counts = [0] * bins
    for value in values:
        slot = min(int(value / hi * bins), bins - 1) if hi else 0
        counts[slot] += 1
    peak = max(counts) or 1
    width = (right - left) / bins
    for i, count in enumerate(counts):
        h = (bottom - top - 40) * count / peak
        x = left + i * width
        draw.rectangle((x + 1, bottom - 40 - h, x + width - 1, bottom - 40), fill=theme.WORLD)
    draw.line((left, bottom - 40, right, bottom - 40), fill=theme.RULE, width=1)
    for i in (0, bins // 2, bins):
        x = left + i * width
        theme.text(
            draw,
            (x, bottom - 32),
            f"{edges[min(i, bins)]:.0f}",
            size=theme.TINY,
            fill=theme.INK_SOFT,
            anchor="ma",
        )
    theme.text(draw, (right, bottom - 10), unit, size=theme.TINY, fill=theme.INK_SOFT, anchor="ra")
    if mark is not None and hi:
        x = left + min(mark / hi, 1.0) * (right - left)
        draw.line((x, top, x, bottom - 40), fill=theme.SHEET, width=3)
        theme.text(draw, (x + 8, top), mark_label, size=theme.SMALL, fill=theme.SHEET)
