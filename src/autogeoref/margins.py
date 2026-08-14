"""Margin sheet-number reading -> adjacency graph -> placement prior windows.

Margin adjacency numbers measured 100% reliable on a validated volume, so a deterministic
adjacency graph from margins is trustworthy given handling for corner double-numbers,
cross-volume edges, and stray glyphs.

Combined with committed neighbour positions, the graph turns corroboration into a
placement PROPOSER: neighbour position plus adjacency direction predict a translation
window for the junction-snap verifier and the rescue machinery. A single E-W neighbour
lands inside the verifier's window; an N-S-only neighbour yields a radius junction-snap
REFUSES by contract — tighten it with a second neighbour or the street index first.

Direction convention: a reading on a sheet's LEFT margin names the sheet continuing to
the WEST, so the page lies on the opposite side of that neighbour.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

#: Empirical sheet-center spacing measured on a validated volume, ground meters.
SPACING_EW_M = 220.0
SPACING_NS_M = 400.0
#: The junction verifier's working prior radius: predicted windows must cover at least this.
MIN_WINDOW_RADIUS_M = 100.0

_WGS84_RADIUS_M = 6378137.0

Side = Literal["top", "bottom", "left", "right"]

_OPPOSITE: dict[Side, Side] = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}
#: (dx, dy) unit direction from the NEIGHBOR toward the page, per the side
#: the neighbor was read on (left margin = neighbor west = page is east of it).
_PAGE_OFFSET_FROM_NEIGHBOR: dict[Side, tuple[float, float]] = {
    "left": (1.0, 0.0),
    "right": (-1.0, 0.0),
    "top": (0.0, -1.0),
    "bottom": (0.0, 1.0),
}

_SEE_VOLUME = re.compile(r"\bSEE\s+VOL(?:UME|S)?\b\.?\s*([A-Z0-9]+)", re.IGNORECASE)


@dataclass(frozen=True)
class MarginReading:
    """One margin text reading: which side of the sheet, and the raw text."""

    side: Side
    text: str


@dataclass(frozen=True)
class MarginEdge:
    """A parsed neighbor sheet number."""

    sheet: int


@dataclass(frozen=True)
class CrossVolumeEdge:
    """A parsed "SEE VOLUME X" continuation into another volume."""

    volume: str


def parse_margin(text: str) -> tuple[MarginEdge | CrossVolumeEdge, ...]:
    """Parse one margin reading into zero, one, or two edges.

    Handles the real-sheet cases: a plain sheet number gives one :class:`MarginEdge`, a corner
    double-number gives two, a cross-volume continuation (``"SEE VOLUME A"``) gives one
    :class:`CrossVolumeEdge`, and stray glyphs give none. A reading with ANY non-numeric token
    is rejected whole rather than salvaged — the reliability measured here came from clean reads
    only.
    """
    t = text.strip().upper()
    if not t:
        return ()
    m = _SEE_VOLUME.search(t)
    if m:
        return (CrossVolumeEdge(volume=m.group(1)),)
    tokens = [tok.strip(".,;:") for tok in t.split()]
    tokens = [tok for tok in tokens if tok]
    if not tokens or len(tokens) > 2 or not all(tok.isdigit() for tok in tokens):
        logger.debug("rejected margin text %r", text)
        return ()
    return tuple(MarginEdge(sheet=int(tok)) for tok in tokens)


@dataclass(frozen=True)
class AdjacencyEdge:
    """Directed adjacency: ``page`` names ``neighbor`` on its ``side`` margin."""

    page: str
    side: Side
    neighbor: str


@dataclass(frozen=True)
class CrossVolumeAdjacency:
    """Directed cross-volume continuation read on ``page``'s ``side`` margin."""

    page: str
    side: Side
    volume: str


@dataclass(frozen=True)
class PriorWindow:
    """A translation prior for the junction-snap verifier / rescue.

    Attributes:
        center_3857: Predicted sheet-center ``(x, y)`` in EPSG:3857.
        radius_m: Window radius in ground meters.
    """

    center_3857: tuple[float, float]
    radius_m: float


def _mercator_scale(y_3857: float) -> float:
    """Ground meters per 3857 unit at a northing: ``cos(lat)``."""
    return math.cos(math.atan(math.sinh(y_3857 / _WGS84_RADIUS_M)))


@dataclass(frozen=True)
class AdjacencyGraph:
    """Directed sheet-adjacency graph parsed from margin readings.

    Attributes:
        edges: All in-volume adjacency edges.
        cross_volume: All cross-volume continuations.
    """

    edges: tuple[AdjacencyEdge, ...]
    cross_volume: tuple[CrossVolumeAdjacency, ...]

    @classmethod
    def from_readings(cls, readings: Mapping[str, Sequence[MarginReading]]) -> AdjacencyGraph:
        """Build the graph from per-page margin readings.

        Args:
            readings: ``{page: [MarginReading, ...]}``; pages are the
                pipeline's string page ids (``"92"``).
        """
        edges: list[AdjacencyEdge] = []
        cross: list[CrossVolumeAdjacency] = []
        for page, page_readings in readings.items():
            for reading in page_readings:
                for parsed in parse_margin(reading.text):
                    if isinstance(parsed, MarginEdge):
                        edges.append(
                            AdjacencyEdge(page=page, side=reading.side, neighbor=str(parsed.sheet))
                        )
                    else:
                        cross.append(
                            CrossVolumeAdjacency(page=page, side=reading.side, volume=parsed.volume)
                        )
        logger.info(
            "adjacency graph: %d pages, %d edges, %d cross-volume",
            len(readings),
            len(edges),
            len(cross),
        )
        return cls(edges=tuple(edges), cross_volume=tuple(cross))

    def neighbors(self, page: str) -> tuple[AdjacencyEdge, ...]:
        """All adjacency edges read on a page's margins."""
        return tuple(e for e in self.edges if e.page == page)

    def mutual_edges(self) -> tuple[tuple[AdjacencyEdge, AdjacencyEdge], ...]:
        """Edge pairs where both sheets name each other on opposite sides.

        Mutual confirmation is the strongest adjacency evidence: both margins
        were read independently and agree on the shared boundary.
        """
        index = {(e.page, e.side, e.neighbor) for e in self.edges}
        out: list[tuple[AdjacencyEdge, AdjacencyEdge]] = []
        for e in self.edges:
            back = (e.neighbor, _OPPOSITE[e.side], e.page)
            if back in index and (e.page, e.neighbor) < (e.neighbor, e.page):
                out.append(
                    (e, AdjacencyEdge(page=e.neighbor, side=_OPPOSITE[e.side], neighbor=e.page))
                )
        return tuple(out)

    def predict_window(
        self,
        page: str,
        committed: Mapping[str, tuple[float, float]],
        spacing_ew_m: float = SPACING_EW_M,
        spacing_ns_m: float = SPACING_NS_M,
    ) -> PriorWindow | None:
        """Predict a translation prior for a page from committed neighbors.

        For each margin neighbour with a committed centre — ``committed`` maps page to an
        EPSG:3857 centre — the page centre is predicted one sheet spacing away in the adjacency
        direction. Spacings are ground metres, converted at the neighbour's latitude. The window
        radius is ``max(MIN_WINDOW_RADIUS_M, spacing/2)`` for the relevant axis; with several
        neighbours the centres are averaged and the TIGHTEST radius kept. Returns the prior
        window, or None when no margin neighbour is committed.
        """
        centers: list[tuple[float, float]] = []
        radii: list[float] = []
        for edge in self.neighbors(page):
            pos = committed.get(edge.neighbor)
            if pos is None:
                continue
            spacing = spacing_ew_m if edge.side in ("left", "right") else spacing_ns_m
            dx, dy = _PAGE_OFFSET_FROM_NEIGHBOR[edge.side]
            scale = _mercator_scale(pos[1])  # ground m -> 3857 units
            centers.append((pos[0] + dx * spacing / scale, pos[1] + dy * spacing / scale))
            radii.append(max(MIN_WINDOW_RADIUS_M, spacing / 2))
        if not centers:
            return None
        cx = sum(c[0] for c in centers) / len(centers)
        cy = sum(c[1] for c in centers) / len(centers)
        window = PriorWindow(center_3857=(cx, cy), radius_m=min(radii))
        logger.info(
            "predicted window for p%s from %d neighbors: center (%.0f, %.0f), radius %.0f m",
            page,
            len(centers),
            cx,
            cy,
            window.radius_m,
        )
        return window


__all__ = [
    "MIN_WINDOW_RADIUS_M",
    "SPACING_EW_M",
    "SPACING_NS_M",
    "AdjacencyEdge",
    "AdjacencyGraph",
    "CrossVolumeAdjacency",
    "CrossVolumeEdge",
    "MarginEdge",
    "MarginReading",
    "PriorWindow",
    "parse_margin",
]
