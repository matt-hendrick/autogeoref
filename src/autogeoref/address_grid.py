"""A city's house-number grid: printed address numbers against geography.

Where a city numbers its houses outward from one origin at a fixed count per
mile, a printed numeral is a *coordinate*. This module is that arithmetic: given
a candidate street's geometry it answers which axis that street's house numbers
run along, and what range its length covers, for the alias proposer.

City-neutral by construction — the origin and count per mile are CONFIGURED —
and a city without a declared grid gets no numeral check, which SHRINKS the
proposer's auto-write tier rather than loosening it. Nothing here is an
acceptance threshold and the placement pipeline never reads it.

**The model is linear and real grids are not**, so a district can run several
blocks short; ``tests/test_address_grid.py`` pins that discrepancy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Statute miles are the unit US address grids are cut in.
METERS_PER_MILE = 1609.344

#: Metres per degree of latitude (WGS84 mean). Longitude scales by cos(lat),
#: taken at the grid ORIGIN: a city-sized span moves this by well under the
#: one-block tolerance every consumer already applies.
M_PER_DEG_LAT = 111_132.0

#: Direction letters whose numbers grow with increasing coordinate.
POSITIVE_LETTERS = frozenset({"N", "E"})
NEGATIVE_LETTERS = frozenset({"S", "W"})

#: The two axes a street's house numbers can run along. ``"ns"`` = a
#: north-south street, numbered N/S; ``"ew"`` = an east-west street, numbered
#: E/W. Which one a street uses is a fact about its GEOMETRY, and
#: :meth:`AddressGrid.span` is the only place it is decided.
NS = "ns"
EW = "ew"

#: Axis a direction letter belongs to.
LETTER_AXIS = {"N": NS, "S": NS, "E": EW, "W": EW}


def signed(letter: str, value: float) -> float:
    """``('N', 2365) -> 2365``; ``('S', 1200) -> -1200``.

    One signed line per axis makes range containment plain arithmetic and
    keeps a street that crosses the origin (numbers running S then N) from
    needing two cases at every call site.
    """
    if letter in POSITIVE_LETTERS:
        return float(value)
    if letter in NEGATIVE_LETTERS:
        return -float(value)
    raise ValueError(f"not a direction letter: {letter!r}")


@dataclass(frozen=True)
class StreetSpan:
    """Where a street's own geometry puts it on the grid, in signed numbers.

    ``axis`` is the axis its house numbers run along; ``along_lo``/``along_hi``
    the numbers its length covers; ``cross_lo``/``cross_hi`` its position on
    the other axis (a north-south street's "1420W"). Both ranges are signed
    per :func:`signed`.
    """

    axis: str
    along_lo: float
    along_hi: float
    cross_lo: float
    cross_hi: float

    def along_covers(self, number: float, pad: float = 0.0) -> bool:
        """Is a signed number inside the along-street range, padded?"""
        return self.along_lo - pad <= number <= self.along_hi + pad

    def along_overlaps(self, lo: float, hi: float, pad: float = 0.0) -> bool:
        """Does a signed number range meet the along-street range, padded?"""
        return min(lo, hi) - pad <= self.along_hi and max(lo, hi) + pad >= self.along_lo

    def cross_covers(self, number: float, pad: float = 0.0) -> bool:
        """Is a signed number inside the cross-axis position, padded?"""
        return self.cross_lo - pad <= number <= self.cross_hi + pad

    @property
    def along_letters(self) -> tuple[str, str]:
        """The two direction letters this street's numbers can carry."""
        return ("S", "N") if self.axis == NS else ("W", "E")


@dataclass(frozen=True)
class AddressGrid:
    """House numbers per mile outward from a city's numbering origin.

    ``units_per_mile`` is the count of address numbers one mile spans — 800
    where the 100-per-block convention gives eight blocks to the mile.
    ``axis_ratio`` is how much longer a street must be on one axis than
    the other before its house numbers are read as running along that axis —
    the guard that keeps a diagonal or an L-shaped fragment from being
    assigned an axis it does not have.
    """

    origin_lon: float
    origin_lat: float
    units_per_mile: float
    axis_ratio: float = 1.5

    @property
    def m_per_deg_lon(self) -> float:
        return M_PER_DEG_LAT * math.cos(math.radians(self.origin_lat))

    def numbers_at(self, lon: float, lat: float) -> tuple[float, float]:
        """``(east-west, north-south)`` signed grid numbers for a point."""
        ew = (lon - self.origin_lon) * self.m_per_deg_lon / METERS_PER_MILE
        ns = (lat - self.origin_lat) * M_PER_DEG_LAT / METERS_PER_MILE
        return ew * self.units_per_mile, ns * self.units_per_mile

    def span(self, bounds: tuple[float, float, float, float]) -> StreetSpan | None:
        """The grid span of a street's bounding box, or None with no clear axis.

        ``bounds`` is ``(west, south, east, north)`` in EPSG:4326 — what
        Shapely hands back for a merged centerline geometry. None means the
        box is too square to call: the caller must then treat the street as
        having no usable numbering axis rather than guessing one.
        """
        west, south, east, north = bounds
        ew_lo, ns_lo = self.numbers_at(west, south)
        ew_hi, ns_hi = self.numbers_at(east, north)
        ew_extent, ns_extent = abs(ew_hi - ew_lo), abs(ns_hi - ns_lo)
        if ns_extent >= self.axis_ratio * max(ew_extent, 1e-9):
            return StreetSpan(NS, ns_lo, ns_hi, ew_lo, ew_hi)
        if ew_extent >= self.axis_ratio * max(ns_extent, 1e-9):
            return StreetSpan(EW, ew_lo, ew_hi, ns_lo, ns_hi)
        return None


__all__ = [
    "EW",
    "LETTER_AXIS",
    "METERS_PER_MILE",
    "M_PER_DEG_LAT",
    "NEGATIVE_LETTERS",
    "NS",
    "POSITIVE_LETTERS",
    "AddressGrid",
    "StreetSpan",
    "signed",
]
