"""The address-grid model: printed house numbers against geography.

Chicago's grid is the measured basis (State & Madison, 800 numbers per mile),
so the arithmetic is pinned against streets whose grid numbers are published
facts. The tolerance is one block, which is also the slack every consumer
applies — see :mod:`autogeoref.alias.propose`.
"""

from __future__ import annotations

import pytest

from autogeoref.address_grid import EW, NS, AddressGrid, StreetSpan, signed

# State & Madison, 800 numbers to the mile.
CHICAGO = AddressGrid(origin_lon=-87.6278, origin_lat=41.8819, units_per_mile=800)


@pytest.mark.parametrize(
    ("lon", "lat", "axis", "expected"),
    [
        # Fullerton Ave is 2400N; Madison itself is 0.
        (-87.6278, 41.9254, "ns", 2400),
        (-87.6278, 41.8819, "ns", 0),
        # North Ave is 1600N.
        (-87.6278, 41.9103, "ns", 1600),
        # Halsted St is 800W, Ashland 1600W — negative on the signed line.
        (-87.6470, 41.8819, "ew", -800),
        (-87.6667, 41.8819, "ew", -1600),
    ],
)
def test_known_grid_positions_reproduce_within_a_block(
    lon: float, lat: float, axis: str, expected: float
) -> None:
    """North and west of the origin, the linear model is good to a block."""
    ew, ns = CHICAGO.numbers_at(lon, lat)
    got = ns if axis == "ns" else ew
    assert abs(got - expected) <= 100, f"{got} vs {expected}"


def test_the_model_runs_short_south_of_the_origin() -> None:
    """KNOWN LIMIT, pinned so nobody rediscovers it as a bug.

    Chicago's first mile below Madison carries 1200 numbers, not 800: Roosevelt
    Rd is printed 1200S and lands one mile south of Madison. A linear grid
    therefore under-reads by ~400 units there, and every street below inherits
    the offset. Why that is survivable — the proposer never converts a numeral
    through this model — is in the module docstring.
    """
    _ew, roosevelt = CHICAGO.numbers_at(-87.6278, 41.8674)
    assert -900 < roosevelt < -700  # printed 1200S
    _ew, thirty_first = CHICAGO.numbers_at(-87.6278, 41.8378)
    assert -2600 < thirty_first < -2300  # printed 3100S


def test_signed_puts_each_axis_on_one_line() -> None:
    assert signed("N", 2365) == 2365
    assert signed("S", 1200) == -1200
    assert signed("E", 100) == 100
    assert signed("W", 1420) == -1420
    with pytest.raises(ValueError, match="not a direction letter"):
        signed("NE", 100)


def test_span_reads_the_numbering_axis_off_the_geometry() -> None:
    """A street's house numbers run along its length, and only geometry says which."""
    # a north-south street: long in latitude
    ns_span = CHICAGO.span((-87.66, 41.90, -87.66, 41.94))
    assert ns_span is not None
    assert ns_span.axis == NS
    assert ns_span.along_lo < ns_span.along_hi  # N numbers grow northward
    assert ns_span.cross_lo == pytest.approx(ns_span.cross_hi)

    ew_span = CHICAGO.span((-87.70, 41.92, -87.60, 41.92))
    assert ew_span is not None
    assert ew_span.axis == EW
    assert ew_span.along_letters == ("W", "E")


def test_span_refuses_a_box_too_square_to_call() -> None:
    """A diagonal or stub has no numbering axis, and guessing one is worse than None."""
    assert CHICAGO.span((-87.66, 41.90, -87.645, 41.9135)) is None


def test_along_covers_and_overlaps_respect_the_pad() -> None:
    span = StreetSpan(axis=NS, along_lo=1500.0, along_hi=2600.0, cross_lo=-1200.0, cross_hi=-1200.0)
    assert span.along_covers(2000)
    assert not span.along_covers(2700)
    assert span.along_covers(2700, pad=100)
    assert span.along_overlaps(2500, 3000)
    assert not span.along_overlaps(2700, 3000)
    assert span.cross_covers(-1200)
    assert not span.cross_covers(-900)


def test_a_street_crossing_the_origin_spans_both_signs() -> None:
    """Numbers run S then N through Madison; one signed line handles it."""
    span = CHICAGO.span((-87.66, 41.86, -87.66, 41.91))
    assert span is not None
    assert span.along_lo < 0 < span.along_hi
    assert span.along_covers(signed("S", 1000))
    assert span.along_covers(signed("N", 1000))
