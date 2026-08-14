"""Margin adjacency graph + prior-window proposer.

Synthetic graph reproducing the measured real-sheet cases: neighbor numbers on all
four sides, a corner double-number, a "SEE VOLUME A" cross-volume edge, and
a stray glyph rejected. Window geometry uses the measured ``_034`` spacings
(~220 m E-W, ~400 m N-S).
"""

import math

import pytest

from autogeoref.margins import (
    AdjacencyEdge,
    AdjacencyGraph,
    CrossVolumeEdge,
    MarginEdge,
    MarginReading,
    parse_margin,
)

_WGS84_RADIUS_M = 6378137.0


# ---------------------------------------------------------------------------
# parse_margin
# ---------------------------------------------------------------------------


def test_plain_sheet_number() -> None:
    assert parse_margin("114") == (MarginEdge(114),)
    assert parse_margin(" 90. ") == (MarginEdge(90),)


def test_corner_double_number_gives_two_edges() -> None:
    assert parse_margin("113 114") == (MarginEdge(113), MarginEdge(114))


def test_cross_volume_edges() -> None:
    assert parse_margin("SEE VOLUME A") == (CrossVolumeEdge("A"),)
    assert parse_margin("SEE VOL. 5") == (CrossVolumeEdge("5"),)
    assert parse_margin("see volume a") == (CrossVolumeEdge("A"),)


def test_stray_glyphs_rejected() -> None:
    # the measured zero mismatches came from clean reads: any junk rejects the reading
    assert parse_margin("~#") == ()
    assert parse_margin("1I4") == ()
    assert parse_margin("113 1I4") == ()
    assert parse_margin("") == ()
    assert parse_margin("113 114 115") == ()  # more than a corner pair


# ---------------------------------------------------------------------------
# AdjacencyGraph (synthetic volume mirroring the measured cases)
# ---------------------------------------------------------------------------


@pytest.fixture
def graph() -> AdjacencyGraph:
    readings = {
        "91": [
            MarginReading("top", "90"),
            MarginReading("bottom", "92"),
            MarginReading("left", "113 114"),  # corner double-number
            MarginReading("right", "SEE VOLUME A"),  # cross-volume edge
        ],
        "90": [
            MarginReading("bottom", "91"),  # mutual with 91's top
            MarginReading("top", "~#"),  # stray glyph: rejected
        ],
        "92": [MarginReading("left", "91")],  # NOT mutual (91 has 92 on bottom)
    }
    return AdjacencyGraph.from_readings(readings)


def test_from_readings_edges(graph: AdjacencyGraph) -> None:
    n91 = graph.neighbors("91")
    assert AdjacencyEdge("91", "top", "90") in n91
    assert AdjacencyEdge("91", "bottom", "92") in n91
    assert AdjacencyEdge("91", "left", "113") in n91
    assert AdjacencyEdge("91", "left", "114") in n91
    assert len(n91) == 4  # the SEE VOLUME reading is not an in-volume edge
    assert graph.cross_volume[0].page == "91"
    assert graph.cross_volume[0].volume == "A"
    # the stray glyph produced no edge for 90's top
    assert graph.neighbors("90") == (AdjacencyEdge("90", "bottom", "91"),)


def test_mutual_edges(graph: AdjacencyGraph) -> None:
    mutual = graph.mutual_edges()
    pairs = {(a.page, a.neighbor) for a, _ in mutual} | {(b.page, b.neighbor) for _, b in mutual}
    # 90 <-> 91 name each other on opposite sides; 91 -> 92 is one-way
    # (92 names 91 on its LEFT, not its TOP, so it does not reciprocate)
    assert ("91", "90") in pairs or ("90", "91") in pairs
    assert ("91", "92") not in pairs
    assert len(mutual) == 1


# ---------------------------------------------------------------------------
# predict_window geometry (equator positions: 3857 units == ground meters)
# ---------------------------------------------------------------------------


def test_window_from_west_neighbor() -> None:
    graph = AdjacencyGraph.from_readings({"92": [MarginReading("left", "91")]})
    window = graph.predict_window("92", {"91": (1000.0, 0.0)})
    assert window is not None
    # neighbor to the WEST -> window center east of it by spacing_ew
    assert window.center_3857 == pytest.approx((1220.0, 0.0))
    # radius = max(100, half the relevant spacing) -> 110 E-W
    assert window.radius_m == pytest.approx(110.0)


def test_window_from_north_neighbor() -> None:
    graph = AdjacencyGraph.from_readings({"92": [MarginReading("top", "90")]})
    window = graph.predict_window("92", {"90": (0.0, 1000.0)})
    assert window is not None
    # neighbor to the NORTH -> window center south of it by spacing_ns
    assert window.center_3857 == pytest.approx((0.0, 600.0))
    assert window.radius_m == pytest.approx(200.0)  # max(100, 400/2)


def test_window_two_neighbors_average_and_tighten() -> None:
    graph = AdjacencyGraph.from_readings(
        {
            "92": [
                MarginReading("left", "91"),
                MarginReading("top", "90"),
            ]
        }
    )
    window = graph.predict_window("92", {"91": (0.0, 0.0), "90": (220.0, 400.0)})
    assert window is not None
    # both neighbors predict (220, 0); the tighter E-W radius wins
    # (abs tolerance: cos(lat) at y=400 is not exactly 1)
    assert window.center_3857 == pytest.approx((220.0, 0.0), abs=1e-3)
    assert window.radius_m == pytest.approx(110.0)


def test_window_none_without_committed_neighbor() -> None:
    graph = AdjacencyGraph.from_readings({"92": [MarginReading("left", "91")]})
    assert graph.predict_window("92", {}) is None
    assert graph.predict_window("92", {"114": (0.0, 0.0)}) is None


def test_window_mercator_scaling_at_chicago() -> None:
    # at Chicago's latitude a 220 ground-meter offset is ~295 3857 units
    graph = AdjacencyGraph.from_readings({"92": [MarginReading("left", "91")]})
    y = 5_140_000.0  # ~41.85 N
    window = graph.predict_window("92", {"91": (0.0, y)})
    assert window is not None
    lat = math.degrees(math.atan(math.sinh(y / _WGS84_RADIUS_M)))
    expected_dx = 220.0 / math.cos(math.radians(lat))
    assert 280.0 < expected_dx < 310.0
    assert window.center_3857[0] == pytest.approx(expected_dx)
    assert window.radius_m == pytest.approx(110.0)  # radius stays ground meters
