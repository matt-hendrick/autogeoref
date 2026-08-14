"""Address-numeral matching and renumbering contracts."""

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from shapely.geometry import Point

from autogeoref.addresses import (
    EMPTY_RENUMBERING,
    AddressNumeral,
    RenumberingEntry,
    RenumberingTable,
    match_address,
    modern_numeral,
)


@pytest.fixture(scope="module")
def madison_features(centerlines_path: Path) -> list[dict[str, Any]]:
    features = json.loads(centerlines_path.read_text())["features"]
    return [f for f in features if (f["properties"].get("street_nam") or "") == "MADISON"]


def _numeral(value: int) -> AddressNumeral:
    return AddressNumeral(value=value, bbox=(0.0, 0.0, 10.0, 10.0), street_hint="W. MADISON ST.")


def _block_2301(matches: list[Any]) -> list[Any]:
    return [m for m in matches if (m.from_add, m.to_add) == (2301, 2359)]


def test_madison_2301_block_matches_left_frontage(madison_features: list[dict[str, Any]]) -> None:
    # The selected range uses the left frontage.
    matches = _block_2301(match_address(_numeral(2301), "W. MADISON ST.", madison_features))
    assert matches, "expected the W MADISON 2301-2359 segment to match"
    m = matches[0]
    assert m.side == "l"
    assert m.fraction == 0.0
    hi = _block_2301(match_address(_numeral(2359), "W. MADISON ST.", madison_features))
    assert hi and hi[0].fraction == 1.0


def test_madison_interpolation_monotonic(madison_features: list[dict[str, Any]]) -> None:
    values = [2301, 2315, 2329, 2345, 2359]
    fracs = []
    dists = []
    for v in values:
        matches = _block_2301(match_address(_numeral(v), "W. MADISON ST.", madison_features))
        assert matches, f"no match for {v}"
        m = matches[0]
        fracs.append(m.fraction)
        dists.append(m.geometry.project(Point(m.point_4326)))
    assert all(b > a for a, b in pairwise(fracs))
    # interpolated points advance monotonically along the segment geometry
    assert all(b > a for a, b in pairwise(dists))


def test_parity_routes_to_correct_side(madison_features: list[dict[str, Any]]) -> None:
    # even numeral in the same block must land on the r (even) frontage
    even = [
        m
        for m in match_address(_numeral(2302), "W. MADISON ST.", madison_features)
        if (m.from_add, m.to_add) == (2300, 2358)
    ]
    assert even and all(m.side == "r" for m in even)
    odd = _block_2301(match_address(_numeral(2301), "W. MADISON ST.", madison_features))
    assert all(m.side == "l" for m in odd)


def test_numeral_outside_every_range_returns_empty(madison_features: list[dict[str, Any]]) -> None:
    assert match_address(_numeral(999_999), "W. MADISON ST.", madison_features) == []


def test_unknown_street_returns_empty(madison_features: list[dict[str, Any]]) -> None:
    assert match_address(_numeral(2301), "NO SUCH STREET", madison_features) == []


def test_modern_numbers_pass_through_unchanged() -> None:
    assert modern_numeral("W. MADISON ST.", 2301, table=None) == 2301


def test_empty_table_returns_none_never_passthrough() -> None:
    # An unconverted numeral must never pass through.
    assert EMPTY_RENUMBERING.convert("N. HERMITAGE AV.", 2412) is None
    assert modern_numeral("N. HERMITAGE AV.", 2412, table=EMPTY_RENUMBERING) is None


def test_table_driven_conversion() -> None:
    table = RenumberingTable(
        entries=(
            RenumberingEntry(
                street="N. HERMITAGE AV.", old_range=(2400, 2458), new_range=(4300, 4358)
            ),
        )
    )
    assert table.convert("HERMITAGE", 2400) == 4300
    assert table.convert("HERMITAGE", 2412) == 4312
    assert modern_numeral("N HERMITAGE AVENUE", 2458, table=table) == 4358
    # outside the entry, or wrong street: unknown, not passthrough
    assert table.convert("HERMITAGE", 2500) is None
    assert table.convert("PAULINA", 2412) is None


def test_contradiction_tolerance_scales_with_the_city_block_size() -> None:
    """The contradiction bound scales with the configured block size."""
    from autogeoref.addresses import ambiguity_tol_numbers

    table = RenumberingTable(
        entries=(
            RenumberingEntry(street="MAIN", old_range=(100, 199), new_range=(1000, 1099)),
            RenumberingEntry(street="MAIN", old_range=(100, 199), new_range=(1100, 1199)),
        )
    )
    # the two answers differ by 100: past the default tolerance (75), inside 200's (150)
    assert table.convert("MAIN", 150) is None
    assert table.convert("MAIN", 150, block_size=200) == 1050
    assert modern_numeral("MAIN", 150, table=table) is None
    assert modern_numeral("MAIN", 150, table=table, block_size=200) == 1050
    assert ambiguity_tol_numbers() == 75.0
    assert ambiguity_tol_numbers(200) == 150.0


def test_table_driven_conversion_antiparallel_ranges() -> None:
    # Cullom Av: old numbers DESCEND westward while new ascend; ranges are
    # stored in pairing order (old_range[0] partners new_range[0])
    entry = RenumberingEntry(street="CULLOM AV", old_range=(759, 579), new_range=(1601, 1795))
    table = RenumberingTable(entries=(entry,))
    assert table.convert("CULLOM", 759) == 1601
    assert table.convert("CULLOM", 579) == 1781  # 1601 + (759-579)
    assert table.convert("CULLOM", 698) == 1662
    assert table.convert("CULLOM", 578) is None


def test_table_from_json_skips_suffixed_old_schemes(tmp_path: Path) -> None:
    # S/W-suffixed old numbers are a different old scheme on the same street;
    # converting a plain numeral through them would be confidently wrong
    path = tmp_path / "renumbering.json"
    path.write_text(
        json.dumps(
            [
                {
                    "street": "HERMITAGE",
                    "old_range": [86, 90],
                    "new_range": [7, 11],
                    "old_suffix": "S",
                },
                {"street": "HERMITAGE", "old_range": [2400, 2458], "new_range": [4300, 4358]},
            ]
        )
    )
    table = RenumberingTable.from_json(path)
    assert table.convert("HERMITAGE", 88) is None
    assert table.convert("HERMITAGE", 2410) == 4310


def test_table_from_json(tmp_path: Path) -> None:
    path = tmp_path / "renumbering.json"
    path.write_text(
        json.dumps([{"street": "HERMITAGE", "old_range": [2400, 2458], "new_range": [4300, 4358]}])
    )
    table = RenumberingTable.from_json(path)
    assert table.convert("N. HERMITAGE AV.", 2410) == 4310
