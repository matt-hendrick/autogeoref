"""Fail-closed waterway candidate contracts."""

from __future__ import annotations

from autogeoref.centerlines import CenterlineIndex
from conftest import load_script

_w = load_script("experiments/water.py")
WaterIndex = _w.WaterIndex
normalize_water_name = _w.normalize_water_name
water_crossing_candidates = _w.water_crossing_candidates


def test_normalize_water_name_keeps_branch_identity() -> None:
    assert normalize_water_name("South Branch of Chicago River") == "SOUTH BRANCH OF CHICAGO RIVER"
    assert normalize_water_name("North Branch, Chicago River") == "NORTH BRANCH CHICAGO RIVER"


def test_water_candidates_require_exact_gazetteer_binding() -> None:
    centerlines = CenterlineIndex(
        [
            {
                "properties": {"street_nam": "MAIN", "street_typ": "ST"},
                "geometry": {"type": "LineString", "coordinates": [[0.0, -1.0], [0.0, 1.0]]},
            }
        ]
    )
    water = WaterIndex(
        {
            "elements": [
                {
                    "type": "way",
                    "tags": {"name": "Chicago River"},
                    "geometry": [{"lon": -1.0, "lat": 0.0}, {"lon": 1.0, "lat": 0.0}],
                }
            ]
        },
        {"CHICAGO RIVER": ("CHICAGO RIVER",)},
    )
    annotation = {
        "streets": [{"name": "Main St.", "bbox": [490, 100, 510, 500], "orientation": "vertical"}],
        "water_labels": [
            {"name": "Chicago River", "bbox": [100, 290, 400, 310], "orientation": "horizontal"}
        ],
    }

    candidates = water_crossing_candidates(annotation, water, centerlines, None, scale=0.5)

    assert len(candidates) == 1
    assert candidates[0].pixel == (1000.0, 600.0)
    assert candidates[0].world4326 == (0.0, 0.0)
    assert candidates[0].streets == ("WTR CHICAGO RIVER", "Main St.")

    annotation["water_labels"][0]["name"] = "River Chicago"
    assert water_crossing_candidates(annotation, water, centerlines, None, scale=0.5) == []
