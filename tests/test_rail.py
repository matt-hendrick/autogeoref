"""Rail-crossing channel: unit contracts + the p92 golden replay.

The golden test replays the G3 rail-channel experiment
end to end from the cached vectors in ``tests/data/`` (the frozen v2
annotation and the cached OSM rail geometry — no network, no vision budget)
against the frozen ``sanborn01790_034`` fixtures, and pins the measured
result: 12 candidate anchors, 7 within the 12 m rescue tolerance, best
4.49 m vs the truth translation.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString

from autogeoref.affine import TO_3857
from autogeoref.bounds import community_area_bounds
from autogeoref.centerlines import CenterlineIndex
from autogeoref.names import load_aliases
from autogeoref.rail import (
    CATCH_ALL_GROUP,
    RailIndex,
    load_rail_gazetteer,
    normalize_rail_name,
    rail_crossing_candidates,
)
from autogeoref.rescue import TOL_M, has_disjoint_pair, pinned_linear

DATA = Path(__file__).resolve().parent / "data"
GAZETTEER_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "chicago" / "rail-gazetteer-chicago.json"
)


# --- normalize_rail_name ----------------------------------------------------


def test_normalize_rail_name_drops_parentheticals_and_collapses_punctuation() -> None:
    assert normalize_rail_name("C.M.&ST.P.R.R. (EVANSTON DIV.)") == "C M ST P R R EVANSTON DIV"
    # spacing variants of the same dotted initialism canonicalize identically
    assert normalize_rail_name("C. M. & St. P. R.R.") == "C M ST P R R"
    assert normalize_rail_name("  Union  Pacific   Railroad ") == "UNION PACIFIC RAILROAD"


def test_normalize_rail_name_is_not_street_normalization() -> None:
    # street normalize() would strip the leading direction token; rail's own
    # lane must not ("North Western" is the railroad's name, not a heading)
    assert normalize_rail_name("N. Western R.R.") == "N WESTERN R R"
    assert normalize_rail_name("North Side Main Line") == "NORTH SIDE MAIN LINE"
    # ...and must not strip street suffix tokens either ("CT", "PL", "AVE")
    assert normalize_rail_name("Belt Line CT") == "BELT LINE CT"


# --- RailIndex grouping -----------------------------------------------------

OVERPASS = {
    "version": 0.6,
    "elements": [
        # two ways of one railroad under raw-spelling variants: ONE group
        {
            "type": "way",
            "id": 1,
            "tags": {"name": "Union Pacific Railroad"},
            "geometry": [{"lat": 0.0, "lon": 0.0}, {"lat": 0.0, "lon": 1.0}],
        },
        {
            "type": "way",
            "id": 2,
            "tags": {"name": "UNION PACIFIC RAILROAD"},
            "geometry": [{"lat": 0.0, "lon": 1.0}, {"lat": 0.0, "lon": 2.0}],
        },
        # no name: falls back to operator
        {
            "type": "way",
            "id": 3,
            "tags": {"operator": "Metra"},
            "geometry": [{"lat": 1.0, "lon": 0.0}, {"lat": 1.0, "lon": 1.0}],
        },
        # no tags at all: catch-all group
        {
            "type": "way",
            "id": 4,
            "geometry": [{"lat": 2.0, "lon": 0.0}, {"lat": 2.0, "lon": 1.0}],
        },
        # no geometry: skipped
        {"type": "way", "id": 5, "tags": {"name": "Ghost"}},
    ],
}


def test_rail_index_groups_overpass() -> None:
    idx = RailIndex(OVERPASS)
    assert set(idx.groups) == {"UNION PACIFIC RAILROAD", "METRA", CATCH_ALL_GROUP}
    up = idx.merged("UNION PACIFIC RAILROAD")
    assert up is not None
    # the two spelling-variant ways merged into one group geometry
    assert up.bounds == (0.0, 0.0, 2.0, 0.0)
    assert idx.merged("GHOST") is None


def test_rail_index_groups_geojson(tmp_path: Path) -> None:
    gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "C.&N.W.R.R."},
                "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 0.0]]},
            },
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "LineString", "coordinates": [[0.0, 1.0], [1.0, 1.0]]},
            },
            {"type": "Feature", "properties": {"name": "Null"}, "geometry": None},
        ],
    }
    path = tmp_path / "rail.geojson"
    path.write_text(json.dumps(gj))
    idx = RailIndex.from_json(path)
    assert set(idx.groups) == {"C N W R R", CATCH_ALL_GROUP}
    cnw = idx.merged("C N W R R")
    assert cnw is not None
    assert cnw.equals(LineString([(0.0, 0.0), (1.0, 0.0)]))


# --- rail_crossing_candidates (synthetic, hand-computable) ------------------

# a north-south street at lon=0 spanning lat 0..0.01
STREET_FEATURES = [
    {
        "properties": {"street_nam": "MAIN", "street_typ": "AVE"},
        "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [0.0, 0.01]]},
    }
]

# UNION PACIFIC crosses MAIN at (0, 0.005); CSX at (0, 0.002);
# BNSF is far away and crosses nothing
RAIL_DATA = {
    "elements": [
        {
            "type": "way",
            "id": 1,
            "tags": {"name": "Union Pacific"},
            "geometry": [{"lat": 0.005, "lon": -0.01}, {"lat": 0.005, "lon": 0.01}],
        },
        {
            "type": "way",
            "id": 2,
            "tags": {"name": "CSX"},
            "geometry": [{"lat": 0.002, "lon": -0.01}, {"lat": 0.002, "lon": 0.01}],
        },
        {
            "type": "way",
            "id": 3,
            "tags": {"name": "BNSF"},
            "geometry": [{"lat": 0.005, "lon": 0.05}, {"lat": 0.005, "lon": 0.06}],
        },
    ]
}

# vertical street label (axis x=500) x wide-therefore-horizontal rail label
# (axis y=300) -> small-frame pixel point 500, 300; scale 0.5 -> full-res
# 1000, 600
ANNOTATION = {
    "streets": [{"name": "MAIN AV.", "bbox": [480, 100, 520, 500], "orientation": "vertical"}],
    "rail_labels": [{"name": "U.P.R.R.", "bbox": [100, 280, 400, 320]}],
}

# the label binds to all three groups; BNSF crosses nothing so binding to it
# is harmless
GAZETTEER = {"U P R R": ("UNION PACIFIC", "CSX", "BNSF")}


def test_candidates_hand_computed_crossing() -> None:
    cl = CenterlineIndex(STREET_FEATURES)
    rails = RailIndex(RAIL_DATA, gazetteer=GAZETTEER)
    cands = rail_crossing_candidates(ANNOTATION, rails, cl, None, scale=0.5)
    # the printed label pairs with every crossing group its gazetteer row
    # names (rail.py docstring)
    assert {c.streets for c in cands} == {
        ("RR UNION PACIFIC", "MAIN AV."),
        ("RR CSX", "MAIN AV."),
    }
    by_group = {c.streets[0]: c for c in cands}
    up = by_group["RR UNION PACIFIC"]
    assert up.pixel == (1000.0, 600.0)
    assert up.world4326[0] == pytest.approx(0.0, abs=1e-12)
    assert up.world4326[1] == pytest.approx(0.005, abs=1e-12)
    csx = by_group["RR CSX"]
    assert csx.pixel == (1000.0, 600.0)
    assert csx.world4326[1] == pytest.approx(0.002, abs=1e-12)


def test_candidates_tall_bbox_is_vertical() -> None:
    # tall rail bbox -> vertical axis (x=300); horizontal street label
    # (axis y=150) -> pixel (300, 150); the street runs east-west at
    # lat=0.005 so UNION PACIFIC (also east-west) yields no Point crossing,
    # while a north-south railroad crosses it once
    cl = CenterlineIndex(
        [
            {
                "properties": {"street_nam": "OAK", "street_typ": "ST"},
                "geometry": {"type": "LineString", "coordinates": [[-0.01, 0.005], [0.01, 0.005]]},
            }
        ]
    )
    rails = RailIndex(
        {
            "elements": [
                {
                    "type": "way",
                    "id": 1,
                    "tags": {"name": "Wabash"},
                    "geometry": [{"lat": 0.0, "lon": 0.003}, {"lat": 0.01, "lon": 0.003}],
                }
            ]
        },
        gazetteer={"WAB R R": ("WABASH",)},
    )
    ann = {
        "streets": [{"name": "OAK ST.", "bbox": [100, 130, 400, 170], "orientation": "horizontal"}],
        "rail_labels": [{"name": "WAB.R.R.", "bbox": [280, 0, 320, 400]}],
    }
    cands = rail_crossing_candidates(ann, rails, cl, None, scale=1.0)
    assert len(cands) == 1
    assert cands[0].pixel == (300.0, 150.0)
    assert cands[0].world4326 == (pytest.approx(0.003), pytest.approx(0.005))
    assert cands[0].streets == ("RR WABASH", "OAK ST.")


def test_candidates_v1_annotation_returns_empty() -> None:
    cl = CenterlineIndex(STREET_FEATURES)
    rails = RailIndex(RAIL_DATA, gazetteer=GAZETTEER)
    v1 = {"streets": ANNOTATION["streets"]}  # no rail_labels key at all
    assert rail_crossing_candidates(v1, rails, cl, None, scale=0.5) == []
    v2_empty = {"streets": ANNOTATION["streets"], "rail_labels": []}
    assert rail_crossing_candidates(v2_empty, rails, cl, None, scale=0.5) == []


def test_candidates_parallel_axes_yield_nothing() -> None:
    # wide rail label + horizontal street label: parallel axes, no Point
    cl = CenterlineIndex(STREET_FEATURES)
    rails = RailIndex(RAIL_DATA, gazetteer=GAZETTEER)
    ann = {
        "streets": [
            {"name": "MAIN AV.", "bbox": [100, 130, 400, 170], "orientation": "horizontal"}
        ],
        "rail_labels": ANNOTATION["rail_labels"],
    }
    assert rail_crossing_candidates(ann, rails, cl, None, scale=0.5) == []


# --- gazetteer binding (the RAIL-015 fix: bound lookup, fail closed) ---------


def test_unbound_label_yields_zero_candidates() -> None:
    # the label's normalized form has no gazetteer row: silence, not noise
    cl = CenterlineIndex(STREET_FEATURES)
    rails = RailIndex(RAIL_DATA, gazetteer={"WAB R R": ("WABASH",)})
    assert rail_crossing_candidates(ANNOTATION, rails, cl, None, scale=0.5) == []


def test_parenthetical_qualifier_beats_the_generic_binding() -> None:
    cl = CenterlineIndex(STREET_FEATURES)
    rails = RailIndex(
        RAIL_DATA,
        gazetteer={
            "U P R R ELEVATED": ("CSX",),
            "U P R R": ("UNION PACIFIC",),
        },
    )
    annotation = {
        **ANNOTATION,
        "rail_labels": [{"name": "U.P.R.R. (ELEVATED)", "bbox": [100, 280, 400, 320]}],
    }
    cands = rail_crossing_candidates(annotation, rails, cl, None, scale=0.5)
    assert {cand.streets[0] for cand in cands} == {"RR CSX"}


def test_parenthetical_note_falls_back_to_the_generic_binding() -> None:
    cl = CenterlineIndex(STREET_FEATURES)
    rails = RailIndex(RAIL_DATA, gazetteer={"U P R R": ("UNION PACIFIC",)})
    annotation = {
        **ANNOTATION,
        "rail_labels": [{"name": "U.P.R.R. (ELEVATED)", "bbox": [100, 280, 400, 320]}],
    }
    cands = rail_crossing_candidates(annotation, rails, cl, None, scale=0.5)
    assert {cand.streets[0] for cand in cands} == {"RR UNION PACIFIC"}


def test_no_gazetteer_yields_zero_candidates() -> None:
    # an index built without a gazetteer never produces candidates
    cl = CenterlineIndex(STREET_FEATURES)
    rails = RailIndex(RAIL_DATA)
    assert rails.gazetteer is None
    assert rail_crossing_candidates(ANNOTATION, rails, cl, None, scale=0.5) == []


def test_binding_restricts_to_named_groups() -> None:
    # both UNION PACIFIC and CSX cross MAIN, but the row names only one
    cl = CenterlineIndex(STREET_FEATURES)
    rails = RailIndex(RAIL_DATA, gazetteer={"U P R R": ("UNION PACIFIC",)})
    cands = rail_crossing_candidates(ANNOTATION, rails, cl, None, scale=0.5)
    assert {c.streets for c in cands} == {("RR UNION PACIFIC", "MAIN AV.")}


def test_load_rail_gazetteer_normalizes_both_sides(tmp_path: Path) -> None:
    path = tmp_path / "gaz.json"
    path.write_text(
        json.dumps(
            {
                "bindings": {"C.M.&ST.P.R.R. (EVANSTON DIV.)": ["North Side Main Line"]},
                "unbound": {"SWIFT R R": "documentation only, never loaded"},
            }
        )
    )
    gaz = load_rail_gazetteer(path)
    assert gaz == {"C M ST P R R EVANSTON DIV": ("NORTH SIDE MAIN LINE",)}


def test_chicago_gazetteer_loads_and_binds() -> None:
    gaz = load_rail_gazetteer(GAZETTEER_PATH)
    # every side of every row is already in normalized form on disk
    for label, groups in gaz.items():
        assert label == normalize_rail_name(label)
        assert all(g == normalize_rail_name(g) for g in groups)
    # the p92 golden's label family stays bound via its corridor qualifier;
    # generic CM&StP rows deliberately exclude the el's group — with CTA
    # geometry citywide it bound them ~400 m off that road's actual trackage
    assert gaz["C M ST P R R EVANSTON DIV"] == ("NORTH SIDE MAIN LINE",)
    assert "NORTH SIDE MAIN LINE" not in gaz["C M ST P R R"]
    # the adopted CTA elevated families bind to their modern CTA groups
    assert gaz["UNION EL LOOP R R"] == ("LOOP L",)
    assert gaz["S S ELEVATED R R"] == ("SOUTH SIDE ELEVATED",)
    assert gaz["CHIC OAK PARK EL R R"] == ("LAKE BRANCH",)


# --- disjoint-by-construction (the structural bonus) -------------------------


def test_rail_anchor_plus_street_pair_is_disjoint_by_construction() -> None:
    # a rail anchor rides NO street: one rail anchor + one street-pair
    # anchor always satisfies the disjoint-pair rule
    assert has_disjoint_pair([("RR X", "LAWRENCE"), ("LEAVITT", "MADISON")])
    # ...but two anchors riding the SAME physical railroad share its group
    # name — a rail-only cluster from one railroad stays provisional
    assert not has_disjoint_pair([("RR X", "LAWRENCE"), ("RR X", "LELAND")])


# --- golden: the p92 experiment, replayed end to end ------------------------


@pytest.mark.golden
def test_p92_rail_channel_golden(fixtures_dir: Path, aliases_dir: Path) -> None:
    """Replay the G3 rail-channel experiment from the cached vectors and pin it.

    Pinned numbers are the port's measured output, which reproduces the
    recorded experiment EXACTLY (12 candidates, 7 within the 12 m rescue
    tolerance, best 4.49 m): the cached OSM extract carries a single
    physical railroad ("North Side Main Line"), so grouping by name unions
    the same geometry the experiment's blanket ``unary_union`` did —
    candidate multiplicity is unchanged.
    """
    vol = "sanborn01790_034"
    annotation = json.loads((DATA / "p92_v2_fable.json").read_text())
    rail_index = RailIndex.from_json(DATA / "f5_rail.json", gazetteer_path=GAZETTEER_PATH)
    assert list(rail_index.groups) == ["NORTH SIDE MAIN LINE"]

    manifest = json.loads((fixtures_dir / vol / "sheets" / "manifest.json").read_text())
    scale = manifest["p92"]["scale"]
    aliases = load_aliases(aliases_dir / f"aliases-{vol}.json")
    areas = json.loads((fixtures_dir / "reference" / "community_areas.geojson").read_text())[
        "features"
    ]
    bounds = community_area_bounds(areas, ["UPTOWN", "LINCOLN SQUARE", "EDGEWATER", "LAKE VIEW"])
    index = CenterlineIndex.from_geojson(
        fixtures_dir / "reference" / "street_center_lines.geojson",
        aliases=aliases,
        bounds_4326=bounds,
    )

    cands = rail_crossing_candidates(annotation, rail_index, index, aliases, scale)

    # truth translation from the recorded p92 GCPs at the volume's pinned
    # linear part (scale 0.0676 m/px, rotation 1.13 deg) — same math as the
    # experiment script
    rec = json.loads((fixtures_dir / vol / "results" / "p92.json").read_text())
    gcps = []
    for ft in rec["gcps_geojson"]["features"]:
        if "synthetic" in (ft["properties"].get("note") or ""):
            continue
        px, py = ft["properties"]["image"]
        x, y = TO_3857.transform(*ft["geometry"]["coordinates"])
        gcps.append((float(px), float(py), x, y))
    linear = pinned_linear(0.0676, 1.13)
    t_truth = np.mean(
        [
            (
                x - (linear[0][0] * px + linear[0][1] * py),
                y - (linear[1][0] * px + linear[1][1] * py),
            )
            for px, py, x, y in gcps
        ],
        axis=0,
    )

    errs = []
    for c in cands:
        x, y = TO_3857.transform(*c.world4326)
        t = (
            x - (linear[0][0] * c.pixel[0] + linear[0][1] * c.pixel[1]),
            y - (linear[1][0] * c.pixel[0] + linear[1][1] * c.pixel[1]),
        )
        errs.append(float(math.hypot(t[0] - t_truth[0], t[1] - t_truth[1])))
    errs.sort()

    # the experiment's recorded result, reproduced exactly by the port
    assert len(cands) == 12
    assert sum(e <= TOL_M for e in errs) == 7
    assert errs[0] == pytest.approx(4.487, abs=0.01)
    # the full rescue-grade ladder, pinned (script printed 4.5/5.7/6.6/7.6/
    # 9.2/9.6/11.0 rounded)
    expected = [4.487, 5.725, 6.553, 7.593, 9.248, 9.590, 11.041]
    assert errs[:7] == pytest.approx(expected, abs=0.01)

    # every rail candidate is keyed on the physical group, and one rail
    # anchor + one street-pair anchor is disjoint by construction
    assert all(c.streets[0] == "RR NORTH SIDE MAIN LINE" for c in cands)
    assert {c.streets[1] for c in cands} <= {"LAWRENCE AV.", "LELAND AV.", "WILSON AV."}
    assert has_disjoint_pair([cands[0].streets, ("LEAVITT AV.", "MONTROSE AV.")], aliases)
