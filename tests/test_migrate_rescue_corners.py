"""Migration contracts for scripts/migrate_rescue_record_corners.py.

The migration exists because a pipeline re-run cannot reach committed rescues
(``stage_rescue`` processes only REJECTED pages), so its safety rests entirely
on these contracts: production corner grammar, reviewer records untouched,
unreconstructible records untouched, dry run writes nothing, idempotent — and
the quadrant rule, pinned here on `_056` p59's real recorded numbers because
that fixture is the one that DISCRIMINATES it from the plausible-but-wrong
closeness rule. The corpus run is history; these
contracts are for the day the script is re-run after an anchor-producer
change.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from autogeoref.affine import TO_4326, fit_affine, gcps_from_geojson, model_determinant
from autogeoref.rescue import is_rescue_model_corner, is_synthetic_gcp, pinned_linear
from autogeoref.volume import (
    STATUS_RESCUE_REVOKED,
    STATUS_RESCUED,
    STATUS_REVIEWER_VERIFIED,
    STATUS_REVIEWER_VERIFIED_LEGACY,
)

_SPEC = importlib.util.spec_from_file_location(
    "migrate_rescue_record_corners",
    Path(__file__).resolve().parents[1] / "scripts" / "migrate_rescue_record_corners.py",
)
migrate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migrate)

SCALE = 0.067
ROT = 0.4
# near Chicago in 3857, so the 4326 round-trip is exercised at real magnitudes
TX, TY = -9760000.0, 5140000.0
FULL = (7000.0, 5500.0)

# anchors along one pixel column — exactly the no-corner shape phase 2 fixes
COLUMN = [(510.0, 700.0), (510.0, 1900.0), (510.0, 3100.0), (510.0, 4300.0)]


def _run(work: Path, configs: Path, *, apply: bool = False) -> None:
    argv = ["migrate_rescue_record_corners.py", "--work", str(work), "--configs", str(configs)]
    if apply:
        argv.append("--apply")
    old = sys.argv
    sys.argv = argv
    try:
        migrate.main()
    finally:
        sys.argv = old


def _model(rot_extra: float = 0.0) -> list[list[float]]:
    lin = pinned_linear(SCALE, ROT + rot_extra)
    return [[TX, lin[0][0], lin[0][1]], [TY, lin[1][0], lin[1][1]]]


def _feature(
    px: float, py: float, m: list[list[float]], note: str = "auto: A x B"
) -> dict[str, Any]:
    x = m[0][0] + m[0][1] * px + m[0][2] * py
    y = m[1][0] + m[1][1] * px + m[1][2] * py
    lng, lat = TO_4326.transform(x, y)
    return {
        "type": "Feature",
        "properties": {"image": [round(px), round(py)], "username": "admin", "note": note},
        "geometry": {"type": "Point", "coordinates": [lng, lat]},
    }


def _volume(tmp_path: Path, vid: str, records: dict[str, dict[str, Any]]) -> Path:
    vdir = tmp_path / "work" / vid
    (vdir / "results").mkdir(parents=True)
    (vdir / "sheets").mkdir()
    (vdir / "volume-constants.json").write_text(
        json.dumps({"scale_m_per_px": SCALE, "rotation_deg": ROT})
    )
    manifest = {f"p{page}": {"full_size": list(FULL)} for page in records}
    (vdir / "sheets" / "manifest.json").write_text(json.dumps(manifest))
    for page, r in records.items():
        (vdir / "results" / f"p{page}.json").write_text(json.dumps(r, indent=2))
    return tmp_path / "work"


def _rescue_record(
    page: str, m: list[list[float]], pixels: list[tuple[float, float]]
) -> dict[str, Any]:
    return {
        "page": page,
        "status": STATUS_RESCUED,
        "rescue_anchors": [["A", "B"]] * len(pixels),
        "gcps_geojson": {
            "type": "FeatureCollection",
            "features": [_feature(px, py, m) for px, py in pixels],
        },
    }


def test_apply_appends_production_grammar_corners(tmp_path: Path) -> None:
    work = _volume(tmp_path, "vol_a", {"7": _rescue_record("7", _model(), COLUMN)})
    _run(work, tmp_path / "configs", apply=True)

    r = json.loads((work / "vol_a" / "results" / "p7.json").read_text())
    feats = r["gcps_geojson"]["features"]
    assert len(feats) == 7
    # the original anchors are byte-identical, in order, ahead of the corners
    expected_anchor_pixels = [[510, 700], [510, 1900], [510, 3100], [510, 4300]]
    assert [f["properties"]["image"] for f in feats[:4]] == expected_anchor_pixels
    corners = feats[4:]
    assert all(is_synthetic_gcp(f) and is_rescue_model_corner(f) for f in corners)
    assert [f["properties"]["note"] for f in corners] == [
        "auto: synthetic x rescue-model-corner"
    ] * 3
    assert [f["properties"]["image"] for f in corners] == [
        [round(FULL[0] * 0.1), round(FULL[1] * 0.1)],
        [round(FULL[0] * 0.9), round(FULL[1] * 0.1)],
        [round(FULL[0] * 0.1), round(FULL[1] * 0.9)],
    ]
    # the record now REPRODUCES the placing model: upright, sub-metre
    m_aug = fit_affine(gcps_from_geojson(r["gcps_geojson"]))
    assert model_determinant(m_aug) < 0
    m_pin = _model()
    for px, py in ((0.0, 0.0), (FULL[0], FULL[1])):
        dx = (m_aug[0][0] + m_aug[0][1] * px + m_aug[0][2] * py) - (
            m_pin[0][0] + m_pin[0][1] * px + m_pin[0][2] * py
        )
        dy = (m_aug[1][0] + m_aug[1][1] * px + m_aug[1][2] * py) - (
            m_pin[1][0] + m_pin[1][1] * px + m_pin[1][2] * py
        )
        assert math.hypot(dx, dy) < 1.0
    # everything else on the record is untouched
    assert r["status"] == STATUS_RESCUED
    assert r["rescue_anchors"] == [["A", "B"]] * 4


def test_dry_run_is_the_default_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work = _volume(tmp_path, "vol_a", {"7": _rescue_record("7", _model(), COLUMN)})
    before = (work / "vol_a" / "results" / "p7.json").read_bytes()
    _run(work, tmp_path / "configs")
    assert "dry-run: 1 records migrated" in capsys.readouterr().out
    assert (work / "vol_a" / "results" / "p7.json").read_bytes() == before


def test_apply_is_idempotent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    work = _volume(tmp_path, "vol_a", {"7": _rescue_record("7", _model(), COLUMN)})
    _run(work, tmp_path / "configs", apply=True)
    once = (work / "vol_a" / "results" / "p7.json").read_bytes()
    capsys.readouterr()
    _run(work, tmp_path / "configs", apply=True)
    # second pass: the record carries corners now, so nothing is selected
    assert "APPLIED: 0 records migrated" in capsys.readouterr().out
    assert (work / "vol_a" / "results" / "p7.json").read_bytes() == once


@pytest.mark.parametrize("status", [STATUS_REVIEWER_VERIFIED, STATUS_REVIEWER_VERIFIED_LEGACY])
def test_reviewer_verified_records_are_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], status: str
) -> None:
    r = _rescue_record("9", _model(), COLUMN)
    r["status"] = status
    work = _volume(tmp_path, "vol_a", {"9": r})
    before = (work / "vol_a" / "results" / "p9.json").read_bytes()
    _run(work, tmp_path / "configs", apply=True)
    out = capsys.readouterr().out
    assert "APPLIED: 0 records migrated" in out
    assert "reviewer-confirmed, untouched (1): vol_a p9" in out
    assert (work / "vol_a" / "results" / "p9.json").read_bytes() == before


def test_unreconstructible_record_is_named_and_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # three anchors whose implied translations disagree by ~100 m steps at
    # EVERY quadrant: no pinned model reconstructs (the _188 pcbd2 shape)
    m = _model()
    feats = [
        _feature(510.0, 700.0, m),
        _feature(
            510.0,
            1900.0,
            [[m[0][0] + 100.0, m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]]],
        ),
        _feature(
            2500.0,
            3100.0,
            [[m[0][0], m[0][1], m[0][2]], [m[1][0] + 200.0, m[1][1], m[1][2]]],
        ),
    ]
    r = {
        "page": "3",
        "status": STATUS_RESCUED,
        "rescue_anchors": [["A", "B"], ["C", "D"], ["E", "F"]],
        "gcps_geojson": {"type": "FeatureCollection", "features": feats},
    }
    work = _volume(tmp_path, "vol_a", {"3": r})
    before = (work / "vol_a" / "results" / "p3.json").read_bytes()
    _run(work, tmp_path / "configs", apply=True)
    out = capsys.readouterr().out
    assert "APPLIED: 0 records migrated" in out
    assert "not reconstructible, untouched (1): vol_a p3" in out
    assert (work / "vol_a" / "results" / "p3.json").read_bytes() == before


def test_match_and_revoked_records_are_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    m = _model()
    match_r = {
        "page": "1",
        "status": "OK",
        "gcps_geojson": {
            "type": "FeatureCollection",
            "features": [_feature(px, py, m) for px, py in COLUMN],
        },
    }
    revoked_r = _rescue_record("2", m, COLUMN)
    revoked_r["status"] = STATUS_RESCUE_REVOKED
    work = _volume(tmp_path, "vol_a", {"1": match_r, "2": revoked_r})
    before1 = (work / "vol_a" / "results" / "p1.json").read_bytes()
    before2 = (work / "vol_a" / "results" / "p2.json").read_bytes()
    _run(work, tmp_path / "configs", apply=True)
    out = capsys.readouterr().out
    assert "APPLIED: 0 records migrated" in out
    # the revoked provisional is counted out loud, never rewritten
    assert "revoked-provisional rescue records without corners" in out
    assert (work / "vol_a" / "results" / "p1.json").read_bytes() == before1
    assert (work / "vol_a" / "results" / "p2.json").read_bytes() == before2


#: `_056` p59's ORIGINAL committed record (pre-migration), verbatim: four
#: anchors along one pixel column plus two rail crossings at one repeated
#: pixel. The set that produced the mirrored refit 434 m from its placement.
P59_SCALE, P59_ROT = 0.06888844659469442, 1.078180700214162
P59_FULL = (5804.0, 8504.0)
P59_ANCHORS = [
    ((308, 266), (-87.66723981780852, 41.71388981528372)),
    ((308, 2500), (-87.66726938512363, 41.712846661036046)),
    ((308, 4284), (-87.66726559824023, 41.71205583286633)),
    ((308, 6308), (-87.66724022497121, 41.71114624530793)),
    ((404, 266), (-87.66729580927795, 41.71388967990309)),
    ((404, 266), (-87.66733573022833, 41.71388958328933)),
]


def test_p59_trap_quadrant_by_clustering_not_by_refit_closeness(tmp_path: Path) -> None:
    """The `_056` p59 regression, on its real recorded numbers.

    This fixture DISCRIMINATES the two candidate quadrant rules: the anchors'
    implied translations cluster at the base quadrant (8.82 m), while the
    bare-anchor refit — a mirrored model — sits closest to the QUADRANT-270
    pin (315.8 m vs 434.3 m at base). Choosing the quadrant by closeness to
    the refit would pin the corners a quarter-turn wrong; the audit's comment
    records the same trap.
    """
    feats = [
        {
            "type": "Feature",
            "properties": {"image": list(px), "username": "admin", "note": "auto: A x B"},
            "geometry": {"type": "Point", "coordinates": list(world)},
        }
        for px, world in P59_ANCHORS
    ]
    r = {
        "page": "59",
        "status": STATUS_RESCUED,
        "rescue_anchors": [["A", "B"]] * len(feats),
        "gcps_geojson": {"type": "FeatureCollection", "features": feats},
    }
    vdir = tmp_path / "work" / "vol_a"
    (vdir / "results").mkdir(parents=True)
    (vdir / "sheets").mkdir()
    (vdir / "volume-constants.json").write_text(
        json.dumps({"scale_m_per_px": P59_SCALE, "rotation_deg": P59_ROT})
    )
    (vdir / "sheets" / "manifest.json").write_text(
        json.dumps({"p59": {"full_size": list(P59_FULL)}})
    )
    (vdir / "results" / "p59.json").write_text(json.dumps(r, indent=2))

    # the bare record is the defect: a MIRRORED refit
    m_bare = fit_affine(gcps_from_geojson(r["gcps_geojson"]))
    assert model_determinant(m_bare) > 0

    _run(tmp_path / "work", tmp_path / "configs", apply=True)

    out = json.loads((vdir / "results" / "p59.json").read_text())
    assert len(out["gcps_geojson"]["features"]) == 9
    m_aug = fit_affine(gcps_from_geojson(out["gcps_geojson"]))
    assert model_determinant(m_aug) < 0
    # the record now reproduces the BASE-quadrant placing model (handoff §5:
    # "under option A it re-records to within 2.04 m of that placement")
    rot = math.degrees(math.atan2(m_aug[1][1], m_aug[0][1]))
    assert abs(rot - P59_ROT) < 1.0
