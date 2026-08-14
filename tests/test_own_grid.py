"""The per-sheet own-grid estimator and the rescue fallback it feeds."""

import json
from pathlib import Path
from typing import Any

import pytest

from autogeoref.centerlines import CenterlineIndex
from autogeoref.own_grid import OwnGridEstimator
from autogeoref.rail import RailIndex
from autogeoref.rescue import has_disjoint_pair, pinned_linear, translation_fit
from autogeoref.stages import rescue as rescue_stage
from autogeoref.stages.rescue import OWN_GRID_MIN_AGREE, stage_rescue
from autogeoref.volume import STATUS_RESCUE_REVOKED, STATUS_RESCUED
from test_rescue_wiring import (
    ANNOTATION,
    FEATURES,
    RAIL_GAZETTEER,
    RAIL_OVERPASS,
    ROTATED_ANNOTATION,
    SCALE_M_PER_PX,
    VOL,
    _rot90,
    _street,
    _volume,
)

#: A fourth E-W street, so a quarter-turned sheet can produce a cluster of three
#: that all share the one N-S street — the configuration the disjoint-pair rule
#: distrusts, at a size the disjoint route would accept.
FEATURES_4 = [*FEATURES, _street("DDD", (0, 500), (2000, 500))]
SHARED_ROTATED = {
    "streets": [
        *ROTATED_ANNOTATION["streets"],
        {"name": "DDD", "bbox": _rot90([800, 480, 1200, 520]), "orientation": "vertical"},
    ],
    "page_number_seen": "9",
}


def _index(aliases: dict[str, str] | None = None) -> CenterlineIndex:
    return CenterlineIndex(FEATURES, aliases=aliases or {})


def _rail() -> RailIndex:
    return RailIndex(RAIL_OVERPASS, gazetteer=dict(RAIL_GAZETTEER))


def _status(paths: Any) -> str:
    return str(json.loads((paths.results / "p9.json").read_text())["status"])


def test_estimator_reads_an_upright_sheet_as_the_volume_grid() -> None:
    est = OwnGridEstimator(_index())
    theta = est.estimate(ANNOTATION)
    assert theta is not None
    assert abs(((theta + 90) % 180) - 90) < 1e-6


def test_estimator_reads_a_quarter_turned_sheet_as_a_quarter_turn() -> None:
    est = OwnGridEstimator(_index())
    theta = est.estimate(ROTATED_ANNOTATION)
    assert theta is not None
    assert abs(theta % 180.0 - 90.0) < 1e-6


def test_estimator_abstains_below_the_vote_quorum() -> None:
    thin = {"streets": ANNOTATION["streets"][:2]}
    assert OwnGridEstimator(_index()).estimate(thin) is None


def test_estimator_ignores_non_cardinal_labels() -> None:
    diagonal: dict[str, Any] = {
        "streets": [
            {"name": "AAA", "bbox": [800, 980, 1200, 1020], "orientation": "diagonal"},
            {"name": "BBB", "bbox": [480, 300, 520, 700], "orientation": "diagonal"},
            {"name": "CCC", "bbox": [800, 1480, 1200, 1520], "orientation": "diagonal"},
        ]
    }
    assert OwnGridEstimator(_index()).estimate(diagonal) is None


def test_estimator_skips_a_key_several_streets_share() -> None:
    """A key two differently named streets normalize onto has no single axis."""
    merged = CenterlineIndex(FEATURES, aliases={"BBB": "AAA", "CCC": "AAA"})
    # every label now resolves to AAA, whose segments run both ways
    assert OwnGridEstimator(merged).estimate(ANNOTATION) is None


def test_own_grid_fallback_places_a_quarter_turned_sheet(tmp_path: Path) -> None:
    """What `quadrant_rescue` was written for, without the volume opting in."""
    paths = _volume(tmp_path)
    (paths.annotations / "p9.json").write_text(json.dumps(ROTATED_ANNOTATION))
    rescued, provisional = stage_rescue(paths, _index(), VOL, rail_index=_rail())
    assert rescued == ["9"] and provisional == []
    r = json.loads((paths.results / "p9.json").read_text())
    assert r["status"] == STATUS_RESCUED
    assert r["n_inliers"] >= OWN_GRID_MIN_AGREE
    # the recorded placement carries the sheet's own quarter turn, not the volume's
    assert abs(abs(r["rescue_pin_rotation_deg"]) - 90.0) < 1.0


def test_own_grid_fallback_needs_more_anchors_than_the_volume_pin(tmp_path: Path) -> None:
    """Two anchors commit at the volume's declared pin and not at an estimated one."""
    paths = _volume(tmp_path)
    # upright: the volume pin finds the two shared-street anchors and records
    # them provisional, which is MIN_AGREE = 2 doing its job
    rescued, provisional = stage_rescue(paths, _index(), VOL)
    assert provisional == ["9"] and _status(paths) == STATUS_RESCUE_REVOKED

    # quarter-turned and railless: the same two anchors are all the own grid can
    # find, and the fallback's higher bar refuses them
    paths = _volume(tmp_path / "b")
    (paths.annotations / "p9.json").write_text(json.dumps(ROTATED_ANNOTATION))
    rescued, provisional = stage_rescue(paths, _index(), VOL)
    assert rescued == [] and provisional == []
    assert _status(paths).startswith("REJECTED")


def test_a_shared_street_cluster_from_the_own_grid_takes_one_more_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three anchors on one street commit nothing here, where a disjoint three would.

    That cluster cannot commit on its own — it goes to corroboration, which has
    vouched for a parallel-street slide hundreds of metres long.
    """
    index = CenterlineIndex(FEATURES_4, aliases={})
    paths = _volume(tmp_path)
    (paths.annotations / "p9.json").write_text(json.dumps(SHARED_ROTATED))
    rescued, provisional = stage_rescue(paths, index, VOL)
    assert rescued == [] and provisional == []
    assert _status(paths).startswith("REJECTED (no valid")

    # ...and the SAME cluster is recorded provisional once the bar is one lower,
    # so what refused it is the bar and not some other gate
    paths = _volume(tmp_path / "b")
    (paths.annotations / "p9.json").write_text(json.dumps(SHARED_ROTATED))
    monkeypatch.setattr(rescue_stage, "OWN_GRID_SHARED_STREET_MIN_AGREE", OWN_GRID_MIN_AGREE)
    rescued, provisional = stage_rescue(paths, index, VOL)
    assert rescued == [] and provisional == ["9"]
    r = json.loads((paths.results / "p9.json").read_text())
    assert r["status"] == STATUS_RESCUE_REVOKED
    assert r["n_inliers"] == OWN_GRID_MIN_AGREE


def test_a_disjoint_end_outranks_a_bigger_shared_street_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two ends are 180 deg apart, so size alone must not arbitrate them.

    Ranking by size lets a shared-street cluster displace a disjoint one — the
    eviction ``translation_fit``'s rail guard forbids inside a single fit,
    happening between two of them.
    """
    paths = _volume(tmp_path)
    monkeypatch.setattr(OwnGridEstimator, "estimate", lambda _self, _ann: 47.0)

    # Both ends have to produce a cluster, which no real geometry does on this
    # fixture, so the fit is stubbed: nothing at the volume's pin, then a
    # 3-anchor DISJOINT cluster, then a bigger one entirely on street BBB.
    real = translation_fit
    upright = pinned_linear(SCALE_M_PER_PX, 0.0)
    calls: list[int] = []

    def spy(cands: Any, linear: Any, **kwargs: Any) -> Any:
        calls.append(1)
        m, anchors = real(cands, upright, **kwargs)
        assert m is not None and len(anchors) >= 4, "fixture no longer clusters"
        if len(calls) == 1:
            return None, []
        if len(calls) == 2:
            keep = [c for c in anchors if "BBB" not in c.streets][:1]
            keep += [c for c in anchors if "BBB" in c.streets][:2]
            assert has_disjoint_pair([c.streets for c in keep]), "stub is not disjoint"
            return m, keep
        shared = [c for c in anchors if "BBB" in c.streets]
        assert not has_disjoint_pair([c.streets for c in shared]), "stub is not shared-street"
        return m, (shared * 4)[:4]

    monkeypatch.setattr(rescue_stage, "translation_fit", spy)
    rescued, provisional = stage_rescue(paths, _index(), VOL, rail_index=_rail())
    assert len(calls) == 3, "both ends must have been tried"
    assert rescued == ["9"] and provisional == [], "the disjoint three must beat the shared four"
    r = json.loads((paths.results / "p9.json").read_text())
    assert r["n_inliers"] == 3
    assert r["status"] == STATUS_RESCUED


def test_own_grid_fallback_never_evicts_a_cluster_the_volume_pin_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The arm runs only where the volume's pin found nothing, so it evicts nothing."""
    paths = _volume(tmp_path)
    monkeypatch.setattr(OwnGridEstimator, "estimate", lambda _self, _ann: 47.0)
    rescued, provisional = stage_rescue(paths, _index(), VOL, rail_index=_rail())
    assert rescued == ["9"] and provisional == []
    r = json.loads((paths.results / "p9.json").read_text())
    assert "rescue_pin_rotation_deg" not in r, "the volume's pin placed this page"


def test_own_grid_fallback_stays_out_when_the_sheet_is_on_the_volume_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An estimate that agrees with the volume adds only the upside-down end."""
    paths = _volume(tmp_path)
    (paths.annotations / "p9.json").write_text(json.dumps(ROTATED_ANNOTATION))
    monkeypatch.setattr(OwnGridEstimator, "estimate", lambda _self, _ann: 0.5)
    calls: list[float] = []
    real = rescue_stage._fit_on_own_grid

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(1.0)
        return real(*args, **kwargs)

    monkeypatch.setattr(rescue_stage, "_fit_on_own_grid", spy)
    stage_rescue(paths, _index(), VOL, rail_index=_rail())
    assert calls == [], "an on-grid estimate is not a second hypothesis"

    # and the same sheet with an off-grid estimate DOES reach the fallback,
    # so the assertion above is the gate and not a broken spy
    paths = _volume(tmp_path / "b")
    (paths.annotations / "p9.json").write_text(json.dumps(ROTATED_ANNOTATION))
    monkeypatch.setattr(OwnGridEstimator, "estimate", lambda _self, _ann: 90.0)
    stage_rescue(paths, _index(), VOL, rail_index=_rail())
    assert calls == [1.0]


def test_own_grid_fallback_abstains_when_the_estimator_does(tmp_path: Path) -> None:
    """No estimate, no fallback — the page keeps the shipped outcome."""
    paths = _volume(tmp_path)
    thin = dict(ROTATED_ANNOTATION, streets=ROTATED_ANNOTATION["streets"][:2])
    (paths.annotations / "p9.json").write_text(json.dumps(thin))
    rescued, provisional = stage_rescue(paths, _index(), VOL, rail_index=_rail())
    assert rescued == [] and provisional == []


def test_the_quadrant_retry_is_not_recorded_as_an_own_grid_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An estimate that leads nowhere must not label whatever places the page next."""
    paths = _volume(tmp_path)
    (paths.annotations / "p9.json").write_text(json.dumps(ROTATED_ANNOTATION))
    # off grid, so the arm runs, but at a rotation no cluster can form at
    monkeypatch.setattr(OwnGridEstimator, "estimate", lambda _self, _ann: 47.0)
    vol_q = type(VOL)(
        identifier=VOL.identifier,
        scale_m_per_px=VOL.scale_m_per_px,
        rotation_deg=0.0,
        quadrant_rescue=True,
    )
    rescued, _ = stage_rescue(paths, _index(), vol_q, rail_index=_rail())
    assert rescued == ["9"]
    r = json.loads((paths.results / "p9.json").read_text())
    assert "rescue_pin_rotation_deg" not in r, "the quadrant retry placed this page"
