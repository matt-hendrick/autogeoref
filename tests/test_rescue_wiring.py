"""Rescue wiring for rail anchors and street-index priors."""

import json
from pathlib import Path
from typing import Any

import pytest

from autogeoref.addresses import RenumberingTable
from autogeoref.affine import TO_4326
from autogeoref.centerlines import CenterlineIndex
from autogeoref.config.load import load_city_config
from autogeoref.config.model import VolumeConfig
from autogeoref.margins import PriorWindow
from autogeoref.paths import VolumePaths
from autogeoref.rail import RailIndex
from autogeoref.stages.rescue import stage_rescue
from autogeoref.volume import STATUS_RESCUE_REVOKED, STATUS_RESCUED

ROOT = Path(__file__).resolve().parent.parent

X0, Y0 = -9760000.0, 5140000.0
SCALE_M_PER_PX = 2.0


def _w(px: float, py: float) -> tuple[float, float]:
    """World 3857 of a full-res pixel under the test affine (rot 0, y flip)."""
    return (X0 + SCALE_M_PER_PX * px, Y0 - SCALE_M_PER_PX * py)


def _lnglat(px: float, py: float) -> list[float]:
    return list(TO_4326.transform(*_w(px, py)))


def _street(name: str, a: tuple[float, float], b: tuple[float, float]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {"street_nam": name},
        "geometry": {"type": "LineString", "coordinates": [_lnglat(*a), _lnglat(*b)]},
    }


# AAA: E-W along pixel row 1000; BBB: N-S along pixel column 500;
# CCC: E-W along pixel row 1500; the rail: N-S along pixel column 1500.
FEATURES = [
    _street("AAA", (0, 1000), (2000, 1000)),
    _street("BBB", (500, 0), (500, 2000)),
    _street("CCC", (0, 1500), (2000, 1500)),
]

RAIL_OVERPASS = {
    "elements": [
        {
            "type": "way",
            "tags": {"name": "Stock Yards Line"},
            "geometry": [
                {"lon": _lnglat(1500, 0)[0], "lat": _lnglat(1500, 0)[1]},
                {"lon": _lnglat(1500, 2000)[0], "lat": _lnglat(1500, 2000)[1]},
            ],
        }
    ]
}

# The printed-era rail label binds explicitly to its modern geometry group.
RAIL_GAZETTEER = {"C J RY": ("STOCK YARDS LINE",)}

ANNOTATION = {
    "streets": [
        {"name": "AAA", "bbox": [800, 980, 1200, 1020], "orientation": "horizontal"},
        {"name": "BBB", "bbox": [480, 300, 520, 700], "orientation": "vertical"},
        {"name": "CCC", "bbox": [800, 1480, 1200, 1520], "orientation": "horizontal"},
    ],
    "page_number_seen": "9",
    "rail_labels": [{"name": "C.J.RY.", "bbox": [1485, 200, 1515, 600]}],
}


def _volume(tmp_path: Path) -> VolumePaths:
    paths = VolumePaths(root=tmp_path / "vol")
    paths.results.mkdir(parents=True)
    paths.annotations.mkdir()
    paths.sheets.mkdir()
    paths.manifest.write_text(
        json.dumps({"p9": {"full_size": [2000, 2000], "small_size": [2000, 2000], "scale": 1.0}})
    )
    (paths.annotations / "p9.json").write_text(json.dumps(ANNOTATION))
    (paths.results / "p9.json").write_text(
        json.dumps({"page": "9", "status": "REJECTED (no valid RANSAC model)"})
    )
    return paths


VOL = VolumeConfig(identifier="volX", scale_m_per_px=SCALE_M_PER_PX, rotation_deg=0.0)


def test_shared_street_chain_stays_provisional_without_rail(tmp_path: Path) -> None:
    paths = _volume(tmp_path)
    index = CenterlineIndex(FEATURES, aliases={})
    rescued, provisional = stage_rescue(paths, index, VOL)
    assert rescued == [] and provisional == ["9"]
    assert json.loads((paths.results / "p9.json").read_text())["status"] == STATUS_RESCUE_REVOKED


def test_rail_anchor_unlocks_the_same_sheet(tmp_path: Path) -> None:
    paths = _volume(tmp_path)
    index = CenterlineIndex(FEATURES, aliases={})
    rescued, provisional = stage_rescue(
        paths, index, VOL, rail_index=RailIndex(RAIL_OVERPASS, gazetteer=RAIL_GAZETTEER)
    )
    assert rescued == ["9"] and provisional == []
    r = json.loads((paths.results / "p9.json").read_text())
    assert r["status"] == STATUS_RESCUED
    anchor_names = {name for pair in r["rescue_anchors"] for name in pair}
    assert "RR STOCK YARDS LINE" in anchor_names


def test_stage_rescue_passes_aliases_to_the_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the eviction guard tests disjointness inside translation_fit, and it must
    # use the same alias table as the post-fit disjointness decision
    paths = _volume(tmp_path)
    index = CenterlineIndex(FEATURES, aliases={"ROBEY": "DAMEN"})
    from autogeoref.stages import rescue as rescue_stage

    real_fit = rescue_stage.translation_fit
    seen: list[Any] = []

    def spy(cands: Any, linear: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("aliases"))
        return real_fit(cands, linear, **kwargs)

    monkeypatch.setattr(rescue_stage, "translation_fit", spy)
    stage_rescue(paths, index, VOL)
    assert seen and all(a == index.aliases for a in seen)


def _rot90(bbox: list[float], size: float = 2000.0) -> list[float]:
    """The bbox after rotating the sheet 90 deg: (px, py) -> (size - py, px)."""
    x0, y0, x1, y1 = bbox
    return [size - y1, x0, size - y0, x1]


#: the same sheet scanned a quadrant off: every bbox rotated, orientations swapped
ROTATED_ANNOTATION = {
    "streets": [
        {"name": "AAA", "bbox": _rot90([800, 980, 1200, 1020]), "orientation": "vertical"},
        {"name": "BBB", "bbox": _rot90([480, 300, 520, 700]), "orientation": "horizontal"},
        {"name": "CCC", "bbox": _rot90([800, 1480, 1200, 1520]), "orientation": "vertical"},
    ],
    "page_number_seen": "9",
    "rail_labels": [{"name": "C.J.RY.", "bbox": _rot90([1485, 200, 1515, 600])}],
}


def test_quadrant_rescue_wiring_finds_the_rotated_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stage tries the other quadrants only when the volume opts in.

    The own-grid fallback reaches this same sheet from its labels, so it is
    stubbed to abstain here: what is under test is the blind quadrant retry,
    which is what covers a sheet whose labels give no estimate.
    """
    from autogeoref.own_grid import OwnGridEstimator

    monkeypatch.setattr(OwnGridEstimator, "estimate", lambda _self, _ann: None)
    paths = _volume(tmp_path)
    (paths.annotations / "p9.json").write_text(json.dumps(ROTATED_ANNOTATION))
    index = CenterlineIndex(FEATURES, aliases={})
    rail = RailIndex(RAIL_OVERPASS, gazetteer=RAIL_GAZETTEER)

    rescued, provisional = stage_rescue(paths, index, VOL, rail_index=rail)
    assert rescued == [] and provisional == []
    assert json.loads((paths.results / "p9.json").read_text())["status"].startswith("REJECTED")

    vol_q = VolumeConfig(
        identifier="volX", scale_m_per_px=SCALE_M_PER_PX, rotation_deg=0.0, quadrant_rescue=True
    )
    rescued, provisional = stage_rescue(paths, index, vol_q, rail_index=rail)
    assert rescued == ["9"] and provisional == []
    assert json.loads((paths.results / "p9.json").read_text())["status"] == STATUS_RESCUED


def test_index_window_check_is_log_only(tmp_path: Path) -> None:
    paths = _volume(tmp_path)
    index = CenterlineIndex(FEATURES, aliases={})
    far_window = PriorWindow(center_3857=(X0 + 50000.0, Y0), radius_m=600.0)
    rescued, _ = stage_rescue(
        paths,
        index,
        VOL,
        rail_index=RailIndex(RAIL_OVERPASS, gazetteer=RAIL_GAZETTEER),
        index_windows={"9": far_window},
    )
    # still rescued (log-only), but the suspicion is recorded for audit
    assert rescued == ["9"]
    r = json.loads((paths.results / "p9.json").read_text())
    assert r["status"] == STATUS_RESCUED
    assert r["index_window_offset_m"] > 40000


def test_a_rescue_whose_record_would_warp_mirrored_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The commit-time handedness tripwire on the RECORDED GCPs.

    It cannot fire on today's code — ``pinned_linear`` is upright at every quadrant and the
    synthetic corners hold the refit to it — so the anchor producer is stubbed to strip the
    corners and reflect the world points, which is the shape the defect took on `_056` p59: a
    record that does not reproduce the model that placed the sheet. The page must stay flagged
    rather than commit, and must SAY it was refused, or `status` and `report` cannot tell this
    apart from "no cluster agreed".
    """
    from autogeoref.matching import Candidate
    from autogeoref.stages import rescue as rescue_stage

    paths = _volume(tmp_path)
    index = CenterlineIndex(FEATURES, aliases={})

    def mirror(anchors: Any, m: Any, full_size: Any) -> Any:
        # reflect about the world x axis: pixel y stops flipping, so the refit
        # over these correspondences comes out positive-determinant
        return [
            Candidate(
                pixel=c.pixel,
                world4326=TO_4326.transform(*_w(c.pixel[0], -c.pixel[1])),
                streets=c.streets,
            )
            for c in anchors
        ]

    monkeypatch.setattr(rescue_stage, "with_synthetic_corners", mirror)
    rescued, provisional = stage_rescue(
        paths, index, VOL, rail_index=RailIndex(RAIL_OVERPASS, gazetteer=RAIL_GAZETTEER)
    )
    assert rescued == [] and provisional == []
    r = json.loads((paths.results / "p9.json").read_text())
    assert r["status"].startswith("REJECTED")
    assert "gcps_geojson" not in r, "a refused record must not carry a placement"
    assert r["rescue_record_refused"].startswith("+"), r["rescue_record_refused"]


def test_addresses_modern_rejects_non_boolean(tmp_path: Path) -> None:
    """The era declaration accepts TOML booleans only."""
    from autogeoref.config.model import ConfigError

    cfg = tmp_path / "city.toml"
    cfg.write_text(
        '[city]\nname = "X"\ncenterlines = "c.geojson"\naliases_dir = "a"\n'
        '[volumes.v1]\naddresses_modern = "false"\n'
    )
    with pytest.raises(ConfigError):
        load_city_config(cfg)


def test_removed_addresses_post_1909_key_errors_loudly(tmp_path: Path) -> None:
    """The pre-rename key must fail with a rename hint, never be silently
    ignored (a silently dropped era flag would make the channel abstain
    when the config author declared an era)."""
    from autogeoref.config.model import ConfigError

    cfg = tmp_path / "city.toml"
    cfg.write_text(
        '[city]\nname = "X"\ncenterlines = "c.geojson"\naliases_dir = "a"\n'
        "[volumes.v1]\naddresses_post_1909 = true\n"
    )
    with pytest.raises(ConfigError, match="addresses_modern"):
        load_city_config(cfg)


def test_chicago_config_ships_the_partial_renumbering_table() -> None:
    """The configured renumbering table loads and converts known entries."""
    city = load_city_config(ROOT / "configs" / "chicago" / "chicago.toml")
    assert city.renumbering_table_path is not None
    table = RenumberingTable.from_json(city.renumbering_table_path)
    assert len(table.entries) >= 20
    assert table.convert("N. Hermitage Av.", 2464) == 4303
    assert table.convert("HERMITAGE", 2410) == 4211
    assert table.convert("N. Paulina St", 2466) == 4303
    # numbers outside the transcribed ranges abstain — never pass through
    assert table.convert("HERMITAGE", 1200) is None
    assert table.convert("UNKNOWN STREET", 2464) is None
