"""Contracts for the extent measurement that says when a pin score is not a displacement.

It exists because a metre figure over two affines of different implied size is a
scale disagreement wearing a distance's units. What must hold: the ratio tracks a
known injected scale, a page the grader scores is COUNTED rather than dropped in
silence, the suspect side is the one departing from its own volume, and volume
membership comes from a config rather than from a directory listing.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from autogeoref.affine import TO_4326
from conftest import load_script

SHEET_W, SHEET_H = 1000, 2000
#: 3857 metres per pixel the synthetic pipeline side always places at.
BASE_SCALE = 0.5
ORIGIN_X, ORIGIN_Y = -9_757_000.0, 5_138_000.0


@pytest.fixture(scope="module")
def audit() -> ModuleType:
    """The measurement library; the audit beside it only prints and bootstraps."""
    return load_script("score_extent.py")


def _gcps(scale: float) -> dict[str, Any]:
    features = []
    for px, py in ((0, 0), (SHEET_W, 0), (0, SHEET_H), (SHEET_W, SHEET_H)):
        lng, lat = TO_4326.transform(ORIGIN_X + px * scale, ORIGIN_Y - py * scale)
        features.append(
            {
                "type": "Feature",
                "properties": {"image": [px, py]},
                "geometry": {"type": "Point", "coordinates": [lng, lat]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _extent() -> list[float]:
    west, south = TO_4326.transform(ORIGIN_X, ORIGIN_Y - SHEET_H * BASE_SCALE)
    east, north = TO_4326.transform(ORIGIN_X + SHEET_W * BASE_SCALE, ORIGIN_Y)
    return [west, south, east, north]


def _volume(
    root: Path,
    gt_dir: Path,
    volume: str,
    pages: dict[str, tuple[float, float]],
    status: str = "OK",
) -> None:
    """Write a synthetic tree, one page per ``{page: (pipeline scale, human scale)}``."""
    tree = root / volume
    (tree / "results").mkdir(parents=True)
    (tree / "sheets").mkdir(parents=True)
    manifest = {}
    layers = []
    for page, (pipeline, human) in pages.items():
        (tree / "results" / f"p{page}.json").write_text(
            json.dumps({"status": status, "gcps_geojson": _gcps(pipeline)})
        )
        manifest[f"p{page}"] = {"full_size": [SHEET_W, SHEET_H]}
        layers.append(
            {
                "slug": f"town_1900_vol_1_p{page}",
                "extent": _extent(),
                "gcps_geojson": _gcps(human),
            }
        )
    (tree / "sheets" / "manifest.json").write_text(json.dumps(manifest))
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / f"api-layers-{volume}.json").write_text(json.dumps(layers))


def test_the_ratio_tracks_an_injected_scale_and_is_1_when_the_sides_agree(
    audit: ModuleType, tmp_path: Path
) -> None:
    """A human side placed at k x the pipeline's reads back as k, both axes."""
    work, gt = tmp_path / "work", tmp_path / "gt"
    _volume(
        work,
        gt,
        "vol_a",
        {"1": (BASE_SCALE, BASE_SCALE), "2": (BASE_SCALE, BASE_SCALE * 0.7)},
    )
    pages = {p.page: p for p in audit.volume_extents("vol_a", work, [gt])}

    assert pages["1"].ratio_w == pytest.approx(1.0, abs=1e-6)
    assert pages["1"].ratio_h == pytest.approx(1.0, abs=1e-6)
    assert pages["2"].ratio_w == pytest.approx(0.7, abs=1e-4)
    assert pages["2"].ratio_h == pytest.approx(0.7, abs=1e-4)
    assert pages["1"].in_band(audit.BAND)
    assert not pages["2"].in_band(audit.BAND)


def test_the_extents_are_true_ground_metres_not_projected_ones(
    audit: ModuleType, tmp_path: Path
) -> None:
    """A projected figure reads a third high at these latitudes and must not be quoted."""
    work, gt = tmp_path / "work", tmp_path / "gt"
    _volume(work, gt, "vol_a", {"1": (BASE_SCALE, BASE_SCALE)})
    page = audit.volume_extents("vol_a", work, [gt])[0]

    projected = SHEET_W * BASE_SCALE
    assert page.pipeline_w_m < projected * 0.8
    assert page.pipeline_w_m == pytest.approx(projected * 0.744, rel=0.01)


def test_the_suspect_is_the_side_that_departs_from_its_own_volume(
    audit: ModuleType, tmp_path: Path
) -> None:
    """One page off in a volume of agreeing pages names the side that moved.

    Both directions, because an off-band page is not evidence against the
    pipeline: the audit's job is to say which side to go and look at.
    """
    work, gt = tmp_path / "work", tmp_path / "gt"
    _volume(
        work,
        gt,
        "vol_a",
        {
            "1": (BASE_SCALE, BASE_SCALE),
            "2": (BASE_SCALE, BASE_SCALE),
            "3": (BASE_SCALE, BASE_SCALE),
            "4": (BASE_SCALE, BASE_SCALE * 0.7),
            "5": (BASE_SCALE * 1.4, BASE_SCALE),
        },
    )
    measured = audit.volume_extents("vol_a", work, [gt])
    audit.attribute_sides(measured)
    pages = {p.page: p for p in measured}

    assert pages["4"].suspect == "human"
    assert pages["5"].suspect == "pipeline"
    assert pages["1"].suspect == "ambiguous"


def test_a_flagged_page_is_never_measured_and_a_page_with_no_pin_is_not_invented(
    audit: ModuleType, tmp_path: Path
) -> None:
    """The population is the score pass's own, so a refused sheet stays out."""
    work, gt = tmp_path / "work", tmp_path / "gt"
    _volume(work, gt, "vol_a", {"1": (BASE_SCALE, BASE_SCALE)}, status="REJECTED")
    assert audit.volume_extents("vol_a", work, [gt]) == []

    _volume(work, gt, "vol_b", {"1": (BASE_SCALE, BASE_SCALE)})
    (work / "vol_b" / "results" / "p2.json").write_text(
        json.dumps({"status": "OK", "gcps_geojson": _gcps(BASE_SCALE)})
    )
    (work / "vol_b" / "sheets" / "manifest.json").write_text(
        json.dumps(
            {"p1": {"full_size": [SHEET_W, SHEET_H]}, "p2": {"full_size": [SHEET_W, SHEET_H]}}
        )
    )
    assert [p.page for p in audit.volume_extents("vol_b", work, [gt])] == ["1"]


def test_a_page_the_grader_scores_and_this_cannot_measure_is_counted_not_dropped(
    audit: ModuleType, tmp_path: Path
) -> None:
    """The scorer's solver grades a degenerate pin set; this one refuses it.

    That difference is the only way the two populations can come apart, so it is
    reported by name — a denominator that moved in silence is the failure this
    whole audit exists to stop.
    """
    work, gt = tmp_path / "work", tmp_path / "gt"
    _volume(work, gt, "vol_a", {"1": (BASE_SCALE, BASE_SCALE), "2": (BASE_SCALE, BASE_SCALE)})
    collinear = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"image": [px, 0]},
                "geometry": {
                    "type": "Point",
                    "coordinates": list(TO_4326.transform(ORIGIN_X + px * BASE_SCALE, ORIGIN_Y)),
                },
            }
            for px in (0, 400, 800)
        ],
    }
    layers = json.loads((gt / "api-layers-vol_a.json").read_text())
    layers[1]["gcps_geojson"] = collinear
    (gt / "api-layers-vol_a.json").write_text(json.dumps(layers))

    unmeasurable: list[str] = []
    measured = audit.volume_extents("vol_a", work, [gt], unmeasurable)

    assert [p.page for p in measured] == ["1"]
    assert unmeasurable == ["vol_a p2"]


def test_a_volume_belongs_to_the_city_that_declares_it_not_to_a_work_root(
    audit: ModuleType, tmp_path: Path
) -> None:
    """Cities share a work root, so a directory scan counts one tree once per city.

    Declaration is what attributes it. A tree no config declares is reported by
    `unclaimed_volumes` rather than silently left out of a corpus figure.
    """
    work, gt = tmp_path / "work", tmp_path / "gt"
    _volume(work, gt, "vol_a", {"1": (BASE_SCALE, BASE_SCALE)})
    _volume(work, gt, "vol_b", {"1": (BASE_SCALE, BASE_SCALE)})
    _volume(work, gt, "vol_undeclared", {"1": (BASE_SCALE, BASE_SCALE)})
    _volume(work, tmp_path / "unused", "vol_unpinned", {"1": (BASE_SCALE, BASE_SCALE)})
    (work / "vol_nothing").mkdir()

    def spec(city: str, declared: str) -> Any:
        config = tmp_path / f"{city}.toml"
        config.write_text(f"[volumes.{declared}]\nbounds_bbox = [0, 0, 1, 1]\n")
        return audit.CitySpec(city=city, config=config, work=work, ground_truth=(gt,))

    one, two = spec("one", "vol_a"), spec("two", "vol_b")

    assert audit.scoreable_volumes(one) == ["vol_a"]
    assert audit.scoreable_volumes(two) == ["vol_b"]
    assert audit.unclaimed_volumes([one, two]) == [f"{work}/vol_undeclared"]


def test_a_configured_city_with_no_recorded_roots_fails_rather_than_vanishing(
    audit: ModuleType, tmp_path: Path
) -> None:
    """A corpus figure that quietly covers one city is this project's classic defect."""
    configs = tmp_path / "configs"
    (configs / "atlantis").mkdir(parents=True)
    (configs / "atlantis" / "atlantis.toml").write_text("[city]\nname = 'Atlantis'\n")

    with pytest.raises(SystemExit, match="atlantis"):
        audit.city_specs(configs)


def test_the_interval_widens_when_one_volume_carries_the_whole_tail(tmp_path: Path) -> None:
    """The bootstrap resamples whole volumes, so a one-volume effect stays uncertain.

    A by-sheet interval would call this settled, which is the reason the driver
    resamples the way it does.
    """
    audit = load_script("audit_score_extent.py")
    work, gt = tmp_path / "work", tmp_path / "gt"
    for volume, bad in (("vol_a", 0), ("vol_b", 0), ("vol_c", 6)):
        pages = {str(i): (BASE_SCALE, BASE_SCALE) for i in range(6)}
        _volume(work, gt, volume, pages)
        for page in list(pages)[:bad]:
            layer = json.loads((gt / f"api-layers-{volume}.json").read_text())
            layer[int(page)]["gcps_geojson"] = _gcps(BASE_SCALE * 1.5)
            (gt / f"api-layers-{volume}.json").write_text(json.dumps(layer))
    measured = [p for v in ("vol_a", "vol_b", "vol_c") for p in audit.volume_extents(v, work, [gt])]
    point, low, high = audit.volume_bootstrap(measured, None)

    assert 0.0 < point < 100.0
    assert low == 0.0
    assert high > point


def test_a_configured_volume_may_belong_to_only_one_city(audit: ModuleType, tmp_path: Path) -> None:
    """Two configs declaring one volume would count its pages twice."""
    configs = tmp_path / "configs"
    for city in ("alpha", "beta"):
        (configs / city).mkdir(parents=True)
        (configs / city / f"{city}.toml").write_text("[volumes.vol_shared]\n")
    audit.CITY_ROOTS["alpha"] = ("work", ("fixtures/ground-truth",))
    audit.CITY_ROOTS["beta"] = ("work", ("fixtures/ground-truth",))
    try:
        with pytest.raises(SystemExit, match="vol_shared"):
            audit.city_specs(configs)
    finally:
        del audit.CITY_ROOTS["alpha"], audit.CITY_ROOTS["beta"]


def test_the_published_interval_is_pinned_to_a_fixed_seed() -> None:
    """A seed change would silently move an interval this project has published."""
    driver = load_script("audit_score_extent.py")

    assert driver.SEED == 20260807
    assert driver.RESAMPLES == 10_000
