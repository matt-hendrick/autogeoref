"""The declared overview class: district-scale sheets held out of the mosaic.

Two halves, decided by :
the paint half (a per-volume ``overview_pages`` declaration driving mask
clipping, a separate bake artifact nothing serves, and its own report row) and
the seam half (declared overview pages are withheld from ``stage_seam``'s tie
set and solve, with previously applied shifts reverted). Vouching is untouched —
the corroboration pool never reads the declaration.

The bake-side behaviour (mask clip, mosaic partition) is covered in
``test_mosaic_stratification.py``; this file owns the config key, the seam
withdrawal, and the report accounting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autogeoref.affine import TO_3857, TO_4326
from autogeoref.config.load import load_city_config
from autogeoref.config.model import ConfigError
from autogeoref.paths import VolumePaths
from autogeoref.report import build_report
from autogeoref.seam import solve
from autogeoref.stages.seam import stage_seam

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


# ------------------------------------------------------------- config key ----


def test_shipped_config_declares_the_overview_volumes() -> None:
    cfg = load_city_config(CONFIGS / "chicago" / "chicago.toml")
    declared = (
        "sanborn01790_017",
        "sanborn01790_018",
        "sanborn01790_086",  # the 1950 revision of _017's Loop volume
        "sanborn01790_188",
        "sanborn01790_189",
    )
    for vid in declared:
        assert cfg.volume(vid).overview_pages == ("cbd1", "cbd2"), vid
    assert cfg.volume("sanborn01790_016").overview_pages == ("cbda", "cbdb")
    # a regular volume declares nothing and the class is empty
    assert cfg.volume("sanborn01790_024").overview_pages == ()
    assert cfg.volume("sanborn01790_999").overview_pages == ()  # undeclared default


def test_config_rejects_malformed_overview_declarations(tmp_path: Path) -> None:
    """Non-lists, empty ids and non-canonical spellings all fail at load.

    A typo here would silently declare a page no result record can match, so
    the same narrow page grammar review uses is enforced at the boundary.
    """
    head = '[city]\nname = "X"\naliases_dir = "a"\n[volumes.v1]\n'
    for bad in ('"cbd1"', "[1]", '[""]', '["pcbd1"]', '["CBD1"]', '["10_1"]'):
        path = tmp_path / "bad.toml"
        path.write_text(f"{head}overview_pages = {bad}\n")
        with pytest.raises(ConfigError):
            load_city_config(path)


def test_config_accepts_numeric_overview_ids_and_dedupes(tmp_path: Path) -> None:
    # `_190`-style segments carry plain numeric ids: the declaration is the
    # only door for them (the id grammar must never widen — slugs docstring)
    path = tmp_path / "ok.toml"
    path.write_text(
        '[city]\nname = "X"\naliases_dir = "a"\n[volumes.v1]\n'
        'overview_pages = ["1", "2", "cbd1", "1"]\n'
    )
    assert load_city_config(path).volume("v1").overview_pages == ("1", "2", "cbd1")


# -------------------------------------------------------- seam withdrawal ----

#: A tiny consistent world: two detail sheets and one district-scale overview
#: sharing exact centerline nodes. All fits agree, so the solve itself applies
#: nothing — the withdrawal mechanics are what is under test.
M = 10.0  # meters between grid nodes
X0, Y0 = -9_760_000.0, 5_140_000.0


def _fc(points: list[tuple[float, float, float, float]]) -> dict[str, Any]:
    feats = []
    for px, py, x_m, y_m in points:
        lng, lat = TO_4326.transform(X0 + x_m, Y0 + y_m)
        feats.append(
            {
                "type": "Feature",
                "properties": {"image": [px, py]},
                "geometry": {"type": "Point", "coordinates": [lng, lat]},
            }
        )
    return {"type": "FeatureCollection", "features": feats}


def _world_of(fc: dict[str, Any]) -> list[tuple[float, float]]:
    out = []
    for f in fc["features"]:
        lng, lat = f["geometry"]["coordinates"]
        out.append(TO_3857.transform(lng, lat))
    return out


def _volume(tmp_path: Path, overview_status: str = "OK") -> VolumePaths:
    paths = VolumePaths(root=tmp_path / "volume")
    paths.results.mkdir(parents=True)
    # p1 owns the left cell, p2 the right; they share their boundary nodes
    p1 = _fc([(0, 0, 0, 0), (100, 0, M, 0), (100, 100, M, -M), (0, 100, 0, -M)])
    p2 = _fc([(0, 0, M, 0), (100, 0, 2 * M, 0), (100, 100, 2 * M, -M), (0, 100, M, -M)])
    # the overview spans both cells at a quarter of the pixel density and
    # shares nodes with each detail sheet
    cbd = _fc([(0, 0, 0, 0), (50, 0, M, 0), (100, 0, 2 * M, 0), (50, 25, M, -M)])
    (paths.results / "p1.json").write_text(
        json.dumps({"page": "1", "status": "OK", "gcps_geojson": p1})
    )
    (paths.results / "p2.json").write_text(
        json.dumps({"page": "2", "status": "OK", "gcps_geojson": p2})
    )
    (paths.results / "pcbd1.json").write_text(
        json.dumps({"page": "cbd1", "status": overview_status, "gcps_geojson": cbd})
    )
    return paths


def test_declared_overview_pages_are_withheld_from_the_tie_set(tmp_path: Path) -> None:
    paths = _volume(tmp_path)
    record = stage_seam(paths, overview_pages=("cbd1",))
    # withheld, only the two detail-detail ties remain (the boundary nodes
    # (M,0) and (M,-M) that p1 and p2 genuinely share)
    assert record["ties"] == 2
    assert record["overview_withheld"] == ["cbd1"]
    persisted = json.loads(paths.seam_deltas.read_text())
    assert persisted["overview_withheld"] == ["cbd1"]
    # undeclared, the district-scale sheet ties itself to every node it spans:
    # 8 ties, of which 6 involve the overview — the §1.4 shape in miniature
    assert stage_seam(paths)["ties"] == 8


def test_withdrawal_reverts_a_previously_applied_overview_shift(tmp_path: Path) -> None:
    paths = _volume(tmp_path)
    rp = paths.results / "pcbd1.json"
    record = json.loads(rp.read_text())
    unshifted = _world_of(record["gcps_geojson"])
    # as an earlier solve would have left it: world coords moved by the shift
    from autogeoref.seam import shift_gcps_geojson

    shift_gcps_geojson(record["gcps_geojson"], 3.0, -1.0)
    record["seam_adjusted"] = {"dx_m": 3.0, "dy_m": -1.0}
    rp.write_text(json.dumps(record))

    stage_seam(paths, overview_pages=("cbd1",))

    after = json.loads(rp.read_text())
    assert "seam_adjusted" not in after
    for (x, y), (rx, ry) in zip(_world_of(after["gcps_geojson"]), unshifted, strict=True):
        assert abs(x - rx) < 1e-6 and abs(y - ry) < 1e-6
    # idempotent: a second run has nothing left to revert and changes nothing
    before_second = rp.read_text()
    stage_seam(paths, overview_pages=("cbd1",))
    assert rp.read_text() == before_second


def test_overview_only_volume_empties_the_tie_set_safely(tmp_path: Path) -> None:
    # `_016`: every committed sheet is a declared overview page. The stage
    # answers gate N/A before reaching the solve, and still reverts.
    paths = VolumePaths(root=tmp_path / "volume")
    paths.results.mkdir(parents=True)
    fc = _fc([(0, 0, 0, 0), (100, 0, M, 0), (100, 100, M, -M), (0, 100, 0, -M)])
    from autogeoref.seam import shift_gcps_geojson

    shift_gcps_geojson(fc, 5.0, 5.0)
    (paths.results / "pcbda.json").write_text(
        json.dumps(
            {
                "page": "cbda",
                "status": "OK",
                "gcps_geojson": fc,
                "seam_adjusted": {"dx_m": 5.0, "dy_m": 5.0},
            }
        )
    )
    record = stage_seam(paths, overview_pages=("cbda", "cbdb"))
    assert record == {"ties": 0, "overview_withheld": ["cbda"], "gate": "N/A"}
    assert "seam_adjusted" not in json.loads((paths.results / "pcbda.json").read_text())


@pytest.mark.parametrize("pinned_status_name", ["STATUS_CORROBORATED", "STATUS_REVIEWER_VERIFIED"])
def test_pinned_overview_sheet_keeps_its_placement(tmp_path: Path, pinned_status_name: str) -> None:
    # the status skips come first: a corroborated placement was vouched AS IS
    # and a reviewer-verified one was verified AS IS, applied shift included —
    # the withdrawal must not move either (this protects `_188`/`_189`)
    from autogeoref import volume as volume_mod

    pinned_status = getattr(volume_mod, pinned_status_name)
    paths = _volume(tmp_path, overview_status=pinned_status)
    rp = paths.results / "pcbd1.json"
    record = json.loads(rp.read_text())
    record["seam_adjusted"] = {"dx_m": 3.0, "dy_m": -1.0}
    rp.write_text(json.dumps(record))
    result = stage_seam(paths, overview_pages=("cbd1",))
    assert "overview_withheld" not in result
    assert json.loads(rp.read_text())["seam_adjusted"] == {"dx_m": 3.0, "dy_m": -1.0}


def test_uncommitted_overview_sheet_still_loses_its_stale_shift(tmp_path: Path) -> None:
    # a revoked overview page must not carry a leftover solve shift either —
    # it would pin a later reinstatement at the shifted placement. It is
    # reverted but NOT counted as withheld (it was never in the tie set).
    paths = _volume(tmp_path, overview_status="REJECTED (revoked)")
    rp = paths.results / "pcbd1.json"
    record = json.loads(rp.read_text())
    unshifted = _world_of(record["gcps_geojson"])
    from autogeoref.seam import shift_gcps_geojson

    shift_gcps_geojson(record["gcps_geojson"], 3.0, -1.0)
    record["seam_adjusted"] = {"dx_m": 3.0, "dy_m": -1.0}
    rp.write_text(json.dumps(record))

    result = stage_seam(paths, overview_pages=("cbd1",))
    assert "overview_withheld" not in result
    after = json.loads(rp.read_text())
    assert "seam_adjusted" not in after
    for (x, y), (rx, ry) in zip(_world_of(after["gcps_geojson"]), unshifted, strict=True):
        assert abs(x - rx) < 1e-6 and abs(y - ry) < 1e-6


def test_seam_solve_is_empty_safe() -> None:
    # `np.vstack([])` raises; an overview-only volume must answer "no deltas"
    assert solve({}, []) == ({}, [], [])


# ----------------------------------------------------------- report row ----


def test_report_counts_committed_overview_sheets(tmp_path: Path) -> None:
    paths = _volume(tmp_path)
    results = {
        page: json.loads((paths.results / f"p{page}.json").read_text())
        for page in ("1", "2", "cbd1")
    }
    report = build_report("volume", results, overview_pages=("cbd1",))
    assert report.overview_committed == 1
    assert report.committed == 3
    # the row prints only when the class is declared and committed
    from autogeoref.report import report_markdown

    assert "declared overview sheets committed" in report_markdown(report)
    plain = build_report("volume", results)
    assert plain.overview_committed == 0
    assert "declared overview sheets" not in report_markdown(plain)
