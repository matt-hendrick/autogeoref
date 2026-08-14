"""What the payload offers a reviewer beyond the placement itself.

Reference centerlines are clipped to the volume without dropping a street that
merely crosses it, the seed falls back through committed neighbours to a
declared box, and the pipeline's own evidence rides along. A recorded fit that
collapsed to one world point is not a placement, and the payload seeds it rather
than serving an affine that would crash the map.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from autogeoref.affine import (
    TO_3857,
    TO_4326,
    apply_affine,
    fit_affine,
    gcps_from_geojson,
)
from autogeoref.review.app import review_queue
from autogeoref.review.apply import apply_reviews
from autogeoref.review.materialize import (
    apply_ops_point,
    compose_ops,
)
from autogeoref.volume import STATUS_REVIEWER_VERIFIED
from review_support import (
    BASE,
    look,
    make_app,
    make_volume,
    ops_translate,
    save_ui_sidecar,
)

if TYPE_CHECKING:
    from autogeoref.paths import VolumePaths


def test_centerlines_keep_streets_that_start_outside_the_volume(tmp_path: Path) -> None:
    """The bbox filter tests every vertex: a street that CROSSES the volume
    but starts outside it is reference geometry, not clutter — first-vertex
    clipping silently thinned the layer exactly at volume edges."""
    app, _ = make_app(tmp_path)
    lng, lat = TO_4326.transform(*apply_affine(BASE, 2000, 4000))
    fc = json.loads(app.city.centerlines_path.read_text())
    fc["features"].append(
        {
            "type": "Feature",
            "properties": {"street_nam": "CROSSING", "street_typ": "AVE"},
            "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [lng, lat]]},
        }
    )
    app.city.centerlines_path.write_text(json.dumps(fc))
    names = [f["properties"]["name"] for f in app.centerlines_payload("volX")["features"]]
    assert "CROSSING AVE" in names
    assert "FARAWAY ST" not in names  # fully-outside features still clipped


def test_seed_falls_back_to_bounds_bbox_when_nothing_committed(tmp_path: Path) -> None:
    """A volume with ZERO committed sheets is still placeable by hand: the
    seed centers on the configured bounds bbox instead of the (absent)
    committed centroid."""
    from autogeoref.config.model import VolumeConfig

    app, paths = make_app(tmp_path)
    (paths.results / "p2.json").unlink()  # the only committed sheet
    paths.constants.write_text(json.dumps({"scale_m_per_px": 0.55, "rotation_deg": 1.0}))
    lng, lat = TO_4326.transform(*apply_affine(BASE, 2500.0, 4000.0))
    app.city.volumes["volX"] = VolumeConfig(
        identifier="volX", bounds_bbox=(lng - 0.01, lat - 0.01, lng + 0.01, lat + 0.01)
    )
    p = app.sheet_payload("volX", "6")
    assert p["seeded"] and p["affine"] is not None
    m = np.asarray(p["affine"])
    center = apply_affine(m, 5860 / 2, 8505 / 2)
    assert center == pytest.approx(TO_3857.transform(lng, lat), abs=1.0)


def test_seed_without_committed_sheets_or_bbox_still_none(tmp_path: Path) -> None:
    app, paths = make_app(tmp_path)
    (paths.results / "p2.json").unlink()
    paths.constants.write_text(json.dumps({"scale_m_per_px": 0.55, "rotation_deg": 1.0}))
    p = app.sheet_payload("volX", "6")
    assert p["seeded"] and p["affine"] is None


def test_placed_payload_lists_committed_sheets_only(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    out = app.placed_payload("volX")
    assert [e["page"] for e in out["placed"]] == ["2"]  # rejected/unplaced stay out
    entry = out["placed"][0]
    assert entry["small_url"] == "/refmedia/volX/p2_small.jpg"
    assert len(entry["corners"]) == 4
    x, y = apply_affine(BASE, 0.0, 0.0)
    assert entry["corners"][0] == pytest.approx(list(TO_4326.transform(x, y)), abs=1e-6)


def test_refmedia_never_marks_the_overlay_shown(tmp_path: Path) -> None:
    """Neighbor-context rasters must not count as 'the reviewer looked at
    this sheet' — otherwise opening p12 quietly unlocks verdicts on p11."""
    app, _ = make_app(tmp_path)
    payload = app.sheet_payload("volX", "4")
    app.resolve_media("volX", "p4_small.jpg")  # the /refmedia route's resolver
    assert not app.overlay_shown("volX", "4")
    body = {
        "base_result_sha256": payload["base_result_sha256"],
        "verdict": "accept",
        "ops": [],
        "affine": payload["affine"],
        "mask_px": None,
    }
    code, _ = app.save("volX", "4", body)
    assert code == 428
    app.media_path("volX", "p4_small.jpg")  # the /media route's resolver
    assert app.overlay_shown("volX", "4")


def test_apply_deferred_warp_still_says_what_to_rerun(tmp_path: Path) -> None:
    """The HTTP apply (UI button / console) defers warps; its summary must
    still say the back half is owed, or the hint only ever appears on CLI
    runs."""
    paths = make_volume(tmp_path)
    save_ui_sidecar(paths, "4", verdict="adjusted", ops=ops_translate(-400.0, 0.0))
    summary = apply_reviews(paths, "volX", do_warp=False)
    assert summary["applied"] == ["4"]
    assert summary["rerun_hint"] and "--warp" in summary["rerun_hint"]
    # an idempotent re-apply materializes nothing new and hints nothing
    summary2 = apply_reviews(paths, "volX", do_warp=False)
    assert summary2["already_applied"] == ["4"] and summary2["rerun_hint"] is None


def test_sheet_payload_carries_pipeline_evidence(tmp_path: Path) -> None:
    app, paths = make_app(tmp_path)
    rp = paths.results / "p4.json"
    r = json.loads(rp.read_text())
    r.update(
        {
            "n_streets": 12,
            "n_candidates": 80,
            "n_inliers": 4,
            "auto_residuals_m": [1.0, 3.0, "bad", 2.0],
            "junction_snap": {"skipped": "only 0 junctions extracted"},
            "verified_accept": {"votes": {"junction": None}, "accepted": False},
        }
    )
    rp.write_text(json.dumps(r))
    p = app.sheet_payload("volX", "4")
    ev = p["evidence"]
    assert ev["n_streets"] == 12 and ev["n_candidates"] == 80 and ev["n_inliers"] == 4
    assert ev["auto_residuals_m"] == {"n": 3, "max": 3.0, "mean": 2.0}
    assert ev["junction_snap"]["skipped"].startswith("only 0")
    assert ev["verified_accept"]["accepted"] is False
    # absent fields stay absent — the panel shows recorded facts, not nulls
    assert "rescue_anchors" not in ev


def test_pin_fit_decomposition_is_expressible_in_ops() -> None:
    """The UI's pin fit emits scale(c) . rotate(c) . translate over the pair
    centroids. Verify that decomposition reproduces the similarity it fits:
    the existing op vocabulary is closed over pin fits, so the server-side
    compose/apply contract needed no new op type."""
    rng = np.random.default_rng(7)
    src = rng.uniform(-500, 500, (4, 2)) + np.array([-9756000.0, 5139000.0])
    k, theta = 1.07, math.radians(3.5)
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    t = np.array([120.0, -40.0])
    tgt = (k * (rot @ src.T)).T + t
    sm, wm = src.mean(axis=0), tgt.mean(axis=0)
    ops = [
        {"type": "scale", "factor": k, "center_3857": [float(sm[0]), float(sm[1])]},
        {"type": "rotate", "deg": math.degrees(theta), "center_3857": [float(sm[0]), float(sm[1])]},
        {"type": "translate", "dx_m": float(wm[0] - sm[0]), "dy_m": float(wm[1] - sm[1])},
    ]
    for s, w in zip(src, tgt, strict=True):
        assert apply_ops_point(float(s[0]), float(s[1]), ops) == pytest.approx(
            (float(w[0]), float(w[1])), abs=1e-6
        )


def add_degenerate_sheet(paths: VolumePaths) -> None:
    """A rescue-revoked record whose GCPs share ONE world point: spread pixels,
    coincident world — fit_affine_checked's pixel-rank test passes but the fit
    is near-singular (this is the real shape of 'anchors share one street')."""
    manifest = json.loads(paths.manifest.read_text())
    manifest["p8"] = dict(manifest["p4"], file="p8_small.jpg")
    paths.manifest.write_text(json.dumps(manifest))
    lng, lat = TO_4326.transform(*apply_affine(BASE, 2000.0, 4000.0))
    features = [
        {
            "type": "Feature",
            "properties": {"image": [px, py], "username": "admin", "note": "auto: A x B"},
            "geometry": {"type": "Point", "coordinates": [lng, lat]},
        }
        for px, py in ((500, 700), (5000, 800), (4800, 8000), (700, 7600))
    ]
    (paths.results / "p8.json").write_text(
        json.dumps(
            {
                "page": "8",
                "status": "REJECTED (rescue revoked: anchors share one street)",
                "gcps_geojson": {"type": "FeatureCollection", "features": features},
            }
        )
    )


def test_degenerate_recorded_fit_is_not_a_placement(tmp_path: Path) -> None:
    """The collapsed affine crashed MapLibre (every corner is one point) and
    would gate saves on garbage — the payload must seed it instead."""
    from autogeoref.review.materialize import displayable_affine

    app, paths = make_app(tmp_path)
    add_degenerate_sheet(paths)
    r = json.loads((paths.results / "p8.json").read_text())
    from autogeoref.review.materialize import affine_from_record

    assert affine_from_record(r) is not None  # the fit itself succeeds...
    assert displayable_affine(affine_from_record(r)) is None  # ...unusably

    q = review_queue(paths, "volX")
    assert next(e for e in q if e["page"] == "8")["has_placement"] is False

    paths.constants.write_text(json.dumps({"scale_m_per_px": 0.55, "rotation_deg": 1.0}))
    p = app.sheet_payload("volX", "8")
    assert p["seeded"] and p["affine"] is not None
    corners = [apply_affine(np.asarray(p["affine"]), px, py) for px, py in p["corners_px"]]
    spans = [max(c[i] for c in corners) - min(c[i] for c in corners) for i in (0, 1)]
    assert min(spans) > 100  # a real quad again, not one collapsed point


def test_degenerate_record_saves_and_applies_the_seeded_placement(tmp_path: Path) -> None:
    """Ops composed on the SEED must save (no drift refusal against the
    garbage base) and materialize as synthetic corners of the shown affine —
    never as recorded-GCPs-moved-through-ops."""
    app, paths = make_app(tmp_path)
    add_degenerate_sheet(paths)
    paths.constants.write_text(json.dumps({"scale_m_per_px": 0.55, "rotation_deg": 1.0}))
    payload = look(app, "volX", "8")
    assert payload["seeded"]
    ops = ops_translate(120.0, -60.0)
    shown = compose_ops(np.asarray(payload["affine"]), ops)
    body = {
        "base_result_sha256": payload["base_result_sha256"],
        "verdict": "adjusted",
        "ops": ops,
        "affine": [[float(v) for v in row] for row in shown],
        "mask_px": None,
    }
    code, resp = app.save("volX", "8", body)
    assert code == 200, resp
    summary = apply_reviews(paths, "volX", do_warp=False)
    assert "8" in summary["applied"]
    r = json.loads((paths.results / "p8.json").read_text())
    assert r["status"] == STATUS_REVIEWER_VERIFIED
    assert all("synthetic" in f["properties"]["note"] for f in r["gcps_geojson"]["features"])
    refit = fit_affine(gcps_from_geojson(r["gcps_geojson"]))
    assert np.allclose(refit, shown, atol=1e-4)  # what landed IS what was shown
