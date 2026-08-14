"""The review app's payload and save, exercised without a server socket.

The payload is what the UI draws: the served affine, the corners, the sheet's
media URL, and a seed where there is no placement. Save is the gate — it refuses
a verdict on a sheet whose overlay was never painted, a base result that has
drifted, an op log that disagrees with the affine it claims, and a mask whose
dry run against the region fails.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from autogeoref.errors import ReviewError
from autogeoref.review.app import review_queue
from autogeoref.review.apply import apply_reviews
from autogeoref.review.sidecars import (
    sidecar_path,
)
from autogeoref.volume import (
    STATUS_REVIEWER_VERIFIED,
)
from review_support import (
    BASE,
    PIXELS,
    composed_affine,
    gcps_fc,
    look,
    make_app,
    ops_translate,
)

if TYPE_CHECKING:
    from autogeoref.paths import VolumePaths


def test_sheet_payload_shape(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    p = app.sheet_payload("volX", "4")
    assert p["status"].startswith("REJECTED (rescue revoked")
    assert not p["seeded"]
    assert np.asarray(p["affine"]).shape == (2, 3)
    assert len(p["corners_px"]) == 4
    assert p["corners_px"][0] == pytest.approx([0.0, 0.0])
    assert p["small_url"] == "/media/volX/p4_small.jpg"
    assert len(p["gcps"]) == 4


def test_sheet_payload_seeds_placementless_pages(tmp_path: Path) -> None:
    app, paths = make_app(tmp_path)
    paths.constants.write_text(json.dumps({"scale_m_per_px": 0.55, "rotation_deg": 1.0}))
    p = app.sheet_payload("volX", "6")
    assert p["seeded"] and p["affine"] is not None
    # seeded at the committed neighbors' centroid, at the pinned constants
    from autogeoref.affine import model_scales

    assert model_scales(np.asarray(p["affine"]))[0] == pytest.approx(0.55)


def test_sheet_payload_serves_uppercase_suffixed_pages(tmp_path: Path) -> None:
    # the stockyards volume's sheets are p10S..p17S: a lowercase-only page
    # allowlist made every one of them a 400 in the UI (unreviewable)
    app, paths = make_app(tmp_path)
    manifest = json.loads(paths.manifest.read_text())
    manifest["p7S"] = dict(manifest["p4"], file="p7S_small.jpg")
    paths.manifest.write_text(json.dumps(manifest))
    (paths.results / "p7S.json").write_text(
        json.dumps({"page": "7S", "status": "OK (rescued)", "gcps_geojson": gcps_fc(BASE, PIXELS)})
    )
    p = app.sheet_payload("volX", "7S")
    assert p["small_url"] == "/media/volX/p7S_small.jpg"
    assert np.asarray(p["affine"]).shape == (2, 3)


def add_named_sheet(paths: VolumePaths) -> None:
    """Bind a Congested District sheet (`pcbd1`) to the synthetic volume."""
    manifest = json.loads(paths.manifest.read_text())
    manifest["pcbd1"] = dict(manifest["p4"], file="pcbd1_small.jpg")
    paths.manifest.write_text(json.dumps(manifest))
    (paths.results / "pcbd1.json").write_text(
        json.dumps(
            {
                "page": "cbd1",
                "status": "REJECTED (no valid RANSAC model)",
                "gcps_geojson": gcps_fc(BASE, PIXELS),
            }
        )
    )


def test_a_verdict_needs_the_overlay_to_have_been_shown(tmp_path: Path) -> None:
    """The rule the acting console would not ship without
    .

        A verdict asserts a human LOOKED. The server can only know that if it painted the
        sheet, so a POST for a page whose ghost raster it never served is refused — which
        is what stops an accept button wired to a row in a summary table, or a bulk
        "accept all", from recording `reviewer_verified` on sheets nobody ever saw.
    """
    app, paths = make_app(tmp_path)
    payload = app.sheet_payload("volX", "4")  # fetched the placement; drew nothing
    body = {
        "base_result_sha256": payload["base_result_sha256"],
        "verdict": "accept",
        "ops": [],
        "affine": payload["affine"],
        "mask_px": None,
    }
    code, resp = app.save("volX", "4", body)
    assert code == 428 and "never took" in resp["error"].replace("nobody took", "never took")
    assert not sidecar_path(paths, "4").exists()

    # ... and a REJECT is refused too: an unseen reject is the same false assertion
    # pointed the other way, and it is recorded on the result just the same.
    code, _ = app.save("volX", "4", {**body, "verdict": "reject", "affine": None})
    assert code == 428

    # once the overlay is actually painted, the verdict lands
    app.media_path("volX", Path(payload["small_url"]).name)
    code, _ = app.save("volX", "4", body)
    assert code == 200 and sidecar_path(paths, "4").exists()


def test_save_rejects_drift_and_persists_sidecar(tmp_path: Path) -> None:
    app, paths = make_app(tmp_path)
    payload = look(app, "volX", "4")
    ops = ops_translate(-400.0, 0.0)
    body = {
        "base_result_sha256": "not-the-sha",
        "verdict": "adjusted",
        "ops": ops,
        "affine": composed_affine(payload, ops),
        "mask_px": None,
    }
    code, resp = app.save("volX", "4", body)
    assert code == 409 and "changed" in resp["error"]
    body["base_result_sha256"] = payload["base_result_sha256"]
    code, resp = app.save("volX", "4", body)
    assert code == 200 and resp["ok"]
    assert sidecar_path(paths, "4").exists()
    # the queue now shows the saved verdict
    q = review_queue(paths, "volX")
    assert next(e for e in q if e["page"] == "4")["verdict"] == "adjusted"


def test_save_rejects_affine_op_log_mismatch(tmp_path: Path) -> None:
    """The client affine is display math; if it disagrees with the op log the
    reviewer saw one placement and apply would materialize another — refuse."""
    app, paths = make_app(tmp_path)
    payload = look(app, "volX", "4")
    body = {
        "base_result_sha256": payload["base_result_sha256"],
        "verdict": "adjusted",
        "ops": ops_translate(-400.0, 0.0),
        "affine": payload["affine"],  # base, NOT composed: 400 m of drift
        "mask_px": None,
    }
    code, resp = app.save("volX", "4", body)
    assert code == 400 and "op log" in resp["error"]
    assert not sidecar_path(paths, "4").exists()


def test_save_mask_dryrun_skipped_without_region(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    payload = look(app, "volX", "4")
    body = {
        "base_result_sha256": payload["base_result_sha256"],
        "verdict": "adjusted",
        "ops": [],
        "affine": payload["affine"],
        "mask_px": [[100.0, 100.0], [5000.0, 100.0], [5000.0, 8000.0], [100.0, 8000.0]],
    }
    code, resp = app.save("volX", "4", body)
    assert code == 200
    assert resp["dryrun"] == "skipped (no full-res image)"


def test_save_mask_dryrun_failure_rejects_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, paths = make_app(tmp_path)
    paths.regions.mkdir()
    (paths.regions / "volX_p4.jpg").write_bytes(b"\xff\xd8fake")
    import autogeoref.review.app as review_app

    def fake_dryrun(*args: Any, **kwargs: Any) -> tuple[bool, str]:
        return False, "ERROR 1: cutline transformer failed"

    monkeypatch.setattr(review_app, "dryrun_against_region", fake_dryrun)
    payload = look(app, "volX", "4")
    body = {
        "base_result_sha256": payload["base_result_sha256"],
        "verdict": "adjusted",
        "ops": [],
        "affine": payload["affine"],
        "mask_px": [[100.0, 100.0], [5000.0, 100.0], [5000.0, 8000.0]],
    }
    code, resp = app.save("volX", "4", body)
    assert code == 422 and "cutline transformer failed" in resp["error"]
    assert not sidecar_path(paths, "4").exists()  # a rejected save persists nothing


def test_named_page_cbd1_ui_and_save_round_trip(tmp_path: Path) -> None:
    """The valid named sheets are reviewable end to end: queue -> payload ->
    save -> apply. The old numeric-only page rule made them 400s in the UI."""
    app, paths = make_app(tmp_path)
    add_named_sheet(paths)
    q = review_queue(paths, "volX")
    assert "cbd1" in [e["page"] for e in q]
    payload = look(app, "volX", "cbd1")
    assert payload["small_url"] == "/media/volX/pcbd1_small.jpg"
    ops = ops_translate(-400.0, 0.0)
    body = {
        "base_result_sha256": payload["base_result_sha256"],
        "verdict": "adjusted",
        "ops": ops,
        "affine": composed_affine(payload, ops),
        "mask_px": None,
    }
    code, resp = app.save("volX", "cbd1", body)
    assert code == 200 and resp["ok"]
    assert sidecar_path(paths, "cbd1") == paths.root / "review" / "pcbd1.json"
    assert sidecar_path(paths, "cbd1").exists()
    q = review_queue(paths, "volX")
    assert next(e for e in q if e["page"] == "cbd1")["verdict"] == "adjusted"
    summary = apply_reviews(paths, "volX", do_warp=False)
    assert "cbd1" in summary["applied"]
    r = json.loads((paths.results / "pcbd1.json").read_text())
    assert r["status"] == STATUS_REVIEWER_VERIFIED


def test_crop_like_page_ids_stay_rejected(tmp_path: Path) -> None:
    """`10_1`-style crop layers stay out of review: their GCP pixels live in a
    volunteer crop's frame, and a page-shaped id would bind them to the sheet."""
    app, _ = make_app(tmp_path)
    for page in ("10_1", "..", "4/../2"):
        with pytest.raises(ReviewError, match="bad page"):
            app.sheet_payload("volX", page)
        code, resp = app.save("volX", page, {})
        assert code == 400 and "bad page" in resp["error"]


def test_centerlines_clipped_to_volume(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    fc = app.centerlines_payload("volX")
    names = [f["properties"]["name"] for f in fc["features"]]
    assert names == ["MADISON ST"]  # FARAWAY ST clipped away


def test_unknown_volume_refused(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    with pytest.raises(ReviewError):
        app.paths("../../etc")
