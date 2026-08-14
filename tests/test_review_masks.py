"""A reviewer's mask has to survive the back half.

The mask is recorded only after its dry run against the region passes, and it is
late-materialized when the region arrives later. From then on the bake honours
it instead of recomputing one, which is the clobber this class of bug kept
reintroducing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autogeoref.affine import (
    TO_3857,
    apply_affine,
    fit_affine,
    gcps_from_geojson,
    invert_affine,
)
from autogeoref.review.apply import apply_reviews
from autogeoref.volume import (
    STATUS_REVIEWER_VERIFIED,
)
from review_support import (
    PENTAGON_PX,
    make_volume,
    sidecar_with_mask,
)


def test_apply_records_reviewer_mask_only_after_dryrun_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autogeoref.review.apply as review_apply

    paths = make_volume(tmp_path)
    paths.regions.mkdir()
    (paths.regions / "volX_p4.jpg").write_bytes(b"\xff\xd8fake")
    sidecar_with_mask(paths, "4")

    monkeypatch.setattr(
        review_apply, "dryrun_against_region", lambda *_a, **_k: (False, "ERROR 1: bad cutline")
    )
    summary = apply_reviews(paths, "volX", do_warp=False)
    r = json.loads((paths.results / "p4.json").read_text())
    assert r["status"] == STATUS_REVIEWER_VERIFIED  # placement still applied
    assert "reviewer_mask_px" not in r  # a rejected mask is never recorded
    assert summary["masks_written"] == []
    assert any("rejected the edited mask" in w for w in summary["warnings"])

    monkeypatch.setattr(review_apply, "dryrun_against_region", lambda *_a, **_k: (True, ""))
    summary = apply_reviews(paths, "volX", do_warp=False)
    r = json.loads((paths.results / "p4.json").read_text())
    assert r["reviewer_mask_px"] == PENTAGON_PX
    assert summary["masks_written"] == ["volX_p4"]
    assert (paths.masks / "volX_p4.geojson").exists()


def test_apply_late_materializes_mask_when_region_arrives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autogeoref.review.apply as review_apply

    paths = make_volume(tmp_path)
    sidecar_with_mask(paths, "4")
    summary = apply_reviews(paths, "volX", do_warp=False)
    assert summary["applied"] == ["4"]
    assert "reviewer_mask_px" not in json.loads((paths.results / "p4.json").read_text())
    assert any("NOT materialized" in w for w in summary["warnings"])

    paths.regions.mkdir()
    (paths.regions / "volX_p4.jpg").write_bytes(b"\xff\xd8fake")
    monkeypatch.setattr(review_apply, "dryrun_against_region", lambda *_a, **_k: (True, ""))
    summary = apply_reviews(paths, "volX", do_warp=False)
    assert summary["already_applied"] == ["4"]  # placement applied exactly once
    r = json.loads((paths.results / "p4.json").read_text())
    assert r["status"] == STATUS_REVIEWER_VERIFIED
    assert r["reviewer_mask_px"] == PENTAGON_PX
    assert summary["masks_written"] == ["volX_p4"]
    # and a THIRD run stays quiet
    summary = apply_reviews(paths, "volX", do_warp=False)
    assert summary["already_applied"] == ["4"] and summary["applied"] == []


def test_stage_masks_honors_reviewer_mask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression this guards: `run --warp` re-detecting the page bounds and
    silently clobbering the reviewer's dry-run-validated crop."""
    import autogeoref.review.apply as review_apply
    from autogeoref.bake.masks import stage_masks

    paths = make_volume(tmp_path)
    paths.regions.mkdir()
    (paths.regions / "volX_p4.jpg").write_bytes(b"\xff\xd8fake")
    sidecar_with_mask(paths, "4")
    monkeypatch.setattr(review_apply, "dryrun_against_region", lambda *_a, **_k: (True, ""))
    apply_reviews(paths, "volX", do_warp=False)

    paths.warped.mkdir()
    (paths.warped / "volX_p4.tif").write_bytes(b"fake-cog")
    monkeypatch.setattr(
        "autogeoref.mask.geometry.detect_page_bounds",
        lambda *_a, **_k: pytest.fail("reviewer mask present — page-bounds detection must not run"),
    )
    monkeypatch.setattr("autogeoref.warp.gdalwarp_cutline_dryrun", lambda *_a, **_k: True)
    stage_masks(paths, "volX")

    written = json.loads((paths.masks / "volX_p4.geojson").read_text())
    ring = written["features"][0]["geometry"]["coordinates"][0]
    r = json.loads((paths.results / "p4.json").read_text())
    m = fit_affine(gcps_from_geojson(r["gcps_geojson"]))
    inv = invert_affine(m)
    back_px = []
    for lng, lat in ring[:-1]:
        x, y = TO_3857.transform(lng, lat)
        back_px.append(apply_affine(inv, x, y))
    assert len(back_px) == len(PENTAGON_PX)
    for got, want in zip(back_px, PENTAGON_PX, strict=True):
        assert got == pytest.approx(want, abs=2.0)  # snap_clean may nudge slightly
