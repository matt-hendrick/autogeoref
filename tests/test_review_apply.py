"""Applying reviews: reviewer-verified is committed-grade, and never a statistic.

A verdict is applied against the record it was made on, so a drifted result is
skipped rather than overwritten, and a reject never demotes something already
committed. What lands carries its provenance, drops the auto placement's stale
score, and re-running the apply changes nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autogeoref.report import build_report
from autogeoref.review.apply import apply_reviews
from autogeoref.stages.seam import stage_seam
from autogeoref.volume import (
    STATUS_REVIEWER_VERIFIED,
    is_committed,
    is_reviewer_verified,
    reviewer_result_key,
)
from autogeoref.vouchers import committed_vouch_nodes
from review_support import (
    make_volume,
    ops_translate,
    save_ui_sidecar,
)


def test_apply_adjusted_writes_reviewer_verified_with_provenance(tmp_path: Path) -> None:
    paths = make_volume(tmp_path)
    save_ui_sidecar(paths, "4", verdict="adjusted", ops=ops_translate(-400.0, 0.0))
    summary = apply_reviews(paths, "volX", do_warp=False)
    assert summary["applied"] == ["4"]
    r = json.loads((paths.results / "p4.json").read_text())
    assert r["status"] == STATUS_REVIEWER_VERIFIED
    assert is_committed(r)
    assert r["reviewer_review"]["previous_status"].startswith("REJECTED")
    assert r["reviewer_review"]["ops"] == ops_translate(-400.0, 0.0)
    # the -400 m op exactly undoes the +400 m synthetic error: world coords
    # must land back on the committed neighbor's
    ref = json.loads((paths.results / "p2.json").read_text())
    for a, b in zip(r["gcps_geojson"]["features"], ref["gcps_geojson"]["features"], strict=True):
        assert a["geometry"]["coordinates"] == pytest.approx(b["geometry"]["coordinates"], abs=1e-7)


def test_apply_forgets_the_auto_placements_stale_score(tmp_path: Path) -> None:
    """A reviewer's correction must not be judged by the machine's error.

    The sheets a scoring pass names as badly placed are exactly the ones a reviewer
    is meant to fix. A score left in the sidecar after --apply describes a placement
    that no longer exists, and the next demotion pass would read it and take the
    human's own GCPs out of served evidence.
    """
    from autogeoref.scoring import load_scores, record_digest, write_sidecar

    paths = make_volume(tmp_path)

    def entry(page: str, rmse: float) -> dict[str, Any]:
        # stamped as the scorer stamps it, so the SCORE is what invalidates here
        # and not the digest check doing the work for it
        return {
            "rmse_vs_human_m": rmse,
            "record_sha256": record_digest(paths.results / f"p{page}.json"),
        }

    write_sidecar(
        paths,
        {"gate_m": 15.0, "sources": [], "pages": {"4": entry("4", 20.32), "2": entry("2", 3.1)}},
    )

    save_ui_sidecar(paths, "4", verdict="adjusted", ops=ops_translate(-400.0, 0.0))
    apply_reviews(paths, "volX", do_warp=False)

    fixed = json.loads((paths.results / "p4.json").read_text())
    assert fixed["status"] == STATUS_REVIEWER_VERIFIED
    scores = load_scores(paths)
    assert "4" not in scores, "the AUTO score outlived the placement it described"
    # ...and only that page's: a wholesale invalidation would throw away every
    # other sheet's measurement on one reviewer click
    assert scores["2"] == 3.1


def test_apply_is_idempotent(tmp_path: Path) -> None:
    paths = make_volume(tmp_path)
    save_ui_sidecar(paths, "4", verdict="adjusted", ops=ops_translate(-400.0, 0.0))
    apply_reviews(paths, "volX", do_warp=False)
    first = (paths.results / "p4.json").read_text()
    summary = apply_reviews(paths, "volX", do_warp=False)
    assert summary["already_applied"] == ["4"]
    assert (paths.results / "p4.json").read_text() == first


def test_apply_skips_drifted_results(tmp_path: Path) -> None:
    paths = make_volume(tmp_path)
    save_ui_sidecar(paths, "4", verdict="adjusted", ops=ops_translate(-400.0, 0.0))
    rp = paths.results / "p4.json"
    r = json.loads(rp.read_text())
    r["status"] = "OK (rescued, neighbor-corroborated)"  # the pipeline moved first
    rp.write_text(json.dumps(r))
    summary = apply_reviews(paths, "volX", do_warp=False)
    assert summary["skipped"] == ["4"]
    assert json.loads(rp.read_text())["status"] == "OK (rescued, neighbor-corroborated)"


def test_apply_reject_never_demotes_committed(tmp_path: Path) -> None:
    paths = make_volume(tmp_path)
    save_ui_sidecar(paths, "2", verdict="reject", ops=[], affine=None)
    summary = apply_reviews(paths, "volX", do_warp=False)
    r = json.loads((paths.results / "p2.json").read_text())
    assert r["status"] == "OK"  # dissent recorded, status untouched
    assert r["reviewer_review"]["verdict"] == "reject"
    assert any("REJECTED a committed placement" in w for w in summary["warnings"])


def test_reviewer_verified_gate_semantics(tmp_path: Path) -> None:
    """The three-way contract: vouches, pinned out of seam, out of the stats."""
    paths = make_volume(tmp_path)
    save_ui_sidecar(paths, "4", verdict="adjusted", ops=ops_translate(-400.0, 0.0))
    apply_reviews(paths, "volX", do_warp=False)

    # 1. joins the corroboration voucher pool (committed-grade evidence)
    nodes = committed_vouch_nodes(paths)
    vouching_pages = {p for uses in nodes.values() for p, _ in uses}
    assert "4" in vouching_pages

    # 2. pinned out of the seam solve (the human verified THIS placement)
    stage_seam(paths)
    r = json.loads((paths.results / "p4.json").read_text())
    assert "seam_adjusted" not in r

    # 3. counted apart from every auto-acceptance statistic
    results = {p: json.loads((paths.results / f"p{p}.json").read_text()) for p in ("2", "4", "6")}
    report = build_report("volX", results)
    assert report.reviewer_verified == 1
    assert report.strict_accepted == 1  # p2 only
    assert report.accepted_total == 1  # reviewer placement does NOT inflate it
    assert report.flagged == 1  # p6


def test_pre_rename_owner_spellings_still_read(tmp_path: Path) -> None:
    """Results applied before the owner->reviewer rename keep their semantics.

    Locally applied sidecars (gitignored work/ trees) may carry the old
    "OK (owner-verified)" status and owner_review/owner_mask_px keys; every
    reader accepts both spellings so no local state is invalidated.
    """
    assert is_reviewer_verified("OK (owner-verified)")
    assert is_reviewer_verified(STATUS_REVIEWER_VERIFIED)
    assert not is_reviewer_verified("OK")

    legacy = {"owner_mask_px": [[0, 0], [1, 0], [1, 1]], "owner_review": {"verdict": "accept"}}
    assert reviewer_result_key(legacy, "mask_px") == [[0, 0], [1, 0], [1, 1]]
    assert reviewer_result_key(legacy, "review") == {"verdict": "accept"}
    both = dict(legacy, reviewer_mask_px=[[2, 2], [3, 2], [3, 3]])
    assert reviewer_result_key(both, "mask_px") == [[2, 2], [3, 2], [3, 3]]  # new wins
    assert reviewer_result_key({}, "mask_px") is None

    # report accounting: a legacy-status sheet counts as reviewer-verified,
    # never as an auto-acceptance
    report = build_report("volX", {"1": {"status": "OK (owner-verified)"}})
    assert report.reviewer_verified == 1
    assert report.accepted_total == 0


def test_apply_without_region_says_what_to_rerun(tmp_path: Path) -> None:
    paths = make_volume(tmp_path)
    save_ui_sidecar(paths, "4", verdict="adjusted", ops=ops_translate(-400.0, 0.0))
    summary = apply_reviews(paths, "volX", do_warp=True)
    assert summary["warped"] == []
    assert any("not re-warped" in w for w in summary["warnings"])
    assert summary["rerun_hint"] and "--warp" in summary["rerun_hint"]
