"""Seam solve, and the human-pin check that now grades it after the fact.

Recorded (fixtures/ref-volume/seam_deltas.json): seam RMS 4.418 -> 3.612 m
over 211 ties, GT median RMSE 7.122 -> 7.327 m, gate PASSED. The port must
reproduce those numbers. The gate itself no longer refuses a solve — it lives in
the scoring pass and reports — but its arithmetic is the same and is pinned here.
"""

import json
from pathlib import Path

import pytest

from autogeoref.score_pass import SEAM_GATE_M, gt_gate
from autogeoref.seam import SheetFit, build_ties, rms, sheet_fit_from_result, solve

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def ref_sheets(fixtures_dir: Path) -> dict[str, SheetFit]:
    sheets: dict[str, SheetFit] = {}
    for f in sorted((fixtures_dir / "ref-volume" / "results").glob("p*.json")):
        r = json.loads(f.read_text())
        if not r.get("layer"):
            continue
        fit = sheet_fit_from_result(r["page"], r)
        if fit is not None:
            sheets[r["page"]] = fit
    return sheets


def test_seam_solve_matches_recorded_and_passes_gate(
    ref_sheets: dict[str, SheetFit], fixtures_dir: Path
) -> None:
    rec = json.loads((fixtures_dir / "ref-volume" / "seam_deltas.json").read_text())
    ties = build_ties(ref_sheets)
    assert len(ties) == rec["ties"]

    deltas, before, after = solve(ref_sheets, ties)
    assert rms(before) == pytest.approx(rec["rms_before_m"], abs=0.02)
    assert rms(after) == pytest.approx(rec["rms_after_m"], abs=0.02)
    # seam mismatch must actually improve
    assert rms(after) < rms(before)

    gt_layers = json.loads(
        (fixtures_dir / "ground-truth" / "api-layers-sanborn01790_006.5.json").read_text()
    )
    g = gt_gate(gt_layers, ref_sheets, deltas)
    assert g is not None
    med_b, med_a, _n, passed = g
    assert med_b == pytest.approx(rec["gt_median_before_m"], abs=0.02)
    assert med_a == pytest.approx(rec["gt_median_after_m"], abs=0.02)
    assert passed  # recorded gate: PASSED
    assert med_a <= med_b + SEAM_GATE_M


def test_gate_fails_on_large_worsening(ref_sheets: dict[str, SheetFit], fixtures_dir: Path) -> None:
    """A delta set that shoves every sheet 5 m must be REPORTED as a worsening.

    Nothing refuses the solve on this verdict any more — the shift is applied and the
    scoring pass says so afterwards — so what is under test is that the arithmetic
    still calls a 5 m shove what it is.
    """
    gt_layers = json.loads(
        (fixtures_dir / "ground-truth" / "api-layers-sanborn01790_006.5.json").read_text()
    )
    bad_deltas = dict.fromkeys(ref_sheets, (5.0, 5.0))
    g = gt_gate(gt_layers, ref_sheets, bad_deltas)
    assert g is not None
    assert not g[3]
