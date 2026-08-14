"""Advisory junction-verify stage over the recorded _034 rescue family."""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autogeoref.bounds import community_area_bounds
from autogeoref.verify import stage_junction_verify

pytestmark = [pytest.mark.golden]

VOL = "sanborn01790_034"
AREAS = ["UPTOWN", "LINCOLN SQUARE", "EDGEWATER", "LAKE VIEW"]


class _Paths:
    def __init__(self, root: Path) -> None:
        self.results = root / "results"
        self.sheets = root / "sheets"
        self.manifest = self.sheets / "manifest.json"


@pytest.fixture(scope="module")
def vol_copy(tmp_path_factory: pytest.TempPathFactory, fixtures_dir: Path) -> _Paths:
    root = tmp_path_factory.mktemp("verify034")
    shutil.copytree(fixtures_dir / VOL / "results", root / "results")
    shutil.copytree(fixtures_dir / VOL / "sheets", root / "sheets")
    return _Paths(root)


@pytest.fixture(scope="module")
def verdicts(vol_copy: _Paths, fixtures_dir: Path) -> dict[str, dict[str, Any]]:
    pytest.importorskip("cv2")
    features = json.loads((fixtures_dir / "reference" / "street_center_lines.geojson").read_text())[
        "features"
    ]
    areas = json.loads((fixtures_dir / "reference" / "community_areas.geojson").read_text())[
        "features"
    ]
    bounds = community_area_bounds(areas, AREAS)
    return stage_junction_verify(vol_copy, features, bounds)


def test_scores_only_rescue_family(verdicts: dict[str, dict[str, Any]], vol_copy: _Paths) -> None:
    assert verdicts, "no rescue-family pages scored"
    for page in verdicts:
        r = json.loads((vol_copy.results / f"p{page}.json").read_text())
        assert r["status"] != "OK", "strict accepts must not be scored"
        assert "junction_snap" in r


def test_advisory_only_never_changes_status(
    verdicts: dict[str, dict[str, Any]], vol_copy: _Paths, fixtures_dir: Path
) -> None:
    """Statuses are byte-identical to the fixtures after the stage runs."""
    for f in sorted((fixtures_dir / VOL / "results").glob("p*.json")):
        rec = json.loads(f.read_text())
        mine = json.loads((vol_copy.results / f.name).read_text())
        assert mine["status"] == rec["status"]


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotation_normalized_smalls_are_scored(
    tmp_path: Path, fixtures_dir: Path, verdicts: dict[str, dict[str, Any]], rotation: int
) -> None:
    """The coverage leak this stage used to have: rotated scans got no verdict.

    Rebuild the volume with every small written UPRIGHT, as prep does when it normalizes
    orientation, plus the matching ``rotation_applied`` keys; the stage must compose the
    rotation and reach the same pages instead of skipping them. What is invariant across all
    three turns: the same pages are reached, the same number score, and every DECISIVE verdict
    keeps its verdict — decisive meaning a separation ratio outside the ambiguous band around
    :data:`MIN_SEPARATION`. What is NOT invariant, and must not be asserted: extraction is not
    bit-equivariant under a quarter turn, so a knife-edge page can move.
    """
    cv2 = pytest.importorskip("cv2")
    from autogeoref.junction_snap import MIN_SEPARATION

    rot = tmp_path / f"rot{rotation}"
    (rot / "sheets").mkdir(parents=True)
    shutil.copytree(fixtures_dir / VOL / "results", rot / "results")
    manifest = json.loads((fixtures_dir / VOL / "sheets" / "manifest.json").read_text())
    for key, info in manifest.items():
        if not key.startswith("p"):
            continue
        src = fixtures_dir / VOL / "sheets" / f"{key}_small.jpg"
        if not src.exists():
            continue
        gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        upright = np.rot90(gray, k=-(rotation // 90))  # prep's clockwise correction
        cv2.imwrite(str(rot / "sheets" / f"{key}_small.jpg"), upright)
        info["rotation_applied"] = rotation
        if rotation in (90, 270):
            info["small_size"] = [info["small_size"][1], info["small_size"][0]]
    (rot / "sheets" / "manifest.json").write_text(json.dumps(manifest))

    features = json.loads((fixtures_dir / "reference" / "street_center_lines.geojson").read_text())[
        "features"
    ]
    areas = json.loads((fixtures_dir / "reference" / "community_areas.geojson").read_text())[
        "features"
    ]
    got = stage_junction_verify(_Paths(rot), features, community_area_bounds(areas, AREAS))

    assert set(got) == set(verdicts), "rotated smalls must reach the same pages"
    scored_src = {p for p, v in verdicts.items() if "supports" in v}
    scored_rot = {p for p, v in got.items() if "supports" in v}
    # coverage: the turn may push a page across the MIN_JUNCTIONS floor either
    # way (measured: one each), but it must not cost the stage its evidence
    assert len(scored_rot) >= len(scored_src) - 1, "rotated smalls lost coverage"

    def decisive(v: dict[str, Any]) -> bool:
        # Away from the MIN_SEPARATION knife-edge the verdict (support, or
        # abstain for want of separation) must survive a quarter turn — a
        # verdict a 90 deg turn can flip is not a verdict. Since the channel
        # stopped refuting, this is a support<->abstain stability check.
        ratio = v["separation_ratio"]
        return bool(ratio >= MIN_SEPARATION or ratio <= 1.0 / MIN_SEPARATION)

    checked = 0
    for page in scored_src & scored_rot:
        if not decisive(verdicts[page]):
            continue  # knife-edge: no junction signal to be stable about
        checked += 1
        assert got[page]["supports"] == verdicts[page]["supports"], (
            f"p{page} flipped a DECISIVE verdict under a {rotation} deg turn "
            f"(source {verdicts[page]}, rotated {got[page]}) — the frame composition is wrong"
        )
    assert checked >= 10, "too few decisive pages left to prove anything"


def test_verdict_shape_and_signal(verdicts: dict[str, dict[str, Any]]) -> None:
    scored = {p: v for p, v in verdicts.items() if "supports" in v}
    assert scored, "every page skipped — extraction or wiring broken"
    for v in scored.values():
        assert 0.0 <= v["score_at_prior"] <= 1.0
        assert v["n_junctions"] >= 3
        # CONTRACT: the channel supports or abstains. It does not refute.
        assert v["supports"] in (True, None), f"junction channel refuted: {v}"
    # the measured expectation: correctly-placed rescue-family sheets
    # mostly verify; a blanket abstain would mean the affine/frame wiring is
    # wrong (the recorded _034 rescues were human-QA'd)
    support_rate = sum(1 for v in scored.values() if v["supports"]) / len(scored)
    assert support_rate >= 0.5, f"support rate {support_rate:.2f} — check frame wiring"
