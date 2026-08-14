"""Cached full-lifecycle golden replays against frozen fixtures."""

import json
import shutil
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from autogeoref.bounds import load_ground_truth, volume_bounds
from autogeoref.centerlines import CenterlineIndex
from autogeoref.config.model import VolumeConfig
from autogeoref.escalate import stage_escalate
from autogeoref.names import load_aliases
from autogeoref.paths import VolumePaths
from autogeoref.score_pass import score_volume
from autogeoref.scoring import GT_COMMIT_RMSE_M, load_scores, median_rmse
from autogeoref.stages.corroborate import stage_corroborate
from autogeoref.stages.match import stage_match
from autogeoref.stages.rescue import stage_rescue, stage_revoke_shared_street_rescues
from autogeoref.stages.seam import stage_seam
from autogeoref.volume import STATUS_VERIFIED_PREFIX, constraints_from_constants, is_committed
from autogeoref.volume_constants import persisted_constants

pytestmark = pytest.mark.golden


class Replay(NamedTuple):
    """How one frozen volume must be replayed to reproduce its own record."""

    slug_prefix: str
    rail: bool
    #: Did the recorded run escalate? The first group was frozen from runs
    #: with no escalation stage — they carry ZERO `p*.escalated.*.json` — so
    #: escalating them now would reach for a model on the first gated page and
    #: could not reproduce their record anyway. The regression pair was escalated
    #: and its record carries the flips (_040 p12 tier 1, _110 p59 tier 2), so
    #: their replay must escalate — from the frozen tier caches only.
    escalate: bool
    verified: set[str]


TESTBEDS: dict[str, Replay] = {
    "sanborn01790_041": Replay("chicago_ill_1919_vol_22_p", True, False, {"52"}),
    "sanborn01790_089": Replay("chicago_ill_1950_vol_3_p", False, False, {"27", "43", "121"}),
    "sanborn01790_130": Replay("chicago_ill_1950_vol_47_p", False, False, set()),
    "sanborn01790_040": Replay("chicago_ill_1918_vol_21_p", False, True, {"118"}),
    "sanborn01790_110": Replay("chicago_ill_1950_vol_23_p", False, True, {"8", "10", "12", "18"}),
}

#: The city escalation ladder (configs/chicago/chicago.toml), cheapest tier first.
#: :func:`_no_model` is the guarantee that replaying it spends nothing: a cache
#: miss raises instead of reaching for a model, because a test that CAN spend
#: budget is a test that eventually will.
ESCALATION_LADDER = ("claude-sonnet-5", "claude-opus-4-8")

#: The pages whose status DELIBERATELY departs from the frozen record, and the status
#: each must now hold: ``matching.loo_spread_ok`` refuses a strict accept whose
#: perpendicular dimension rests on a single anchor, and each departure here is an
#: improvement against the human GCPs. The fixtures are NOT rewritten — they are the
#: manifest-pinned baseline that proves what the gate did — so any page absent from
#: this map must match its frozen status exactly. A further diff is a regression.
LOO_GATE_DIFFS: dict[str, dict[str, str]] = {
    "sanborn01790_041": {"54": "OK (rescued)"},
    "sanborn01790_089": {"15": "REJECTED (rescue revoked: anchors share one street)"},
    "sanborn01790_130": {"37": "OK (rescued)", "18": "OK (rescued)"},
}

#: The pages the OWN-GRID rescue fallback moves, and what each must now hold. All
#: three are drawn near -57 deg in a volume pinned near 0, so the volume's pin
#: could not reach them and nothing placed them before. p91 is the only one that
#: commits; the scoring assertion below pins it at 8.44 m against the human GCPs,
#: which is what makes this map evidence rather than a rubber stamp.
OWN_GRID_DIFFS: dict[str, dict[str, str]] = {
    "sanborn01790_089": {
        "67": "REJECTED (rescue revoked: anchors share one street)",
        "70": "REJECTED (rescue revoked: anchors share one street)",
        "91": "OK (rescued)",
    },
}

#: What the one own-grid commit must score. Pinned, because the whole argument for
#: the fallback is that it places these sheets WELL, and a status-only assertion
#: would pass just as happily with the sheet a quarter of a mile out.
OWN_GRID_COMMIT_RMSE: dict[str, dict[str, float]] = {"sanborn01790_089": {"91": 8.44}}

#: `_041`'s frozen result records predate the rail gazetteer and include
#: provisional clusters from pairing each printed label with every modern rail
#: group. The fail-closed index used by the replay produces no rail candidates
#: for these unbound labels, so the fabricated provisional records disappear.
#: p14's remaining street-only cluster is likewise provisional. All four stay
#: flagged; no acceptance or served placement changes.
RAIL_GAZETTEER_DIFFS: dict[str, str] = {
    "4": "REJECTED (no valid RANSAC model)",
    "10": "REJECTED (no valid RANSAC model)",
    "14": "REJECTED (rescue revoked: anchors share one street)",
    "20": "REJECTED (no valid RANSAC model)",
}

#: `_041`'s frozen records predate ``centerline_key`` deciding the numbered
#: PLACE/COURT twin BEFORE aliases. Its table renames the numbered street, which
#: used to empty the twin's key into the new name; the twin is back, the volume
#: carries one more index key, and this page places. It is the only golden
#: volume with an alias table, so the only one that can move. The gain is graded
#: against volunteer pins in ``SCORED``, not asserted: p68 lands at 11.68 m.
TWIN_KEY_DIFFS: dict[str, str] = {"68": "OK (rescued)"}


#: What the scoring pass measures on each frozen volume, pinned exactly. A count
#: and a median alone would hold steady while one sheet slid across the volume,
#: so the pages beyond the 15 m commit gate are named — those are the sheets a
#: demotion pass acts on, and three of these volumes have some. `_130` has none,
#: which is the negative control: the gate must not fire where nothing is wrong.
SCORED: dict[str, tuple[int, float, dict[str, float]]] = {
    "sanborn01790_040": (
        27,
        9.22,
        {
            "21": 37.75,
            "97": 26.51,
            "57": 26.23,
            "118": 17.47,
            "87": 17.26,
            "54": 15.71,
            "106": 15.49,
            "51": 15.17,
        },
    ),
    # 15, was 14: the twin guard restores an index key and adds p68 (TWIN_KEY_DIFFS)
    "sanborn01790_041": (15, 8.04, {"46": 17.96}),
    # 17, was 16: the own-grid fallback adds p91 at 8.44 m (OWN_GRID_DIFFS)
    "sanborn01790_089": (17, 6.08, {"40": 30.22, "33": 19.74}),
    "sanborn01790_110": (27, 5.35, {"34": 17.83}),
    "sanborn01790_130": (28, 5.65, {}),
}


def _no_model(image: Path, model: str) -> dict[str, Any]:
    raise AssertionError(f"replay tried to annotate {image.name} with {model}: cache miss")


class _Inputs(NamedTuple):
    """The scratch tree a replay places into, and the indexes its stages read."""

    paths: VolumePaths
    centerlines: CenterlineIndex
    aliases: dict[str, str]
    bounds: tuple[float, float, float, float]
    rail_index: Any | None


def _replay_inputs(
    vid: str, bed: Replay, fixtures_dir: Path, aliases_dir: Path, tmp_path: Path
) -> _Inputs:
    """Copy the frozen annotations and sheets into a scratch tree and index the references."""
    frozen = fixtures_dir / vid

    paths = VolumePaths(root=tmp_path / vid)
    shutil.copytree(frozen / "annotations", paths.annotations)
    shutil.copytree(frozen / "sheets", paths.sheets)
    paths.results.mkdir()

    gt_path = fixtures_dir / "ground-truth" / f"api-layers-{vid}.json"
    gt = load_ground_truth(gt_path, slug_prefix=bed.slug_prefix)
    bounds = volume_bounds(gt)
    aliases_path = aliases_dir / f"aliases-{vid}.json"
    aliases = load_aliases(aliases_path if aliases_path.exists() else None)
    index = CenterlineIndex.from_geojson(
        fixtures_dir / "reference" / "street_center_lines.geojson",
        aliases=aliases,
        bounds_4326=bounds,
    )
    rail_index = None
    if bed.rail:
        from autogeoref.rail import RailIndex

        rail_index = RailIndex.from_json(fixtures_dir / "reference" / "rail-vol22-overpass.json")
    return _Inputs(paths, index, aliases, bounds, rail_index)


def _replay_stages(vid: str, bed: Replay, inputs: _Inputs, fixtures_dir: Path) -> None:
    """Every placement stage the CLI's DAG runs, in DAG order and with no ground truth."""
    paths, index = inputs.paths, inputs.centerlines
    aliases, bounds = inputs.aliases, inputs.bounds

    vcfg = VolumeConfig(identifier=vid)  # constants derived two-pass, as recorded
    stage_match(paths, index, vcfg)
    # escalation sits between match and revoke-stale in the CLI's DAG. It reads
    # the frozen tier caches; a page whose cache is absent never qualified in the
    # recorded run either, so _no_model is unreachable — and fails loudly if the
    # replay ever drifts into wanting a model call.
    if bed.escalate:
        recorded = persisted_constants(paths)
        assert recorded is not None, f"{vid}: stage_match persisted no constants"
        stage_escalate(
            paths,
            index,
            constraints_from_constants(*recorded),
            list(ESCALATION_LADDER),
            annotate_fn=_no_model,
        )
    stage_revoke_shared_street_rescues(paths, aliases)
    # no ground truth reaches any stage, exactly as in the CLI: a run places on its
    # own evidence and the scoring pass grades it afterwards
    stage_rescue(paths, index, vcfg, rail_index=inputs.rail_index, bounds=bounds)
    stage_seam(paths)
    stage_corroborate(paths)

    from autogeoref.geometry import clip_features_4326
    from autogeoref.verified_accept import stage_verified_accept
    from autogeoref.verify import stage_junction_verify

    features = json.loads((fixtures_dir / "reference" / "street_center_lines.geojson").read_text())[
        "features"
    ]
    stage_junction_verify(paths, features, bounds)
    stage_verified_accept(
        paths,
        clip_features_4326(features, bounds),
        aliases,
        address_era="modern",
    )


@pytest.mark.parametrize("vid", sorted(TESTBEDS))
def test_full_lifecycle_replay_matches_frozen_baseline(
    vid: str, fixtures_dir: Path, aliases_dir: Path, tmp_path: Path
) -> None:
    pytest.importorskip("cv2")
    bed = TESTBEDS[vid]
    expected_verified = bed.verified
    frozen = fixtures_dir / vid

    inputs = _replay_inputs(vid, bed, fixtures_dir, aliases_dir, tmp_path)
    paths = inputs.paths
    _replay_stages(vid, bed, inputs, fixtures_dir)

    # every page's status equals the frozen baseline, except the pages the LOO
    # spread gate deliberately moves — those must equal their NEW status exactly,
    # so a fifth page changing (or one of these four changing differently) fails
    # loudly instead of hiding under a relaxed count
    expected_diffs = {
        **LOO_GATE_DIFFS.get(vid, {}),
        **OWN_GRID_DIFFS.get(vid, {}),
        **(RAIL_GAZETTEER_DIFFS if vid == "sanborn01790_041" else {}),
        **(TWIN_KEY_DIFFS if vid == "sanborn01790_041" else {}),
    }
    mismatches: list[str] = []
    verified: set[str] = set()
    for f in sorted((frozen / "results").glob("p*.json")):
        rec: dict[str, Any] = json.loads(f.read_text())
        mine_path = paths.results / f.name
        assert mine_path.exists(), f"{vid} {f.name}: replay produced no record"
        mine = json.loads(mine_path.read_text())
        page = str(mine["page"])
        want = expected_diffs.get(page, rec["status"])
        if mine["status"] != want:
            mismatches.append(f"{f.name}: {mine['status']!r} != expected {want!r}")
        if str(mine["status"]).startswith(STATUS_VERIFIED_PREFIX):
            verified.add(page)
    assert not mismatches, f"{vid}: {len(mismatches)} status diffs vs expected: {mismatches[:5]}"
    assert verified == expected_verified

    # ...and the LOO gate really is what moved its listed pages: each frozen
    # record says OK. (Guards the map against rotting into a rubber stamp for
    # whatever the code happens to do.)
    for page in LOO_GATE_DIFFS.get(vid, {}):
        frozen_status = json.loads((frozen / "results" / f"p{page}.json").read_text())["status"]
        assert frozen_status == "OK", f"{vid} p{page}: frozen status {frozen_status!r}, expected OK"

    if vid == "sanborn01790_041":
        # The gazetteer migration only removes or reshapes provisional records;
        # it must never turn a frozen flagged page into an unreviewed accept.
        for page, current_status in RAIL_GAZETTEER_DIFFS.items():
            frozen_status = json.loads((frozen / "results" / f"p{page}.json").read_text())["status"]
            assert frozen_status.startswith("REJECTED")
            assert current_status.startswith("REJECTED")

    # a run must write NO human score anywhere; the field belongs to the scorer,
    # and a record that carries one is a record a gate could read
    for f in sorted(paths.results.glob("p*.json")):
        scored = "rmse_vs_human_m" in json.loads(f.read_text())
        assert not scored, f"{vid} {f.name}: a run wrote a human score onto a result record"

    # ...and the separate pass grades the finished tree, naming the over-gate accepts
    # rather than withholding them (_089/_130 carry some). They serve until a demotion
    # pass acts on the sidecar — that is the loss taken when GT left the run.
    payload = score_volume(paths, vid, [fixtures_dir / "ground-truth"])
    scores = load_scores(paths)
    assert payload["gate_m"] == GT_COMMIT_RMSE_M
    want_n, want_median, want_over = SCORED[vid]
    assert len(scores) == want_n, f"{vid}: {len(scores)} scored accepts, pinned at {want_n}"
    assert median_rmse(scores) == pytest.approx(want_median, abs=0.05)
    over = {p: round(v, 2) for p, v in scores.items() if v > GT_COMMIT_RMSE_M}
    assert over == want_over, f"{vid}: over the gate {over}, pinned as {want_over}"
    # ...and each of those still COMMITS: the run cannot see a human score, so an
    # over-gate accept vouches, seams and serves until a demotion pass acts on it
    for page in over:
        rec = json.loads((paths.results / f"p{page}.json").read_text())
        assert is_committed(rec), f"{vid} p{page}: an accept must commit on its own funnel"

    # the own-grid fallback's own commits, pinned by SCORE and not only by status
    for page, rmse in OWN_GRID_COMMIT_RMSE.get(vid, {}).items():
        rec = json.loads((paths.results / f"p{page}.json").read_text())
        assert rec.get("rescue_pin_rotation_deg") is not None, (
            f"{vid} p{page}: pinned as an own-grid commit but the volume's pin placed it"
        )
        assert scores[page] == pytest.approx(rmse, abs=0.05), (
            f"{vid} p{page} moved: {scores[page]} m, recorded here as {rmse} m"
        )
