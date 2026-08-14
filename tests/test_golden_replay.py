"""Golden-volume replay + determinism vs the recorded reference run.

No network, no vision: match and gates over the reference volume's cached annotations,
placed WITHOUT ground truth and scored afterwards — the two steps the shipped flow takes
(``run``, then ``score``). The accuracy assertions say where each sheet LANDS.

**Every difference from the frozen record is enumerated by page, not bounded by a count.**
The recorded results predate the rotation contract, so they include sheets recorded OK at
tens to hundreds of metres that today's gates reject, and quadrant-rotated scans that
passed by batch luck. This port accepts FEWER sheets at match, and the honest way to say
so is a named page set with a reason each; the counts are DERIVED from the frozen sets.

:data:`MATCH_DEPARTURES` is the port's cost, :data:`RESCUE_FAMILY` its payoff, and
:data:`LOO_DEMOTED` the leave-one-out gate's cost alone — a checked slice of the first.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from autogeoref import matching
from autogeoref.affine import fit_affine, gcps_from_geojson, grid_rmse_m
from autogeoref.bounds import load_ground_truth, mercator_correction_lat, volume_bounds
from autogeoref.centerlines import CenterlineIndex
from autogeoref.config.model import VolumeConfig
from autogeoref.matching import FitGates, candidate_gcps
from autogeoref.names import Aliases, load_aliases
from autogeoref.paths import VolumePaths
from autogeoref.score_pass import score_volume
from autogeoref.scoring import (
    GT_COMMIT_RMSE_M,
    load_scores,
    median_rmse,
    score_record_vs_ground_truth,
)
from autogeoref.stages.corroborate import stage_corroborate
from autogeoref.stages.match import stage_match
from autogeoref.stages.rescue import stage_rescue, stage_revoke_shared_street_rescues
from autogeoref.stages.seam import stage_seam
from autogeoref.volume import (
    SheetInput,
    derive_constraints,
    is_committed,
    match_sheet,
)

pytestmark = pytest.mark.golden


def run_volume(
    sheets: list[SheetInput],
    index: CenterlineIndex,
    aliases: Aliases | None = None,
) -> dict[str, dict[str, Any]]:
    """Two-pass volume match: derive constraints, then gate. ``{page: record}``."""
    constraints = derive_constraints(sheets, index, aliases)
    return {sheet.page: match_sheet(sheet, index, constraints, aliases) for sheet in sheets}


def accepted_pages(results: dict[str, dict[str, Any]]) -> list[str]:
    """Pages whose status is any OK variant."""
    return [p for p, r in results.items() if str(r.get("status", "")).startswith("OK")]


GT_VOLUME = "sanborn01790_006.5"
GT_FILE = f"api-layers-{GT_VOLUME}.json"
GT_PREFIX = "chicago_ill_1895_vol_16_p"

#: Strict accepts on this volume BEFORE the leave-one-out spread gate shipped.
PRE_GATE_ACCEPTED = 72

#: EVERY page the recorded run accepted that this port does NOT accept at the match
#: stage, mapped to the SET of gates that, peeled ALONE, would let a model through —
#: "loo" the leave-one-out spread gate, "rot" the quadrant-folded rotation window,
#: "scale" the scale window. An empty set means no single peel suffices. These are
#: the whole of the difference and nothing else may appear here; the comment on each
#: row is the recorded RMSE vs the human GCPs -> what this port does with the page.
LOO, ROT, SCALE = "loo", "rot", "scale"
MATCH_DEPARTURES: dict[str, frozenset[str]] = {
    "9": frozenset({SCALE}),  # 15.32 m -> rescued at 6.80 m
    "16": frozenset({ROT}),  # 3.44 m -> rescued at 2.35 m
    "22": frozenset({LOO}),  # 23.56 m -> refused at match, back through rescue at 5.35 m
    "32": frozenset({SCALE}),  # 27.30 m -> rescued at 5.59 m
    "35": frozenset({LOO}),  # 7.67 m -> rescued at 8.63 m
    "59": frozenset({LOO, SCALE}),  # 59.29 m -> rescued at 3.98 m
    "64": frozenset({ROT}),  # 12.57 m -> rescued at 4.38 m
    "75": frozenset(),  # 186.76 m -> rescue REVOKED (anchors share one street)
    "80": frozenset({LOO, SCALE}),  # 13.53 m -> rescued at 4.16 m
    "87": frozenset({LOO, SCALE}),  # 64.06 m -> rescue REVOKED. An escape; gone
    "90": frozenset(),  # 83.91 m -> corroborated at 20.32 m, which is beyond the commit gate
    "97": frozenset({LOO}),  # 53.56 m -> rescued at 8.72 m
}

#: The six pages the leave-one-out spread gate demotes, measured as a diff of two real runs. One
#: was an escape (60.1 m) and is simply gone; the other five come back through rescue at better
#: accuracy, which is the entire argument for the gate and is why
#: :func:`test_loo_demoted_good_sheets_re_enter_via_rescue` exists: without it this suite would
#: assert the gate's cost and leave its payoff undefended.
LOO_DEMOTED = {"87", "22", "80", "35", "97", "59"}

#: ...and of those, the five rescue must give back. p22 comes back on the sheet's OWN grid,
#: which is why it joined late: at 23.56 m it was written off as an escape.
LOO_RE_ENTERED = {"80", "35", "97", "59", "22"}

#: The full pipeline's answer to :data:`MATCH_DEPARTURES`: every non-strict accept on
#: this volume, with its status and its score against the human GCPs. Most departures
#: return here, each closer to the human GCPs than the recorded run got — the port
#: rejects those sheets at match and then places them BETTER. Four rows are placements
#: the recorded run never made at all. Pinned EXACTLY, since a length floor would pass
#: just as happily while three of these silently disappeared; ``was`` marks a rebase.
RESCUE_FAMILY: dict[str, tuple[str, float]] = {
    "9": ("OK (rescued)", 6.80),  # was 7.93
    "16": ("OK (rescued)", 2.35),  # was 2.96
    # placed on the sheet's OWN grid: a quarter turn from a volume pinned near 0,
    # so the volume's pin could not reach it. The frozen record holds it as a
    # strict accept at 23.56 m; the LOO gate refuses that fit, and this arm gives
    # the page back with six anchors, a disjoint pair, and a turn within 0.06 deg
    # of what the human GCPs imply.
    "22": ("OK (rescued)", 5.35),
    "32": ("OK (rescued)", 5.59),
    "35": ("OK (rescued)", 8.63),  # was 7.70
    "59": ("OK (rescued)", 3.98),  # was 5.17
    "64": ("OK (rescued)", 4.38),
    "80": ("OK (rescued)", 4.16),  # was 4.04
    "82": ("OK (rescued, neighbor-corroborated)", 4.26),
    "86": ("OK (rescued, neighbor-corroborated)", 7.37),
    "90": ("OK (rescued, neighbor-corroborated)", 20.32),
    "92": ("OK (rescued)", 4.73),  # was 6.17
    "93": ("OK (rescued, neighbor-corroborated)", 5.59),
    "97": ("OK (rescued)", 8.72),  # was 5.53
}

#: The one rescue-family accept beyond the 15 m commit gate. p90 corroborates — its
#: neighbors vouch for it at >=2 shared nodes within 8 m — and is still 20.3 m from the
#: human GCPs: a coherent local shift they happily agree with. It COMMITS regardless;
#: only a demotion pass reading the sidecar can take it out of served evidence.
RESCUE_FAMILY_OVER_GATE = {"90"}

#: WHERE an accept lands, pinned per page — not bounded by a median. This port
#: reproduces the recorded run's RMSE exactly on all but these two, and both are the
#: port landing the sheet CLOSER to the human GCPs. Enumerating them is what makes the
#: rest assertable: an acceptance set and a median can both hold steady while one sheet
#: slides across the volume and commits anyway, one centimetre inside the gate.
STRICT_RMSE_DIFFS: dict[str, tuple[float, float]] = {
    "79": (10.04, 4.78),  # record -> here
    "95": (4.95, 4.75),
}

#: The most a strict accept's SCORE may differ from the frozen record's without the
#: sheet having moved. Measured over all 66 strict accepts, not assumed: 0.03 m,
#: and the pre-existing latitude restatement is INSIDE that (the frozen values
#: already carry the 41.8 latitude), so it is not added on top.
STRICT_RESTATEMENT_M = 0.03

#: ...and where a strict accept lands AFTER the seam solve, which is what actually
#: serves. :data:`STRICT_RMSE_DIFFS` above is measured on the match-only replay, so
#: nothing pinned the SERVED score of a strict accept until this: a rescue record
#: changing shifts the fit its neighbours' ties are measured against, and no page-level
#: test could see it. Two of these moved with the always-corners change and two did
#: not, so the pin proves it can hold still as well as follow.
POST_SEAM_STRICT_RMSE: dict[str, float] = {
    "91": 10.60,
    "94": 4.73,
    "79": 4.77,
    "95": 4.75,
}


@pytest.fixture(scope="module")
def ref_setup(fixtures_dir: Path, aliases_dir: Path) -> dict[str, Any]:
    aliases = load_aliases(aliases_dir / "aliases-sanborn01790_006.5.json")
    gt = load_ground_truth(
        fixtures_dir / "ground-truth" / GT_FILE,
        slug_prefix=GT_PREFIX,
    )
    index = CenterlineIndex.from_geojson(
        fixtures_dir / "reference" / "street_center_lines.geojson",
        aliases=aliases,
        bounds_4326=volume_bounds(gt),
    )
    manifest = json.loads((fixtures_dir / "ref-volume" / "sheets" / "manifest.json").read_text())
    sheets = []
    for ann_path in sorted((fixtures_dir / "ref-volume" / "annotations").glob("p*.json")):
        page = ann_path.stem.lstrip("p")
        info = manifest.get(f"p{page}")
        if info is None:
            continue
        sheets.append(
            SheetInput(
                page=page,
                annotation=json.loads(ann_path.read_text()),
                full_size=tuple(info["full_size"]),
                scale=info["scale"],
            )
        )
    return {"aliases": aliases, "gt": gt, "index": index, "sheets": sheets, "manifest": manifest}


@pytest.fixture(scope="module")
def recorded(fixtures_dir: Path) -> dict[str, dict[str, Any]]:
    """The frozen record: the reference run's own result files, keyed by page."""
    rec_dir = fixtures_dir / "ref-volume" / "results"
    return {p.stem.lstrip("p"): json.loads(p.read_text()) for p in rec_dir.glob("p*.json")}


def _score(ref_setup: dict[str, Any], results: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Grade an in-memory replay through the shipped scoring engine.

    ``run_volume`` is the match stage alone and writes no volume tree, so there is
    no sidecar to read — but the numbers still have to be the sidecar's, so this
    calls the same leaf ``score_volume`` calls for every page it grades.
    """
    gt, manifest = ref_setup["gt"], ref_setup["manifest"]
    lat = mercator_correction_lat(gt)
    scored: dict[str, float] = {}
    for page in accepted_pages(results):
        rmse = score_record_vs_ground_truth(
            results[page], manifest.get(f"p{page}"), gt.get(page), lat
        )
        if rmse is not None:
            scored[page] = round(rmse, 2)
    return scored


@pytest.fixture(scope="module")
def ref_results(ref_setup: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return run_volume(ref_setup["sheets"], ref_setup["index"], aliases=ref_setup["aliases"])


@pytest.fixture(scope="module")
def ref_scores(
    ref_setup: dict[str, Any], ref_results: dict[str, dict[str, Any]]
) -> dict[str, float]:
    return _score(ref_setup, ref_results)


@pytest.fixture(scope="module")
def ref_results_pre_gate(ref_setup: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The same replay with the leave-one-out spread gate disabled.

    Neutralizing the gate at its helper, rather than giving the replay helper
    above an opt-out flag, keeps the switch out of everything the replay drives —
    ``derive_constraints`` and ``match_sheet`` are the production matcher. This is
    the BEFORE arm: the demotion set is then a measured diff of two real runs, not
    a hard-coded belief about one.
    """
    original = matching.loo_spread_ok
    matching.loo_spread_ok = lambda *_args: True
    try:
        return run_volume(ref_setup["sheets"], ref_setup["index"], aliases=ref_setup["aliases"])
    finally:
        matching.loo_spread_ok = original


def test_loo_spread_gate_demotes_exactly_the_measured_pages(
    ref_results: dict[str, dict[str, Any]],
    ref_results_pre_gate: dict[str, dict[str, Any]],
) -> None:
    """The gate's cost, stated as an exact page set rather than a slackened bound."""
    before = set(accepted_pages(ref_results_pre_gate))
    after = set(accepted_pages(ref_results))
    assert len(before) == PRE_GATE_ACCEPTED, f"pre-gate acceptance moved: {len(before)}"
    assert before - after == LOO_DEMOTED, f"demoted {sorted(before - after)}"
    assert after - before == set(), "the gate may only REMOVE accepts, never add one"

    # ...and the demotion set is exactly the "loo" slice of the departure table, so
    # the two frozen sets cannot drift apart while each keeps passing its own test.
    assert {p for p, gates in MATCH_DEPARTURES.items() if LOO in gates} == LOO_DEMOTED


def test_departure_gates_are_the_measured_ones(ref_setup: dict[str, Any]) -> None:
    """The gate named against each departure is MEASURED, not asserted.

    Peel one gate at a time off pass-2 RANSAC and see which peels admit a model.
    Without this the gate column is a comment: a wrong label could never fail, and
    the first version of the table did carry three (it named a single culprit for
    p59/p80/p87, which each fall to the LOO gate and the scale window alike).
    """
    sheets = {s.page: s for s in ref_setup["sheets"]}
    cons = derive_constraints(ref_setup["sheets"], ref_setup["index"], ref_setup["aliases"])
    wide = (1e-6, 1e6)  # a scale window so wide it cannot reject

    index, aliases = ref_setup["index"], ref_setup["aliases"]
    for page, expected in MATCH_DEPARTURES.items():
        sheet = sheets[page]
        cands = candidate_gcps(sheet.annotation, index, sheet.scale, aliases)

        def model(
            scale_range: tuple[float, float],
            rot: tuple[float, float] | None,
            loo: bool,
            cands: list[Any] = cands,
            sheet: SheetInput = sheet,
        ) -> bool:
            m, _ = matching.ransac_affine(
                cands,
                sheet.full_size,
                scale_range=scale_range,
                rot_range_deg=rot,
                rot_quadrant_fold=True,
                gates=FitGates(loo_spread=loo),
            )
            return m is not None

        assert not model(cons.scale_range, cons.rot_range_deg, True), (
            f"p{page} is listed as a departure but the shipped gates admit a model"
        )
        admits = {
            gate
            for gate, ok in (
                (LOO, model(cons.scale_range, cons.rot_range_deg, False)),
                (ROT, model(cons.scale_range, None, True)),
                (SCALE, model(wide, cons.rot_range_deg, True)),
            )
            if ok
        }
        assert admits == expected, (
            f"p{page}: single-gate peels admitting a model = {sorted(admits)}, "
            f"frozen as {sorted(expected)}"
        )


def test_match_departures_are_exactly_enumerated(
    ref_results: dict[str, dict[str, Any]], recorded: dict[str, dict[str, Any]]
) -> None:
    """Every difference from the frozen record is NAMED — no aggregate floor.

    The old criterion was a count (">=75 of 96", then ``len(accepted) == 66``),
    and a count is where a regression hides: one page can break while another
    improves and the number never moves. Here a departure nobody wrote down is a
    failure, and so is a departure that has silently HEALED — an unexplained
    return is a placement appearing from nowhere, which on this corpus has never
    once been good news.
    """
    accepted = set(accepted_pages(ref_results))
    recorded_ok = {p for p, r in recorded.items() if str(r.get("status", "")).startswith("OK")}

    departed = recorded_ok - accepted
    assert departed == set(MATCH_DEPARTURES), (
        f"unenumerated departures: {sorted(departed - set(MATCH_DEPARTURES), key=int)}; "
        f"enumerated but no longer departing: "
        f"{sorted(set(MATCH_DEPARTURES) - departed, key=int)}"
    )

    # The anti-confidently-wrong invariant, unchanged: the port may accept FEWER
    # sheets than the record, never more. A higher acceptance rate is a red flag.
    assert accepted - recorded_ok == set(), (
        f"accepted here but REJECTED in the record: {sorted(accepted - recorded_ok, key=int)}"
    )


def test_strict_accept_accuracy_is_pinned_per_page(
    ref_results: dict[str, dict[str, Any]],
    ref_scores: dict[str, float],
    recorded: dict[str, dict[str, Any]],
) -> None:
    """Not just WHICH sheets are accepted, but WHERE each one lands.

    The acceptance sets above would all still pass while a strict accept quietly
    slid from 4 m to 14.9 m — inside the commit gate, so it would serve, and the
    median would barely twitch. Every strict accept is therefore held to the
    recorded run's own score for that page, with the two improvements enumerated.
    """
    for page in accepted_pages(ref_results):
        mine = ref_scores[page]
        # the frozen record still CARRIES the field it was written with; nothing
        # produces one any more, and rewriting the fixture to tidy that would
        # destroy a recorded measurement (fixtures are read-only)
        rec = recorded[page]["rmse_vs_human_m"]
        if page in STRICT_RMSE_DIFFS:
            want_rec, want_mine = STRICT_RMSE_DIFFS[page]
            assert rec == pytest.approx(want_rec, abs=0.01), f"p{page}: the RECORD moved"
            assert mine == pytest.approx(want_mine, abs=0.01), f"p{page}: {mine} m, was {want_mine}"
            assert mine < rec, f"p{page} is enumerated as an improvement but is not one"
        else:
            # Two RESTATEMENTS, neither of which moves a sheet: the record reports
            # meters at its run's frozen 41.8 latitude while the replay derives the
            # volume's own (41.7969 — a larger cos), and the score is now taken from
            # the EXPORTED GCPs, the points a warp uses, rather than the in-memory
            # RANSAC model. Their combined effect is what STRICT_RESTATEMENT_M
            # measures; anything past it is a real departure.
            assert abs(round(mine - rec, 2)) <= STRICT_RESTATEMENT_M, (
                f"p{page} lands at {mine} m; the record put it at {rec} m. An unenumerated "
                f"accuracy departure — add it to STRICT_RMSE_DIFFS only if it is an IMPROVEMENT"
            )


def test_acceptance_and_accuracy(
    ref_results: dict[str, dict[str, Any]],
    ref_scores: dict[str, float],
    recorded: dict[str, dict[str, Any]],
) -> None:
    """The counts, DERIVED from the record and the departure table — never asserted.

    Writing ``== 66`` here is what let the aggregate rot in the first place: the
    number is a consequence of the frozen sets, so it is computed from them and
    can only move when a set does (and a set can only move past the test above).
    """
    recorded_ok = {p for p, r in recorded.items() if str(r.get("status", "")).startswith("OK")}
    acc = accepted_pages(ref_results)
    assert set(acc) == recorded_ok - set(MATCH_DEPARTURES), (
        "acceptance is not the record minus the enumerated departures — see "
        "test_match_departures_are_exactly_enumerated for which pages moved"
    )
    assert len(acc) == PRE_GATE_ACCEPTED - len(LOO_DEMOTED)  # 78 recorded - 12 departures = 66

    med = median_rmse(ref_scores)
    assert med is not None and med < 7.5, f"median RMSE {med} >= 7.5 m"

    # p28 is the only strict accept beyond the 15 m gate (16.43 m at this stage,
    # 16.44 after seam); the record accepted it too, at 16.43 m. run_volume is the
    # MATCH stage alone, so the rescue-family returns are not visible here —
    # RESCUE_FAMILY carries them.
    over_gate = sorted((p for p in acc if ref_scores[p] > GT_COMMIT_RMSE_M), key=int)
    assert over_gate == ["28"], f"strict accepts beyond the commit gate: {over_gate}"
    # ...and the pipeline commits it anyway, because nothing in a run reads a human
    # score. This is the loss taken when ground truth left the run, stated as an
    # assertion rather than left to be discovered: a demotion pass is what removes
    # p28 from served evidence now, and until one runs it serves.
    assert is_committed(ref_results["28"]), "an over-gate accept must commit on its own funnel"
    unscored = [p for p in acc if p not in ref_scores]
    assert unscored == [], f"strict accepts with no GT score: {unscored} — the scorer is blind"
    assert len(acc) - len(over_gate) == 65


def test_loo_demoted_good_sheets_re_enter_via_rescue(
    ref_setup: dict[str, Any], fixtures_dir: Path, tmp_path: Path
) -> None:
    """The gate's PAYOFF: the four good demotions come back, and come back RIGHT.

    Asserting the status alone would prove they returned without proving they
    returned to the right place — so each is scored from its own gcps_geojson
    (the points a warp would actually use) against the human GCPs. That score is
    the only one available: rmse_vs_human_m is written by match_sheet only, so a
    rescued record carries none, and is_committed treats an unscored record as
    committed. Diffing the runs on that field would have counted every demotion
    as a free win without ever checking where the sheet landed.
    """
    pytest.importorskip("cv2")
    paths = VolumePaths(root=tmp_path / "ref-volume")
    shutil.copytree(fixtures_dir / "ref-volume" / "annotations", paths.annotations)
    shutil.copytree(fixtures_dir / "ref-volume" / "sheets", paths.sheets)
    paths.results.mkdir(parents=True)

    aliases, gt, index = ref_setup["aliases"], ref_setup["gt"], ref_setup["index"]
    stage_match(paths, index, VolumeConfig(identifier="ref-volume"))
    stage_revoke_shared_street_rescues(paths, aliases)
    stage_rescue(paths, index, VolumeConfig(identifier="ref-volume"))

    for page in sorted(LOO_RE_ENTERED):
        rec = json.loads((paths.results / f"p{page}.json").read_text())
        assert rec["status"] == "OK (rescued)", f"p{page}: {rec['status']!r}, expected rescue"
        assert "rmse_vs_human_m" not in rec, "a run must not write a human score onto a record"

        width, height = ref_setup["manifest"][f"p{page}"]["full_size"]
        mine = fit_affine(gcps_from_geojson(rec["gcps_geojson"]))
        human = fit_affine(gcps_from_geojson(gt[page]["gcps_geojson"]))
        rmse = grid_rmse_m(
            mine, human, width, height, mercator_correction_lat=mercator_correction_lat(gt)
        )
        assert rmse <= GT_COMMIT_RMSE_M, f"p{page} rescued to {rmse:.2f} m — outside commit gate"


def test_the_scorer_reaches_the_rescue_family(
    ref_setup: dict[str, Any], fixtures_dir: Path, tmp_path: Path
) -> None:
    """The whole shipped flow: place with no ground truth, then score the tree.

    The scoring pass must judge EVERY accept, not just the strict ones — `match_sheet`
    scores nothing at all now, and a rescued or corroborated sheet reaches the mosaic on
    its own funnel's verdict. p90 is the sheet that proves the check is not theoretical.
    It corroborates (its neighbors vouch for it at >=2 shared nodes within 8 m) and it is
    still 20.3 m from the human GCPs — a coherent local shift its neighbors happily agree
    with. Nothing in the run notices; the sidecar names it.
    """
    pytest.importorskip("cv2")
    paths = VolumePaths(root=tmp_path / "ref-volume")
    shutil.copytree(fixtures_dir / "ref-volume" / "annotations", paths.annotations)
    shutil.copytree(fixtures_dir / "ref-volume" / "sheets", paths.sheets)
    paths.results.mkdir(parents=True)

    aliases, index = ref_setup["aliases"], ref_setup["index"]
    vcfg = VolumeConfig(identifier="ref-volume")
    stage_match(paths, index, vcfg)
    stage_revoke_shared_street_rescues(paths, aliases)
    stage_rescue(paths, index, vcfg)
    stage_seam(paths)
    stage_corroborate(paths)

    results_by_page = {
        r["page"]: r
        for r in (json.loads(f.read_text()) for f in sorted(paths.results.glob("p*.json")))
    }
    assert not [p for p, r in results_by_page.items() if "rmse_vs_human_m" in r], (
        "a run wrote a human score onto a result record — the field is the scorer's, "
        "and a record that carries one is a record a gate could read"
    )

    # ...and NOW grade it, exactly as `autogeoref score` does. The volume id is the
    # REAL one: it names the export to grade against, and the work tree's directory
    # name is the fixture's own, not a volume the corpus has ever heard of.
    score_volume(paths, GT_VOLUME, [fixtures_dir / "ground-truth"])
    scores = load_scores(paths)

    p90 = results_by_page["90"]
    assert p90["status"] == "OK (rescued, neighbor-corroborated)"
    assert scores["90"] == pytest.approx(20.3, abs=1.0)
    assert scores["90"] > GT_COMMIT_RMSE_M, "p90 is 20 m out and the scorer must say so"
    # the loss, asserted: the run no longer withholds it. p90 vouches, seams and serves
    # until a demotion pass acts on the number above.
    assert is_committed(p90), "an over-gate accept commits on its own funnel's verdict"

    # the scorer did not become indiscriminate: every OTHER rescue-family accept on this
    # volume is scored and DOES clear the gate. Pinned as an exact page set, because a
    # count-only floor would pass just as happily with three of these gone — and three of
    # them are sheets the recorded run placed tens of metres out, so their disappearance
    # would silently cost the port its whole argument.
    family = {p: r for p, r in results_by_page.items() if str(r["status"]).startswith("OK (")}
    assert set(family) == set(RESCUE_FAMILY), (
        f"missing: {sorted(set(RESCUE_FAMILY) - set(family), key=int)}; "
        f"unenumerated: {sorted(set(family) - set(RESCUE_FAMILY), key=int)}"
    )
    # abs=0.05, not the 1.0 m this once used: RANSAC is seeded and the pipeline is
    # byte-deterministic (see test_determinism), so a metre of slack buys nothing and
    # would let a systematic sub-metre degradation across every rescue pass in silence.
    for page, (status, rmse) in RESCUE_FAMILY.items():
        got = family[page]["status"]
        assert got == status, f"p{page}: {got!r} != {status!r}"
        assert page in scores, f"p{page}: rescue-family accept unscored"
        assert scores[page] == pytest.approx(rmse, abs=0.05), (
            f"p{page} moved: {scores[page]} m, recorded here as {rmse} m"
        )
    over_gate = {p for p in family if scores.get(p, 0.0) > GT_COMMIT_RMSE_M}
    assert over_gate == RESCUE_FAMILY_OVER_GATE, f"over the gate: {sorted(over_gate)}"

    # The STRICT accepts move here too, and only here: this is the only golden
    # test that runs the seam solve, and a rescue-family record changing shifts
    # the fit its neighbours' ties are measured against.
    # `test_strict_accept_accuracy_is_pinned_per_page` replays match ONLY, so it
    # cannot see this — which left the post-seam scores of the sheets that
    # actually serve unpinned until the always-corners change moved eleven of them.
    for page, rmse in POST_SEAM_STRICT_RMSE.items():
        assert results_by_page[page]["status"] == "OK"
        assert scores[page] == pytest.approx(rmse, abs=0.05), (
            f"p{page} moved post-seam: {scores[page]} m, recorded here as {rmse} m"
        )


def test_page2_canary(ref_results: dict[str, dict[str, Any]], ref_scores: dict[str, float]) -> None:
    assert ref_results["2"]["status"] == "OK"
    assert ref_scores["2"] == pytest.approx(4.4, abs=1.0)


def test_determinism(ref_setup: dict[str, Any]) -> None:
    """Same inputs -> byte-identical result records (fixed RANSAC seed)."""
    sheets = [s for s in ref_setup["sheets"] if s.page in {"2", "8", "53"}]
    from autogeoref.volume import derive_constraints

    c1 = derive_constraints(sheets, ref_setup["index"], ref_setup["aliases"])
    c2 = derive_constraints(sheets, ref_setup["index"], ref_setup["aliases"])
    assert c1 == c2
    for s in sheets:
        r1 = match_sheet(s, ref_setup["index"], c1, ref_setup["aliases"])
        r2 = match_sheet(s, ref_setup["index"], c2, ref_setup["aliases"])
        assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
