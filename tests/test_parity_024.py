"""Whole-system acceptance: no-ground-truth volume funnel parity.

A fresh end-to-end run through the real CLI from cached inputs with the recorded volume
constants, compared to the recorded funnel within a fixed band per stage.

**This is the ONE test that drives the real CLI against the real city config**, so it is
where the evidence channels' DECISIONS are contracted: the config declares
``evidence_channels``, so junction-verify and verified-accept run with no flag and
``test_evidence_channels_promote_only_the_reviewed_pages`` pins what they conclude.

**It is NOT the shipped command, and the gap is one stage wide.** It passes
``--no-escalate``, and escalation is the OTHER default-ON stage — whose flips become
strict accepts, which become committed vouchers that corroborate then votes off. The
promotion set under the shipped command is therefore unmeasured.
"""

import json
import shutil
from pathlib import Path

import pytest

from autogeoref.cli.entry import main
from autogeoref.report import build_report, load_results_dir

pytestmark = pytest.mark.golden

ROOT = Path(__file__).resolve().parent.parent
VOL = "sanborn01790_024"
#: recorded plain-rescue records that predate the disjoint-pair rule
STALE_PRE_RULE_RESCUES = {"27", "30", "50", "61", "109"}
#: strict accepts the leave-one-out spread gate demotes on this volume — each one
#: re-enters through the rescue family, so the placement survives and only its
#: mechanism changes (measured A/B: accepted_total and flagged both unmoved)
LOO_STRICT_DEMOTIONS = {"9", "42", "52", "105", "106"}


#: The sheets the evidence channels promote here, and the evidence each channel gave
#: for each — reviewed against the ghost overlay and review UI. p82 is NOT a new
#: accept: it was corroborated before and is verified now, at the SAME placement, its
#: own record byte-identical. What moved is its NEIGHBOURS' records. The exactness of
#: this set is aimed at a page no maintainer has looked at BECOMING an accept, which
#: is not what happened there.
VERIFIED_ACCEPTS = {"17", "82"}

#: The run is driven with --no-escalate: a scope decision, not a slip, but NOT
#: outcome-neutral, so read the module docstring before trusting a count here. Copying
#: the sheets and smalls un-starves ESCALATION, which would re-read this volume's
#: evidence-gated rejected pages and spend calls that conftest's tripwire turns into a
#: red test. Escalation's own default is contracted in tests/test_escalate.py and the
#: goldens that replay frozen tier caches.
NO_ESCALATE = "--no-escalate"


def _run(work: Path, fixtures_dir: Path) -> int:
    rc: int = main(
        [
            "run",
            VOL,
            "--city",
            str(ROOT / "configs" / "chicago" / "chicago.toml"),
            "--work",
            str(work),
            "--viewer-manifest",
            str(fixtures_dir / "viewer-manifest.json"),
            NO_ESCALATE,
        ]
    )
    return rc


#: Pages in the revoked pool with NO frozen v2 sidecar, so the addresses channel has
#: nothing to hear from them and abstains. EXACT, not a bound: without it a later
#: change could move five more pages into the revoked pool, cost all five their
#: addresses evidence, stay inside every band and still report PASS. Nothing in the
#: pipeline can buy a numeral read any more, so the no-spend property holds by
#: construction and this assertion is what proves the pool itself has not moved.
ADDRESSES_UNREAD_PAGES = {"82"}


@pytest.fixture(scope="module")
def fresh_run(tmp_path_factory: pytest.TempPathFactory, fixtures_dir: Path) -> Path:
    work = tmp_path_factory.mktemp("parity024")
    vol_dir = work / VOL
    vol_dir.mkdir()
    # annotations carry BOTH the v1 reads the matcher runs on and the frozen v2
    # sidecars the addresses channel votes from; sheets/ carries the smalls the
    # junction channel reads. Without the smalls the channel stages run and score
    # NOTHING, and this test would certify a default it never exercised — which is
    # exactly the hole item 21 closed.
    shutil.copytree(fixtures_dir / VOL / "annotations", vol_dir / "annotations")
    shutil.copytree(fixtures_dir / VOL / "sheets", vol_dir / "sheets")
    assert _run(work, fixtures_dir) == 0

    # No stage may write a numeral reading. This is the standing proof that the run
    # spent nothing on the addresses channel — it is what used to require patching a
    # producer to enforce, and it is why that producer's reach into the real Codex
    # CLI from inside a golden test is now structurally impossible.
    def _files(root: Path) -> set[str]:
        return {f.name for f in (root / "annotations").glob("*.v2.*")}

    def _pages(root: Path) -> set[str]:
        return {name.split(".")[0].lstrip("p") for name in _files(root)}

    # FILE names, not page numbers. A page set cannot see the regression this
    # replaced: the old producer would have written 20 new sidecars under the
    # shipped voices onto the SAME ten pages the fixture already covers, leaving
    # the page set identical while it reached the real Codex CLI.
    assert _files(vol_dir) == _files(fixtures_dir / VOL), (
        "the run wrote (or lost) a v2 sidecar; nothing in the pipeline is allowed to"
    )
    # EXACT, and asserted in the fixture so no test in this module can certify
    # addresses evidence the run never had. See ADDRESSES_UNREAD_PAGES.
    revoked = {
        p
        for p, r in load_results_dir(vol_dir / "results").items()
        if str(r.get("status", "")).startswith("REJECTED (rescue revoked")
        or (r.get("verified_accept") or {})
        .get("previous_status", "")
        .startswith("REJECTED (rescue revoked")
    }
    assert revoked - _pages(vol_dir) == ADDRESSES_UNREAD_PAGES, (
        "the provisional pool moved relative to the frozen sidecars — these pages ran "
        "with NO addresses evidence, and the read-only fixture tree cannot supply any"
    )
    return vol_dir


def test_funnel_parity(fresh_run: Path, fixtures_dir: Path) -> None:
    mine = build_report(VOL, load_results_dir(fresh_run / "results"))
    rec = build_report(VOL, load_results_dir(fixtures_dir / VOL / "results"))

    assert rec.strict_accepted == 67 and rec.rescued == 23 and rec.corroborated == 7

    # the LOO spread gate moves five sheets across the strict/rescue-family line
    # (see module docstring); it moves NONE across the accept/flag line, which is
    # why accepted_total and flagged below are compared against the record as-is
    n_loo = len(LOO_STRICT_DEMOTIONS)
    assert abs(mine.strict_accepted - (rec.strict_accepted - n_loo)) <= 3
    rule_compliant_recorded_rescues = rec.rescued - len(STALE_PRE_RULE_RESCUES)
    assert abs(mine.rescued - rule_compliant_recorded_rescues) <= 3
    # ...the five land 4 as plain rescues and 1 corroborated, so the +5 belongs on
    # the COMBINED rescue family, not on either category alone (see docstring)
    mine_rescue_family = mine.rescued + mine.corroborated
    rec_rescue_family = rec.rescued + rec.corroborated + n_loo
    assert abs(mine_rescue_family - rec_rescue_family) <= 3
    assert abs(mine.accepted_total - rec.accepted_total) <= 3
    assert abs(mine.flagged - rec.flagged) <= 3
    # revoked is a SPLIT of flagged, so the flagged band alone cannot see a page
    # moving between the two. Both drop together here, on the same band.
    assert abs(mine.revoked - rec.revoked) <= 3


def test_loo_demotions_keep_their_placement(fresh_run: Path, fixtures_dir: Path) -> None:
    """The five sheets the LOO gate demotes here must still be PLACED.

    There is no ground truth on this volume, so "placed correctly" is not
    checkable — but "placed at all" is, and it is the invariant that makes the
    gate free here: each of the five was a strict accept in the recorded run and
    must now be an accept via the rescue family, having passed rescue's own
    disjoint-pair gate (or a neighbor's corroboration) on its way back in. One of
    them dropping out to REJECTED would mean this volume paid a price the four
    ground-truth volumes did not, and that is a finding, not a rounding error.
    """
    mine = load_results_dir(fresh_run / "results")
    rec = load_results_dir(fixtures_dir / VOL / "results")
    for page in sorted(LOO_STRICT_DEMOTIONS):
        assert rec[page]["status"] == "OK", f"p{page} was not a recorded strict accept"
        status = str(mine[page]["status"])
        assert status.startswith("OK ("), f"p{page}: {status!r} — lost its placement to the gate"


def test_evidence_channels_promote_only_the_reviewed_pages(fresh_run: Path) -> None:
    """The shipped default's DECISIONS, on _024's real provisional pool.

    The city config declares `evidence_channels`, so the plain run above ran junction-verify
    and verified-accept. This pins what they concluded, and the set is EXACT, not a bound: a
    page appearing here that a maintainer has not looked at is the failure mode the >=2-channel
    rule exists to prevent. Two negatives are contract too: p61 drew a junction SUPPORT but too
    few votable numerals, so addresses abstained and it stayed flagged; p75 was REFUTED by
    addresses, and a refute is a veto no matter what else votes.
    """
    mine = load_results_dir(fresh_run / "results")
    promoted = {p for p, r in mine.items() if str(r.get("status", "")).startswith("OK (verified")}
    assert promoted == VERIFIED_ACCEPTS

    r17 = mine["17"]["verified_accept"]
    assert r17["votes"] == {"corroboration": None, "junction": True, "addresses": True}
    addr = r17["addresses"]
    assert addr["votable"] == addr["in_block"] == 66, "p17's numerals must all land in-block"
    assert len(addr["models"]) >= 2, "the addresses channel needs 2 DISTINCT model voices"

    # the channels stayed honest on the pages they did not promote
    assert mine["61"]["verified_accept"]["votes"] == {
        "corroboration": None,
        "junction": True,
        "addresses": None,
    }
    p75 = mine["75"]["verified_accept"]
    assert p75["votes"]["addresses"] is False and p75["addresses"]["in_block"] == 0
    assert not str(mine["75"]["status"]).startswith("OK")


def test_page_level_diffs_are_only_rule_or_frame_cases(fresh_run: Path, fixtures_dir: Path) -> None:
    """Every accept/flag difference vs the recorded run must be a stale pre-rule
    rescue (we flag it honestly), a corroboration borderline within the seam-frame
    sensitivity, or a REVIEWED verified accept — never a strict-gate page."""
    mine = load_results_dir(fresh_run / "results")
    rec = load_results_dir(fixtures_dir / VOL / "results")
    diffs = []
    for page, r in rec.items():
        a = str(mine.get(page, {}).get("status", "")).startswith("OK")
        b = str(r.get("status", "")).startswith("OK")
        if a != b:
            diffs.append(page)
    assert len(diffs) <= 8
    # The origin had no evidence channels, so promoting a page the record REJECTED
    # is necessarily a diff. A page the record already accepted by another
    # mechanism is not — p82 is exactly that, an accept on both sides whose label
    # moved (see VERIFIED_ACCEPTS). Subtracting the recorded accepts says so,
    # rather than weakening the check to a bare subset.
    recorded_ok = {p for p, r in rec.items() if str(r.get("status", "")).startswith("OK")}
    assert set(diffs) >= VERIFIED_ACCEPTS - recorded_ok
    for page in diffs:
        # never a strict-gate discrepancy
        assert rec[page].get("status") != "OK", f"p{page} was a strict accept"
        assert mine.get(page, {}).get("status") != "OK", f"p{page} became a strict accept"
        # every diff involves the rescue/corroboration lifecycle on both sides
        assert rec[page].get("rescue_anchors") or mine.get(page, {}).get("rescue_anchors"), (
            f"p{page} diff not rescue-related"
        )


def test_second_run_is_a_no_op(fresh_run: Path, fixtures_dir: Path) -> None:
    """Re-running the whole pipeline must not move anything: seam solves in
    the pre-seam frame and applies total-minus-applied deltas, so a repeat
    with unchanged inputs re-derives the same totals and shifts zero."""
    before = {p.name: p.read_text() for p in sorted((fresh_run / "results").glob("p*.json"))}
    seam_before = (fresh_run / "seam_deltas.json").read_text()
    # the channel stages re-run too: junction-verify re-scores and verified-accept
    # re-decides, both from the same inputs — so a second run must still move
    # nothing and spend nothing
    assert _run(fresh_run.parent, fixtures_dir) == 0
    after = {p.name: p.read_text() for p in sorted((fresh_run / "results").glob("p*.json"))}
    assert after == before
    # the seam record may be rewritten but must carry the same solve
    assert json.loads((fresh_run / "seam_deltas.json").read_text()) == json.loads(seam_before)


def test_markers_written(fresh_run: Path) -> None:
    markers = {p.name for p in (fresh_run / "markers").glob("*.marker.json")}
    assert {
        "match.marker.json",
        "rescue.marker.json",
        "seam.marker.json",
        "corroborate.marker.json",
        # the two the city TOML turns on with no flag at all — the item-21 contract
        "junction-verify.marker.json",
        "verified-accept.marker.json",
        "report.marker.json",
    } <= markers
    for m in markers:
        data = json.loads((fresh_run / "markers" / m).read_text())
        assert data["status"] in {"ok", "fresh", "disabled"}


def test_report_artifacts_exist(fresh_run: Path) -> None:
    assert (fresh_run / "report.json").exists()
    assert (fresh_run / "report.md").exists()
    report = json.loads((fresh_run / "report.json").read_text())
    assert report["volume"] == VOL
    assert report["n_sheets"] == 114
