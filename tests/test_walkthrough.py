"""The illustrated walkthrough: its committed assets, and what its page may not do.

The page is display-only by contract. Two things have to hold and neither is
visible to a unit test of anything else: every plate the JSON names is on disk
(a broken asset path is the realistic failure), and no acceptance threshold has
acquired a second copy in JavaScript, where it could drift from the code it
claims to describe.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageStat

from autogeoref.corroborate import MIN_NODES
from autogeoref.corroborate import TOL_M as CORROBORATE_TOL_M
from autogeoref.matching import (
    RANSAC_ITERS,
    RANSAC_MIN_INLIERS,
    RANSAC_TOL_M,
    SPREAD_PERP_FRAC,
    SPREAD_SPAN_FRAC,
)
from autogeoref.rescue import TOL_M as RESCUE_TOL_M
from autogeoref.verified_accept import MIN_CHANNELS
from viewer_support import WALK_CSS, WALK_DIR, WALK_HTML, WALK_JS, WALK_PANELS

#: The six acts, in the order a reader meets them.
ACTS = (
    "I. From a scan to a reading",
    "II. From a reading to a fit",
    "III. Second chances",
    "IV. The volume settles",
    "V. Becoming a map",
    "VI. Grading, afterwards",
)

PANELS = 21


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(WALK_PANELS.read_text(encoding="utf-8"))
    return data


def test_every_stage_of_the_pipeline_gets_a_panel(payload: dict[str, Any]) -> None:
    """The whole process, not a highlights reel. A stage dropped for budget is
    a misrepresentation of the system, so the count and the act structure are
    asserted rather than left to whoever regenerates the assets next."""
    panels = payload["panels"]
    assert [p["number"] for p in panels] == list(range(1, PANELS + 1))
    assert [a for a in ACTS if any(p["act"] == a for p in panels)] == list(ACTS)
    stages = {p["stage"] for p in panels}
    for stage in ("prep", "annotate", "escalate", "match", "rescue", "corroborate", "seam"):
        assert stage in stages, f"no panel depicts {stage}"
    for stage in ("junction-verify", "verified-accept", "warp", "mask", "tile", "report", "score"):
        assert stage in stages, f"no panel depicts {stage}"


def test_every_plate_the_json_names_is_on_disk(payload: dict[str, Any]) -> None:
    """The failure this page can actually have. Nothing in the browser reports
    a 404 image as anything but a blank box."""
    missing = [
        state["file"]
        for panel in payload["panels"]
        for state in panel["states"]
        if not (WALK_DIR / state["file"]).is_file()
    ]
    assert not missing, f"panels.json names plates that are not committed: {missing}"
    for panel in payload["panels"]:
        assert panel["states"], f"panel {panel['number']} has no plate"
        for state in panel["states"]:
            assert state["alt"], f"panel {panel['number']} state {state['key']} has no alt text"


#: A plate with less variation than this is blank or nearly so. The committed
#: set sits far above it; an empty 1600x1000 sheet of paper scores 0.
PLATE_INK_FLOOR = 8.0


def test_no_committed_plate_is_blank(payload: dict[str, Any]) -> None:
    """A file that loads is not a figure that drew.

    The browser tier can only ask whether an image resolved, and a blank JPEG
    resolves. This asks whether there is anything on it.
    """
    thin = []
    for panel in payload["panels"]:
        for state in panel["states"]:
            image = Image.open(WALK_DIR / state["file"]).convert("L")
            spread = ImageStat.Stat(image).stddev[0]
            if spread < PLATE_INK_FLOOR:
                thin.append((state["file"], round(spread, 2)))
    assert not thin, f"plates with almost nothing on them: {thin}"


def test_no_plate_on_disk_is_unreachable(payload: dict[str, Any]) -> None:
    """The other direction: a renamed panel leaves its old plate behind, and
    committed dead weight in a bundle a visitor downloads is worth catching."""
    named = {state["file"] for panel in payload["panels"] for state in panel["states"]}
    on_disk = {path.name for path in WALK_DIR.glob("*.jpg")}
    assert on_disk - named == set()


def test_the_running_tally_accounts_for_every_sheet(payload: dict[str, Any]) -> None:
    """The counter carried across the panels is the volume's own arithmetic:
    a sheet is placed, held as a proposal, or flagged, and never two of those."""
    funnel = payload["funnel"]
    total = funnel["total"]
    assert total > 0
    for stage in funnel["order"]:
        at = funnel["stages"][stage]
        assert at["placed"] + at["provisional"] + at["flagged"] == total, stage
    end = funnel["stages"][funnel["order"][-1]]
    assert end["provisional"] == 0, "the last stage still holds proposals"


def _spellings(value: float) -> set[str]:
    """Every way a person would plausibly write one threshold into prose.

    The first version of this test compared ``repr(25.0)`` and nothing else,
    which no one writes: the page could say `within 25 m` and pass.
    """
    out = {repr(value), f"{value:g}", f"{value:.0f}", f"{value:,g}"}
    if 0 < value < 1:
        out |= {f"{value:.0%}", f"{value * 100:g}"}
    return {spelling for spelling in out if len(spelling) > 1}


THRESHOLDS = sorted(
    spelling
    for value in (
        RANSAC_TOL_M,
        SPREAD_SPAN_FRAC,
        SPREAD_PERP_FRAC,
        RESCUE_TOL_M,
        CORROBORATE_TOL_M,
        float(RANSAC_ITERS),
        float(MIN_NODES),
    )
    for spelling in _spellings(value)
)


def test_the_running_tally_carries_past_the_last_stage(payload: dict[str, Any]) -> None:
    """The counter is meant to carry across every panel, and most panels are not
    a stage: the back half runs on sheets the funnel already resolved. Once a
    stage has resolved, no later panel may fall back to nothing-decided-yet,
    which on a back-half panel would state something false.
    """
    stages = payload["funnel"]["stages"]
    resolved = False
    for panel in payload["panels"]:
        tally = panel["tally"]
        if panel["stage"] in stages:
            resolved = True
        if resolved:
            assert tally in stages, f"panel {panel['number']} lost the counter"
        else:
            assert tally == "", f"panel {panel['number']} claims a stage before any resolved"
    assert resolved, "no panel ever reached a funnel stage"
    assert payload["panels"][-1]["tally"] == "verified-accept"


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_no_acceptance_threshold_is_typed_into_the_page(threshold: str) -> None:
    """Every number a reader sees is generated. A threshold spelled in the page
    source is a second copy of a contract, and the copy is the one that rots."""
    for path in (WALK_HTML, WALK_CSS, WALK_JS):
        # the stylesheet is full of incidental lengths; only prose is checked
        text = path.read_text(encoding="utf-8")
        if path is WALK_CSS:
            continue
        assert threshold not in text, f"{path.name} spells {threshold}"


def test_the_page_carries_no_pipeline_logic() -> None:
    """v1 is illustrative: a computed state is computed in Python and rendered.

    A gate predicate re-implemented here could diverge from the pipeline and
    teach something false with complete confidence. Naming the functions it must
    not call is not enough — a re-implementation under other names reads the
    same to a grep — so this bans the raw material: a bare number the page did
    not get from the JSON. Layout indices (0, 1, 2, 100) survive; a tolerance,
    an iteration count or a minimum does not.
    """
    script = WALK_JS.read_text(encoding="utf-8")
    stripped = re.sub(r"/\*.*?\*/|//[^\n]*", "", script, flags=re.DOTALL)
    allowed = {"0", "1", "2", "100"}
    numbers = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?", stripped)) - allowed
    assert not numbers, f"walkthrough.js carries numeric literals of its own: {sorted(numbers)}"
    for banned in ("Math.hypot", "Math.sqrt", "determinant", "affine"):
        assert banned not in script, f"walkthrough.js computes something ({banned})"


#: What a comparison in the stepper is allowed to be about: how many panels or
#: states there are, and where in that list we are. Anything else means the page
#: is deciding something about the pipeline's data.
LAYOUT_OPERANDS = re.compile(r"\.length$|indexOf\(|^(?:index|at|i|next|0|1|2)$")


def test_the_page_compares_only_layout_quantities() -> None:
    """The companion to the literal ban, and the hole it leaves.

    Banning bare numbers stops `spread < 25.0` but not `spread < p.limit`, which
    is the same forbidden thing with the constant moved into the JSON. Both
    operands of every comparison must therefore be layout material — a list
    length, a lookup result, or a cursor — so a predicate over pipeline data has
    nowhere to put either side.
    """
    script = WALK_JS.read_text(encoding="utf-8")
    stripped = re.sub(r"/\*.*?\*/|//[^\n]*", "", script, flags=re.DOTALL)
    stripped = re.sub(r"`[^`]*`|\"[^\"]*\"|'[^']*'", "''", stripped)
    operand = r"([\w.$]+(?:\([^()]*\))?)"
    comparison = re.compile(operand + r"\s*(?<![=!<>])(<=|>=|<|>)(?!=)\s*" + operand)
    offenders = [
        f"{left} {op} {right}"
        for left, op, right in comparison.findall(stripped)
        if not (LAYOUT_OPERANDS.search(left) and LAYOUT_OPERANDS.search(right))
    ]
    assert not offenders, f"walkthrough.js compares something that is not layout: {offenders}"


def test_the_page_reads_everything_from_the_generated_json() -> None:
    """The single source of truth, and the wiring that keeps it single."""
    script = WALK_JS.read_text(encoding="utf-8")
    assert 'fetch("walkthrough/panels.json")' in script
    assert "textContent" in script
    assert "innerHTML" not in script, "a caption off disk must reach the page as text"


def test_the_generated_figures_are_labelled(payload: dict[str, Any]) -> None:
    """A figure with no label is a number a reader cannot use."""
    for panel in payload["panels"]:
        assert panel["caption"], panel["number"]
        assert panel["dek"], panel["number"]
        for figure in panel["figures"]:
            assert figure["label"] and figure["value"] != "", panel["number"]


def test_the_status_vocabulary_appears_verbatim(payload: dict[str, Any]) -> None:
    """An operator reading the result files and a reader of this page have to
    see the same words, so the funnel panel quotes the statuses exactly."""
    from autogeoref.volume import STATUS_OK, STATUS_REJECTED, STATUS_RESCUED

    blob = json.dumps(payload)
    for status in (STATUS_OK, STATUS_RESCUED, STATUS_REJECTED):
        assert status in blob


def test_the_two_channel_contract_is_stated(payload: dict[str, Any]) -> None:
    """The keystone number, generated rather than written into prose."""
    panel = next(p for p in payload["panels"] if p["slug"] == "verified-accept")
    values = {f["label"]: f["value"] for f in panel["figures"]}
    assert values["Agreement needed"] == str(MIN_CHANNELS)


def test_the_corroboration_gate_is_stated(payload: dict[str, Any]) -> None:
    panel = next(p for p in payload["panels"] if p["slug"] == "corroborate")
    values = {f["label"]: f["value"] for f in panel["figures"]}
    assert values["Needed"] == f"{MIN_NODES} within {CORROBORATE_TOL_M:g} m"


def test_the_deploy_bundle_carries_the_plates(tmp_path: Path, payload: dict[str, Any]) -> None:
    """A page file missing from the bundle is a hard error; a missing TREE was
    silent, and a walkthrough deployed without its plates is twenty-one broken
    images and a `panels.json` that 404s."""
    from autogeoref.viewer.deploy import build_deploy_bundle

    viewer = WALK_DIR.parent
    staged = tmp_path / "viewer"
    staged.mkdir()
    for path in viewer.iterdir():
        if path.name == "manifest.json":
            continue
        (staged / path.name).symlink_to(path)
    city = staged / "testville"
    city.mkdir()
    (city / "manifest.json").write_text(
        json.dumps({"volumes": [{"id": "v1", "era": "1950", "pmtiles": "v1.pmtiles"}]}),
        encoding="utf-8",
    )
    out = tmp_path / "bundle"
    build_deploy_bundle(staged, out, "https://tiles.example.invalid", city="testville")

    assert (out / "walkthrough.html").is_file()
    assert (out / "walkthrough" / "panels.json").is_file()
    first = payload["panels"][0]["states"][0]["file"]
    assert (out / "walkthrough" / first).is_file(), "the plates did not travel with the page"


#: The exemplar pins the generator reads. Panel 4 is the one a reader can judge
#: against the drawing, so the sheet it names carries a release condition.
EXEMPLARS = WALK_DIR.parents[1] / "scripts" / "walkthrough" / "walkthrough_exemplars.json"


def test_panel_four_shows_a_reader_error_and_not_a_gap_of_ours() -> None:
    """The name panel 4 crosses out must be a misread the model made.

    An earlier exemplar showed `BALMORAL 45 AV.` for Balmoral Avenue - a house
    number run into the name, which dropping the digits repairs at no cost. The
    page then told the public that the reader was at fault for something the
    pipeline could mend, on the one page built to make a good impression.
    """
    from autogeoref.names import normalize

    ex = json.loads(EXEMPLARS.read_text(encoding="utf-8"))
    wrong, right = ex["escalated_wrong"], ex["escalated_right"]
    plain = re.sub(r"\s+", " ", re.sub(r"\d+", "", wrong)).strip()
    assert plain != right, f"{wrong} is {right} with a number in it, which is our defect"
    assert normalize(plain, {}) != normalize(right, {}), (
        f"{wrong} reaches {right} through the normaliser, so no reader is to blame"
    )


def test_panel_four_names_the_atlas_its_sheet_came_from(payload: dict[str, Any]) -> None:
    """It is the one panel drawn from another volume, and it has to say so."""
    ex = json.loads(EXEMPLARS.read_text(encoding="utf-8"))
    panel = next(p for p in payload["panels"] if p["slug"] == "escalate")
    assert ex["escalated_volume"] != ex["volume"]
    assert ex["escalated_title"] in json.dumps(panel)
    assert ex["escalated_title"] not in payload["meta"]["title"]


def test_the_gate_panel_names_every_check(payload: dict[str, Any]) -> None:
    """Seven checks, each shown refusing, plus the state where all seven pass."""
    panel = next(p for p in payload["panels"] if p["slug"] == "the-gates")
    keys = [state["key"] for state in panel["states"]]
    assert keys[0] == "pass"
    assert set(keys[1:]) == {
        "handedness",
        "scale_window",
        "rotation_window",
        "aspect",
        "span",
        "perp",
        "leave_one_out",
    }
    values = {f["label"]: f["value"] for f in panel["figures"]}
    assert values["Checks"] == str(len(keys) - 1)
    assert values["Minimum corners"] == str(RANSAC_MIN_INLIERS)
