"""The scoring pass, and the structural rule it exists to make unbreakable.

Ground truth may GRADE a finished placement and must never INFLUENCE one. That
rule used to be enforced by remembering not to pass a flag, and in practice it
was not remembered: the queue appended ``--ground-truth`` to every place leg,
which made a volume's search box come from hand-placed pins. So the input was
deleted rather than documented, and these tests are what keep it deleted.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

import autogeoref.bake.layers  # noqa: F401  (bake.layers is reached as an attribute)
from autogeoref import (
    bake,
    corroborate,
    escalate,
    matching,
    rescue,
    run_inputs,
    runplan,
    seam,
    stages,
    verified_accept,
    verify,
    volume,
    vouchers,
)
from autogeoref.paths import VolumePaths
from autogeoref.score_pass import load_sources, merge_pages, resolve_pages, score_volume
from autogeoref.scoring import GT_COMMIT_RMSE_M, drop_score, load_scores, write_sidecar
from autogeoref.stages import corroborate as stage_corroborate_mod
from autogeoref.stages import match as stage_match_mod
from autogeoref.stages import rescue as stage_rescue_mod
from autogeoref.stages import seam as stage_seam_mod

#: Every module a placement flows through. Ground truth reaching ANY of them is
#: the defect this work removed, so the check is over the source text: a
#: parameter, an import, or a helpfully-named local all count. A PACKAGE here
#: means every module in it — naming one and sweeping only its `__init__` is
#: how this check would quietly stop covering a stage that moved into a package.
PLACEMENT_MODULES = (
    run_inputs,
    stages,
    runplan,
    seam,
    escalate,
    volume,
    rescue,
    corroborate,
    matching,
    verified_accept,
    verify,
    vouchers,
    bake,
)


def _swept_files() -> list[Path]:
    """Every source file `PLACEMENT_MODULES` covers, packages expanded."""
    found: list[Path] = []
    for module in PLACEMENT_MODULES:
        path = Path(inspect.getfile(module))
        found.extend(sorted(path.parent.rglob("*.py")) if path.name == "__init__.py" else [path])
    return found


#: ...and the functions that actually DECIDE, which are held to more than a name:
#: none of them may reach the sidecar either. ``stage_report`` is absent because
#: reading the sidecar is its job — printing what the scorer found. The sweep
#: above still covers its module, since that bans naming ground truth, not
#: reading a score.
PLACEMENT_DECISIONS = (
    run_inputs.resolve_bounds,
    stage_match_mod.stage_match,
    stage_rescue_mod.stage_rescue,
    stage_corroborate_mod.stage_corroborate,
    stage_seam_mod.stage_seam,
    stage_seam_mod._collect_seam_inputs,
    escalate.stage_escalate,
    volume.is_committed,
    volume.match_sheet,
    bake.layers.committed_layers,
    vouchers.committed_vouch_nodes,
)


def _volume_tree(tmp_path: Path, records: dict[str, dict[str, Any]]) -> VolumePaths:
    paths = VolumePaths(root=tmp_path / "work" / "vol_a")
    paths.results.mkdir(parents=True)
    paths.sheets.mkdir(parents=True)
    paths.manifest.write_text(
        json.dumps({f"p{page}": {"full_size": [1000, 1000], "scale": 0.25} for page in records})
    )
    for page, record in records.items():
        (paths.results / f"p{page}.json").write_text(json.dumps({"page": page, **record}))
    return paths


def _gcps(features: list[tuple[float, float, float, float]]) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"image": [px, py]},
                "geometry": {"type": "Point", "coordinates": [lng, lat]},
            }
            for px, py, lng, lat in features
        ],
    }


#: An identity-ish placement and the same points nudged east, so the volume has
#: a real, nonzero grid RMSE without any fixture.
_AUTO = _gcps([(0, 0, -87.60, 41.80), (1000, 0, -87.59, 41.80), (0, 1000, -87.60, 41.79)])
_HUMAN = _gcps([(0, 0, -87.60, 41.80), (1000, 0, -87.59, 41.80), (0, 1000, -87.60, 41.7901)])


def _export(directory: Path, volume_id: str, pages: dict[str, dict[str, Any]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"api-layers-{volume_id}.json"
    path.write_text(
        json.dumps(
            [
                {
                    "slug": f"chicago_ill_1895_vol_1_p{page}",
                    "extent": [-87.60, 41.79, -87.59, 41.80],
                    "gcps_geojson": fc,
                }
                for page, fc in pages.items()
            ]
        )
    )
    return path


def test_no_placement_module_can_reach_ground_truth() -> None:
    """The done condition, asserted rather than grepped by hand once."""
    for path in _swept_files():
        source = path.read_text()
        assert "ground_truth" not in source, f"{path} still names ground truth"
        assert "rmse_vs_human_m" not in source, f"{path} still names the score"


def test_the_sweep_reaches_inside_every_package_it_names() -> None:
    """One module became five, then a package. A name-level list loses that silently."""
    swept = {path.name for path in _swept_files()}

    for expected in ("match.py", "rescue.py", "corroborate.py", "seam.py", "report.py"):
        assert expected in swept, f"stages/{expected} escaped the ground-truth sweep"
    for expected in ("warp.py", "masks.py", "mosaic.py", "tiles.py", "layers.py"):
        assert expected in swept, f"bake/{expected} escaped the ground-truth sweep"
    assert "placement.py" in swept and "backhalf.py" in swept


def test_no_placement_decision_can_reach_the_sidecar() -> None:
    """The rule the module sweep above cannot state.

    Deleting the input made feedback impossible through a result RECORD; the
    sidecar is a file any of these could open. A gate reading it would be the
    same defect wearing a new filename, and would pass every other test here.
    """
    banned = ("load_scores", "load_sidecar", "score_pass", "results-scores", "rmse")
    for func in PLACEMENT_DECISIONS:
        # the docstring is stripped first: pointing a reader AT the scorer is
        # exactly what these functions should do, and is not reaching for it
        body = ast.parse(textwrap.dedent(inspect.getsource(func))).body[0]
        assert isinstance(body, ast.FunctionDef)
        if ast.get_docstring(body) is not None:
            body.body = body.body[1:]
        code = ast.unparse(body)
        hit = [name for name in banned if name in code]
        assert not hit, f"{func.__module__}.{func.__qualname__} reaches the scorer: {hit}"


def test_run_has_no_ground_truth_flag() -> None:
    from autogeoref.cli.parser import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "vol_a", "--city", "c.toml", "--ground-truth", "gt.json"])


def test_the_scorer_writes_a_sidecar_and_leaves_records_alone(tmp_path: Path) -> None:
    paths = _volume_tree(tmp_path, {"1": {"status": "OK", "gcps_geojson": _AUTO}})
    before = (paths.results / "p1.json").read_text()
    _export(tmp_path / "gt", "vol_a", {"1": _HUMAN})

    payload = score_volume(paths, "vol_a", [tmp_path / "gt"])

    assert (paths.results / "p1.json").read_text() == before, "the scorer edited a result record"
    assert payload["gate_m"] == GT_COMMIT_RMSE_M
    assert load_scores(paths) == {"1": payload["pages"]["1"]["rmse_vs_human_m"]}
    assert load_scores(paths)["1"] > 0.0


def test_only_accepted_non_reviewer_placements_are_scored(tmp_path: Path) -> None:
    """A flagged sheet has no placement to grade, and a reviewer-verified one is
    human work — grading it against other human work measures nothing."""
    paths = _volume_tree(
        tmp_path,
        {
            "1": {"status": "OK", "gcps_geojson": _AUTO},
            "2": {"status": "REJECTED (no valid RANSAC model)", "gcps_geojson": _AUTO},
            "3": {"status": "OK (reviewer-verified)", "gcps_geojson": _AUTO},
        },
    )
    _export(tmp_path / "gt", "vol_a", {"1": _HUMAN, "2": _HUMAN, "3": _HUMAN})

    assert sorted(score_volume(paths, "vol_a", [tmp_path / "gt"])["pages"]) == ["1"]


def test_an_empty_export_is_checked_not_absent(tmp_path: Path) -> None:
    """The marker convention, carried into the scorer.

    A 0-byte export means "the volunteer corpus was checked: this volume was never
    pinned" — a different fact from no file at all, and the sidecar must keep them
    apart or a checked-and-unpinned volume reads as one nobody has looked at.
    """
    paths = _volume_tree(tmp_path, {"1": {"status": "OK", "gcps_geojson": _AUTO}})
    checked, unchecked = tmp_path / "checked", tmp_path / "unchecked"
    checked.mkdir()
    unchecked.mkdir()
    (checked / "api-layers-vol_a.json").write_text("")

    assert score_volume(paths, "vol_a", [checked])["sources"] == [
        {"path": str(checked / "api-layers-vol_a.json"), "pinned_pages": 0, "empty_marker": True}
    ]
    assert score_volume(paths, "vol_a", [unchecked])["sources"] == []


def test_a_second_corpus_can_be_graded_in_the_same_pass(tmp_path: Path) -> None:
    """_090/_093 keep their pins in fixtures/prod/, not fixtures/ground-truth/."""
    first, second = tmp_path / "gt", tmp_path / "prod"
    _export(first, "vol_a", {"1": _HUMAN})
    _export(second, "vol_a", {"1": _AUTO, "2": _HUMAN})
    sources = load_sources("vol_a", [first, second])

    assert [s.path.parent.name for s in sources] == ["gt", "prod"]
    merged = merge_pages(sources)
    # the FIRST directory named wins a contested page, and the second still
    # contributes the pages the first never had
    assert merged["1"] is sources[0].pages["1"]
    assert sorted(merged) == ["1", "2"]


def test_a_reviewer_move_invalidates_only_that_pages_score(tmp_path: Path) -> None:
    paths = _volume_tree(tmp_path, {"1": {"status": "OK"}})
    write_sidecar(
        paths,
        {
            "gate_m": GT_COMMIT_RMSE_M,
            "sources": [],
            "pages": {"1": {"rmse_vs_human_m": 20.3}, "2": {"rmse_vs_human_m": 4.0}},
        },
    )

    assert drop_score(paths, "1") is True
    assert load_scores(paths) == {"2": 4.0}
    assert drop_score(paths, "1") is False, "dropping a score that is not there is not a change"


def test_a_score_cannot_outlive_the_placement_it_describes(tmp_path: Path) -> None:
    """A re-place invalidates its own scores, without anyone remembering to.

    ``stage_report`` reads the sidecar on EVERY run, so a graded volume that is
    then re-placed would otherwise publish a median — and an over-gate count —
    computed against placements that no longer exist. The entry carries the
    digest of the record it graded; when that stops matching, the score is gone.
    """
    paths = _volume_tree(tmp_path, {"1": {"status": "OK", "gcps_geojson": _AUTO}})
    _export(tmp_path / "gt", "vol_a", {"1": _HUMAN})
    score_volume(paths, "vol_a", [tmp_path / "gt"])
    assert load_scores(paths) == {"1": pytest.approx(load_scores(paths)["1"])}

    rp = paths.results / "p1.json"
    record = json.loads(rp.read_text())
    record["seam_adjusted"] = {"dx_m": 3.0, "dy_m": 0.0}  # the sheet moved
    rp.write_text(json.dumps(record))

    assert load_scores(paths) == {}, "a stale score survived the placement it described"


def test_a_missing_export_never_erases_a_real_grading(tmp_path: Path) -> None:
    """`_093`'s pins live in fixtures/prod/, so the default directory has no file
    for it. That invocation must not wipe what a correct one wrote."""
    paths = _volume_tree(tmp_path, {"1": {"status": "OK", "gcps_geojson": _AUTO}})
    _export(tmp_path / "prod", "vol_a", {"1": _HUMAN})
    score_volume(paths, "vol_a", [tmp_path / "prod"])
    graded = load_scores(paths)
    assert graded

    empty = tmp_path / "gt"
    empty.mkdir()
    payload = score_volume(paths, "vol_a", [empty])

    assert payload["sources"] == []
    assert load_scores(paths) == graded, "a mistyped directory erased a real grading"


def test_the_seam_diagnostic_counts_a_contested_page_once(tmp_path: Path) -> None:
    """The seam check reads the MERGED pins, not every source's raw layer list —
    a page two corpora both pin would otherwise vote twice in the median."""
    paths = _volume_tree(tmp_path, {"1": {"status": "OK", "gcps_geojson": _AUTO}})
    paths.seam_deltas.write_text(json.dumps({"ties": 1, "deltas": {"1": [1.0, 0.0]}}))
    first, second = tmp_path / "gt", tmp_path / "prod"
    _export(first, "vol_a", {"1": _HUMAN})
    _export(second, "vol_a", {"1": _HUMAN})

    one = score_volume(paths, "vol_a", [first])["seam"]
    both = score_volume(paths, "vol_a", [first, second])["seam"]

    assert both == one, f"a contested page changed the seam diagnostic: {both} != {one}"


def test_a_pin_spelled_in_the_other_case_still_grades_its_sheet(tmp_path: Path) -> None:
    """`_086` placed 95 sheets against 90 pins and scored NONE of them.

    Its pages are letter-suffixed (`p12S`, from the LOC master), and the
    volunteer export spells the same sheet `..._p12s`. The join was exact, so
    every pin missed and the pass reported "no accepted placement has a pinned
    counterpart" — a data absence, which it was not.
    """
    paths = _volume_tree(tmp_path, {"12S": {"status": "OK", "gcps_geojson": _AUTO}})
    _export(tmp_path / "gt", "vol_a", {"12s": _HUMAN})

    payload = score_volume(paths, "vol_a", [tmp_path / "gt"])

    assert list(payload["pages"]) == ["12S"], "a lower-case pin lost its sheet"
    assert payload["pages"]["12S"]["rmse_vs_human_m"] > 0.0


def test_the_seam_diagnostic_folds_case_the_same_way(tmp_path: Path) -> None:
    """The seam gate joins pins to sheets too, and missed them identically."""
    paths = _volume_tree(tmp_path, {"12S": {"status": "OK", "gcps_geojson": _AUTO}})
    paths.seam_deltas.write_text(json.dumps({"ties": 1, "deltas": {"12S": [1.0, 0.0]}}))
    _export(tmp_path / "gt", "vol_a", {"12s": _HUMAN})

    seam_record = score_volume(paths, "vol_a", [tmp_path / "gt"]).get("seam")

    assert seam_record is not None, "the seam diagnostic lost its only sheet to case"
    assert seam_record["n_sheets"] == 1


def test_an_exact_key_wins_and_a_contested_pin_is_folded_onto_nobody() -> None:
    """Folding case repairs one spelling of one sheet; it is not a licence to
    guess. An exact match always wins and is never displaced, and a pin two
    pages could both claim goes to neither — case is sometimes the only thing
    separating two real sheets."""
    assert resolve_pages(["12S"], {"12s": 1}) == {"12S": "12s"}
    assert resolve_pages(["7a", "7A"], {"7a": 1, "7A": 2}) == {"7a": "7a", "7A": "7A"}
    # the exact holder keeps its pin; the other page gets nothing rather than
    # borrowing it
    assert resolve_pages(["7a", "7A"], {"7a": 1}) == {"7a": "7a"}
    assert resolve_pages(["7Ab", "7aB"], {"7ab": 1}) == {}
    assert resolve_pages([], {"12s": 1}) == {}
    assert resolve_pages(["12S"], {}) == {}


def test_a_sheet_with_no_pin_never_borrows_another_sheets(tmp_path: Path) -> None:
    """The fold must not manufacture a grade. Two pages differing only in case
    are two sheets: if only one is pinned, the other is unscored — grading it
    against its neighbour's pins reports a wild residual for a placement nobody
    measured, and a residual past the gate is what marks a sheet for demotion."""
    paths = _volume_tree(
        tmp_path,
        {
            "12s": {"status": "OK", "gcps_geojson": _AUTO},
            "12S": {"status": "OK", "gcps_geojson": _AUTO},
        },
    )
    _export(tmp_path / "gt", "vol_a", {"12s": _HUMAN})

    payload = score_volume(paths, "vol_a", [tmp_path / "gt"])

    assert list(payload["pages"]) == ["12s"], "an unpinned sheet was graded against another's pins"


def test_the_seam_diagnostic_counts_a_folded_page_once(tmp_path: Path) -> None:
    """The contested-page rule, in the spelling that motivated the fold: two
    corpora pinning one sheet under different case must still vote once."""
    paths = _volume_tree(tmp_path, {"12S": {"status": "OK", "gcps_geojson": _AUTO}})
    paths.seam_deltas.write_text(json.dumps({"ties": 1, "deltas": {"12S": [1.0, 0.0]}}))
    first, second = tmp_path / "gt", tmp_path / "prod"
    _export(first, "vol_a", {"12s": _HUMAN})
    _export(second, "vol_a", {"12S": _HUMAN})

    seam_record = score_volume(paths, "vol_a", [first, second])["seam"]

    assert seam_record["n_sheets"] == 1, "one sheet voted twice under two spellings"


def test_the_preferred_corpus_wins_even_spelled_differently(tmp_path: Path) -> None:
    """`load_sources` promises the FIRST directory named wins a contested page.
    That promise is about the sheet, not the spelling, so a later corpus must
    not take the page back by spelling it the way the placement does."""
    paths = _volume_tree(tmp_path, {"12S": {"status": "OK", "gcps_geojson": _AUTO}})
    first, second = tmp_path / "gt", tmp_path / "prod"
    _export(first, "vol_a", {"12s": _HUMAN})
    _export(second, "vol_a", {"12S": _AUTO})  # a perfect, and wrong, second opinion

    payload = score_volume(paths, "vol_a", [first, second])

    assert payload["pages"]["12S"]["rmse_vs_human_m"] > 0.0, "the later corpus won the page"


def test_a_damaged_export_is_skipped_not_fatal(tmp_path: Path) -> None:
    """One unreadable file in a 108-export corpus must not sink the pass —
    the same tolerance `status` already applies over the same tree."""
    paths = _volume_tree(tmp_path, {"1": {"status": "OK", "gcps_geojson": _AUTO}})
    damaged, good = tmp_path / "damaged", tmp_path / "gt"
    damaged.mkdir()
    (damaged / "api-layers-vol_a.json").write_text("{not json")
    _export(good, "vol_a", {"1": _HUMAN})

    payload = score_volume(paths, "vol_a", [damaged, good])

    assert [Path(s["path"]).parent.name for s in payload["sources"]] == ["gt"]
    assert payload["pages"], "the good source was lost with the damaged one"


def test_a_volume_with_no_sidecar_scores_as_unscored(tmp_path: Path) -> None:
    """Every reader must tolerate the volume nobody has scored — which, on the
    day this shipped, was every volume in the tree."""
    assert load_scores(_volume_tree(tmp_path, {"1": {"status": "OK"}})) == {}
