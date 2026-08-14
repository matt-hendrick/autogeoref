"""The alias-gap tripwire: match-rate sidecar at match, advisory note at report.

Contracts pinned here (bars measured in
):

- the run-time count is the corpus instrument's count, aliases included;
- ``stage_match`` persists it as a sidecar and never as a result field;
- the note is ADVISORY — no counter, status, or gate moves with it, and
  nothing about it can fail a run or a report;
- a volume matched before the sidecar existed keeps the zero-candidate half
  and stays silent about the metric it does not have.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autogeoref.centerlines import CenterlineIndex
from autogeoref.config.model import VolumeConfig
from autogeoref.name_match import (
    MIN_PAGES_FOR_NOTE,
    MIN_READS_FOR_NOTE,
    SWEEP_RECORD,
    alias_gap_note,
    count_name_matches,
    load_name_match,
)
from autogeoref.paths import VolumePaths
from autogeoref.stages import match as match_stage
from autogeoref.stages.match import stage_match
from autogeoref.stages.report import stage_report
from autogeoref.volume import SheetInput

# a three-street reference; enough for index keys, not for a fit — this file
# measures vocabulary coverage, not placement
FEATURES = [
    {
        "type": "Feature",
        "properties": {"street_nam": "W 57TH", "street_typ": "ST"},
        "geometry": {"type": "LineString", "coordinates": [[-87.7, 41.79], [-87.6, 41.79]]},
    },
    {
        "type": "Feature",
        "properties": {"street_nam": "S STATE", "street_typ": "ST"},
        "geometry": {"type": "LineString", "coordinates": [[-87.65, 41.75], [-87.65, 41.83]]},
    },
    {
        "type": "Feature",
        "properties": {"street_nam": "S RACINE", "street_typ": "AVE"},
        "geometry": {"type": "LineString", "coordinates": [[-87.66, 41.75], [-87.66, 41.83]]},
    },
]


def _index(aliases: dict[str, str] | None = None) -> CenterlineIndex:
    return CenterlineIndex(FEATURES, aliases=aliases)


def _sheet(page: str, names: list[str]) -> SheetInput:
    streets = [{"name": n, "bbox": [0, 0, 10, 10], "orientation": "horizontal"} for n in names]
    return SheetInput(
        page=page, annotation={"streets": streets}, full_size=(100.0, 100.0), scale=1.0
    )


def test_name_match_counts_reads_and_credits_the_alias_table() -> None:
    """The count is the sweep instrument's count: an aliased read IS a match.

    Measuring raw names instead would flag exactly the volumes whose gap has
    already been closed by an alias table."""
    sheets = [_sheet("1", ["57TH", "S. STATE ST."]), _sheet("2", ["CENTRE AV.", "OGDEN AV."])]

    bare = count_name_matches(sheets, _index())
    assert bare.document() == {"reads": 4, "matched": 2, "match_rate": 0.5}
    # the unmatched leaders the corpus instrument reports come from this loop,
    # so the two can never disagree about what "unmatched" means
    assert bare.unmatched.most_common() == [("CENTRE AV.", 1), ("OGDEN AV.", 1)]

    aliased = count_name_matches(sheets, _index({"CENTRE": "RACINE"}))
    assert aliased.document() == {"reads": 4, "matched": 3, "match_rate": 0.75}

    # nameless reads are not reads; a volume with none scores nothing
    assert count_name_matches([_sheet("1", [])], _index()).match_rate is None


def test_the_corpus_instrument_shares_the_measurement_and_the_bars() -> None:
    """One implementation, or a sweep number and a run number are not the
    same number — the whole point of printing the run one."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "audit_alias_coverage", root / "scripts" / "audit_alias_coverage.py"
    )
    assert spec is not None and spec.loader is not None
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)

    from autogeoref import name_match

    assert audit.MATCH_FLAG is name_match.LOW_MATCH_RATE
    assert audit.ZERO_SHARE_FLAG is name_match.HIGH_ZERO_CANDIDATE_SHARE
    assert audit.count_name_matches is name_match.count_name_matches


def _volume(root: Path, pages: dict[str, list[str]]) -> VolumePaths:
    paths = VolumePaths(root=root)
    paths.annotations.mkdir(parents=True)
    paths.sheets.mkdir(parents=True)
    manifest = {}
    for page, names in pages.items():
        sheet = _sheet(page, names)
        (paths.annotations / f"p{page}.json").write_text(json.dumps(sheet.annotation))
        manifest[f"p{page}"] = {
            "full_size": [100, 100],
            "small_size": [50, 50],
            "scale": 1.0,
            "file": f"p{page}_small.jpg",
        }
    paths.manifest.write_text(json.dumps(manifest))
    return paths


def test_stage_match_persists_the_sidecar_and_keeps_it_current(tmp_path: Path) -> None:
    """Written where the index is already in hand, recomputed every run."""
    paths = _volume(tmp_path / "v", {"1": ["57TH", "S. STATE ST."], "2": ["CENTRE AV.", "OGDEN"]})
    stage_match(paths, _index(), VolumeConfig(identifier="v"))

    assert load_name_match(paths) == {"volume": "v", "reads": 4, "matched": 2, "match_rate": 0.5}
    assert paths.name_match == paths.root / "name-match.json"

    # the metric never touches the results schema
    for record in paths.results.glob("p*.json"):
        assert "match_rate" not in json.loads(record.read_text())

    # A RESUMED run skips committed pages in the match loop — but their reads
    # still count, because the count reads `sheets`, not the loop. Commit p1
    # (skip_committed short-circuits on status_ok) and the recount must still
    # see all four reads, not the two belonging to the page that re-matched.
    committed = json.loads((paths.results / "p1.json").read_text())
    committed["status"] = "OK"
    (paths.results / "p1.json").write_text(json.dumps(committed))

    stage_match(paths, _index({"CENTRE": "RACINE", "OGDEN": "STATE"}), VolumeConfig(identifier="v"))
    assert json.loads((paths.results / "p1.json").read_text())["status"] == "OK"  # was skipped
    assert load_name_match(paths) == {"volume": "v", "reads": 4, "matched": 4, "match_rate": 1.0}


def test_a_failed_sidecar_write_cannot_fail_the_match_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Advisory means advisory: every result is already on disk when this
    runs, so losing rescue/seam/report over an unwritable metric is the tail
    wagging the dog. The report then behaves as it does for any pre-sidecar
    volume — silently."""

    paths = _volume(tmp_path / "v", {"1": ["57TH"], "2": ["CENTRE AV."]})

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("read-only work tree")

    monkeypatch.setattr(match_stage, "write_name_match", boom)
    results = stage_match(paths, _index(), VolumeConfig(identifier="v"))

    assert set(results) == {"1", "2"}
    assert sorted(p.name for p in paths.results.glob("p*.json")) == ["p1.json", "p2.json"]
    assert not paths.name_match.exists()
    assert load_name_match(paths) is None


def test_sidecar_replacement_preserves_the_previous_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same publication contract as volume-constants.json and seam_deltas.json:
    a torn write never replaces a complete document, and leaves no temp file."""
    from autogeoref.name_match import write_name_match

    paths = VolumePaths(root=tmp_path / "v")
    paths.root.mkdir(parents=True)
    old = json.dumps({"volume": "v", "reads": 4, "matched": 4, "match_rate": 1.0}, indent=2)
    paths.name_match.write_text(old)

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replacement failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        write_name_match(paths, "v", [_sheet("1", ["CENTRE AV."])], _index())

    assert paths.name_match.read_text() == old
    assert not list(paths.root.glob(".name-match.json.*.tmp"))


def _results(paths: VolumePaths, n_candidates: list[int]) -> None:
    paths.results.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(n_candidates, start=1):
        record: dict[str, Any] = {
            "page": str(i),
            "n_streets": 4,
            "n_candidates": n,
            "status": "REJECTED" if n == 0 else "OK",
        }
        (paths.results / f"p{i}.json").write_text(json.dumps(record))


def _notes(paths: VolumePaths, volume: str = "v") -> list[str]:
    stage_report(paths, volume)
    return list(json.loads((paths.root / "report.json").read_text())["notes"])


def test_report_warns_on_a_starved_volume_in_both_formats(tmp_path: Path) -> None:
    """Both halves of the signal, in report.json and report.md alike."""
    paths = VolumePaths(root=tmp_path / "v")
    _results(paths, [0] * 8 + [5, 6])
    paths.name_match.write_text(
        json.dumps({"volume": "v", "reads": 1076, "matched": 624, "match_rate": 0.58})
    )

    notes = _notes(paths)
    assert len(notes) == 1
    note = notes[0]
    assert "suspect a historic street-name alias gap" in note
    assert "58% of 1076 street reads" in note and "8 of 10 pages" in note
    assert "ADVISORY only" in note and SWEEP_RECORD in note
    assert "audit_alias_coverage.py" in note
    assert f"- NOTE: {note}" in (paths.root / "report.md").read_text()


def test_report_is_silent_on_a_healthy_volume(tmp_path: Path) -> None:
    paths = VolumePaths(root=tmp_path / "v")
    _results(paths, [5] * 19 + [0])  # 5% zero-candidate, below the bar
    paths.name_match.write_text(
        json.dumps({"volume": "v", "reads": 1058, "matched": 899, "match_rate": 0.85})
    )
    assert _notes(paths) == []


def test_each_half_of_the_signal_fires_on_its_own(tmp_path: Path) -> None:
    """A low match rate with a clean funnel warns, and so does the reverse."""
    starved_names = VolumePaths(root=tmp_path / "names")
    _results(starved_names, [5] * 20)
    starved_names.name_match.write_text(
        json.dumps({"reads": 950, "matched": 741, "match_rate": 0.78})
    )
    note = _notes(starved_names, "names")[0]
    assert "78% of 950 street reads" in note and "pages produced no match candidates" not in note

    starved_funnel = VolumePaths(root=tmp_path / "funnel")
    _results(starved_funnel, [0] * 3 + [5] * 7)
    starved_funnel.name_match.write_text(
        json.dumps({"reads": 989, "matched": 811, "match_rate": 0.82})
    )
    note = _notes(starved_funnel, "funnel")[0]
    assert "3 of 10 pages produced no match candidates (30%" in note
    assert "street reads match" not in note


def test_each_half_is_floored_on_the_sample_it_actually_measures(tmp_path: Path) -> None:
    """The two halves are shares of different things, so they floor on
    different things. A five-segment special sheet reading 5 of 5
    zero-candidate is not a funnel measurement — but a two-segment sheet can
    still carry a real vocabulary sample, and gating its match rate on page
    count would suppress a true positive with a proxy for a question it does
    not measure (the corpus has exactly that volume: 2 pages, 139 reads,
    genuine retired names among the unmatched)."""
    special = VolumePaths(root=tmp_path / "special")
    _results(special, [0] * (MIN_PAGES_FOR_NOTE - 1))
    special.name_match.write_text(json.dumps({"reads": 16, "matched": 4, "match_rate": 0.25}))
    assert _notes(special, "special") == [], "too few pages AND too few reads"

    # few pages, ample reads: the vocabulary half must survive
    small_book = VolumePaths(root=tmp_path / "small-book")
    _results(small_book, [0, 0])
    small_book.name_match.write_text(
        json.dumps({"reads": 139, "matched": 103, "match_rate": 0.741})
    )
    note = _notes(small_book, "small-book")[0]
    assert "74% of 139 street reads" in note
    assert "pages produced no match candidates" not in note, "2 pages cannot floor a funnel"

    # ample pages, few reads: the funnel half must survive
    thin_reads = VolumePaths(root=tmp_path / "thin-reads")
    _results(thin_reads, [0] * MIN_PAGES_FOR_NOTE)
    thin_reads.name_match.write_text(
        json.dumps({"reads": MIN_READS_FOR_NOTE - 1, "matched": 4, "match_rate": 0.04})
    )
    note = _notes(thin_reads, "thin-reads")[0]
    assert f"{MIN_PAGES_FOR_NOTE} of {MIN_PAGES_FOR_NOTE} pages" in note
    assert "street reads match" not in note


def test_a_pre_sidecar_volume_keeps_the_funnel_half_and_never_mentions_itself(
    tmp_path: Path,
) -> None:
    """The absent metric is silent by design: warning about it would fire on
    every already-matched volume and train the reader to skip the note."""
    starved = VolumePaths(root=tmp_path / "starved")
    _results(starved, [0] * 3 + [5] * 7)
    note = _notes(starved, "starved")[0]
    assert "3 of 10 pages produced no match candidates" in note
    assert "street reads" not in note and "sidecar" not in note

    healthy = VolumePaths(root=tmp_path / "healthy")
    _results(healthy, [5] * 20)
    assert _notes(healthy, "healthy") == []


@pytest.mark.parametrize(
    "body",
    [
        "{ truncated",
        '["not", "an", "object"]',
        '{"match_rate": 0.5}',  # formatted with reads: must not KeyError
        '{"reads": 10, "matched": 5, "match_rate": "0.5"}',
        '{"reads": 10, "matched": 5, "match_rate": false}',  # bool is an int
        '{"reads": 10, "matched": 5, "match_rate": -3}',
        '{"reads": "ten", "matched": 5, "match_rate": 0.5}',
    ],
)
def test_a_malformed_sidecar_is_silent_not_fatal(tmp_path: Path, body: str) -> None:
    """The report stage runs AFTER every result is on disk; a hand-edited or
    half-written sidecar may only cost the advisory line, never the report."""
    paths = VolumePaths(root=tmp_path / "v")
    _results(paths, [5] * 20)
    paths.name_match.write_text(body)
    assert load_name_match(paths) is None
    assert _notes(paths) == []


def test_the_note_is_advisory_and_moves_no_counter(tmp_path: Path) -> None:
    """The bars misclassify in both directions, so nothing may depend on them:
    the same results produce the same funnel, warned or not."""
    plain = VolumePaths(root=tmp_path / "plain")
    _results(plain, [0] * 8 + [5, 6])
    _notes(plain, "v")
    without = json.loads((plain.root / "report.json").read_text())

    warned = VolumePaths(root=tmp_path / "warned")
    _results(warned, [0] * 8 + [5, 6])
    warned.name_match.write_text(json.dumps({"reads": 100, "matched": 58, "match_rate": 0.58}))
    _notes(warned, "v")
    with_note = json.loads((warned.root / "report.json").read_text())

    assert with_note["notes"] != without["notes"]
    assert {k: v for k, v in with_note.items() if k != "notes"} == {
        k: v for k, v in without.items() if k != "notes"
    }


def test_the_note_is_total_on_any_input_it_is_handed() -> None:
    """It is public, so its validation cannot live only in the loader: an
    empty volume, a None sidecar and a half-written one all warn about
    nothing rather than raising inside the report stage."""
    assert alias_gap_note(None, {}) is None
    assert alias_gap_note({"reads": 0, "matched": 0, "match_rate": None}, {}) is None
    # match_rate present, reads absent — the shape that used to KeyError
    assert alias_gap_note({"match_rate": 0.1}, {"1": {"n_candidates": 5}}) is None
    assert alias_gap_note({"match_rate": 0.1, "reads": "many"}, {}) is None
