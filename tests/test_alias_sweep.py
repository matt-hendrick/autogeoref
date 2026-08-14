"""The alias-sweep command end to end, on a synthetic city.

Contracts pinned here:

- structural rules are one implementation, shared with the sweep;
- a volume above the tripwire's bars, on the skip list, already marked, or in a city with
  no rename source is NEVER proposed for, and every skip is reported with its reason;
- the clean tier MERGES over a landed table and never overwrites or removes an entry
  carrying owner sign-off;
- a validator failure ABORTS the volume, writing nothing, and exits nonzero;
- the written file's bytes are a pure function of its inputs, so a second sweep rewrites
  nothing and the fixture manifest does not churn.

No fixtures, no network, no model call.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from autogeoref.alias.propose import CLEAN, HELD, Candidate, Proposal
from autogeoref.alias.sweep import (
    SKIP_ABOVE_BARS,
    SKIP_DECLARED,
    SKIP_MARKED,
    SKIP_NO_BOUNDS,
    SKIP_NO_SOURCE,
    alias_document,
    annotated_volumes,
    marker_path,
    render_report,
    run_sweep,
)
from autogeoref.alias.validate import alias_files, volume_of
from autogeoref.config.load import load_city_config
from autogeoref.config.model import ConfigError
from autogeoref.paths import VolumePaths

VOLUME = "sanborn00000_001"

FEATURES: list[dict[str, Any]] = [
    {
        "type": "Feature",
        "properties": {"street_nam": "ALPHA", "street_typ": "ST"},
        "geometry": {"type": "LineString", "coordinates": [[-87.660, 41.900], [-87.660, 41.940]]},
    },
    {
        "type": "Feature",
        "properties": {"street_nam": "BETA", "street_typ": "ST"},
        "geometry": {"type": "LineString", "coordinates": [[-87.670, 41.920], [-87.640, 41.920]]},
    },
    {
        "type": "Feature",
        "properties": {"street_nam": "SOLO", "street_typ": "AVE"},
        "geometry": {"type": "LineString", "coordinates": [[-87.659, 41.912], [-87.659, 41.928]]},
    },
    {
        "type": "Feature",
        "properties": {"street_nam": "SECOND", "street_typ": "AVE"},
        "geometry": {"type": "LineString", "coordinates": [[-87.657, 41.912], [-87.657, 41.928]]},
    },
    # the numbered-PLACE twin: a bare "22ND" alias key would drag this
    # centerline into the value's bucket, silently
    {
        "type": "Feature",
        "properties": {"street_nam": "W 22ND", "street_typ": "PL"},
        "geometry": {"type": "LineString", "coordinates": [[-87.670, 41.905], [-87.640, 41.905]]},
    },
]

SOURCE = """\
-Only St., Solo Ave., 1250W fr. 1600 to 2600N
-Other St., Second Ave., 1230W fr. 1600 to 2600N
-22nd St., Solo Ave., 1250W fr. 1600 to 2600N
"""

# every read that is NOT a reference street is an unmatched family, which is how
# the tripwire's match rate lands below its bar on this synthetic volume
READS = ["ALPHA ST.", "BETA ST.", "ONLY ST.", "OTHER ST.", "22ND ST."]


def _write_volume(work: Path, volume: str = VOLUME, pages: int = 12) -> VolumePaths:
    """A volume with enough pages and reads to clear the tripwire's floors."""
    paths = VolumePaths(root=work / volume)
    paths.annotations.mkdir(parents=True, exist_ok=True)
    paths.sheets.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for i in range(1, pages + 1):
        page = str(i)
        annotation = {
            "streets": [
                {"name": name, "bbox": [0, 0, 10, 10], "orientation": "horizontal"}
                for name in READS * 2
            ]
        }
        (paths.annotations / f"p{page}.json").write_text(json.dumps(annotation))
        manifest[f"p{page}"] = {"full_size": [100, 100], "scale": 1.0}
    paths.manifest.write_text(json.dumps(manifest))
    return paths


def _city_toml(
    tmp_path: Path,
    *,
    rename_source: bool = True,
    skip: dict[str, str] | None = None,
) -> Path:
    (tmp_path / "aliases").mkdir(exist_ok=True)
    (tmp_path / "centerlines.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": FEATURES})
    )
    (tmp_path / "source.txt").write_text(SOURCE)
    lines = [
        "[city]",
        'name = "Synthetic"',
        'centerlines = "centerlines.geojson"',
        'aliases_dir = "aliases"',
        "address_grid_origin = [-87.6278, 41.8819]",
        "address_grid_units_per_mile = 800",
    ]
    if rename_source:
        lines += [
            'rename_source_parser = "martin1948"',
            'rename_source_text = "source.txt"',
            'rename_source_citation = "synthetic source, 1948"',
        ]
    if skip:
        lines.append("[city.alias_sweep_skip]")
        lines += [f'{vid} = "{reason}"' for vid, reason in skip.items()]
    lines += [
        f"[volumes.{VOLUME}]",
        "bounds_bbox = [-87.68, 41.89, -87.63, 41.95]",
        "addresses_modern = true",
        "[volumes.sanborn00000_002]",
        "bounds_bbox = [-87.68, 41.89, -87.63, 41.95]",
        "addresses_modern = true",
    ]
    path = tmp_path / "city.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def city_path(tmp_path: Path) -> Path:
    return _city_toml(tmp_path)


@pytest.fixture
def work(tmp_path: Path) -> Path:
    root = tmp_path / "work"
    _write_volume(root)
    return root


def _sweep(city_path: Path, work: Path, **kwargs: Any) -> Any:
    return run_sweep(
        load_city_config(city_path), city_path, work, today=date(2026, 7, 26), **kwargs
    )


def test_the_clean_tier_is_written_with_provenance_and_a_marker(
    city_path: Path, work: Path
) -> None:
    result = _sweep(city_path, work)
    city = load_city_config(city_path)
    written = {o.volume: dict(o.written) for o in result.written}
    assert written == {VOLUME: {"ONLY": "SOLO", "OTHER": "SECOND"}}

    document = json.loads(city.aliases_path(VOLUME).read_text())
    assert document["ONLY"] == "SOLO"
    generated = " ".join(document["_generated"])
    assert "synthetic source, 1948" in generated
    assert "alias-sweep" in generated
    assert "docs/INTERNALS.md" in generated
    # the held tier is carried in the file too, so a reader of the fixture sees
    # what was considered and refused
    assert any("22ND" in line for line in document["_generated_held"])
    assert document["_generated_keys"] == ["ONLY", "OTHER"]
    # ASCII only, like every other alias table in the tree
    raw = city.aliases_path(VOLUME).read_bytes()
    assert raw.decode("ascii")

    marker = json.loads(marker_path(VolumePaths(root=work / VOLUME)).read_text())
    assert marker["run_date"] == "2026-07-26"
    assert marker["entries_written"] == {"ONLY": "SOLO", "OTHER": "SECOND"}
    assert marker["match_rate_before"] < marker["match_rate_after"]


def test_a_bare_numbered_key_is_held_not_written(city_path: Path, work: Path) -> None:
    """A rename of 22nd Street is not a rename of 22nd Place, so a person decides.

    The index no longer moves either way — the twin keeps its own key whatever
    the table says — so this is the auto-writer declining a claim it cannot
    make, not the table failing a rule.
    """
    result = _sweep(city_path, work)
    outcome = result.outcomes[0]
    assert "22ND" not in outcome.written
    held = {p.key: p.reason for p in outcome.held}
    assert "place or court twin" in held["22ND"]
    # and the volume kept its other, sound entries
    assert outcome.written == {"ONLY": "SOLO", "OTHER": "SECOND"}


def test_a_second_sweep_rewrites_nothing(city_path: Path, work: Path) -> None:
    """Idempotence, and the reason the file carries no run id or date."""
    _sweep(city_path, work)
    city = load_city_config(city_path)
    path = city.aliases_path(VOLUME)
    first = path.read_bytes()
    marker = marker_path(VolumePaths(root=work / VOLUME))
    assert marker.is_file()

    # the marker latches: a plain re-run skips the volume entirely
    again = _sweep(city_path, work)
    assert again.outcomes[0].skipped == SKIP_MARKED
    assert path.read_bytes() == first

    # and --force re-derives the same table, so there is nothing left to write:
    # the entries are now the volume's own, and the file is untouched
    forced = _sweep(city_path, work, force=True)
    assert forced.outcomes[0].written == {}
    assert forced.outcomes[0].match_rate_before == forced.outcomes[0].match_rate_after
    assert path.read_bytes() == first


def test_a_dry_run_writes_nothing_but_still_projects_the_rate(city_path: Path, work: Path) -> None:
    result = _sweep(city_path, work, dry_run=True)
    outcome = result.outcomes[0]
    assert outcome.written
    assert outcome.match_rate_after > outcome.match_rate_before
    assert not load_city_config(city_path).aliases_path(VOLUME).exists()
    assert not marker_path(VolumePaths(root=work / VOLUME)).is_file()
    assert "DRY RUN" in render_report(result, dry_run=True)


def test_an_existing_entry_is_never_overwritten_or_removed(city_path: Path, work: Path) -> None:
    """Landed entries carry owner sign-off this command does not have."""
    city = load_city_config(city_path)
    city.aliases_path(VOLUME).write_text(
        json.dumps({"_comment": "hand-built. Deliberately NOT aliased: OTHER.", "ONLY": "SECOND"})
    )
    result = _sweep(city_path, work)
    document = json.loads(city.aliases_path(VOLUME).read_text())
    assert document["ONLY"] == "SECOND", "the sweep must not contradict a landed value"
    assert document["_comment"].startswith("hand-built")
    assert result.outcomes[0].written == {"OTHER": "SECOND"}
    # the prose disposition it just superseded is surfaced, not silently broken
    assert result.outcomes[0].supersedes == ("OTHER",)
    assert "Supersedes a recorded disposition" in render_report(result)


@pytest.mark.parametrize(
    "comment",
    [
        "SOLONGONLY is a different street.",  # ONLY as a SUFFIX (the MILTON/HAMILTON bug)
        "ONLYX and SOLONG are fine.",  # ONLY as a PREFIX
    ],
)
def test_a_substring_of_a_comment_is_not_a_superseded_disposition(
    city_path: Path, work: Path, comment: str
) -> None:
    """``MILTON`` is not dispositioned by a comment that mentions ``HAMILTON``.

    Both sides of the token boundary, because the real false positive was a
    SUFFIX match and a lookahead alone would not have caught it.
    """
    city = load_city_config(city_path)
    city.aliases_path(VOLUME).write_text(json.dumps({"_comment": comment}))
    result = _sweep(city_path, work)
    assert result.outcomes[0].supersedes == ()


def test_a_declared_skip_wins_over_force(tmp_path: Path, work: Path) -> None:
    city_path = _city_toml(tmp_path, skip={VOLUME: "an index sheet, not an alias gap"})
    result = _sweep(city_path, work, force=True)
    assert result.outcomes[0].skipped is not None
    assert result.outcomes[0].skipped.startswith(SKIP_DECLARED)
    assert "an index sheet" in result.outcomes[0].skipped
    assert not load_city_config(city_path).aliases_path(VOLUME).exists()
    assert "an index sheet" in render_report(result)


def test_a_city_with_no_rename_source_is_reported_never_proposed_for(
    tmp_path: Path, work: Path
) -> None:
    """Degrade to visibility, never to invention."""
    city_path = _city_toml(tmp_path, rename_source=False)
    result = _sweep(city_path, work)
    outcome = result.outcomes[0]
    assert outcome.skipped == SKIP_NO_SOURCE
    assert outcome.reads > 0, "it is still MEASURED"
    assert not outcome.written


def test_a_volume_above_the_bars_is_left_alone(city_path: Path, tmp_path: Path) -> None:
    work = tmp_path / "healthy"
    paths = VolumePaths(root=work / VOLUME)
    paths.annotations.mkdir(parents=True)
    paths.sheets.mkdir(parents=True)
    manifest = {}
    for i in range(1, 13):
        annotation = {
            "streets": [
                {"name": n, "bbox": [0, 0, 10, 10], "orientation": "horizontal"}
                for n in ["ALPHA ST.", "BETA ST."] * 5
            ]
        }
        (paths.annotations / f"p{i}.json").write_text(json.dumps(annotation))
        manifest[f"p{i}"] = {"full_size": [100, 100], "scale": 1.0}
    paths.manifest.write_text(json.dumps(manifest))
    result = _sweep(city_path, work)
    assert result.outcomes[0].skipped == SKIP_ABOVE_BARS
    assert result.outcomes[0].match_rate_before == 1.0


def test_a_broken_volume_does_not_end_the_sweep(city_path: Path, work: Path) -> None:
    """A corpus pass must survive one unreadable volume, and name it."""
    broken = VolumePaths(root=work / "sanborn00000_002")
    broken.sheets.mkdir(parents=True)
    broken.manifest.write_text("{ not json")
    broken.annotations.mkdir(parents=True)
    (broken.annotations / "p1.json").write_text("{}")

    result = _sweep(city_path, work)
    assert {o.volume for o in result.outcomes} == {VOLUME, "sanborn00000_002"}
    assert result.outcomes[0].written  # the healthy one still ran
    failed = next(o for o in result.outcomes if o.volume == "sanborn00000_002")
    assert failed.error is not None
    assert failed.error in render_report(result)
    # no marker, so a retry re-does it rather than latching the failure
    assert not marker_path(broken).is_file()


def test_a_validator_failure_aborts_the_volume_and_writes_nothing(
    city_path: Path, work: Path
) -> None:
    """The per-entry guard vets ONE entry; the whole-table gate vets the merge.

    A landed ``SOLO -> SECOND`` keys a street that is itself in bounds, so the
    table it is part of does not satisfy the rules — and the sweep refuses to
    add to a table that does not. Nothing is written, the rule is named, and the
    command exits nonzero: the validator rejecting a merge is a defect to look
    at, not a routine outcome like a skip or a held entry.
    """
    city = load_city_config(city_path)
    original = json.dumps({"SOLO": "SECOND"})
    city.aliases_path(VOLUME).write_text(original)

    result = _sweep(city_path, work)
    outcome = result.outcomes[0]
    assert outcome.aborted
    assert any("SHADOWS AN IN-BOUNDS STREET" in failure for failure in outcome.aborted)
    assert not outcome.written
    # NO marker: a marker here would latch the failure, and the next run would
    # skip the volume, report nothing, and exit 0 on a live defect.
    assert not marker_path(VolumePaths(root=work / VOLUME)).is_file()
    assert city.aliases_path(VOLUME).read_text() == original
    assert result.aborted, "an abort is what makes the command exit nonzero"
    assert "Aborted" in render_report(result)


def test_a_volume_with_no_resolvable_bounds_is_skipped(city_path: Path, work: Path) -> None:
    """Bounds provenance decides value-in-bounds verdicts; guessing is not an option."""
    undeclared = VolumePaths(root=work / "sanborn00000_099")
    undeclared.sheets.mkdir(parents=True)
    undeclared.manifest.write_text(json.dumps({"p1": {"full_size": [100, 100], "scale": 1.0}}))
    undeclared.annotations.mkdir(parents=True)
    (undeclared.annotations / "p1.json").write_text(json.dumps({"streets": []}))

    result = _sweep(city_path, work)
    outcome = next(o for o in result.outcomes if o.volume == "sanborn00000_099")
    assert outcome.skipped == SKIP_NO_BOUNDS


def test_annotated_volumes_ignores_cache_siblings(work: Path) -> None:
    paths = VolumePaths(root=work / VOLUME)
    (paths.annotations / "p1.annotation.active.json").write_text("{}")
    assert annotated_volumes(work) == [VOLUME]
    assert annotated_volumes(work / "absent") == []


def test_the_report_lists_every_scan_write_and_held_row(city_path: Path, work: Path) -> None:
    report = render_report(_sweep(city_path, work, dry_run=True), dry_run=True)
    assert "## Scan" in report and "## Written" in report and "## Held" in report
    assert "`ONLY` -> `SOLO`" in report
    assert "`22ND`" in report
    # the placement follow-up is printed, never taken
    assert "--no-annotate" in report
    assert "demotes" in report and "docs/OPERATIONS.md" in report


def test_a_typo_in_volumes_is_an_error_not_a_silent_skip(city_path: Path, work: Path) -> None:
    result = _sweep(city_path, work, volumes=["sanborn00000_999"])
    assert result.outcomes[0].error is not None
    assert "typo in --volumes" in result.outcomes[0].error
    assert result.aborted, "it must make the command exit nonzero"


def test_an_already_swept_volume_still_reports_its_numbers(city_path: Path, work: Path) -> None:
    """The scan table is the audit surface; a row of n/a audits nothing."""
    _sweep(city_path, work)
    again = _sweep(city_path, work).outcomes[0]
    assert again.skipped == SKIP_MARKED
    assert again.reads > 0
    assert again.match_rate_before is not None


def test_alias_document_preserves_comments_and_is_byte_stable() -> None:
    """Byte-stable, not merely equal: the manifest must not churn on a re-run."""
    previous = {"_comment": "hand-built", "OLD": "VALUE"}
    clean = (
        Proposal(
            key="NEW",
            reads=3,
            tier=CLEAN,
            reason="unique in-bounds candidate; locality 12 m over 2 sheets",
            value="OTHER",
            candidates=(
                Candidate(value="OTHER", sheets=2, locality_m=12.0, source_line="-New St., Other"),
            ),
        ),
    )
    held = (Proposal(key="HELDKEY", reads=1, tier=HELD, reason="locality on 1 sheet", value="X"),)
    table = {"OLD": "VALUE", "NEW": "OTHER"}

    def render() -> bytes:
        return json.dumps(
            alias_document(VOLUME, "cite", table, clean, held, previous), indent=2
        ).encode()

    assert render() == render()
    document = alias_document(VOLUME, "cite", table, clean, held, previous)
    assert next(iter(document)) == "_comment", "the human comment stays first"
    assert document["OLD"] == "VALUE"
    assert document["_generated_keys"] == ["NEW"]
    # key order is part of the bytes, so it is asserted, not assumed
    assert list(document) == [
        "_comment",
        "_generated",
        "_generated_keys",
        "_generated_held",
        "OLD",
        "NEW",
    ]


# --- the structural rules, as the sweep uses them --------------------------------------
# The rules themselves are pinned one at a time in `test_alias_validate.py`.


def test_every_candidate_is_judged_on_its_own(city_path: Path, work: Path) -> None:
    """No landed entry can vouch for a new one, however long it has been there."""
    result = _sweep(city_path, work)
    held = {p.key: p.reason for p in result.outcomes[0].held}
    assert "22ND" in held
    assert "22ND" not in result.outcomes[0].written


# --- the two city-config seams -------------------------------------------------------


def _bare_city(tmp_path: Path, extra: list[str]) -> Path:
    (tmp_path / "aliases").mkdir(exist_ok=True)
    (tmp_path / "centerlines.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": FEATURES})
    )
    path = tmp_path / "city.toml"
    path.write_text(
        "\n".join(
            [
                "[city]",
                'name = "Synthetic"',
                'centerlines = "centerlines.geojson"',
                'aliases_dir = "aliases"',
                *extra,
            ]
        )
        + "\n"
    )
    return path


def test_a_city_declaring_neither_seam_loads_with_both_absent(tmp_path: Path) -> None:
    city = load_city_config(_bare_city(tmp_path, []))
    assert city.rename_source is None
    assert city.address_grid is None
    assert city.alias_sweep_skip == {}


def test_a_half_declared_rename_source_is_a_config_error(tmp_path: Path) -> None:
    """A parser with nothing to parse is a slip, not "this city has no source"."""
    with pytest.raises(ConfigError, match="together"):
        load_city_config(_bare_city(tmp_path, ['rename_source_parser = "martin1948"']))


def test_an_unknown_parser_is_refused_at_load(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown rename_source_parser"):
        load_city_config(
            _bare_city(
                tmp_path,
                [
                    'rename_source_parser = "handwaving"',
                    'rename_source_text = "s.txt"',
                    'rename_source_citation = "x"',
                ],
            )
        )


def test_an_empty_citation_is_refused(tmp_path: Path) -> None:
    """The citation is copied verbatim into every file the source writes."""
    with pytest.raises(ConfigError, match="rename_source_citation"):
        load_city_config(
            _bare_city(
                tmp_path,
                [
                    'rename_source_parser = "martin1948"',
                    'rename_source_text = "s.txt"',
                    'rename_source_citation = "   "',
                ],
            )
        )


def test_a_half_declared_address_grid_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="configured together"):
        load_city_config(_bare_city(tmp_path, ["address_grid_units_per_mile = 800"]))


@pytest.mark.parametrize(
    "extra",
    [
        ["address_grid_origin = [-87.6]", "address_grid_units_per_mile = 800"],
        ["address_grid_origin = [-200.0, 41.9]", "address_grid_units_per_mile = 800"],
        ["address_grid_origin = [-87.6, 41.9]", "address_grid_units_per_mile = 0"],
    ],
)
def test_a_malformed_address_grid_is_refused(tmp_path: Path, extra: list[str]) -> None:
    with pytest.raises(ConfigError, match="address_grid"):
        load_city_config(_bare_city(tmp_path, extra))


def test_a_skip_list_entry_must_carry_a_reason(tmp_path: Path) -> None:
    """A skip whose reason nobody wrote down is indistinguishable from a bug."""
    with pytest.raises(ConfigError, match="alias_sweep_skip"):
        load_city_config(_bare_city(tmp_path, ["[city.alias_sweep_skip]", f'{VOLUME} = ""']))


def test_the_chicago_config_declares_both_seams() -> None:
    """The shipped city is the measured basis; if it stops declaring, say so loudly."""
    path = Path(__file__).resolve().parents[1] / "configs" / "chicago" / "chicago.toml"
    city = load_city_config(path)
    assert city.rename_source is not None
    assert city.rename_source.parser == "martin1948"
    assert "Martin 1948" in city.rename_source.citation
    assert city.address_grid is not None
    assert city.address_grid.units_per_mile == 800
    assert set(city.alias_sweep_skip) >= {
        "sanborn01790_013",
        "sanborn01790_014",
        "sanborn01790_016",
        "sanborn01790_190",
    }


def test_alias_files_discovers_rather_than_hardcodes(tmp_path: Path) -> None:
    (tmp_path / "aliases-sanborn01790_039.json").write_text("{}")
    (tmp_path / "aliases-sanborn01790_019.json").write_text("{}")
    (tmp_path / "not-an-alias-file.json").write_text("{}")
    found = alias_files(tmp_path)
    assert [volume_of(p) for p in found] == ["sanborn01790_019", "sanborn01790_039"]
