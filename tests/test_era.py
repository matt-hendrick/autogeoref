"""`autogeoref era`: propose an address era from a year, and never guess one.

Declaring `addresses_modern` is one TOML line, and it walled off 29 of Chicago's 34
place candidates. It is also the single most dangerous line in the config: an
undeclared era means MODERN, and on a pre-renumbering volume that reads printed
numerals against a grid ~900 m away while the addresses channel — the only one
permitted to REFUTE — vetoes CORRECT sheets.

So this module proposes, shows its working, and stops. The tests below pin the three
things that keep it honest: it refuses rather than guesses, it says where its year came
from, and a config it writes is one that reloads as the value it proposed.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from autogeoref import era
from autogeoref.config.load import load_city_config
from autogeoref.config.model import CityConfig, VolumeConfig


def _city(
    *, renumbering_year: int | None = 1909, table: bool = True, **volumes: bool
) -> CityConfig:
    return CityConfig(
        name="Chicago, Ill.",
        centerlines_path=Path("cl.geojson"),
        aliases_dir=Path("aliases"),
        renumbering_table_path=Path("renumbering-chicago-1909.json") if table else None,
        renumbering_year=renumbering_year,
        volumes={
            vid: VolumeConfig(identifier=vid, addresses_modern=modern)
            for vid, modern in volumes.items()
        },
    )


def _catalog(year: int | None, source: str = "date") -> dict[str, dict[str, object]]:
    return {"vol_a": {"year": year, "year_source": source}} if year else {}


def test_the_provenance_flag_fails_closed_not_open() -> None:
    """An absent or unrecognised `year_source` must warn, not pass as a catalogued date.

    `propose` takes any catalog dict. If the flag only fired on the exact string
    "description", then a catalog that carries no provenance — or a rename of the key in
    `viewer.sources.loc_titles` — would silently render a scraped year as a trusted one, with
    every test still green. A signal a typo can switch off is not a safety signal.
    """
    trusted = era.propose(_city(), {"vol_a": {"year": 1896, "year_source": "date"}}, "vol_a")
    assert trusted.year_is_trusted

    for untrusted in (
        {"year": 1896, "year_source": "description"},  # scraped from the blurb
        {"year": 1896},  # a catalog that does not say
        {"year": 1896, "year_source": "prose"},  # a key that drifted
    ):
        p = era.propose(_city(), {"vol_a": untrusted}, "vol_a")
        assert not p.year_is_trusted, untrusted
        assert "NOT A CATALOGUED DATE" in era.render(p, _city()), untrusted


def test_a_volume_printed_before_the_renumbering_is_proposed_false() -> None:
    p = era.propose(_city(), _catalog(1896), "vol_a")
    assert p.ok and p.modern is False
    text = era.render(p, _city())
    assert "1896" in text and "BEFORE" in text
    assert "addresses_modern = false" in text
    # a `false` still owes the operator the question of WHICH book: Chicago renumbered
    # the Loop separately in 1911, and the two disagree by a median 934 m
    assert "CHECK WHICH BOOK" in text


def test_a_volume_printed_after_it_is_proposed_true() -> None:
    p = era.propose(_city(), _catalog(1917), "vol_a")
    assert p.ok and p.modern is True
    assert "addresses_modern = true" in era.render(p, _city())


def test_no_year_refuses_rather_than_guessing() -> None:
    """The engine never infers an era from silence; nor does this, from a year it lacks."""
    p = era.propose(_city(), _catalog(None), "vol_a")
    assert not p.ok and p.modern is None
    assert "no edition year" in (p.refusal or "")
    with pytest.raises(era.EraError):
        era.declare(Path("unused.toml"), p)


def test_an_edition_dated_in_the_renumbering_year_refuses() -> None:
    """A year cannot say which side of the switch its sheets were printed on.

    Chicago has two (`_069`, `_072`). Coin-flipping them would arm a channel that can
    veto correct sheets, to save a human ten seconds.
    """
    p = era.propose(_city(), _catalog(1909), "vol_a")
    assert not p.ok
    assert "the very year" in (p.refusal or "")


def test_a_scraped_year_is_flagged_to_the_human_confirming_it() -> None:
    """LOC left `date` null on its four Chicago subject maps; the year comes from prose.

    That fallback was written to letter a title in the viewer, where a wrong year is
    cosmetic. Here it would set `addresses_modern`. The operator is told which kind of
    year they are looking at — the confirm is the only place that fact can matter.
    """
    p = era.propose(_city(), _catalog(1901, source="description"), "vol_a")
    assert p.ok and p.modern is False and not p.year_is_trusted

    text = era.render(p, _city())
    assert "NOT A CATALOGUED DATE" in text
    assert "title page" in text

    # ...and an ordinary catalogued year says nothing of the sort
    assert "NOT A CATALOGUED DATE" not in era.render(
        era.propose(_city(), _catalog(1896), "vol_a"), _city()
    )


@pytest.mark.golden  # needs the frozen catalog
def test_the_real_catalog_reaches_propose_with_its_provenance_intact(fixtures_dir: Path) -> None:
    """The SEAM. `viewer.sources.loc_titles` produces the provenance and `era.propose` consumes it.

    Both halves were tested apart — test_viewer pinned what loc_titles emits, and the tests
    above hand `propose` a dict they built themselves. Neither would notice if the key were
    renamed on one side, which is the one way this feature dies quietly. So: the real
    parser, the real frozen catalog, all the way through to the rendered proposal.
    """
    from autogeoref.viewer.sources import loc_titles

    catalog = loc_titles(fixtures_dir / "loc-catalog-chicago.json", "Chicago, Ill.")
    city = _city()

    # _014: a SUBJECT map (grain elevators). LOC's `date` is null; the year is prose.
    scraped = era.propose(city, catalog, "sanborn01790_014")
    assert scraped.year == 1901 and scraped.modern is False
    assert not scraped.year_is_trusted
    assert "NOT A CATALOGUED DATE" in era.render(scraped, city)

    # _024: an ordinary bound volume with a real catalogued date. No warning.
    ordinary = era.propose(city, catalog, "sanborn01790_024")
    assert ordinary.year == 1917 and ordinary.year_is_trusted
    assert "NOT A CATALOGUED DATE" not in era.render(ordinary, city)

    # _190: LOC's month-year description and created_published corroborate 1933.
    # It is still a description year, so era automation must warn rather than treat it
    # like a catalogued structured date.
    month_year = era.propose(city, catalog, "sanborn01790_190")
    assert month_year.year == 1933 and month_year.modern is True
    assert not month_year.year_is_trusted
    assert "NOT A CATALOGUED DATE" in era.render(month_year, city)


def test_a_city_that_never_renumbered_needs_no_declaration() -> None:
    p = era.propose(_city(table=False), _catalog(1896), "vol_a")
    assert not p.ok
    assert "ships no renumbering table" in (p.refusal or "")


def test_a_city_that_renumbered_but_declares_no_year_refuses() -> None:
    """The calendar is the CITY's. This tool will not assume one."""
    p = era.propose(_city(renumbering_year=None), _catalog(1896), "vol_a")
    assert not p.ok
    assert "renumbering_year" in (p.refusal or "")


def test_an_already_declared_volume_is_left_alone() -> None:
    p = era.propose(_city(vol_a=True), _catalog(1896), "vol_a")
    assert p.declared is True
    assert "already declares" in era.render(p, _city(vol_a=True))


def test_declare_refuses_to_overrule_an_existing_declaration(tmp_path: Path) -> None:
    """A declared proposal is `ok` (it has a value and no refusal), so the write guard has
    to catch it by name.

    Without that, `declare` inserts a SECOND `addresses_modern` into the block — a duplicate
    key. tomllib rejects it and the config is restored, so it fails safe; but the operator
    gets a TOMLDecodeError about a column number instead of the sentence they need.
    """
    cfg = _write(tmp_path)
    city = load_city_config(cfg)
    before = cfg.read_text()
    # _024 already declares addresses_modern = true; the catalog year would say false
    p = era.propose(city, {"sanborn01790_024": {"year": 1896}}, "sanborn01790_024")
    assert p.declared is True

    with pytest.raises(era.EraError, match="already declares"):
        era.declare(cfg, p)
    assert cfg.read_text() == before


# ----------------------------------------------------------------- the write

TOML = """\
[city]
name = "Chicago, Ill."
centerlines = "cl.geojson"
aliases_dir = "aliases"
renumbering_table = "renumbering.json"
renumbering_year = 1909

[volumes.sanborn01790_024]
bounds_bbox = [-87.66, 41.87, -87.60, 41.90]
addresses_modern = true
"""


def _write(tmp_path: Path, body: str = TOML) -> Path:
    p = tmp_path / "city.toml"
    p.write_text(body)
    (tmp_path / "cl.geojson").write_text('{"type":"FeatureCollection","features":[]}')
    (tmp_path / "renumbering.json").write_text("[]")
    (tmp_path / "aliases").mkdir(exist_ok=True)
    return p


def test_declaring_a_new_volume_appends_a_block_that_reloads(tmp_path: Path) -> None:
    cfg = _write(tmp_path)
    city = load_city_config(cfg)
    p = era.propose(city, {"sanborn01790_002": {"year": 1896}}, "sanborn01790_002")

    era.declare(cfg, p)

    assert load_city_config(cfg).volume("sanborn01790_002").addresses_modern is False
    # the volume that was already declared is untouched
    assert load_city_config(cfg).volume("sanborn01790_024").addresses_modern is True


def test_a_dotted_volume_id_is_quoted_or_it_declares_the_wrong_table(tmp_path: Path) -> None:
    """`sanborn01790_006.5` is a real volume id.

    A bare `[volumes.sanborn01790_006.5]` is a NESTED table in TOML — it would declare an
    era for a volume that does not exist while the operator believed they had fixed the one
    that does, and the run would go on refusing.
    """
    cfg = _write(tmp_path)
    city = load_city_config(cfg)
    vid = "sanborn01790_006.5"
    p = era.propose(city, {vid: {"year": 1895}}, vid)

    era.declare(cfg, p)

    assert f'[volumes."{vid}"]' in cfg.read_text()
    assert load_city_config(cfg).volume(vid).addresses_modern is False


def test_a_write_that_does_not_reload_as_proposed_restores_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool that mis-declares an era is worse than the wall it replaces.

    The wall fails loudly; a wrong declaration vetoes correct sheets in the confident
    language of evidence. So the write is verified by RELOADING, and anything unexpected
    puts the original file back.
    """
    cfg = _write(tmp_path)
    original = cfg.read_text()
    city = load_city_config(cfg)
    p = era.propose(city, {"sanborn01790_002": {"year": 1896}}, "sanborn01790_002")

    # a text edit that lands somewhere harmless: the reload will not see the value
    monkeypatch.setattr(era, "_with_declaration", lambda text, _vol, _modern: text + "\n# nope\n")

    with pytest.raises(era.EraError, match="reloads as"):
        era.declare(cfg, p)
    assert cfg.read_text() == original, "the config must be exactly as it was"


def test_an_existing_block_gains_the_key_rather_than_a_second_block(tmp_path: Path) -> None:
    """A volume can already have a block for another reason (bounds_bbox) and no era.

    The declaration goes INTO that block. Appending a second `[volumes.X]` table would be a
    duplicate table, which tomllib rejects outright.
    """
    cfg = _write(
        tmp_path,
        TOML + '\n[volumes."sanborn01790_006.5"]\nbounds_bbox = [-87.7, 41.8, -87.6, 41.9]\n',
    )
    city = load_city_config(cfg)
    vid = "sanborn01790_006.5"
    p = era.propose(city, {vid: {"year": 1895, "year_source": "date"}}, vid)

    era.declare(cfg, p)

    reloaded = load_city_config(cfg)
    assert reloaded.volume(vid).addresses_modern is False
    assert reloaded.volume(vid).bounds_bbox is not None, "the block's other keys survive"
    assert cfg.read_text().count(f'[volumes."{vid}"]') == 1, "one block, not two"


# ------------------------------------------------------------------ the CLI


def _cli(tmp_path: Path, *args: str) -> int:
    from autogeoref.cli.entry import main

    cfg = _write(tmp_path)
    catalog = tmp_path / "cat.json"
    catalog.write_text(
        json.dumps(
            [
                {  # an ordinary bound volume: a real catalogued date
                    "id": "http://www.loc.gov/item/sanborn01790_002/",
                    "description": ["Vol. 2, 1896. 100 sheet(s). Bound."],
                    "date": "1896",
                },
                {  # a SUBJECT map: LOC left `date` null, the year is prose
                    "id": "http://www.loc.gov/item/sanborn01790_014/",
                    "description": ["1901. 56 sheet(s). Bound. Grain elevators."],
                    "date": None,
                },
            ]
        )
    )
    return main(["era", *args, "--city", str(cfg), "--loc-catalog", str(catalog)])


def test_cli_yes_will_not_rubber_stamp_a_year_that_is_not_a_catalogued_date(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--yes must not be able to buy back the one signal this feature exists to raise.

    A prose-scraped year written silently into `addresses_modern` arms the only evidence
    channel allowed to REFUTE, from a regex over a catalogue blurb. The confirm IS the
    safety mechanism, so --yes covers the catalogued years and refuses the rest.
    """
    rc = _cli(tmp_path, "sanborn01790_014", "--yes")

    assert rc == 1
    err = capsys.readouterr().err
    assert "will not confirm" in err and "sanborn01790_014" in err
    # and nothing was written
    cfg = tmp_path / "city.toml"
    assert "sanborn01790_014" not in cfg.read_text()


def test_cli_yes_still_batches_the_ordinary_catalogued_volumes(tmp_path: Path) -> None:
    """The 161 normal volumes are exactly what --yes is for."""
    assert _cli(tmp_path, "sanborn01790_002", "--yes") == 0
    assert (
        load_city_config(tmp_path / "city.toml").volume("sanborn01790_002").addresses_modern
        is False
    )


def test_cli_writes_nothing_when_the_confirm_is_declined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The confirm is the point. "n" means nothing is written."""
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
    rc = _cli(tmp_path, "sanborn01790_002")
    assert rc == 1
    assert "sanborn01790_002" not in (tmp_path / "city.toml").read_text()


def test_cli_writes_on_an_explicit_yes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))
    assert _cli(tmp_path, "sanborn01790_002") == 0
    assert (
        load_city_config(tmp_path / "city.toml").volume("sanborn01790_002").addresses_modern
        is False
    )


def test_cli_a_run_of_pure_refusals_is_a_failure(tmp_path: Path) -> None:
    """The operator asked for a declaration and did not get one — the volume still
    will not start, so exit 0 would be a lie."""
    assert _cli(tmp_path, "sanborn01790_999", "--yes") == 1


def test_cli_derives_the_catalog_from_the_city_toml_loc_catalog(tmp_path: Path) -> None:
    """`--loc-catalog` used to default to a Chicago fixture path baked into the
    parser; the city TOML now names its own catalog (`loc_catalog`) and the flag
    is an override."""
    from autogeoref.cli.entry import main

    body = TOML.replace(
        "renumbering_year = 1909", 'renumbering_year = 1909\nloc_catalog = "cat.json"'
    )
    cfg = _write(tmp_path, body)
    (tmp_path / "cat.json").write_text(
        json.dumps(
            [
                {
                    "id": "http://www.loc.gov/item/sanborn01790_002/",
                    "description": ["Vol. 2, 1896. 100 sheet(s). Bound."],
                    "date": "1896",
                }
            ]
        )
    )
    assert main(["era", "sanborn01790_002", "--city", str(cfg), "--yes"]) == 0
    assert load_city_config(cfg).volume("sanborn01790_002").addresses_modern is False


def test_cli_refuses_with_neither_catalog_flag_nor_config_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No catalog means no year, and era never guesses — it says where the
    catalog can be named and stops."""
    from autogeoref.cli.entry import main

    cfg = _write(tmp_path)
    assert main(["era", "sanborn01790_002", "--city", str(cfg)]) == 2
    assert "loc_catalog" in capsys.readouterr().err
