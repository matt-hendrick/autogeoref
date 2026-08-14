"""Rename-source parsing and the alias-key conventions.

Every shape in ``SOURCE`` below is one the real document actually contains, and
most of them are traps the recorded build pass named
(,
-4): the source
lists renames in BOTH directions, wraps entries across lines, trails prose after
the names, spells and abbreviates ordinals four ways, and distinguishes a street
from its same-named court. The parser has to survive all of it offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autogeoref.address_grid import EW, NS
from autogeoref.alias.source import (
    RenameSource,
    RenameSourceError,
    address_ranges,
    alias_key_for,
    canon_key,
    edit_distance_one,
    parse_martin1948,
    snap_value,
    tokens_match,
)

SOURCE = """\
-Cornelia St., Walton St., 932N fr. 100 to 400W.  Named after a landowner.
-Carl St., Burton Pl. 1500N
-High St., Janssen Ave., 1420W fr. 2200 to 2799N
-Marcy St., Marcey St. 1000W
-Grove Pl., Shakespeare Ave. 600W
-Pleasant St., Frontier Ave., 1200 to 1250 N.
Fulton St., 300N 438 to 6000W  Frink St., Pleasant St.
St., Anna Pl., Bingo St., Central St.
-Wrapped St., Continued
   Ave., 1200W fr. 100 to 200N
-Fortieth Ave., Pulaski Rd. 4000W
-Fifty Second Ave., Laramie Ave. 5200W
"""


@pytest.fixture
def source(tmp_path: Path) -> RenameSource:
    text = tmp_path / "source.txt"
    text.write_text(SOURCE)
    return RenameSource(parser="martin1948", text_path=text, citation="test source, 1948")


def _pairs(source: RenameSource) -> set[tuple[str, str]]:
    return {(p.old, p.new) for p in source.pairs()}


def test_former_name_lines_give_old_to_new(source: RenameSource) -> None:
    pairs = _pairs(source)
    assert ("CORNELIA ST", "WALTON ST") in pairs
    assert ("HIGH ST", "JANSSEN AV") in pairs
    assert ("CARL ST", "BURTON PL") in pairs


def test_current_street_lines_give_the_same_pairs_reversed(source: RenameSource) -> None:
    """Using both directions is what lifts recovery to 98% of a manual pass."""
    pairs = _pairs(source)
    assert ("FRINK ST", "FULTON ST") in pairs
    assert ("PLEASANT ST", "FULTON ST") in pairs


def test_a_continuation_line_flows_into_its_entry(source: RenameSource) -> None:
    assert ("WRAPPED ST", "CONTINUED AV") in _pairs(source)


def test_prose_leaks_a_candidate_that_index_membership_then_kills(
    source: RenameSource,
) -> None:
    """Parsing does NOT eliminate prose junk, and the record says so.

    : junk
    candidates from prose reach the ambiguity tier occasionally and are
    eliminated by the value-in-bounds filter and locality, not by the parser.
    Pinned as behaviour so a future parser change is measured against what
    actually protects the output, rather than against a tidier story.
    """
    junk = {new for _old, new in _pairs(source) if "LANDOWNER" in new}
    assert junk, "the prose clause is expected to leak a name run"
    # ...and nothing outside the volume's own bounded street list survives
    assert all(snap_value(name, {"WALTON", "JANSSEN"}) is None for name in junk)


def test_a_suffix_only_current_street_line_is_skipped(source: RenameSource) -> None:
    """``St., Anna Pl., Bingo St.`` would pair every listed name to bare 'ST'.

    ``normalize('ST')`` is the EMPTY string, which is a real bucket in a bounded
    index (unnamed segments land there), so every name on such a line would
    score against it as if it were a street.
    """
    assert not any(new in {"ST", "AV", "PL"} for _old, new in _pairs(source))
    assert not any(old == "ANNA PL" and new == "ST" for old, new in _pairs(source))


def test_candidates_apply_the_court_and_twin_rules(source: RenameSource) -> None:
    values = {new for new, _line in source.candidates("GROVE PL")}
    assert "SHAKESPEARE AV" in values
    # a suffix-less key must NOT accept a court/place line: that documents the
    # TWIN street, not this one
    assert not source.candidates("GROVE")


def test_candidates_tolerate_one_edit_on_a_long_token(source: RenameSource) -> None:
    """Source spellings drift against sheet reads (Marcy/Marcey, Rees/Reese)."""
    assert any(new == "MARCEY ST" for new, _ in source.candidates("MARCY"))


def test_missing_source_text_names_the_cached_document(tmp_path: Path) -> None:
    source = RenameSource(
        parser="martin1948",
        text_path=tmp_path / "absent.txt",
        citation="x",
        pdf_path=tmp_path / "cached.pdf",
    )
    with pytest.raises(RenameSourceError, match=r"cached.pdf"):
        source.pairs()


def test_unknown_parser_is_refused_not_guessed(tmp_path: Path) -> None:
    text = tmp_path / "s.txt"
    text.write_text(SOURCE)
    with pytest.raises(RenameSourceError, match="unknown rename_source_parser"):
        RenameSource(parser="nope", text_path=text, citation="x").pairs()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("40TH", "40TH"),
        ("N. 52D AV.", "52ND AV"),  # period ordinal
        ("N. 41 ST. AV.", "41ST AV"),  # OCR-split ordinal
        ("FORTIETH", "40TH"),  # spelled out
        ("FIFTY SECOND AV", "52ND AV"),
        ("FIFTH", "5TH"),
        ("12", "12TH"),  # bare number, teens exception
        ("W. GARFIELD BOULEVARD", "GARFIELD BLVD"),
    ],
)
def test_canon_key_normalizes_every_ordinal_spelling(raw: str, expected: str) -> None:
    assert canon_key(raw) == expected


def test_alias_key_for_follows_the_recorded_conventions() -> None:
    # court-like reads take the FULL cleaned string: normalize refuses to alias
    # a court through its stripped stem
    assert alias_key_for("W. 22ND PL.") == "W 22ND PL"
    assert alias_key_for("CUSTOM HOUSE C'T") == "CUSTOM HOUSE CT"
    # everything else takes the normalized key
    assert alias_key_for("W. 22ND ST.") == "22ND"
    assert alias_key_for("N. ROBEY") == "ROBEY"


def test_address_ranges_separates_an_extent_from_a_position() -> None:
    ranges = address_ranges("-High St., Janssen Ave., 1420W fr. 2200 to 2799N")
    assert ranges.spans_on(NS) == ((2200.0, 2799.0),)
    assert ranges.points_on(EW) == (-1420.0,)
    # the span consumed its own numbers, so 2799N is not also a position
    assert 2799.0 not in ranges.points_on(NS)


def test_address_ranges_reads_the_bare_to_form() -> None:
    """The source writes extents both with and without 'fr.'."""
    assert address_ranges("-Pleasant St., Frontier Ave., 1200 to 1250 N.").spans_on(NS) == (
        (1200.0, 1250.0),
    )


def test_address_ranges_signs_south_and_west_negative() -> None:
    ranges = address_ranges("Pulaski Rd., 4000W 1 to 8700S")
    assert ranges.spans_on(NS) == ((-8700.0, -1.0),)
    assert ranges.points_on(EW) == (-4000.0,)


def test_snap_value_is_bounded_by_index_membership() -> None:
    """One edit absorbs source typos; nothing outside the index can be produced."""
    assert snap_value("DELAWARE", {"DELAWARE", "OGDEN"}) == "DELAWARE"
    assert snap_value("DELWARE", {"DELAWARE", "OGDEN"}) == "DELAWARE"
    assert snap_value("DELWARE", {"OGDEN"}) is None
    # short values get no fuzz at all
    assert snap_value("OGDEM", {"OGDEN"}) is None


def test_token_matching_refuses_short_near_misses() -> None:
    assert tokens_match(["MARCY"], ["MARCEY"])
    assert not tokens_match(["ELM"], ["ELK"])
    assert not tokens_match(["ELM", "ST"], ["ELM"])
    assert edit_distance_one("REES", "REESE")
    assert not edit_distance_one("REES", "REESES")


def test_parse_is_a_pure_function_of_the_text() -> None:
    assert parse_martin1948(SOURCE) == parse_martin1948(SOURCE)
