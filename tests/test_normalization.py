"""Normalization invariants.

Cases sampled from the fixture annotations plus the documented ordering
traps: aliases before direction stripping; apostrophes deleted before
punctuation spacing; numbered ST/PL twins distinct; long-form suffixes strip.
"""

import json
from pathlib import Path

import pytest

from autogeoref.names import join_split_ordinal, load_aliases, normalize, ordinal_suffix


def test_apostrophe_deleted_before_punctuation() -> None:
    # BL'VD must become BLVD (strippable); spacing the apostrophe would leave "BL VD"
    assert normalize("OAKLEY BL'VD") == normalize("OAKLEY BLVD") == "OAKLEY"
    assert normalize("N. OAKLEY BLV'D") == "OAKLEY"
    assert normalize("SHERIDAN R'D") == "SHERIDAN"


def test_numbered_st_pl_twins_stay_distinct() -> None:
    assert normalize("31ST ST") == "31ST"
    assert normalize("31ST PL") == "31ST PL"
    assert normalize("31ST ST") != normalize("31ST PL")
    assert normalize("W. 31ST PL.") == "31ST PL"
    assert normalize("W. 32ND PL.") == "32ND PL"
    assert normalize("48TH CT") == "48TH CT"


def test_alias_before_direction_stripping_south_park() -> None:
    aliases = {"SOUTH PARK": "DR MARTIN LUTHER KING JR"}
    # SOUTH is part of the name here, not a direction word: the alias must
    # fire on the suffix-stripped-but-direction-kept form.
    assert normalize("SOUTH PARK AV", aliases) == "DR MARTIN LUTHER KING JR"
    assert normalize("SOUTH PARK AV.", aliases) == "DR MARTIN LUTHER KING JR"
    # without the alias, SOUTH is treated as a direction and stripped
    assert normalize("SOUTH PARK AV") == "PARK"


def test_long_form_suffixes_strip() -> None:
    assert normalize("FOSTER ROAD") == "FOSTER"
    assert normalize("SHERIDAN DRIVE") == "SHERIDAN"
    assert normalize("ARCADE TERRACE") == "ARCADE"
    assert normalize("MARSHALL PARKWAY") == "MARSHALL"
    assert normalize("W. GARFIELD BOUL.") == "GARFIELD"
    assert normalize("MICHIGAN BOULEVARD") == "MICHIGAN"


def test_parentheticals_dropped() -> None:
    assert normalize("S. RACINE AV. (S. CENTRE AV.)") == "RACINE"
    assert normalize("W. 38TH ST. (PRIVATE)") == "38TH"
    assert normalize("N. ROBEY (NOT OPEN)") == "ROBEY"


def test_case_and_directions() -> None:
    assert normalize("Washburn Av.") == "WASHBURN"
    assert normalize("N. WESTERN AV") == "WESTERN"
    assert normalize("S. WESTERN AV.") == "WESTERN"
    assert normalize("W. MONROE ST.") == "MONROE"
    assert normalize("W. MADISON") == "MADISON"


def test_hyphens_become_spaces() -> None:
    assert normalize("BELLE-PLAINE AVE") == "BELLE PLAINE"


def test_postfix_directionals_strip_after_suffix() -> None:
    # Cleveland-style postfix quadrants (city-#2 probe): the direction trails
    # the suffix token, and both must strip for the name to key the index
    assert normalize("MILES AV. S.E.") == "MILES"
    assert normalize("BUCKEYE RD. S.E.") == "BUCKEYE"
    assert normalize("KINSMAN ROAD S.E.") == "KINSMAN"
    assert normalize("E. 93RD ST.") == "93RD"  # prefix numbered streets unchanged
    # compound single-token quadrant and full words strip too
    assert normalize("MILES AV SE") == "MILES"
    assert normalize("UNION AVENUE SOUTHEAST") == "UNION"
    # both prefix and postfix present
    assert normalize("W. MILES AV. S.E.") == "MILES"


def test_postfix_strip_requires_preceding_suffix() -> None:
    # names that merely END in a direction word are not postfix directionals
    assert normalize("PARK WEST") == "PARK WEST"
    assert normalize("LINCOLN PARK WEST") == "LINCOLN PARK WEST"
    # a bare suffix + direction is a NAME ("AVENUE E"), never stripped to empty
    assert normalize("AVENUE E") == "AVENUE E"
    # lettered avenues with their official prefix keep keying the centerline
    # name; without the non-direction-token guard the postfix strip would
    # leave a bare "S" for the leading strip to empty
    assert normalize("S. AVENUE E") == "AVENUE E"
    assert normalize("S. AVE. E") == "AVE E"


def test_postfix_court_like_twins_stay_distinct() -> None:
    # the ordinal-twin rule survives the postfix strip
    assert normalize("E. 31ST PL. S.E.") == "31ST PL"
    assert normalize("E. 31ST ST. S.E.") == "31ST"


def test_direction_and_suffix_names_never_reduce_to_the_empty_key() -> None:
    """A name made only of direction or suffix words keeps a key of its own.

    The empty key is not inert. The centerline index collects every street whose
    name strips away under one key, so their geometries merge and a lookup
    against it returns intersections belonging to none of them.
    """
    # the leading-direction strip: the module contract promised this guard and
    # only the postfix branch enforced it
    assert normalize("SOUTH ST.") == "SOUTH"
    assert normalize("NORTH AVE") == "NORTH"
    assert normalize("EAST DR") == "EAST"
    assert normalize("WEST ST") == "WEST"
    assert normalize("NORTH CT") == "NORTH"
    # ...and the suffix strip, which TERRACE reaches from both sides
    assert normalize("TERRACE DR") == "TERRACE"
    assert normalize("TERRACE") == "TERRACE"
    assert normalize("STREET") == "STREET"
    assert normalize("BOULEVARD") == "BOULEVARD"
    # distinct streets stay distinct rather than sharing one blob
    assert normalize("SOUTH ST.") != normalize("NORTH AVE") != normalize("TERRACE DR")


def test_direction_suffix_guard_changes_nothing_that_already_had_a_key() -> None:
    """The guard fires only where the answer was empty; ordinary names are untouched."""
    assert normalize("W. MONROE ST.") == "MONROE"
    assert normalize("S. WESTERN AV.") == "WESTERN"
    assert normalize("W. 31ST PL.") == "31ST PL"
    assert normalize("SOUTH PARK AV") == "PARK"
    assert normalize("AVENUE E") == "AVENUE E"
    assert normalize("SHERIDAN TER") == "SHERIDAN"


def test_ordinal_suffix_teens_exception() -> None:
    """The %100 branch is the one the reimplementation could get wrong."""
    assert [ordinal_suffix(n) for n in (1, 2, 3, 4)] == ["ST", "ND", "RD", "TH"]
    assert [ordinal_suffix(n) for n in (11, 12, 13)] == ["TH", "TH", "TH"]
    # the hundreds repeat the exception, which testing 11/12/13 alone misses:
    # 211 is two hundred ELEVENTH, so it takes TH and not ST
    assert [ordinal_suffix(n) for n in (111, 112, 113, 211, 1012)] == ["TH"] * 5
    # ...and the tens outside it do not
    assert [ordinal_suffix(n) for n in (21, 22, 23, 121, 1001)] == ["ST", "ND", "RD", "ST", "ST"]


def test_split_ordinal_joins_only_the_tail_the_number_takes() -> None:
    """The guard is the whole safety argument: ST and RD are also suffixes."""
    assert join_split_ordinal("W. 73 RD ST.") == "W. 73RD ST."
    assert join_split_ordinal("E. 121 ST.") == "E. 121ST."
    assert join_split_ordinal("72 ND") == "72ND"
    assert join_split_ordinal("W. 74 TH ST.") == "W. 74TH ST."
    # 12 takes TH, so that ST is a Street suffix the strip loop owns
    assert join_split_ordinal("12 ST") == "12 ST"
    assert join_split_ordinal("2 ST") == "2 ST"
    assert join_split_ordinal("11 RD") == "11 RD"
    # nothing to join
    assert join_split_ordinal("DOUGLAS PARK") == "DOUGLAS PARK"
    assert join_split_ordinal("BALMORAL 45 AV.") == "BALMORAL 45 AV."


def test_split_ordinal_reaches_the_key_a_bare_number_could_not() -> None:
    assert normalize("W. 73 RD ST.") == "73RD"
    assert normalize("E. 121 ST.") == "121ST"
    assert normalize("W. 74 TH ST.") == "74TH"
    # and the numbered PLACE/COURT twin rule still fires on the joined form
    assert normalize("73 RD PL.") == "73RD PL"
    assert normalize("N. 41 ST. CT.") == "41ST CT"


def test_split_ordinal_joins_after_punctuation_is_spaced() -> None:
    """The join runs on the spaced form, so any separator breaks the number from its tail."""
    # a hyphen or slash reads as a break, exactly as a space and a period do
    assert normalize("W 73-RD ST") == "73RD"
    assert normalize("W 73/RD ST") == "73RD"
    # a parenthetical between the two is dropped first, then the join sees them adjacent
    assert normalize("W 73 (ALT) RD ST") == "73RD"
    # joining on the spaced form is what makes the key a fixed point
    assert normalize(normalize("W 73-RD ST")) == normalize("W 73-RD ST")
    # the tail the number does NOT take is still left alone whatever separates it
    assert normalize("12-ST") == "12"


def test_split_ordinal_leaves_a_real_suffix_alone() -> None:
    assert normalize("12 ST") == "12"
    assert normalize("11 RD") == "11"
    assert normalize("W 46 ST.") == "46"
    assert normalize("DOUGLAS PARK") == "DOUGLAS PARK"
    assert normalize("31ST PL") == "31ST PL"
    # a name a rule emptied would collect every such street under one key
    for raw in ("W 46 ST.", "45", "45 AV", "12 ST", "N. 41 ST. AV.", "72 ND"):
        assert normalize(raw)


def test_split_ordinal_keeps_the_037_spaced_ordinal_aliases_reaching_their_value(
    aliases_dir: Path,
) -> None:
    """The volume's spaced-ordinal catch keys have ordinal twins; both must hit."""
    aliases = load_aliases(aliases_dir / "aliases-sanborn01790_037.json")
    assert normalize("N. 41 ST. AV.", aliases) == "KARLOV"
    assert normalize("N. 41ST AV.", aliases) == "KARLOV"
    assert normalize("42 ND", aliases) == "KEELER"
    assert normalize("N. 52D AV.", aliases) == "LARAMIE"


def test_split_ordinal_lets_a_spaced_court_read_reach_its_own_alias(
    aliases_dir: Path,
) -> None:
    """The join runs before the FIRST alias lookup, which is a full-string one.

    A numbered-court read spelled with a split ordinal used to strip to a bare
    number and key nothing. It now matches the direction-qualified court key the
    table carries for exactly this street, which is what those keys are for.
    """
    aliases = load_aliases(aliases_dir / "aliases-sanborn01790_037.json")
    assert normalize("N. 51 ST. CT.", aliases) == "LEAMINGTON"
    assert normalize("N. 42 ND. CT.", aliases) == "TRIPP"
    assert normalize("S. 43 RD. CT.", aliases) == "KOLIN"
    assert normalize("N. 50 TH. CT.", aliases) == "LAWLER"
    # the court guard is what still keeps it off the AVENUE's alias: 41ST CT has
    # no court key of its own, so it keys the court and never reaches KARLOV
    assert "N 41ST CT" not in aliases
    assert normalize("N. 41 ST. CT.", aliases) == "41ST CT"
    assert normalize("N. 41 ST. AV.", aliases) == "KARLOV"


def test_named_place_never_aliased() -> None:
    # aliases target the avenue; a same-named COURT/PLACE must not follow it
    aliases = {"COTTAGE": "SOMETHING ELSE"}
    assert normalize("COTTAGE PL.", aliases) == "COTTAGE"
    assert normalize("COTTAGE GROVE AV", aliases) == "COTTAGE GROVE"


def test_full_string_alias_fires_before_court_guard() -> None:
    # an outright-renamed PL street aliases via the exact full-string pass
    aliases = {"MACALISTER PL": "LEXINGTON"}
    assert normalize("MACALISTER PL.", aliases) == "LEXINGTON"


def test_numeric_ordinals() -> None:
    assert normalize("58TH ST") == "58TH"
    assert normalize("W. 13TH ST.") == "13TH"
    assert normalize("FIFTY SEVENTH") == "FIFTY SEVENTH"  # spelled-out stays


def test_vol16_alias_table(aliases_dir: Path) -> None:
    # scoped names are isolated; load_aliases is silent
    # on a missing path, so assert existence rather than pass {} quietly
    path = aliases_dir / "aliases-sanborn01790_006.5.json"
    assert path.exists(), path
    aliases = load_aliases(path)
    assert normalize("SOUTH PARK AV", aliases) == "DR MARTIN LUTHER KING JR"
    assert normalize("GRAND BOUL", aliases) == "DR MARTIN LUTHER KING JR"
    # LAKE -> LAKE PARK (vol 16 scope)
    assert normalize("LAKE AV", aliases) == "LAKE PARK"


def test_every_alias_table_parses(aliases_dir: Path) -> None:
    """Every shipped alias file loads and carries at least one rename.

    Files land here in batches
    and nothing else in the suite enumerates the directory: FIXTURE-SHA256SUMS
    only hashes the bytes of the paths it already lists, so a new file with a
    trailing comma would surface for the first time at pipeline runtime. This
    test replaces the corpus-wide parse that the viewer's search canary used to
    provide incidentally (removed with the search strip, see
    ).
    """
    paths = sorted(aliases_dir.glob("aliases-*.json"))
    assert paths, aliases_dir
    for path in paths:
        aliases = load_aliases(path)
        # load_aliases is silent on a missing or unreadable path; an empty
        # table would pass quietly and feed the pipeline nothing
        assert aliases, path
        for key, value in aliases.items():
            assert key == key.strip() and value == value.strip(), (path, key)


def test_024_alias_table_full_string_pl(aliases_dir: Path) -> None:
    aliases = load_aliases(aliases_dir / "aliases-sanborn01790_024.json")
    pl_keys = [k for k in aliases if k.endswith(" PL")]
    if not pl_keys:
        pytest.skip("no PL-suffixed alias keys in _024 table")
    for k in pl_keys:
        assert normalize(f"{k}.", aliases) == aliases[k]


def test_fixture_annotation_sample(fixtures_dir: Path) -> None:
    """Every street name in a real annotation normalizes to something non-empty
    that contains no direction prefix or trailing standard suffix."""
    ann = json.loads((fixtures_dir / "sanborn01790_024" / "annotations" / "p1.json").read_text())
    for street in ann["streets"]:
        n = normalize(street["name"])
        assert n
        toks = n.split(" ")
        assert toks[0] not in {"N", "S", "E", "W"}
        # trailing suffix survives only for court-like twins (31ST PL etc.)
        if toks[-1] in {"AVE", "AV", "ST", "BLVD"}:
            raise AssertionError(f"unstripped suffix: {street['name']} -> {n}")
