"""The candidate numeral rules, and the traps each one has to survive.

The rules are measurement-only (`scripts/experiments/numeral_rules.py`), but
their failure modes are the expensive kind — a key that resolves to the WRONG
street is invisible downstream — so the negative cases are pinned here.
"""

from __future__ import annotations

import pytest

from autogeoref import names
from conftest import load_script

rules = load_script("experiments/numeral_rules.py")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # the tail is the ordinal that number takes: join it
        ("W. 73 RD ST.", "W. 73RD ST."),
        ("E. 121 ST.", "E. 121ST."),
        ("72 ND", "72ND"),
        ("W. 74 TH ST.", "W. 74TH ST."),
        # the tail is NOT that number's ordinal: it is a suffix, leave it
        ("12 ST", "12 ST"),
        ("13 TH ST", "13TH ST"),
        ("11 RD", "11 RD"),
        ("2 ST", "2 ST"),
        # nothing to join
        ("DOUGLAS PARK", "DOUGLAS PARK"),
        ("BALMORAL 45 AV.", "BALMORAL 45 AV."),
    ],
)
def test_join_split_ordinal(raw: str, expected: str) -> None:
    assert rules.join_split_ordinal(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BALMORAL 45 AV.", "BALMORAL AV"),
        ("W. CHICAGO 48 AV.", "W CHICAGO AV"),
        ("BARTLETT 70 AV. S.E.", "BARTLETT AV S E"),
        # a direction is not a real left flank: this is a split ordinal
        ("W. 73 RD ST.", "W. 73 RD ST."),
        ("E. 63 STREET", "E. 63 STREET"),
        # nothing follows the numeral
        ("PLANKED ALLEY 6", "PLANKED ALLEY 6"),
        ("BARTLETT 70", "BARTLETT 70"),
        # the follower is an ordinal tail, so the numeral is part of a name
        ("BERWYN 72 ND", "BERWYN 72 ND"),
        # a two-letter left flank is not a word
        ("AV 45 ST", "AV 45 ST"),
    ],
)
def test_drop_interpolated_numeral(raw: str, expected: str) -> None:
    assert rules.drop_interpolated_numeral(raw) == expected


def test_interp_suffix_narrows_the_right_flank() -> None:
    """The narrow arm fires only where the follower is a known suffix."""
    assert rules.drop_interpolated_numeral("PRIVATE 4 ALLEY", True) == "PRIVATE 4 ALLEY"
    assert rules.drop_interpolated_numeral("PRIVATE 4 ALLEY") == "PRIVATE ALLEY"
    assert rules.drop_interpolated_numeral("BALMORAL 45 AV.", True) == "BALMORAL AV"


@pytest.mark.parametrize("rule", ["split", "interp", "interp-suffix"])
def test_no_rule_empties_a_key_that_was_not_empty(rule: str) -> None:
    """The empty string is a real key in the modern index; a rule must not reach it."""
    variant = rules.make_variant(frozenset({rule}))
    for raw in ("W 46 ST.", "45", "45 AV", "(66' WIDE)", " ", "12 ST", "N. 41 ST. AV."):
        if names.normalize(raw):
            assert variant(raw), f"{rule} emptied {raw!r}"


def test_numbered_place_twins_stay_distinct_under_split() -> None:
    variant = rules.make_variant(frozenset({"split"}))
    assert variant("73 RD PL.") == "73RD PL"
    assert variant("31ST PL.") == "31ST PL"


def test_alias_defer_skips_the_rule_when_an_alias_claims_the_name() -> None:
    aliases = {"BARTLETT 70 AV S E": "BARTLETT 70"}
    plain = rules.make_variant(frozenset({"interp-suffix"}))
    deferred = rules.make_variant(frozenset({"interp-suffix"}), defer_to_alias=True)
    assert plain("BARTLETT 70 AV. S.E.", aliases) == "BARTLETT"
    assert deferred("BARTLETT 70 AV. S.E.", aliases) == "BARTLETT 70"
    # a name no alias claims is unaffected by the deferral
    assert deferred("BALMORAL 45 AV.", aliases) == "BALMORAL"


def test_patch_targets_excludes_names_itself() -> None:
    """Patching the variant's own home would recurse forever."""
    targets = rules.patch_targets()
    assert targets
    assert not any(m.__name__.endswith(".names") for m in targets)
