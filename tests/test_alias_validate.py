"""The structural rules an alias table must satisfy, one rule at a time.

Every rule asks what an entry DOES, and both halves of that are pinned here:
the shapes that damage the index are refused, and the shapes `docs/INTERNALS.md`
recommends are asserted CLEAN, because a rule that fires on the recommended
shape is a rule nobody can act on. The third case has its own tests — an entry
that takes reads off a surviving street is REPORTED and not refused, since
nothing structural separates the recommended shape from a mistake wearing it.

The sweep's use of these rules lives in `test_alias_sweep.py`. No fixtures, no
network, no model call.
"""

from __future__ import annotations

from typing import Any

import pytest

from autogeoref.alias.validate import is_inert, redirects, twin_hold, validate_table
from autogeoref.centerlines import CenterlineIndex, centerline_key
from autogeoref.names import normalize

#: NORTH BETA and BETA are two streets the normalizer files under one key, and
#: "W 22ND" with type PL is the numbered twin of a 22ND that is not here.
NAMED = (("ALPHA", "AVE"), ("BETA", "AVE"), ("SOLO", "AVE"), ("SECOND", "AVE"))
EXTRA = (("NORTH BETA", "AVE"), ("W 22ND", "PL"))


def _index() -> CenterlineIndex:
    features: list[dict[str, Any]] = [
        {
            "type": "Feature",
            "properties": {"street_nam": name, "street_typ": street_typ},
            "geometry": {"type": "LineString", "coordinates": [[-87.66, 41.90], [-87.66, 41.94]]},
        }
        for name, street_typ in NAMED + EXTRA
    ]
    return CenterlineIndex(features, aliases={})


def _check(table: dict[str, str]) -> list[str]:
    return validate_table(table, _index(), "street_nam", "street_typ")


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ({"ONLY": "SOLO"}, ""),
        ({"ONLY": "NOT A STREET"}, "VALUE NOT AN INDEX KEY"),
        ({"ALPHA": "SOLO"}, "KEY SHADOWS AN IN-BOUNDS STREET"),
        ({"NORTH BETA": "SOLO"}, "RE-KEYED"),
        ({"ONLY": "SOLO", "SOLO": "SECOND"}, "VALUE IS ALSO A KEY"),
    ],
)
def test_each_structural_rule_is_enforced(table: dict[str, str], expected: str) -> None:
    problems = _check(table)
    if not expected:
        assert problems == []
    else:
        assert any(expected in p for p in problems), problems


def test_a_key_that_keeps_the_printed_direction_is_reported_but_not_refused() -> None:
    """The recommended shape for a one-directional rename, and its cost.

    ``N ALPHA`` does take reads off the surviving ``ALPHA`` — that is what it is
    for — so it is REPORTED. It is not refused, because a table cannot tell this
    apart from a mistake with the same shape, and refusing it would refuse the
    shape the guide recommends.
    """
    assert _check({"N ALPHA": "SOLO"}) == []
    assert redirects({"N ALPHA": "SOLO"}, _index())


def test_the_report_covers_a_key_that_only_the_full_string_lookup_reaches() -> None:
    """A suffix-carrying key is consulted FIRST, before anything is stripped.

    So ``ALPHA AVE -> SOLO`` splits one street's reads across two geometries by
    spelling, and no structural rule sees it: the key is not an index key, the
    value is one, and the ``ALPHA`` segments never move.
    """
    hijack = {"ALPHA AVE": "SOLO"}
    assert _check(hijack) == []
    assert redirects(hijack, _index()) == [
        "'ALPHA AVE' -> 'SOLO': takes 'ALPHA' reads off a street still in bounds"
    ]
    assert normalize("ALPHA AVE.", hijack) == "SOLO"
    assert normalize("ALPHA ST.", hijack) == "ALPHA"


def test_an_entry_that_only_restates_the_normalizer_is_not_reported() -> None:
    """An inert entry sends the read where it was already going, so there is nothing to see."""
    assert redirects({"ALPHA AVE": "ALPHA"}, _index()) == []


def test_a_bare_key_that_is_itself_a_street_is_still_refused() -> None:
    """The half of the shadow rule that has teeth, beside the half above."""
    assert any("KEY SHADOWS" in p for p in _check({"ALPHA": "SOLO"}))


def test_keeping_the_direction_does_not_make_a_key_safe_on_its_own() -> None:
    """``NORTH BETA`` is a real street's own spelling, so aliasing it moves geometry.

    The shape being recommended is not a licence: what matters is whether a
    centerline is reached, and here one is.
    """
    assert any("RE-KEYED" in p for p in _check({"NORTH BETA": "SOLO"}))


def test_an_entry_that_separates_two_streets_filed_together_is_clean() -> None:
    """An entry whose value is its own key is not a missing value and not a chain.

    ``NORTH BETA`` is filed under ``BETA`` because the normalizer drops the
    leading direction, so the two streets share one geometry. Pointing the
    spelling at itself gives it a key of its own. The value is absent from the
    alias-free index by construction — this entry is what creates it — and it
    resolves to itself whichever lookup runs first.
    """
    assert _check({"NORTH BETA": "NORTH BETA"}) == []
    assert _check({"NORTH BETA": "NORTH BETA", "N BETA": "NORTH BETA"}) == []


def test_a_real_chain_is_still_refused() -> None:
    assert any("chain" in p for p in _check({"ONLY": "SOLO", "SOLO": "SECOND"}))


def test_an_alias_for_a_numbered_street_leaves_its_place_twin_alone() -> None:
    """A read of the twin cannot reach a bare numbered key; the index side matches it.

    Without that, a rename of 22nd Street empties 22nd Place into the new name
    and the merged geometry answers for neither street.
    """
    assert "22ND PL" in _index().by_name
    assert _check({"22ND": "SOLO"}) == []
    assert (
        centerline_key({"street_nam": "W 22ND", "street_typ": "PL"}, {"22ND": "SOLO"}) == "22ND PL"
    )


def test_a_place_renamed_onto_a_number_keeps_its_own_key_too() -> None:
    """The same rule from the other side, which decides on the aliased name.

    An entry pointing a named place at a numbered one must not pool it into
    that number's street. Deciding twin-ness on the bare name alone would.
    """
    props = {"street_nam": "GARFIELD", "street_typ": "PL"}
    assert centerline_key(props, {"GARFIELD": "55TH"}) == "55TH PL"
    assert centerline_key({"street_nam": "GARFIELD", "street_typ": "ST"}, {"GARFIELD": "55TH"}) == (
        "55TH"
    )


def test_the_auto_writer_still_holds_that_key_for_a_person_to_decide() -> None:
    """Safe is not the same as decided: the twin's own rename is a separate claim."""
    held = twin_hold("22ND", _index())
    assert held and "place or court twin" in held[0]
    assert twin_hold("ALPHA", _index()) == []
    assert twin_hold("22ND", CenterlineIndex([], aliases={})) == []


def test_an_inert_entry_is_exempt_until_its_value_becomes_a_key() -> None:
    """An entry handing back its key's own stripped form decides nothing.

    ``NOWHERE`` is not an index key, so the rules would object — but the read
    reaches it with or without the entry. Add a second entry claiming that form
    and the first one starts deciding, and is judged again.
    """
    assert is_inert("NOWHERE AVE", {"NOWHERE AVE": "NOWHERE"})
    assert _check({"NOWHERE AVE": "NOWHERE"}) == []
    both = {"NOWHERE AVE": "NOWHERE", "NOWHERE": "SOLO"}
    assert not is_inert("NOWHERE AVE", both)
    assert any("chain" in p for p in _check(both))


def test_inertness_accounts_for_the_lookup_that_runs_between_the_two() -> None:
    """The middle lookup uses a THIRD string, and it can be a key as well.

    ``N NOWHERE AVE`` hands back ``NOWHERE``, so on its own it decides nothing.
    But a miss would have tried ``N NOWHERE`` on the way past, so with that
    entry present the first one is deciding the outcome and must be judged.
    """
    alone = {"N NOWHERE AVE": "NOWHERE"}
    assert is_inert("N NOWHERE AVE", alone)
    both = {"N NOWHERE AVE": "NOWHERE", "N NOWHERE": "SOLO"}
    assert normalize("N NOWHERE AVE", both) == "NOWHERE"
    assert normalize("N NOWHERE AVE", {"N NOWHERE": "SOLO"}) == "SOLO"
    assert not is_inert("N NOWHERE AVE", both)
    assert any("VALUE NOT AN INDEX KEY" in p for p in _check(both))
