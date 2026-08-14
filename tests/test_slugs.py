"""The canonical page<->slug mapping (one function, four former copies).

Pins the semantics every consumer (prep, bounds, seam, mask, back half)
now shares: recorded production slugs, fresh-volume slugs, letter-suffixed
pages, and the token-boundary rule that keeps ``map3`` from reading as
page 3 (the old ``rsplit("_p")`` copies disagreed on exactly these edges).
"""

from __future__ import annotations

from autogeoref.slugs import (
    DuplicateCoverage,
    duplicate_coverage_page,
    duplicate_coverage_slug,
    overview_page,
    overview_slug,
    page_from_slug,
    page_sort_key,
    skeleton_pages,
    slug_for_page,
    valid_review_page,
)


def test_page_from_recorded_production_slug() -> None:
    assert page_from_slug("chicago_ill_1917_vol_7_p1") == "1"
    assert page_from_slug("chicago_ill_1895_vol_16_p39") == "39"


def test_page_from_fresh_volume_slug_roundtrip() -> None:
    slug = slug_for_page("sanborn01790_041", "12")
    assert slug == "sanborn01790_041_p12"
    assert page_from_slug(slug) == "12"


def test_page_from_bare_and_letter_suffixed_names() -> None:
    assert page_from_slug("p12") == "12"
    assert page_from_slug("chicago_ill_1901_vol_x_p7a") == "7a"


def test_named_page_canonicalizes_to_lower_case() -> None:
    # `_016`'s LOC segment tags are `CBDa`/`CBDb`; one sheet must not become
    # two page ids, so the named id is folded to its canonical form
    assert page_from_slug("sanborn01790_016_pCBDa") == "cbda"
    assert page_from_slug("sanborn01790_016_pcbdb") == "cbdb"
    assert page_from_slug("sanborn01790_017_pCBD1") == "cbd1"


def test_non_pages_return_none() -> None:
    assert page_from_slug("titlesheet") is None
    assert page_from_slug("sanborn01790_024") is None  # volume id, not a page
    assert page_from_slug("x_planned") is None  # contains "_p" mid-word
    assert page_from_slug("map3") is None  # p-token must follow start or "_"


def test_page_sort_key_is_natural_order() -> None:
    slugs = ["v_p10", "v_p2", "v_p1"]
    assert sorted(slugs, key=page_sort_key) == ["v_p1", "v_p2", "v_p10"]


def test_page_sort_key_non_numeric_tails_sort_last_stably() -> None:
    slugs = ["v_p7a", "v_p3", "index_sheet", "v_p12"]
    ordered = sorted(slugs, key=page_sort_key)
    assert ordered[:2] == ["v_p3", "v_p12"]
    assert set(ordered[2:]) == {"v_p7a", "index_sheet"}
    assert ordered[2:] == sorted(ordered[2:])  # stable within the tail bucket


def test_valid_review_page_accepts_the_canonical_page_grammar() -> None:
    # exactly what page_from_slug can produce: numeric + optional letter
    # suffix (either case), plus the literal named sheets
    for page in ("1", "12", "7a", "7A", "13S", "0a", "cbd1", "cbd2", "cbda", "cbdb"):
        assert valid_review_page(page), page
        assert page_from_slug(slug_for_page("v", page)) is not None, page


def test_valid_review_page_rejects_crop_ids_and_traversal() -> None:
    # `10_1`-style crop layers bind crop pixels to full-page pixels (the
    # module-docstring refusal), and review interpolates page ids into paths
    rejected = (
        "10_1",
        "p10",
        "cbd3",
        "cbdc",
        "CBD1",
        "CBDa",
        "",
        "..",
        "../2",
        "2/..",
        "1;rm",
        "1 ",
    )
    for page in rejected:
        assert not valid_review_page(page), page


#: A volume's declared overview class, as chicago.toml declares it for the
#: Loop volumes (VolumeConfig.overview_pages).
DECLARED = ("cbd1", "cbd2")

#: The mixed volume the skeleton rule was written for: numeric sheets plus
#: uncolored outline twins of some of them.
MIXED = ("12", "13", "13S", "14", "15", "15S", "16")

#: A volume that prints the division letter as part of the sheet number, so
#: every map page carries the suffix and none duplicates anything.
DIVISION_LETTERED = ("0", "1S", "2S", "13S", "20S", "94S", "cbd1", "cbd2")


def test_skeleton_pages_needs_the_numeric_twin_in_the_same_volume() -> None:
    assert skeleton_pages(MIXED) == {"13S", "15S"}
    # the form alone settles nothing: with no numeric twin the suffix is part
    # of the printed sheet number, and none of these is a duplicate
    assert skeleton_pages(DIVISION_LETTERED) == frozenset()
    # lowercase is the continuation form, a DISTINCT map area, never a twin
    assert skeleton_pages(("13", "13s")) == frozenset()
    assert skeleton_pages(("7", "7a")) == frozenset()


def test_duplicate_coverage_pages_are_skeletons_and_declared_overviews() -> None:
    mixed = DuplicateCoverage.resolve(MIXED)
    assert duplicate_coverage_page("13S", mixed)  # printed "(SKELETON)" twin of p13
    assert duplicate_coverage_page("15S", mixed)
    declared = DuplicateCoverage(frozenset(DECLARED))
    assert duplicate_coverage_page("cbd1", declared)  # 4x-scale Congested District overview
    assert duplicate_coverage_page("cbd2", declared)
    segments = DuplicateCoverage(frozenset({"cbda", "cbdb"}))
    assert duplicate_coverage_page("cbda", segments)  # `_016` block-line map segments
    assert duplicate_coverage_page("1", DuplicateCoverage(frozenset({"1"})))  # numeric, declared


def test_duplicate_coverage_excludes_regular_undeclared_and_continuation_pages() -> None:
    mixed = DuplicateCoverage.resolve(MIXED, DECLARED)
    assert not duplicate_coverage_page("13", mixed)
    assert not duplicate_coverage_page("7a", mixed)  # continuation: a DISTINCT map area
    assert not duplicate_coverage_page("0a", mixed)  # front matter, lowercase
    assert not duplicate_coverage_page("13s", mixed)  # lowercase = continuation, not skeleton
    assert not duplicate_coverage_page("S", mixed)  # no numeric twin to duplicate
    # a named page is NOT duplicate coverage unless its volume declares it:
    # the class moved from the id grammar to VolumeConfig.overview_pages
    assert not duplicate_coverage_page("cbd1", DuplicateCoverage.resolve(MIXED))
    # and every S-suffixed page of a division-lettered volume is a REGULAR sheet
    lettered = DuplicateCoverage.resolve(DIVISION_LETTERED, DECLARED)
    for page in ("1S", "13S", "20S", "94S"):
        assert not duplicate_coverage_page(page, lettered), page


def test_duplicate_coverage_slug_lifts_page_and_refuses_unpageable() -> None:
    mixed = DuplicateCoverage.resolve(MIXED)
    declared = DuplicateCoverage(frozenset(DECLARED))
    assert duplicate_coverage_slug("sanborn01790_015_p13S", mixed)
    assert duplicate_coverage_slug("sanborn01790_017_pcbd1", declared)
    assert not duplicate_coverage_slug("sanborn01790_017_pcbd1", mixed)
    assert not duplicate_coverage_slug("sanborn01790_015_p13", declared)
    assert not duplicate_coverage_slug("index_sheet", declared)


def test_overview_pages_are_the_declared_kind_only() -> None:
    # overview = district-scale duplicate coverage; skeletons coincide with
    # their twins and are NOT overviews (the bake clips only overview masks)
    declared = DuplicateCoverage.resolve(("13", "13S"), DECLARED)
    assert overview_page("cbd1", declared)
    assert overview_page("cbd2", declared)
    assert not overview_page("cbd1", DuplicateCoverage())  # undeclared = regular sheet
    assert not overview_page("13S", declared)
    assert not overview_page("13", declared)


def test_overview_slug_lifts_page_and_refuses_unpageable() -> None:
    declared = DuplicateCoverage(frozenset(DECLARED))
    assert overview_slug("sanborn01790_017_pcbd1", declared)
    # the named id canonicalizes before the declared set is consulted
    assert overview_slug("sanborn01790_017_pCBD1", declared)
    assert not overview_slug("sanborn01790_015_p13S", declared)
    assert not overview_slug("index_sheet", declared)
