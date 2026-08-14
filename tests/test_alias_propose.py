"""The clean/held/no-candidate tiering — the whole safety story of auto-writes.

Contracts pinned here: a UNIQUE in-bounds candidate with a locality inside the bar on two
or more sheets is auto-writable; an AMBIGUOUS one is auto-writable ONLY when the locality
decides by the recorded margin AND printed numerals independently agree; every way the
numeral check can decline is a HOLD (no city grid, a pre-renumbering address era, no
documented range, a rival numbering on the other axis, too few numerals, or numerals that
also fit the rival); a family whose reads carry diverging printed directions is held,
because the key an automatic proposal can form is the direction-stripped one and cannot
express a per-side rename; and the structural entry guard holds one bad key without
costing the volume its other entries.

A synthetic four-street grid stands in for a city. No fixtures, no network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autogeoref.address_grid import AddressGrid
from autogeoref.alias.propose import (
    CLEAN,
    HELD,
    NO_CANDIDATE,
    REFUSED,
    Candidate,
    documented_along,
    propose_volume,
    sheet_reads,
)
from autogeoref.alias.source import RenamePair, RenameSource
from autogeoref.centerlines import CenterlineIndex
from autogeoref.volume import SheetInput

GRID = AddressGrid(origin_lon=-87.6278, origin_lat=41.8819, units_per_mile=800)

# WIN and LOSE are both north-south and both documented, but over different
# latitude bands, so printed numerals can separate them. CROSS runs east-west:
# its house numbers are on the other axis, which no magnitude comparison can
# bridge.
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
        "properties": {"street_nam": "WIN", "street_typ": "AVE"},
        "geometry": {"type": "LineString", "coordinates": [[-87.658, 41.910], [-87.658, 41.930]]},
    },
    {
        "type": "Feature",
        "properties": {"street_nam": "LOSE", "street_typ": "AVE"},
        "geometry": {"type": "LineString", "coordinates": [[-87.640, 41.885], [-87.640, 41.900]]},
    },
    {
        "type": "Feature",
        "properties": {"street_nam": "SOLO", "street_typ": "AVE"},
        "geometry": {"type": "LineString", "coordinates": [[-87.659, 41.912], [-87.659, 41.928]]},
    },
    {
        "type": "Feature",
        "properties": {"street_nam": "CROSS", "street_typ": "AVE"},
        "geometry": {"type": "LineString", "coordinates": [[-87.670, 41.930], [-87.650, 41.930]]},
    },
    # far from every sheet locality: the REFUSED case
    {
        "type": "Feature",
        "properties": {"street_nam": "DISTANT", "street_typ": "AVE"},
        "geometry": {"type": "LineString", "coordinates": [[-87.500, 41.910], [-87.500, 41.930]]},
    },
]

# WIN spans ~1550-2660N and sits ~1240W; LOSE spans ~170-1000N at ~500W;
# CROSS spans ~910-1740W at ~2650N.
SOURCE = """\
-Test St., Win Ave., 1240W fr. 1560 to 2650N
-Test St., Lose Ave., 500W fr. 180 to 990N
-Only St., Solo Ave., 1250W fr. 1600 to 2600N
-Nowhere St., Absent Ave. 100N
-Split St., Win Ave., 1240W fr. 1560 to 2650N
-Axis St., Win Ave., 1240W fr. 1560 to 2650N
-Axis St., Cross Ave., 2650N fr. 1000 to 2000W
-Gap St., Win Ave. 200 to 800N and 1560 to 2650N
-Gap St., Lose Ave., 500W fr. 180 to 990N
-Far St., Distant Ave., 1240W fr. 1560 to 2650N
"""


@pytest.fixture
def source(tmp_path: Path) -> RenameSource:
    text = tmp_path / "source.txt"
    text.write_text(SOURCE)
    return RenameSource(parser="martin1948", text_path=text, citation="synthetic source, 1948")


def _index(aliases: dict[str, str] | None = None) -> CenterlineIndex:
    return CenterlineIndex(FEATURES, aliases=aliases or {})


def _sheet(page: str, names: list[str], numerals: dict[str, list[int]] | None = None) -> SheetInput:
    annotation: dict[str, Any] = {
        "streets": [
            {"name": n, "bbox": [0, 0, 10, 10], "orientation": "horizontal"} for n in names
        ],
        "address_numerals": [
            {"value": v, "bbox": [0, 0, 2, 2], "street": street}
            for street, values in (numerals or {}).items()
            for v in values
        ],
    }
    return SheetInput(page=page, annotation=annotation, full_size=(100.0, 100.0), scale=1.0)


def _sheets(read: str, numerals: list[list[int]] | None = None, pages: int = 2) -> list[SheetInput]:
    """``pages`` sheets each carrying ALPHA + BETA (a locality) and the read.

    ``numerals`` is per sheet, because the check pools numerals ACROSS the
    sheets carrying the read — a one-numeral-per-sheet volume still reaches the
    floor, and a test that shared one list would hide that.
    """
    per_page = numerals or [[] for _ in range(pages)]
    return [
        _sheet(str(i + 1), ["ALPHA ST.", "BETA ST.", read], {read: values} if values else None)
        for i, values in enumerate(per_page)
    ]


def _propose(source: RenameSource, sheets: list[SheetInput], **kwargs: Any) -> dict[str, Any]:
    proposal = propose_volume(
        "synthetic_001", source, _index(), sheets, grid=GRID, block=100.0, **kwargs
    )
    return {p.key: p for p in proposal.proposals}


def test_a_unique_candidate_on_two_sheets_is_auto_writable(source: RenameSource) -> None:
    """The measured 79-of-79-correct slice: no numerals needed, none consulted."""
    by_key = _propose(source, _sheets("ONLY ST."))
    only = by_key["ONLY"]
    assert only.tier == CLEAN
    assert only.value == "SOLO"
    assert only.numerals is None
    assert "unique in-bounds candidate" in only.reason


def test_a_unique_candidate_on_one_sheet_is_held(source: RenameSource) -> None:
    """One sheet's intersection cloud is a single sample; the record says so."""
    by_key = _propose(source, _sheets("ONLY ST.", pages=1))
    assert by_key["ONLY"].tier == HELD
    assert "1 sheet(s)" in by_key["ONLY"].reason


def test_an_ambiguity_numerals_agree_with_is_auto_writable(source: RenameSource) -> None:
    """The second arm of the clean tier: locality decides, numerals corroborate."""
    by_key = _propose(source, _sheets("TEST ST.", [[2000], [2100]]), era_modern=True)
    test = by_key["TEST"]
    assert test.tier == CLEAN
    assert test.value == "WIN"
    assert test.numerals is not None
    assert test.numerals.supported
    assert test.numerals.numerals == (2000, 2100)
    assert (test.numerals.inside_winner, test.numerals.inside_runner) == (2, 0)
    assert test.numerals.winner_range == "1560-2650N"


def test_an_ambiguity_is_held_when_numerals_also_fit_the_rival(source: RenameSource) -> None:
    by_key = _propose(source, _sheets("TEST ST.", [[500], [600]]), era_modern=True)
    test = by_key["TEST"]
    assert test.tier == HELD
    assert test.numerals is not None
    assert (test.numerals.inside_winner, test.numerals.inside_runner) == (0, 2)
    assert "also fit LOSE" in test.reason


def test_an_ambiguity_is_held_on_a_pre_renumbering_volume(source: RenameSource) -> None:
    """Source ranges are POST-renumbering numbers; the era gate is absolute.

    This is the qualification the whole numeral requirement exists for: the one
    measured wrong auto-decision sits on a pre-renumbering volume, where the
    check cannot run and the family can therefore only be held.
    """
    by_key = _propose(source, _sheets("TEST ST.", [[2000], [2100]]), era_modern=False)
    assert by_key["TEST"].tier == HELD
    assert "address era is not modern" in by_key["TEST"].reason


def test_an_ambiguity_is_held_with_no_city_grid(source: RenameSource) -> None:
    proposal = propose_volume(
        "synthetic_001",
        source,
        _index(),
        _sheets("TEST ST.", [[2000], [2100]]),
        grid=None,
        era_modern=True,
    )
    held = {p.key: p for p in proposal.held}
    assert "TEST" in held
    assert "no address grid configured" in held["TEST"].reason


def test_an_ambiguity_is_held_when_the_rival_numbers_on_the_other_axis(
    source: RenameSource,
) -> None:
    """Two ranges on different axes are different number lines.

    Nothing about a printed 2000 being outside ``1000-2000W`` follows from it
    being inside ``1560-2650N``. What separates a north-south street from an
    east-west one is the read's own printed orientation, which is a different
    channel and not this one's to claim.
    """
    by_key = _propose(source, _sheets("AXIS ST.", [[2000], [2100]]), era_modern=True)
    axis = by_key["AXIS"]
    assert axis.tier == HELD
    assert "different grid line" in axis.reason
    assert axis.numerals is not None
    assert (axis.numerals.winner_range, axis.numerals.runner_range) == (
        "1560-2650N",
        "1000-2000W",
    )


def test_an_ambiguity_is_held_with_too_few_numerals(source: RenameSource) -> None:
    """One numeral is an OCR read, not a corroboration."""
    by_key = _propose(source, _sheets("TEST ST.", [[2000], []]), era_modern=True)
    assert by_key["TEST"].tier == HELD
    assert "1 attributed numerals" in by_key["TEST"].reason


def test_diverging_printed_directions_are_held(source: RenameSource) -> None:
    """A stripped key cannot express a per-side rename, so it must not be written.

    Chicago's 51st Avenue diverges (N to LeClaire, S to Leamington) and the
    build record keyed the qualified forms by hand.
    """
    sheets = [
        _sheet("1", ["ALPHA ST.", "BETA ST.", "N. SPLIT ST."]),
        _sheet("2", ["ALPHA ST.", "BETA ST.", "S. SPLIT ST."]),
        _sheet("3", ["ALPHA ST.", "BETA ST.", "N. SPLIT ST."]),
    ]
    by_key = _propose(source, sheets)
    assert by_key["SPLIT"].tier == HELD
    assert "diverging printed directions" in by_key["SPLIT"].reason


def test_one_printed_direction_is_not_divergence(source: RenameSource) -> None:
    """A volume that reads only ``W. …`` forms is not a per-side rename."""
    sheets = [_sheet(str(i), ["ALPHA ST.", "BETA ST.", "N. SPLIT ST."]) for i in (1, 2)]
    assert _propose(source, sheets)["SPLIT"].tier == CLEAN


def test_the_entry_guard_holds_one_key_without_costing_the_others(
    source: RenameSource,
) -> None:
    """The recorded case is a single family; a whole-volume abort would be worse."""
    calls: list[tuple[str, str]] = []

    def guard(key: str, value: str) -> list[str]:
        calls.append((key, value))
        return ["would re-key a centerline"] if key == "ONLY" else []

    proposal = propose_volume(
        "synthetic_001",
        source,
        _index(),
        _sheets("ONLY ST.") + _sheets("TEST ST.", [[2000], [2100]]),
        grid=GRID,
        era_modern=True,
        entry_guard=guard,
    )
    by_key = {p.key: p for p in proposal.proposals}
    assert by_key["ONLY"].tier == HELD
    assert "would re-key a centerline" in by_key["ONLY"].reason
    assert by_key["TEST"].tier == CLEAN
    assert ("ONLY", "SOLO") in calls


def test_two_disjoint_documented_extents_are_not_merged_into_a_hull(
    source: RenameSource,
) -> None:
    """The source writes disjoint segments on one line; a hull invents the gap.

    ``-Gap St., Win Ave. 200 to 800N and 1560 to 2650N`` documents two stretches
    with 760 units of nothing between them. Merged into ``200-2650N`` a numeral
    at 1000 would "corroborate" a rename the source does not document there.
    """
    extent = documented_along(
        GRID,
        _index(),
        Candidate(
            value="WIN",
            sheets=2,
            locality_m=10.0,
            source_line="-Gap St., Win Ave. 200 to 800N and 1560 to 2650N",
        ),
        block=100.0,
    )
    assert extent is not None
    assert extent.ranges == ((200.0, 800.0), (1560.0, 2650.0))
    assert extent.covers(700, pad=0.0)
    assert extent.covers(2000, pad=0.0)
    assert not extent.covers(1000, pad=0.0), "the gap is not documented"
    assert extent.label == "200-800N+1560-2650N"

    by_key = _propose(source, _sheets("GAP ST.", [[1000], [1100]]), era_modern=True)
    assert by_key["GAP"].tier == HELD
    assert by_key["GAP"].numerals is not None
    assert by_key["GAP"].numerals.inside_winner == 0


def test_a_numeral_is_never_placed_on_the_side_the_range_happens_to_be_on() -> None:
    """A printed house number carries no direction; only geometry can place it.

    A street lying wholly north of the origin gets no numeral check off a range
    quoted SOUTH of it, and a street crossing the origin gets none at all.
    """
    index = _index()
    south_range = Candidate(
        value="WIN", sheets=2, locality_m=10.0, source_line="-X St., Win Ave. fr. 600 to 1200S"
    )
    assert documented_along(GRID, index, south_range, block=100.0) is None

    straddling = [
        *FEATURES,
        {
            "type": "Feature",
            "properties": {"street_nam": "STRADDLE", "street_typ": "AVE"},
            "geometry": {
                "type": "LineString",
                "coordinates": [[-87.658, 41.860], [-87.658, 41.910]],
            },
        },
    ]
    crossing = Candidate(
        value="STRADDLE",
        sheets=2,
        locality_m=10.0,
        source_line="-X St., Straddle Ave. fr. 100 to 1800N",
    )
    assert documented_along(GRID, CenterlineIndex(straddling), crossing, block=100.0) is None


def test_a_family_whose_every_candidate_is_far_is_refused_not_held(
    source: RenameSource,
) -> None:
    """More corroboration cannot reach a value with no geometry near the reads."""
    by_key = _propose(source, _sheets("FAR ST."))
    far = by_key["FAR"]
    assert far.tier == REFUSED
    assert "bar 400 m" in far.reason


def test_a_prefix_of_a_key_is_not_a_sheet_carrying_it(source: RenameSource) -> None:
    """``PARK RIDGE AV`` must not answer for the family ``PARK``.

    A prefix match would let sheets that never mention the read satisfy the
    two-sheet floor and set the locality from the wrong intersection cloud.
    """
    reads = sheet_reads([_sheet("1", ["ONLY RIDGE AV."])])
    assert not reads[0].has_key("ONLY")
    # ...so a volume whose only "ONLY" reads are a different street is not clean
    sheets = [_sheet("1", ["ALPHA ST.", "BETA ST.", "ONLY ST."])] + [
        _sheet(str(i), ["ALPHA ST.", "BETA ST.", "ONLY RIDGE AV."]) for i in (2, 3)
    ]
    assert _propose(source, sheets)["ONLY"].tier == HELD


def test_north_and_n_are_one_printed_direction(source: RenameSource) -> None:
    sheets = [
        _sheet("1", ["ALPHA ST.", "BETA ST.", "N. SPLIT ST."]),
        _sheet("2", ["ALPHA ST.", "BETA ST.", "NORTH SPLIT ST."]),
    ]
    assert _propose(source, sheets)["SPLIT"].tier == CLEAN


def test_a_read_the_source_does_not_document_lands_in_no_candidate(
    source: RenameSource,
) -> None:
    by_key = _propose(source, _sheets("UNDOCUMENTED ST."))
    assert by_key["UNDOCUMENTED"].tier == NO_CANDIDATE
    assert by_key["UNDOCUMENTED"].value is None


def test_a_candidate_out_of_bounds_is_never_proposed(source: RenameSource) -> None:
    """ABSENT AVE is documented but is not a key of this volume's index."""
    by_key = _propose(source, _sheets("NOWHERE ST."))
    assert by_key["NOWHERE"].tier == NO_CANDIDATE


def test_an_already_aliased_key_is_never_re_proposed(source: RenameSource) -> None:
    """An owner-signed entry cannot be silently restated or contradicted."""
    proposal = propose_volume(
        "synthetic_001",
        source,
        _index({"ONLY": "SOLO"}),
        _sheets("ONLY ST."),
        grid=GRID,
        existing={"ONLY": "SOLO"},
    )
    assert proposal.table == {}
    assert not proposal.held


def test_an_already_aliased_family_is_its_own_tier_not_a_held_row(
    source: RenameSource,
) -> None:
    """The held tier is an agent queue; a family already covered is not work.

    Reached when the landed entry does not resolve — here an inert value that is
    not an index key, so the reads stay unmatched and the family reappears. It
    used to be reported as ``HELD, not aliased: … best candidate None``, written
    into the fixture, in a file that aliased the key a few lines down.
    """
    proposal = propose_volume(
        "synthetic_001",
        source,
        _index({"ONLY": "GONE FROM THE GRID"}),
        _sheets("ONLY ST."),
        grid=GRID,
        existing={"ONLY": "GONE FROM THE GRID"},
    )
    assert [p.key for p in proposal.already_aliased] == ["ONLY"]
    assert not proposal.held
    assert proposal.table == {}


def test_the_empty_index_key_is_never_a_candidate_value() -> None:
    """The EMPTY key is still a real bucket, and a source phrase can normalize into it.

    A reference row whose whole name strips away keys as ``''``. Any source
    phrase that also strips to nothing then "snaps" onto it and scores as a
    candidate street, which is how a locality-decided proposal for a nameless
    bucket appeared in a corpus run. ``normalize`` no longer empties a real
    street name, so only punctuation-only rows reach the bucket now; the
    proposer's own guard is what protects a future parser, and it is pinned here.
    """
    features = [
        *FEATURES,
        {
            "type": "Feature",
            "properties": {"street_nam": "-", "street_typ": ""},
            "geometry": {"type": "LineString", "coordinates": [[-87.66, 41.91], [-87.66, 41.92]]},
        },
    ]
    index = CenterlineIndex(features, aliases={})
    assert "" in index.by_name  # the hazard is real
    source = RenameSource(
        parser="martin1948",
        text_path=Path("unused"),
        citation="x",
        _pairs=[RenamePair("ONLY ST", "--", "-Only St., --. 100N")],
    )
    proposal = propose_volume("synthetic_001", source, index, _sheets("ONLY ST."), grid=GRID)
    by_key = {p.key: p for p in proposal.proposals}
    assert by_key["ONLY"].tier == NO_CANDIDATE
    assert all(p.value != "" for p in proposal.proposals)


def test_documented_along_needs_a_range_at_least_a_block_wide() -> None:
    """A prose coincidence ('in 1909 to 1911 N. Clark') is not a street extent."""
    index = _index()
    narrow = Candidate(
        value="WIN", sheets=2, locality_m=10.0, source_line="renamed in 1909 to 1911 N. Clark"
    )
    assert documented_along(GRID, index, narrow, block=100.0) is None
    wide = Candidate(
        value="WIN", sheets=2, locality_m=10.0, source_line="-Test St., Win Ave. fr. 1560 to 2650N"
    )
    got = documented_along(GRID, index, wide, block=100.0)
    assert got is not None
    assert got.ranges == ((1560.0, 2650.0),)


def test_sheet_reads_ignores_a_numeral_with_no_street_attribution() -> None:
    reads = sheet_reads(
        [
            _sheet("1", ["ALPHA ST."]),
        ]
    )
    assert reads[0].numerals == {}
    reads = sheet_reads([_sheet("1", ["ALPHA ST."], {"ALPHA ST.": [100, 200]})])
    assert reads[0].numerals == {"ALPHA ST": (100, 200)}
