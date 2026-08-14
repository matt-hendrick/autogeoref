"""Source discovery: LOC catalog titles, community-area names, pmtiles classes.

The catalog parse is where a volume gets its title, year and the provenance of
that year; the subject maps are the awkward half, since LOC leaves their date
field empty and the year has to come out of the description prose. The pmtiles
classifier decides what is a volume and what is a bake leftover nothing serves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autogeoref.viewer.sources import AreaIndex, classify_pmtiles, loc_titles
from viewer_support import _write_catalog


def test_loc_titles_vol_prefix(tmp_path: Path) -> None:
    catalog = _write_catalog(
        tmp_path / "cat.json",
        [
            {
                "id": "http://www.loc.gov/item/sanborn01790_001/",
                "description": ["Vol. 9, 1894. 139 sheet(s). Bound."],
                "date": "1894",
            }
        ],
    )
    meta = loc_titles(catalog, "Test City")
    assert meta["sanborn01790_001"] == {
        "title": "Test City | 1894 | Vol. 9",
        "year": 1894,
        "year_source": "date",  # LOC's structured field
        "volume_number": "9",
        "subject": None,
    }


def test_loc_titles_specials_quirk(tmp_path: Path) -> None:
    """Grain elevators / stockyards: no "Vol." prefix, empty date field —
    the year leads the description and the subject is what remains once the
    format boilerplate (date, sheet count, binding, dimensions) is stripped."""
    catalog = _write_catalog(
        tmp_path / "cat.json",
        [
            {
                "id": "http://www.loc.gov/item/sanborn01790_015/",
                "description": ["1901. 54 sheet(s). Union Stock Yards."],
                "date": "",
            },
            {  # no year anywhere -> honestly excluded
                "id": "http://www.loc.gov/item/sanborn01790_099/",
                "description": ["Bound."],
                "date": "",
            },
        ],
    )
    meta = loc_titles(catalog, "Test City")
    assert meta["sanborn01790_015"] == {
        "title": "Test City | 1901 | Union Stock Yards",
        "year": 1901,
        "year_source": "description",  # scraped from the blurb — see below
        "volume_number": None,
        "subject": "Union Stock Yards",
    }
    assert "sanborn01790_099" not in meta


def test_catalog_subject_prefers_the_maps_own_name_over_other_places(tmp_path: Path) -> None:
    """LOC's "Other places as they appear on original:" clause is a
    cross-reference, not the name. The exposition sheet lost its name to that
    clause under last-sentence parsing ("Burnham Park" shipped as the title of
    the World's Fair map); prose before the clause must win, and the referenced
    place stands in only when nothing else survives (the CBD sheets)."""
    catalog = _write_catalog(
        tmp_path / "cat.json",
        [
            {
                "id": "http://www.loc.gov/item/sanborn01790_190/",
                "description": [
                    "Apr 1933. 1 sheet(s). 63 X 272 cm. A Century of Progress "
                    "International Exposition. Other places as they appear on "
                    "original: Burnham Park."
                ],
                "date": "",
                "item": {"created_published": "Sanborn Map Company, Apr 1933"},
            },
            {
                "id": "http://www.loc.gov/item/sanborn01790_188/",
                "description": [
                    "1927. 1 sheet(s). 112 X 77 cm. Other places as they appear "
                    "on original: Central Business District."
                ],
                "date": "1927",
            },
            {  # month forms LOC actually writes: "Sept. 1916.", not "Sep 1916."
                "id": "http://www.loc.gov/item/sanborn01790_191/",
                "description": ["Sept. 1916. 2 sheet(s). New, additional sheets."],
                "date": "1916",
            },
            {  # a subject's own leading place name is NOT date boilerplate,
                # even when it starts with a month prefix and cites a year
                "id": "http://www.loc.gov/item/sanborn01790_192/",
                "description": ["1901. 1 sheet(s). Maywood 1901. Annex district."],
                "date": "1901",
            },
            {  # a mid-subject "Vol." is prose, not a volume number
                "id": "http://www.loc.gov/item/sanborn01790_016/",
                "description": [
                    "1903. 1 sheet(s). 89 X 68 cm. Block line map of Chicago, "
                    "Vol. 1 Heavy valued district."
                ],
                "date": "1903",
            },
        ],
    )
    meta = loc_titles(catalog, "Test City")
    assert meta["sanborn01790_190"]["subject"] == "A Century of Progress International Exposition"
    assert meta["sanborn01790_188"]["subject"] == "Central Business District"
    assert meta["sanborn01790_191"]["subject"] == "New, additional sheets"
    assert meta["sanborn01790_192"]["subject"] == "Maywood 1901. Annex district"
    assert meta["sanborn01790_016"]["volume_number"] is None
    assert meta["sanborn01790_016"]["subject"] == (
        "Block line map of Chicago, Vol. 1 Heavy valued district"
    )


def test_loc_titles_marks_a_conflicting_subject_map_year_untrusted(tmp_path: Path) -> None:
    catalog = _write_catalog(
        tmp_path / "cat.json",
        [
            {
                "id": "http://www.loc.gov/item/sanborn01790_188/",
                "description": ["1927. 1 sheet(s). Central Business District."],
                "date": "1963-11",
                "item": {"created_published": "Sanborn Map Company, 1927"},
            }
        ],
    )

    assert loc_titles(catalog, "Test City")["sanborn01790_188"] == {
        "title": "Test City | 1927 | Central Business District",
        "year": 1927,
        "year_source": "description-conflict",
        "volume_number": None,
        "subject": "Central Business District",
    }


def test_loc_titles_keeps_an_uncorroborated_structured_date(tmp_path: Path) -> None:
    catalog = _write_catalog(
        tmp_path / "cat.json",
        [
            {
                "id": "http://www.loc.gov/item/sanborn01790_999/",
                "description": ["1927. 1 sheet(s). Test district."],
                "date": "1963-11",
            }
        ],
    )

    meta = loc_titles(catalog, "Test City")["sanborn01790_999"]
    assert meta["year"] == 1963 and meta["year_source"] == "date"


def test_a_scraped_year_is_labelled_as_one(fixtures_dir: Path) -> None:
    """The year's PROVENANCE, because one caller now bets on the answer.

    LOC populated the structured `date` field only for the bound city volumes. Its four
    Chicago SUBJECT maps carry `date: null`, and their year is recovered by a regex over
    the description prose. That was free while the year only lettered a title in the
    viewer; `era.py` now proposes `addresses_modern` from it, which arms the only evidence
    channel allowed to REFUTE. So each entry says which kind of year it is.
    """
    meta = loc_titles(fixtures_dir / "loc-catalog-chicago.json", "Chicago, Ill.")

    # the ordinary bound volumes: a real catalogued date
    assert meta["sanborn01790_024"]["year_source"] == "date"
    # the subject maps: LOC left `date` null, so the year came out of the blurb
    for special in ("sanborn01790_014", "sanborn01790_015", "sanborn01790_016", "sanborn01790_190"):
        assert meta[special]["year_source"] == "description", special

    scraped = [v for v, m in meta.items() if m["year_source"] == "description"]
    assert len(scraped) == 4, f"exactly four, and they are known: {scraped}"
    assert meta["sanborn01790_188"]["year"] == 1927
    assert meta["sanborn01790_188"]["year_source"] == "description-conflict"
    assert meta["sanborn01790_190"]["year"] == 1933


def test_a_month_year_needs_created_published_corroboration(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    """`_190`'s month-year description is accepted only because LOC corroborates it."""
    meta = loc_titles(fixtures_dir / "loc-catalog-chicago.json", "Chicago, Ill.")
    assert meta["sanborn01790_190"]["year"] == 1933

    catalog = _write_catalog(
        tmp_path / "unverified-month-year.json",
        [
            {
                "id": "http://www.loc.gov/item/sanborn01790_999/",
                "description": ["Apr 1933. 1 sheet(s). Unverified."],
                "date": "",
            },
            {
                "id": "http://www.loc.gov/item/sanborn01790_998/",
                "description": ["Foo 1933. 1 sheet(s). Not a month."],
                "date": "",
                "item": {"created_published": "Publisher, 1933"},
            },
            {
                "id": "http://www.loc.gov/item/sanborn01790_997/",
                "description": ["Apr 1933. 1 sheet(s). Mismatched."],
                "date": "",
                "item": {"created_published": "Publisher, 1932"},
            },
        ],
    )
    meta = loc_titles(catalog, "Test City")
    assert not {"sanborn01790_999", "sanborn01790_998", "sanborn01790_997"} & set(meta)


def test_loc_titles_real_chicago_catalog(fixtures_dir: Path) -> None:
    """Against the frozen catalog: the parity volume must parse exactly."""
    meta = loc_titles(fixtures_dir / "loc-catalog-chicago.json", "Chicago, Ill.")
    assert meta["sanborn01790_024"]["year"] == 1917
    assert meta["sanborn01790_024"]["volume_number"] == "7"
    assert meta["sanborn01790_024"]["title"] == "Chicago, Ill. | 1917 | Vol. 7"
    # the served specials, by their catalogued subjects
    assert meta["sanborn01790_015"]["subject"] == "Packing Houses. Union Stock Yards"
    assert meta["sanborn01790_188"]["subject"] == "Central Business District"
    assert meta["sanborn01790_189"]["subject"] == "North Central Business District"
    assert meta["sanborn01790_190"]["subject"] == "A Century of Progress International Exposition"
    # exactly one naming claim per item: a volume number or a subject
    for ident, m in meta.items():
        assert (m["volume_number"] is None) != (m["subject"] is None), ident


def test_area_index_orders_by_overlap(tmp_path: Path) -> None:
    def square(x0: float, y0: float, x1: float, y1: float) -> dict[str, Any]:
        return {
            "type": "Polygon",
            "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
        }

    geo = {
        "type": "FeatureCollection",
        "features": [
            {"properties": {"community": "MOSTLY"}, "geometry": square(0, 0, 10, 10)},
            {"properties": {"community": "EDGE"}, "geometry": square(9, 0, 20, 10)},
            {"properties": {"community": "ELSEWHERE"}, "geometry": square(50, 50, 60, 60)},
            {"properties": {"community": ""}, "geometry": square(0, 0, 1, 1)},
        ],
    }
    path = tmp_path / "areas.geojson"
    path.write_text(json.dumps(geo))
    index = AreaIndex(path)
    assert index.names([1, 1, 9.5, 9]) == ["Mostly", "Edge"]
    assert index.names([1, 1, 9.5, 9], top=1) == ["Mostly"]


def test_classify_pmtiles_volume_vs_overview_vs_in_progress(tmp_path: Path) -> None:
    (tmp_path / "sanborn01790_124.pmtiles").write_bytes(b"x")
    # a bake still writes this beside the archive; nothing serves it, and it
    # must not become a volume of its own
    (tmp_path / "sanborn01790_124-overview.pmtiles").write_bytes(b"x")
    (tmp_path / "sanborn01790_125.pmtiles").write_bytes(b"")  # zero-byte: mid-bake
    assert set(classify_pmtiles(tmp_path)) == {"sanborn01790_124"}


def test_classify_pmtiles_takes_a_hyphenated_stem_as_a_volume(tmp_path: Path) -> None:
    """`-overview` is the ONE suffix with a meaning. A stem that merely ends in
    a hyphenated segment — including a year, which used to name a citywide era
    archive — is an ordinary volume identifier now that nothing bakes one."""
    (tmp_path / "chicago-1950.pmtiles").write_bytes(b"x")
    (tmp_path / "sanborn01790_086-preskeletonfix.pmtiles").write_bytes(b"x")
    assert set(classify_pmtiles(tmp_path)) == {
        "chicago-1950",
        "sanborn01790_086-preskeletonfix",
    }
