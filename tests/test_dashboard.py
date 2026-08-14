"""Coverage and provenance dashboard contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from autogeoref.config.load import load_city_config
from autogeoref.dashboard import (
    COVERAGE_STATES,
    UNLOCATED,
    build_coverage,
    coverage_json,
    coverage_state,
    render_html,
)
from autogeoref.status import VolumeStatus
from autogeoref.viewer.config import ViewerConfig
from autogeoref.viewer.era import EraBucket

CITY_TOML = Path("configs/chicago/chicago.toml")


def _status(volume: str, **kw: object) -> VolumeStatus:
    """A status row with everything absent unless the test names it."""
    fields: dict[str, object] = {
        "volume": volume,
        "sheets": None,
        "gt": None,
        "gt_unscoreable": None,
        "unaddressable": None,
        "lost_sheets": [],
        "damaged_results": None,
        "damaged_frozen_results": None,
        "reads": None,
        "results": None,
        "accepted": None,
        "flagged": None,
        "reviewer_verified": None,
        "pinned_orientation": None,
        "mask_qa_flagged": None,
        "mask_qa_volume_flags": None,
        "frozen_source": None,
        "frozen_sheets": None,
        "frozen_accepted": None,
        "tiles": None,
        "serve_stale": None,
        "stale_record": None,
        "note": "",
    }
    fields.update(kw)
    # what build_status derives: the row is ours when its archive sits in the
    # directory this pipeline publishes into
    fields.setdefault("ours", fields["tiles"] == "autogeoref")
    return VolumeStatus(**fields)  # type: ignore[arg-type]


VIEWER = ViewerConfig(era_buckets=(EraBucket(first_year=1894, last_year=1897, label="1895"),))


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (_status("a", tiles="autogeoref"), "autogeoref"),
        (_status("d", results=10, accepted=5), "processed-unserved"),
        (_status("e", sheets=40), "ready"),
        (_status("f"), "no-scans"),
        # human pins are not scans: a pinned volume whose images were never
        # fetched has nothing to run on, and must not read as a runnable queue item
        (_status("g", gt=90), "no-scans"),
    ],
)
def test_coverage_state_buckets(row: VolumeStatus, expected: str) -> None:
    assert coverage_state(row) == expected


def test_serving_provenance_beats_processing() -> None:
    """Serving provenance and local processing remain independent, in both
    directions: results on disk are not a published layer, and a published layer
    is not evidence the work tree is still here."""
    unpublished = _status("v", results=97, accepted=59, flagged=38, sheets=97)
    assert coverage_state(unpublished) == "processed-unserved"
    assert unpublished.processed_here is True

    pruned = _status("w", tiles="autogeoref", sheets=97)
    assert coverage_state(pruned) == "autogeoref"
    assert pruned.processed_here is False


def test_states_all_have_a_stylesheet_colour() -> None:
    css = (Path("src/autogeoref/dashboard_ui/dashboard.css")).read_text(encoding="utf-8")
    for state in COVERAGE_STATES:
        assert f"--s-{state.key}:" in css, f"{state.key} has no colour: its segment is invisible"


@pytest.fixture
def coverage(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Two volumes: one locatable by human pins, one locatable by nothing."""
    gt_dir = tmp_path / "ground-truth"
    gt_dir.mkdir()
    # a minimal pinned export: one layer with an extent, which is all the
    # footprint needs (bounds.volume_bounds reads `extent`)
    (gt_dir / "api-layers-pinned.json").write_text(
        json.dumps(
            [
                {
                    "slug": "pinned_p1",
                    "extent": [-87.65, 41.88, -87.62, 41.90],
                    "gcps_geojson": {"type": "FeatureCollection", "features": []},
                }
            ]
        )
    )
    # Provide an area index for location assertions.
    areas_path = tmp_path / "community_areas.geojson"
    areas_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"community": "TESTVILLE"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [-87.70, 41.85],
                                    [-87.60, 41.85],
                                    [-87.60, 41.95],
                                    [-87.70, 41.95],
                                    [-87.70, 41.85],
                                ]
                            ],
                        },
                    }
                ],
            }
        )
    )
    return build_coverage(
        [
            _status("pinned", gt=1),
            _status("nowhere", sheets=30),
        ],
        replace(load_city_config(CITY_TOML), community_areas_path=areas_path),
        VIEWER,
        ground_truth_dir=gt_dir,
    )


def test_unlocatable_volume_is_bucketed_not_dropped(coverage) -> None:  # type: ignore[no-untyped-def]
    """A volume nobody can place still appears — in its own honest bucket."""
    rows = {r.volume: r for r in coverage.rows}
    assert rows["nowhere"].bounds_source is None
    assert rows["nowhere"].areas == []

    buckets = {b.name: b for b in coverage.by_area}
    assert UNLOCATED in buckets, "an unlocatable volume vanished from the area rollup"
    assert buckets[UNLOCATED].counts["ready"] == 1
    # and it is the LAST row: a gap in the map is not a place
    assert coverage.by_area[-1].name == UNLOCATED


def test_pinned_volume_is_located_by_its_human_pins(coverage) -> None:  # type: ignore[no-untyped-def]
    row = next(r for r in coverage.rows if r.volume == "pinned")
    assert row.bounds_source == "volunteer pins"
    assert row.areas, "a pinned volume must land in at least one community area"


def test_scoreable_needs_both_pins_and_scans() -> None:
    """The reason there is no accuracy panel: pins and scans are near-disjoint,
    and only their intersection can be scored at all."""
    coverage = build_coverage(
        [_status("pins", gt=90), _status("scans", sheets=90), _status("both", gt=90, sheets=90)],
        load_city_config(CITY_TOML),
        VIEWER,
        ground_truth_dir=Path("does-not-exist"),
    )
    assert {r.volume for r in coverage.rows if r.scoreable} == {"both"}


def test_render_is_self_contained_and_lists_every_volume(coverage) -> None:  # type: ignore[no-untyped-def]
    html = render_html(coverage, generated="2026-07-12")
    for volume in ("pinned", "nowhere"):
        assert volume in html
    assert "2026-07-12" in html, "an undated coverage page reads as a current one"
    # self-contained like the viewer bundle: no CDN, no fetch, no external asset
    for leak in ("http://", "https://", "<script"):
        assert leak not in html, f"page reaches outside itself: {leak}"
    # the refusals are on the page, not just in the docstring
    assert "not an accuracy report" in html
    assert "volume footprint" in html


def test_json_is_the_same_numbers(coverage) -> None:  # type: ignore[no-untyped-def]
    data = json.loads(coverage_json(coverage))
    assert data["totals"]["volumes"] == 2
    assert {v["volume"] for v in data["volumes"]} == {"pinned", "nowhere"}
    assert [s["key"] for s in data["states"]] == [s.key for s in COVERAGE_STATES]
