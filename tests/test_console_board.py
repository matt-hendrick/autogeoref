"""The board payload: the four columns, the per-volume context, and what it admits.

The queue section is `queue.progress.board`'s payload verbatim, so the terminal
view and the page cannot drift into disagreeing about one tree. The context
block carries the year and the neighbourhoods a volume's bounds fall in, and
tolerates a missing area file. With no city there was no era check, and the
payload says so rather than presenting a backlog nobody vetted.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from autogeoref.config.model import VolumeConfig
from autogeoref.console import payload as console_payload
from autogeoref.queue import progress as qprogress
from autogeoref.queue import store as qstore
from autogeoref.status import build_status
from console_support import _city, _images, _results, _status, _tiles, _tree


def test_board_carries_the_four_columns_and_never_disagrees_with_the_queue(
    tmp_path: Path,
) -> None:
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_runnable", 4)
    _images(roots["work"], "vol_served", 4)
    _results(roots["work"], "vol_served", accepted=3, flagged=1)
    _tiles(roots["tiles"], "autogeoref", "vol_served")
    _images(roots["work"], "vol_queued", 4)
    qstore.add(roots["work"], "vol_queued", "place")

    payload = console_payload.board(
        work=roots["work"], rows=_status(roots), city=_city(renumbering=False)
    )
    assert [c["volume"] for c in payload["runnable"]] == ["vol_runnable"]
    assert [v["volume"] for v in payload["served"]] == ["vol_served"]
    # `entries` is queue.progress.board's payload verbatim — the terminal view and the page
    # cannot drift into disagreeing about the same tree
    assert payload["entries"] == qprogress.board(roots["work"])["entries"]
    assert payload["links"]["review"] and payload["links"]["viewer"]


def test_board_context_labels_volumes_and_tolerates_a_missing_area_file(tmp_path: Path) -> None:
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 4)
    areas = tmp_path / "community-areas.geojson"
    areas.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
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
    city = replace(
        _city(renumbering=False),
        community_areas_path=areas,
        volumes={
            "vol_a": VolumeConfig(identifier="vol_a", bounds_bbox=(-87.70, 41.85, -87.60, 41.95))
        },
    )

    payload = console_payload.board(
        work=roots["work"],
        rows=_status(roots),
        city=city,
        catalog={"vol_a": {"year": 1917}},
    )

    assert payload["context"]["vol_a"] == {
        "city": "Chicago, Ill.",
        "year": 1917,
        "neighborhoods": ["Testville"],
    }
    missing = replace(city, community_areas_path=tmp_path / "missing-community-areas.geojson")
    payload = console_payload.board(
        work=roots["work"], rows=_status(roots), city=missing, catalog={"vol_a": {"year": 1917}}
    )
    assert payload["context"]["vol_a"]["neighborhoods"] == []


def test_board_context_reads_the_supplied_manifest(tmp_path: Path) -> None:
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 4)
    manifest = tmp_path / "custom-manifest.json"
    manifest.write_text(json.dumps({"volumes": [{"id": "vol_a", "bounds": [0, 0, 1, 1]}]}))
    areas = tmp_path / "community-areas.geojson"
    areas.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "properties": {"community": "TESTVILLE"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-88, 41, -87, 41, -87, 42, -88, 42, -88, 41]]],
                        },
                    }
                ],
            }
        )
    )
    city = replace(
        _city(renumbering=False),
        community_areas_path=areas,
        volumes={"vol_a": VolumeConfig(identifier="vol_a", bounds_bbox=(-87.7, 41.8, -87.6, 41.9))},
    )

    payload = console_payload.board(
        work=roots["work"], rows=_status(roots), city=city, viewer_manifest=manifest
    )

    assert payload["context"]["vol_a"]["neighborhoods"] == []


def test_the_board_payload_names_the_queues_it_expects_rendered(tmp_path: Path) -> None:
    """The page loops over this, so it is the contract: pipeline order, by name, and
    every one of them has an element in the markup (the test above)."""
    roots = _tree(tmp_path)
    rows = build_status(work=roots["work"], fixtures=roots["fixtures"], tiles=roots["tiles"])
    board = console_payload.board(work=roots["work"], rows=rows)
    assert board["tracks"] == ["fetch", "place", "serve"]


def test_the_board_says_so_when_no_city_config_vetted_the_backlog(tmp_path: Path) -> None:
    """With no city there is no era check, so every candidate renders as ready — and on
    a renumbering city most of those are runs that refuse on line one. The payload must
    ADMIT that rather than present a backlog nobody vetted."""
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 4)

    assert console_payload.board(work=roots["work"], rows=_status(roots))["era_check"] is False
    vetted = console_payload.board(work=roots["work"], rows=_status(roots), city=_city())
    assert vetted["era_check"] is True
