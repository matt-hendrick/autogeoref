"""The coverage gate: a caption may not point at blank paper.

A stop names a camera and the eras it is meant to be read in, and this decides
whether any served layer actually draws there. The volume's envelope is not
enough — a third of its sheets can be flagged and the hole still falls inside
the box — so the placed sheets are checked where an export tree exists, and the
report says so when there is none.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autogeoref.config.model import ConfigError
from autogeoref.viewer.config import load_viewer_config
from autogeoref.viewer.coverage import SheetFootprints, assert_stops_are_covered
from viewer_support import _stop, _story, _story_world


def _served(**overrides: Any) -> dict[str, Any]:
    volume: dict[str, Any] = {
        "id": "vireo_002",
        "title": "Port Vireo | 1905 | Vol. 2",
        "era": "1905",
        "bounds": [172.60, -43.56, 172.68, -43.50],
        "pmtiles": "vireo_002.pmtiles",
    }
    volume.update(overrides)
    return volume


def _exports(root: Path, volume: str, boxes: list[tuple[float, float, float, float]]) -> Path:
    """One export tree whose sheets' control points span ``boxes``."""
    gcps = root / volume / "gcps"
    gcps.mkdir(parents=True)
    for index, (west, south, east, north) in enumerate(boxes, start=1):
        corners = [(west, south), (east, north), (west, north)]
        (gcps / f"p{index}.json").write_text(
            json.dumps(
                {
                    "page": f"p{index}",
                    "gcps_geojson": {
                        "type": "FeatureCollection",
                        "features": [
                            {"geometry": {"type": "Point", "coordinates": [lng, lat]}}
                            for lng, lat in corners
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )
    return root


def test_a_stop_in_a_hole_inside_the_envelope_stops_the_build(tmp_path: Path) -> None:
    """The volume's box is not enough: a third of its sheets may be flagged and
    the hole still falls inside the box. Check the placed sheets."""
    footprints = SheetFootprints(
        _exports(tmp_path / "exports", "vireo_002", [(172.60, -43.56, 172.62, -43.54)])
    )
    toml = _story_world(tmp_path, [_story(_stop())])  # camera at 172.64, -43.53
    with pytest.raises(ConfigError, match="no placed sheet reaches it"):
        assert_stops_are_covered(load_viewer_config(toml).stories, [_served()], footprints)


def test_coverage_names_the_volume_that_will_draw_the_stop(tmp_path: Path) -> None:
    """Presence is not identity: an envelope that looks like an ordinary
    district can belong to a special that draws something else."""
    footprints = SheetFootprints(
        _exports(tmp_path / "exports", "vireo_002", [(172.63, -43.54, 172.66, -43.52)])
    )
    toml = _story_world(tmp_path, [_story(_stop())])
    lines = assert_stops_are_covered(load_viewer_config(toml).stories, [_served()], footprints)
    assert lines == ["s1/one: vireo_002 Port Vireo | 1905 | Vol. 2"]


def test_a_layer_with_no_sheet_record_is_covered_but_says_so(tmp_path: Path) -> None:
    """A foreign or not-yet-exported layer can only be judged on its envelope,
    and the report has to admit that rather than imply a sheet was checked."""
    toml = _story_world(tmp_path, [_story(_stop())])
    lines = assert_stops_are_covered(load_viewer_config(toml).stories, [_served()])
    assert lines == ["s1/one: vireo_002 Port Vireo | 1905 | Vol. 2 [no sheet record]"]


def test_a_stop_in_the_wrong_era_is_not_covered(tmp_path: Path) -> None:
    """A stop is covered only in the eras it declares — turning the chips to a
    decade with no layer there is exactly the blank pane this gate exists for."""
    toml = _story_world(tmp_path, [_story(_stop())])
    with pytest.raises(ConfigError, match="which no served layer covers"):
        assert_stops_are_covered(load_viewer_config(toml).stories, [_served(era="1930")])


def test_a_listed_but_unserved_volume_does_not_cover_a_stop(tmp_path: Path) -> None:
    """The rule the deploy bundle uses: a volume covers a stop only through its
    own archive. A list-only volume publishes no imagery to stand on."""
    toml = _story_world(tmp_path, [_story(_stop())])
    listed = _served()
    del listed["pmtiles"]
    with pytest.raises(ConfigError, match="which no served layer covers"):
        assert_stops_are_covered(load_viewer_config(toml).stories, [listed])
    # and the same volume with its archive back does cover it
    assert assert_stops_are_covered(load_viewer_config(toml).stories, [_served()])
