"""rotation_applied composition: annotations on normalized smalls reach the
matcher in the SOURCE-scan frame (adversarial-review G1 finding 3)."""

import json
from pathlib import Path
from typing import Any

from autogeoref.frames import rotate_annotation
from autogeoref.paths import VolumePaths
from autogeoref.sheet_inputs import annotation_in_source_frame, load_sheet_inputs

SRC_SMALL = (1326, 2000)  # source-scan small frame (portrait)
UPRIGHT_SMALL = (2000, 1326)  # after the 90 CW correction (landscape)

ANN_SOURCE: dict[str, Any] = {
    "streets": [
        {"name": "MAIN ST", "bbox": [100, 200, 300, 240], "orientation": "horizontal"},
        {"name": "OAK AV", "bbox": [500, 100, 540, 400], "orientation": "vertical"},
    ],
    "rail_labels": [{"name": "C.M.&ST.P.R.R.", "bbox": [200, 300, 800, 340]}],
    "address_numerals": [{"value": 151, "bbox": [110, 205, 130, 217], "street": "MAIN ST"}],
    "page_number_seen": "5",
}


def test_identity_without_rotation_key() -> None:
    info = {"small_size": list(SRC_SMALL), "scale": 0.25, "full_size": [5304, 8000]}
    assert annotation_in_source_frame(ANN_SOURCE, info) is ANN_SOURCE


def test_round_trip_through_upright_frame() -> None:
    """Annotation made on the upright small maps back to the source frame."""
    # forward: what the annotator would see after prep applied 90 CW
    ann_upright = rotate_annotation(ANN_SOURCE, 90, SRC_SMALL)
    info = {
        "small_size": list(UPRIGHT_SMALL),  # prep records the as-written frame
        "scale": 0.25,
        "full_size": [5304, 8000],
        "rotation_applied": 90,
    }
    back = annotation_in_source_frame(ann_upright, info)
    for orig, got in zip(ANN_SOURCE["streets"], back["streets"], strict=True):
        assert got["orientation"] == orig["orientation"]
        assert tuple(got["bbox"]) == tuple(orig["bbox"])


def test_every_bbox_feature_class_round_trips() -> None:
    """rail_labels and address_numerals turn too — not just streets.

    rail.rail_crossing_candidates intersects a rail label's axis with the
    STREET axes: if streets were composed and rail labels were not, the
    crossing would be computed across two coordinate systems and feed the
    rescue fit a bogus pixel (a wrong placement, not a skip).
    """
    ann_upright = rotate_annotation(ANN_SOURCE, 90, SRC_SMALL)
    info = {
        "small_size": list(UPRIGHT_SMALL),
        "scale": 0.25,
        "full_size": [5304, 8000],
        "rotation_applied": 90,
    }
    # the upright frame really is a different frame — else this proves nothing
    assert ann_upright["rail_labels"][0]["bbox"] != ANN_SOURCE["rail_labels"][0]["bbox"]

    back = annotation_in_source_frame(ann_upright, info)
    for key in ("rail_labels", "address_numerals"):
        for orig, got in zip(ANN_SOURCE[key], back[key], strict=True):
            assert tuple(got["bbox"]) == tuple(orig["bbox"]), key
            assert got.get("name", got.get("value")) == orig.get("name", orig.get("value"))


def test_load_sheet_inputs_composes_rotation(tmp_path: Path) -> None:
    paths = VolumePaths(root=tmp_path)
    paths.annotations.mkdir(parents=True)
    paths.sheets.mkdir(parents=True)
    ann_upright = rotate_annotation(ANN_SOURCE, 90, SRC_SMALL)
    (paths.annotations / "p5.json").write_text(json.dumps(ann_upright))
    manifest = {
        "p5": {
            "small_size": list(UPRIGHT_SMALL),
            "scale": 0.25,
            "full_size": [5304, 8000],
            "rotation_applied": 90,
        }
    }
    paths.manifest.write_text(json.dumps(manifest))
    sheets = load_sheet_inputs(paths)
    assert len(sheets) == 1
    got = sheets[0].annotation["streets"]
    assert tuple(got[0]["bbox"]) == (100, 200, 300, 240)
    assert got[0]["orientation"] == "horizontal"
