"""Matcher inputs: a malformed model bbox must not reach a consumer."""

from autogeoref.sheet_inputs import drop_malformed_labels, sheet_input_from

INFO = {"full_size": [2000, 3000], "small_size": [1326, 2000], "scale": 4.0}
#: A rotated page is what makes the sanitize/rotate ORDER observable: the frame
#: map unpacks a bbox itself, so it crashes on the malformed one if it goes first.
ROTATED = dict(INFO, rotation_applied=90)


def test_a_nested_bbox_is_dropped_not_unpacked() -> None:
    """A doubly-wrapped bbox crashed the rescue stage several stages later.

    ``rail_crossing_candidates`` unpacks ``x0, y0, x1, y1 = rail["bbox"]``, so
    a one-element outer list raised a ValueError naming neither page nor
    volume, after every model call for the volume had been paid for.
    """
    rails = [
        {"name": "C. & N. W. R'Y.", "bbox": [[745, 1780, 902, 1812]]},
        {"name": "C. & W. I. R.R.", "bbox": [10, 20, 30, 40]},
    ]
    ann = {
        "streets": [{"name": "W. MADISON", "bbox": [1, 2, 3, 4], "orientation": "horizontal"}],
        "rail_labels": rails,
    }

    cleaned = drop_malformed_labels("25", ann)

    assert [r["name"] for r in cleaned["rail_labels"]] == ["C. & W. I. R.R."]
    assert cleaned["streets"] == ann["streets"]
    # the caller's dict is never mutated: it is the cached read on disk
    assert len(rails) == 2


def test_a_clean_annotation_is_passed_through_unchanged() -> None:
    ann = {"streets": [{"name": "W. MADISON", "bbox": [1, 2, 3, 4], "orientation": "horizontal"}]}
    assert drop_malformed_labels("1", ann) is ann


def test_every_bbox_carrying_key_is_checked() -> None:
    ann = {
        "streets": [{"name": "A", "bbox": [1, 2, 3], "orientation": "horizontal"}],
        "rail_labels": [{"name": "B", "bbox": "1,2,3,4"}],
        "park_labels": [{"name": "C", "bbox": None}],
        "address_numerals": [{"value": 1, "bbox": [[1, 2, 3, 4]]}],
    }
    cleaned = drop_malformed_labels("7", ann)
    assert all(cleaned[key] == [] for key in ann)


def test_booleans_are_not_accepted_as_coordinates() -> None:
    ann = {"streets": [{"name": "A", "bbox": [True, 2, 3, 4], "orientation": "horizontal"}]}
    assert drop_malformed_labels("1", ann)["streets"] == []


def test_sheet_input_sanitizes_before_the_frame_map() -> None:
    """Order is load-bearing: the frame map unpacks bboxes of its own.

    Sanitizing second lets the rotation reach the malformed bbox first and
    raise there instead, which is the same outage one stage earlier.
    """
    ann = {"rail_labels": [{"name": "R", "bbox": [[1, 2, 3, 4]]}], "streets": []}
    sheet = sheet_input_from("25", ann, ROTATED)
    assert sheet.annotation["rail_labels"] == []
