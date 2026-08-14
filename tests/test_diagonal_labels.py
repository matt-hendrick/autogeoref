"""A street label that is neither horizontal nor vertical: schema, axis, frame, cache."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from autogeoref.annotate.failures import AnnotateError
from autogeoref.annotate.providers import model_cache_key, model_from_cache_key
from autogeoref.annotate.schema import (
    DIAGONAL_PROMPT_TEMPLATE,
    EXTENDED_PROMPT_TEMPLATE,
    Annotation,
    StreetLabel,
    parse_extended_response,
    prompt_template,
)
from autogeoref.frames import rotate_annotation, rotate_direction_cw, rotate_point_cw
from autogeoref.matching import label_axis


def _street(**over: Any) -> dict[str, Any]:
    return {"name": "MAPLE AV.", "bbox": [100, 200, 300, 260], "orientation": "horizontal"} | over


def test_diagonal_is_a_readable_orientation_and_keeps_its_vector() -> None:
    ann = Annotation.from_dict(
        {
            "streets": [_street(orientation="diagonal", direction=[0.8, -0.6])],
            "page_number_seen": None,
        }
    )
    assert ann.streets[0].orientation == "diagonal"
    assert ann.streets[0].direction == (0.8, -0.6)
    assert ann.to_dict()["streets"][0]["direction"] == [0.8, -0.6]


def test_a_cardinal_label_round_trips_without_growing_a_direction_key() -> None:
    """Every cached read on disk is cardinal; widening must not change its shape."""
    label = StreetLabel.from_dict(_street())
    assert label.direction is None
    assert "direction" not in label.to_dict()


@pytest.mark.parametrize(
    "direction", [None, "north-east", [1], [1, 2, 3], [0, 0], ["a", "b"], [True, False]]
)
def test_an_unusable_direction_degrades_the_label_and_never_the_sheet(direction: object) -> None:
    """One bad vector must not cost the page.

    ``parse_annotation_response`` raises on a bad orientation, and that raise
    takes the whole sheet with it: a page is one call. A direction is
    additive evidence, so it degrades to no vector instead.
    """
    payload = {
        "streets": [_street(orientation="diagonal", direction=direction)],
        "page_number_seen": "3",
    }
    ext = parse_extended_response(json.dumps(payload))
    assert ext.annotation.streets[0].orientation == "diagonal"
    assert ext.annotation.streets[0].direction is None


def test_an_orientation_that_is_none_of_the_three_is_still_a_malformed_read() -> None:
    payload = {"streets": [_street(orientation="slanted")], "page_number_seen": None}
    with pytest.raises(Exception, match="orientation must be"):
        Annotation.from_dict(payload)


# --- the pixel axis -------------------------------------------------------


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_a_cardinal_axis_is_unchanged_by_the_widening(orientation: str) -> None:
    """The 88% of the corpus that is cardinal must be byte-identical."""
    axis = label_axis(_street(orientation=orientation))
    cx, cy = 200.0, 230.0
    expected = (
        [(cx - 100000, cy), (cx + 100000, cy)]
        if orientation == "horizontal"
        else [(cx, cy - 100000), (cx, cy + 100000)]
    )
    assert list(axis.coords) == expected


def test_a_diagonal_axis_runs_along_its_vector() -> None:
    axis = label_axis(_street(orientation="diagonal", direction=[3.0, -4.0]))
    (ax, ay), (bx, by) = axis.coords
    assert math.isclose(math.atan2(by - ay, bx - ax), math.atan2(-4.0, 3.0), abs_tol=1e-9)
    # the label centre stays on the line
    assert math.isclose((ax + bx) / 2, 200.0, abs_tol=1e-6)
    assert math.isclose((ay + by) / 2, 230.0, abs_tol=1e-6)


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [([100, 200, 300, 260], "horizontal"), ([100, 200, 160, 700], "vertical")],
)
def test_a_diagonal_without_a_vector_falls_back_to_its_bbox_axis(
    bbox: list[int], expected: str
) -> None:
    """Never a dropped label: the fallback is exactly what a cardinal read gives."""
    diagonal = label_axis({"name": "X", "bbox": bbox, "orientation": "diagonal"})
    cardinal = label_axis({"name": "X", "bbox": bbox, "orientation": expected})
    assert list(diagonal.coords) == list(cardinal.coords)


# --- the frame ------------------------------------------------------------


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_a_direction_turns_with_the_frame(rotation: int) -> None:
    """``rotate_annotation`` used to leave a vector alone: a silent wrong axis.

    The vector is the linear part of the point map, so it must agree with two
    points rotated individually.
    """
    size = (1000.0, 2000.0)
    p0, p1 = (10.0, 20.0), (40.0, 60.0)
    q0 = rotate_point_cw(*p0, rotation, size)
    q1 = rotate_point_cw(*p1, rotation, size)
    turned = rotate_direction_cw((p1[0] - p0[0], p1[1] - p0[1]), rotation)
    assert turned == pytest.approx((q1[0] - q0[0], q1[1] - q0[1]))


def test_rotate_annotation_turns_the_direction_it_carries() -> None:
    ann = {"streets": [_street(orientation="diagonal", direction=[1.0, 0.0])]}
    out = rotate_annotation(ann, 90, (1000.0, 2000.0))
    assert out["streets"][0]["direction"] == [0.0, 1.0]
    assert out["streets"][0]["orientation"] == "diagonal"  # no cardinal flip applies


def test_rotate_annotation_drops_a_vector_it_cannot_turn() -> None:
    """A stale vector is a confidently wrong axis; the matcher degrades without one."""
    ann = {"streets": [_street(orientation="diagonal", direction="north")]}
    out = rotate_annotation(ann, 270, (1000.0, 2000.0))
    assert "direction" not in out["streets"][0]


# --- the prompt -----------------------------------------------------------


def test_the_diagonal_prompt_differs_from_the_frozen_one_in_exactly_two_places() -> None:
    """A ``.replace`` that matches nothing would leave the two prompts identical.

    That is the null result this experiment exists to avoid: an arm that asks
    for a diagonal and a control that does not have to actually differ.
    """
    assert DIAGONAL_PROMPT_TEMPLATE != EXTENDED_PROMPT_TEMPLATE
    assert "diagonal" not in EXTENDED_PROMPT_TEMPLATE
    assert '"horizontal|vertical|diagonal"' in DIAGONAL_PROMPT_TEMPLATE
    assert '"direction": [dx, dy]' in DIAGONAL_PROMPT_TEMPLATE
    assert "20 degrees off" in DIAGONAL_PROMPT_TEMPLATE
    # everything the frozen prompt says about the other channels survives
    for clause in ("ADDRESS NUMERALS:", "RAIL:", "PARKS:", "margin_numbers"):
        assert clause in DIAGONAL_PROMPT_TEMPLATE


def test_prompt_selection_is_by_name_and_an_unknown_name_is_refused() -> None:
    assert prompt_template(None) is EXTENDED_PROMPT_TEMPLATE
    assert prompt_template("diagonal") is DIAGONAL_PROMPT_TEMPLATE
    with pytest.raises(AnnotateError, match="unknown annotation prompt"):
        prompt_template("diagonals")


def test_the_frozen_prompt_has_no_name_so_it_cannot_key_a_second_cache() -> None:
    """Identity is keyed on the prompt NAME, not its text.

    A second spelling for the frozen prompt would therefore miss every read
    already on disk and re-buy the volume, while returning the same answers.
    """
    with pytest.raises(AnnotateError, match="unknown annotation prompt"):
        prompt_template("v2")


# --- the cache key --------------------------------------------------------


def test_the_prompt_is_part_of_the_cache_identity() -> None:
    """Without this, a second arm replays the first arm's reads for free.

    The ``v2`` in the key is the cache ENCODING version, not a prompt version,
    so two prompts at one model and effort collided on one path.
    """
    plain = model_cache_key("codex:gpt-5.6-terra", "high")
    diagonal = model_cache_key("codex:gpt-5.6-terra", "high", "diagonal")
    assert plain != diagonal
    assert model_from_cache_key(diagonal) == "codex:gpt-5.6-terra"


def test_omitting_the_prompt_preserves_every_existing_cache_path() -> None:
    assert model_cache_key("codex:gpt-5.6-terra", "high") == (
        "v2-WyJjb2RleDpncHQtNS42LXRlcnJhIiwiaGlnaCJd"
    )
    assert model_cache_key("codex:gpt-5.6-terra") == "codex:gpt-5.6-terra"


def test_a_prompt_key_without_a_variant_still_decodes_to_its_model() -> None:
    """A null variant beside a prompt must not leak the raw key as a model name.

    Callers use this to attribute a read to a voice, and two voices that are
    really one model must never look distinct.
    """
    key = model_cache_key("codex:gpt-5.6-terra", None, "diagonal")
    assert key.startswith("v2-")
    assert model_from_cache_key(key) == "codex:gpt-5.6-terra"
