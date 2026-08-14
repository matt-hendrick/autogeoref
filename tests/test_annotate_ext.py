"""v2 extended-annotation parser: superset schema, tolerant extras, strict core."""

import json

import pytest

from autogeoref.annotate.failures import (
    AnnotationCallError,
    BudgetLimitError,
    EmptyResponseError,
    MalformedResponseError,
)
from autogeoref.annotate.schema import parse_extended_response

V2 = {
    "streets": [{"name": "W. MADISON", "bbox": [705, 213, 935, 253], "orientation": "horizontal"}],
    "page_number_seen": "1",
    "address_numerals": [
        {"value": 2301, "bbox": [700, 300, 720, 310], "street": "W. MADISON"},
        {"value": 2303, "bbox": [730, 300, 750, 310], "street": None},
        {"value": "not-a-number", "bbox": [1, 1, 2, 2], "street": "X"},  # dropped
    ],
    "margin_numbers": [
        {"side": "top", "text": "114"},
        {"side": "middle", "text": "?"},  # dropped: bad side
    ],
    "rail_labels": [{"name": "C.M.&ST.P.R.R.", "bbox": [10, 10, 90, 20]}],
    "park_labels": [],
}


def test_full_v2_response_parses() -> None:
    ext = parse_extended_response(json.dumps(V2))
    assert len(ext.annotation.streets) == 1
    assert ext.annotation.page_number_seen == "1"
    assert [n.value for n in ext.address_numerals] == [2301, 2303]
    assert ext.address_numerals[0].street_hint == "W. MADISON"
    assert ext.address_numerals[1].street_hint is None
    assert len(ext.margin_readings) == 1 and ext.margin_readings[0].side == "top"
    assert ext.rail_labels[0][0] == "C.M.&ST.P.R.R."
    assert ext.park_labels == ()


def test_v1_only_response_parses_with_empty_extras() -> None:
    v1 = {"streets": V2["streets"], "page_number_seen": None}
    ext = parse_extended_response(json.dumps(v1))
    assert ext.address_numerals == ()
    assert ext.margin_readings == ()


def test_code_fences_stripped() -> None:
    ext = parse_extended_response("```json\n" + json.dumps(V2) + "\n```")
    # the whole payload, not just the street count: a strip that eats one
    # character too many either side still leaves exactly one street
    assert ext.annotation.to_dict() == {
        "streets": V2["streets"],
        "page_number_seen": V2["page_number_seen"],
    }


def test_empty_and_malformed() -> None:
    with pytest.raises(EmptyResponseError):
        parse_extended_response("   ")
    with pytest.raises(MalformedResponseError):
        parse_extended_response("this is not json")
    with pytest.raises(MalformedResponseError):
        parse_extended_response("Sure! Here are the streets I can see on the map.")


@pytest.mark.parametrize(
    "streets",
    [
        "not a list",
        [{"name": 5, "bbox": [1, 2, 3, 4], "orientation": "horizontal"}],
        [{"name": "X", "bbox": [1, 2, 3], "orientation": "horizontal"}],
        # "diagonal" IS readable now (tests/test_diagonal_labels.py); a fourth
        # orientation is still not.
        [{"name": "X", "bbox": [1, 2, 3, 4], "orientation": "slanted"}],
    ],
)
def test_core_schema_rejects_the_read_even_when_the_extras_are_fine(streets: object) -> None:
    with pytest.raises(MalformedResponseError):
        parse_extended_response(json.dumps(dict(V2, streets=streets)))


def test_zero_streets_is_a_legitimate_result_not_a_failure() -> None:
    ext = parse_extended_response('{"streets": [], "page_number_seen": null}')
    assert ext.annotation.streets == ()
    assert ext.annotation.page_number_seen is None


def test_core_schema_violation_is_a_call_error_not_a_crash() -> None:
    """A schema-violating read must be RETRYABLE, not fatal to the calling stage.

    One escalation read came back with a street whose ``orientation`` was ``null``. The core
    parser raised a bare ``ValueError``, which is not in the ``AnnotationCallError`` taxonomy
    that :func:`annotate_with_retry` catches — so it escaped the retry policy, failed the
    escalate stage, and took rescue/seam/corroborate/verified-accept down with it, after ~20
    model calls had been paid for. The v1 parser had always classified this correctly; v2 did
    not. A malformed model response is a failed READ (retry, then a `.failed.json` marker, then
    carry on) — never a crashed RUN.
    """
    null_orientation = dict(
        V2, streets=[{"name": "W. MADISON", "bbox": [1, 2, 3, 4], "orientation": None}]
    )
    with pytest.raises(MalformedResponseError) as exc:
        parse_extended_response(json.dumps(null_orientation))
    # the taxonomy is the point: this is what the shared retry policy catches
    assert isinstance(exc.value, AnnotationCallError)
    assert not isinstance(exc.value, BudgetLimitError)  # not terminal for the stage
