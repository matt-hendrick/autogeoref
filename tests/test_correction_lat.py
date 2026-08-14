"""The cos(lat) correction latitude is DERIVED from volume bounds, never
defaulted: a wrong or defaulted latitude silently rescales every reported
ground-meter and the 15 m commit gate with it (the retired ``CHICAGO_LAT``
constant did exactly that for any non-Chicago city).

It is a SCORING input and only that. ``match_sheet`` no longer takes one, and
:func:`test_matching_never_sees_a_latitude` is what keeps that true.
"""

import inspect
import math
from typing import Any

import pytest

from autogeoref.bounds import BoundsError, mercator_correction_lat
from autogeoref.scoring import score_record_vs_ground_truth
from autogeoref.volume import match_sheet


def _layer(extent: list[float]) -> dict[str, Any]:
    return {"extent": extent, "gcps_geojson": {"features": []}}


def test_correction_lat_is_the_extent_union_midpoint() -> None:
    gt: dict[str, dict[str, Any]] = {
        "1": _layer([-87.60, 41.70, -87.59, 41.72]),
        "2": _layer([-87.58, 41.80, -87.57, 41.90]),
    }
    assert mercator_correction_lat(gt) == pytest.approx((41.70 + 41.90) / 2)


def test_ground_truth_without_extents_is_an_explicit_error() -> None:
    gt: dict[str, dict[str, Any]] = {"1": {"gcps_geojson": {"features": []}}}
    with pytest.raises(BoundsError):
        mercator_correction_lat(gt)


def test_matching_never_sees_a_latitude() -> None:
    """The correction latitude is derived from human pins, so a placement path that
    could take one is a placement path pins can reach. ``match_sheet`` cannot."""
    params = inspect.signature(match_sheet).parameters
    assert "correction_lat" not in params
    assert "gt_layer" not in params


def test_score_scales_by_cos_of_the_given_latitude() -> None:
    def gcps(dx: float) -> dict[str, Any]:
        return {
            "features": [
                {"properties": {"image": [px, py]}, "geometry": {"coordinates": [x + dx, y]}}
                for px, py, x, y in [(0, 0, 0, 0), (100, 0, 1e-3, 0), (0, 100, 0, 1e-3)]
            ]
        }

    record = {"gcps_geojson": gcps(0.0)}
    layer = {"gcps_geojson": gcps(1e-4)}
    info = {"full_size": [100, 100]}
    at_equator = score_record_vs_ground_truth(record, info, layer, 0.0)
    at_60 = score_record_vs_ground_truth(record, info, layer, 60.0)
    assert at_equator is not None and at_equator > 0
    assert at_60 == pytest.approx(at_equator * math.cos(math.radians(60.0)))
