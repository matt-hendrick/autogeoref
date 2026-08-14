"""Pure quarter-turn frame transforms stay independent of optional CV code."""

from __future__ import annotations

import subprocess
import sys

import pytest

from autogeoref.frames import (
    full_px_to_small,
    rotate_bbox,
    rotate_point_cw,
    small_corners_full_px,
)


def test_frames_import_does_not_load_cv2() -> None:
    code = "import sys; import autogeoref.frames; assert 'cv2' not in sys.modules"
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
    )


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_rotate_bbox_is_defined_by_rotated_corners(rotation: int) -> None:
    bbox = (10.5, 20.25, 30.75, 60.5)
    size = (100.0, 200.0)
    x0, y0, x1, y1 = bbox
    corners = [
        rotate_point_cw(x, y, rotation, size) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    ]
    xs, ys = zip(*corners, strict=True)
    assert rotate_bbox(bbox, rotation, size) == (min(xs), min(ys), max(xs), max(ys))


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_full_px_to_small_inverts_small_corners(rotation: int) -> None:
    w, h = 1376.0, 2000.0
    info = {"small_size": [w, h], "scale": 0.25, "rotation_applied": rotation}
    upright = ((0.0, 0.0), (w, 0.0), (w, h), (0.0, h))
    for (fx, fy), (ux, uy) in zip(small_corners_full_px(info), upright, strict=True):
        assert full_px_to_small(fx, fy, info) == pytest.approx((ux, uy))


def test_full_px_to_small_no_rotation_is_pure_scale() -> None:
    info = {"small_size": [1376, 2000], "scale": 0.25}
    assert full_px_to_small(400.0, 100.0, info) == pytest.approx((100.0, 25.0))


def test_full_px_to_small_90_turns_scaled_point() -> None:
    # upright small 2000x1376 came from a 1376x2000 source small rotated CW
    info = {"small_size": [2000, 1376], "scale": 0.25, "rotation_applied": 90}
    assert full_px_to_small(0.0, 0.0, info) == pytest.approx((2000.0, 0.0))
    assert full_px_to_small(400.0, 100.0, info) == pytest.approx((1975.0, 100.0))
