"""The compose/invert math, and the pixel frames it is expressed in.

An op log composed onto a base affine has to agree pointwise with applying the
ops one at a time, and inverting the chain has to land back on the original
corners — this is where the bugs would live. The frame tests pin the other half:
an upright small image's corners in full-resolution source pixels, for a scan
that was rotated on the way in.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from autogeoref.affine import (
    TO_3857,
    apply_affine,
    invert_affine,
)
from autogeoref.frames import rotate_point_cw, small_corners_full_px
from autogeoref.review.materialize import (
    apply_ops_point,
    compose_ops,
    corners_4326,
    seed_affine,
)
from review_support import (
    BASE,
    FULL_SIZE,
    ops_translate,
)


@pytest.mark.parametrize(
    ("modname", "names"),
    [
        (
            "autogeoref.review.app",
            (
                "affine_from_record",
                "compose_ops",
                "corners_4326",
                "displayable_affine",
                "dryrun_against_region",
                "final_gcps_geojson",
                "mask_px_from_ring_4326",
                "seed_affine",
            ),
        ),
        (
            "autogeoref.review.apply",
            (
                "affine_from_record",
                "dryrun_against_region",
                "final_gcps_geojson",
                "mask_ring_4326",
            ),
        ),
        ("autogeoref.bake.masks", ("mask_ring_4326",)),
    ],
)
def test_consumers_bind_the_authoritative_materializers(
    modname: str, names: tuple[str, ...]
) -> None:
    """Save, apply, and bake call the same materialization functions, through
    their own module globals — which is also where these tests monkeypatch
    them, so this doubles as a map of the patch targets."""
    import importlib

    materialize = importlib.import_module("autogeoref.review.materialize")
    mod = importlib.import_module(modname)
    for name in names:
        assert getattr(mod, name) is getattr(materialize, name)


def test_translate_moves_every_corner_exactly() -> None:
    m2 = compose_ops(BASE, ops_translate(120.0, -35.0))
    for px, py in ((0, 0), (FULL_SIZE[0], 0), FULL_SIZE, (0, FULL_SIZE[1])):
        x0, y0 = apply_affine(BASE, px, py)
        x1, y1 = apply_affine(m2, px, py)
        assert (x1 - x0, y1 - y0) == pytest.approx((120.0, -35.0), abs=1e-9)


def test_scale_about_center_pins_center_and_scales_distances() -> None:
    cx, cy = apply_affine(BASE, FULL_SIZE[0] / 2, FULL_SIZE[1] / 2)
    op = {"type": "scale", "factor": 1.25, "center_3857": [cx, cy]}
    m2 = compose_ops(BASE, [op])
    assert apply_affine(m2, FULL_SIZE[0] / 2, FULL_SIZE[1] / 2) == pytest.approx((cx, cy))
    x1, y1 = apply_affine(m2, 0, 0)
    x0, y0 = apply_affine(BASE, 0, 0)
    assert math.hypot(x1 - cx, y1 - cy) == pytest.approx(
        1.25 * math.hypot(x0 - cx, y0 - cy), rel=1e-12
    )


def test_rotate_about_center_preserves_distances() -> None:
    cx, cy = apply_affine(BASE, 100, 100)
    op = {"type": "rotate", "deg": 3.5, "center_3857": [cx, cy]}
    m2 = compose_ops(BASE, [op])
    for px, py in ((0, 0), FULL_SIZE):
        x0, y0 = apply_affine(BASE, px, py)
        x1, y1 = apply_affine(m2, px, py)
        assert math.hypot(x1 - cx, y1 - cy) == pytest.approx(math.hypot(x0 - cx, y0 - cy), rel=1e-9)


def test_op_chain_then_inverse_ops_is_identity() -> None:
    cx, cy = apply_affine(BASE, 2000, 3000)
    chain = [
        {"type": "translate", "dx_m": 55.0, "dy_m": -12.0},
        {"type": "scale", "factor": 1.1, "center_3857": [cx, cy]},
        {"type": "rotate", "deg": 2.0, "center_3857": [cx, cy]},
    ]
    inverse = [
        {"type": "rotate", "deg": -2.0, "center_3857": [cx, cy]},
        {"type": "scale", "factor": 1 / 1.1, "center_3857": [cx, cy]},
        {"type": "translate", "dx_m": -55.0, "dy_m": 12.0},
    ]
    m2 = compose_ops(compose_ops(BASE, chain), inverse)
    assert np.allclose(m2, BASE, atol=1e-6)


def test_invert_affine_round_trips_points() -> None:
    inv = invert_affine(BASE)
    for px, py in ((0.0, 0.0), (123.4, 567.8), FULL_SIZE):
        x, y = apply_affine(BASE, px, py)
        assert apply_affine(inv, x, y) == pytest.approx((px, py), abs=1e-6)


def test_compose_ops_matches_pointwise_transform() -> None:
    """The affine composition and the per-point op chain are THE SAME map —
    this is what makes fit_affine(moved GCPs) reproduce the composed affine."""
    cx, cy = apply_affine(BASE, 3000, 4000)
    chain = [
        {"type": "rotate", "deg": -7.0, "center_3857": [cx, cy]},
        {"type": "translate", "dx_m": -300.0, "dy_m": 80.0},
        {"type": "scale", "factor": 0.93, "center_3857": [cx - 50, cy + 20]},
    ]
    m2 = compose_ops(BASE, chain)
    for px, py in ((0, 0), (5860, 0), (0, 8505), (2222, 3333)):
        x, y = apply_affine(BASE, px, py)
        assert apply_affine(m2, px, py) == pytest.approx(apply_ops_point(x, y, chain), abs=1e-6)


def test_seed_affine_center_scale_rotation() -> None:
    m = seed_affine(0.6, 5.0, (-9756000.0, 5139000.0), FULL_SIZE)
    assert apply_affine(m, FULL_SIZE[0] / 2, FULL_SIZE[1] / 2) == pytest.approx(
        (-9756000.0, 5139000.0)
    )
    from autogeoref.affine import model_rotation_deg, model_scales

    sx, sy = model_scales(np.asarray(m))
    assert (sx, sy) == pytest.approx((0.6, 0.6))
    assert model_rotation_deg(np.asarray(m)) == pytest.approx(5.0)


def test_rotate_point_cw_matches_orient_corner_math() -> None:
    # the documented forms from orient.rotate_annotation, at the corners
    w, h = 100.0, 200.0
    assert rotate_point_cw(0, 0, 90, (w, h)) == (h, 0)
    assert rotate_point_cw(w, h, 90, (w, h)) == (0, w)
    assert rotate_point_cw(0, 0, 180, (w, h)) == (w, h)
    assert rotate_point_cw(0, 0, 270, (w, h)) == (0, w)


@pytest.mark.parametrize("rot", [0, 90, 180, 270])
def test_rotation_round_trip_is_identity(rot: int) -> None:
    w, h = 137.0, 200.0
    size_rotated = (h, w) if rot in (90, 270) else (w, h)
    for pt in ((0.0, 0.0), (w, 0.0), (w, h), (13.5, 77.0)):
        once = rotate_point_cw(*pt, rot, (w, h))
        back = rotate_point_cw(*once, (360 - rot) % 360, size_rotated)
        assert back == pytest.approx(pt)


def test_small_corners_no_rotation_is_pure_scale() -> None:
    info = {"small_size": [1376, 2000], "scale": 0.25, "full_size": [5504, 8000]}
    corners = small_corners_full_px(info)
    assert corners[0] == pytest.approx((0, 0))
    assert corners[2] == pytest.approx((5504, 8000))


def test_small_corners_180_flips_frame() -> None:
    # a 180-rotated scan: the upright small's top-left is the SOURCE's
    # bottom-right (this is p0 of the _041 golden volume)
    info = {"small_size": [1378, 2000], "scale": 0.235, "rotation_applied": 180}
    corners = small_corners_full_px(info)
    w_full, h_full = 1378 / 0.235, 2000 / 0.235
    assert corners[0] == pytest.approx((w_full, h_full))
    assert corners[2] == pytest.approx((0.0, 0.0))


def test_corners_4326_round_trip_through_3857() -> None:
    pts = corners_4326(BASE, [(0.0, 0.0), FULL_SIZE])
    for (lng, lat), (px, py) in zip(pts, [(0.0, 0.0), FULL_SIZE], strict=True):
        x, y = TO_3857.transform(lng, lat)
        assert (x, y) == pytest.approx(apply_affine(BASE, px, py), abs=1e-3)
