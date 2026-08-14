"""Dependency-free quarter-turn pixel-frame transforms.

Coordinates are continuous pixel edges: a frame of ``(width, height)`` has
corners at ``(0, 0)`` and ``(width, height)``. No integer pixel-center offset
is applied.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any


def rotate_point_cw(x: float, y: float, deg: int, size: tuple[float, float]) -> tuple[float, float]:
    """Map a point through a clockwise quarter-turn of its image.

    ``size`` is the pre-rotation frame; 90 and 270 degree turns swap the
    output dimensions.
    """
    w, h = size
    if deg == 0:
        return x, y
    if deg == 90:
        return h - y, x
    if deg == 180:
        return w - x, h - y
    if deg == 270:
        return y, w - x
    raise ValueError(f"rotation must be one of 0/90/180/270, got {deg}")


def rotate_bbox(
    bbox: Sequence[float], rotation_deg: int, size: tuple[float, float]
) -> tuple[float, float, float, float]:
    """Map a bbox into its image rotated CLOCKWISE.

    The four continuous pixel-edge corners are transformed with
    :func:`rotate_point_cw`, then normalized back into bbox order.
    """
    if rotation_deg not in (0, 90, 180, 270):
        raise ValueError(f"rotation_deg must be one of 0/90/180/270, got {rotation_deg}")
    x0, y0, x1, y1 = (float(v) for v in bbox)
    corners = (
        rotate_point_cw(x0, y0, rotation_deg, size),
        rotate_point_cw(x1, y0, rotation_deg, size),
        rotate_point_cw(x1, y1, rotation_deg, size),
        rotate_point_cw(x0, y1, rotation_deg, size),
    )
    xs, ys = zip(*corners, strict=True)
    return min(xs), min(ys), max(xs), max(ys)


def small_corners_full_px(info: Mapping[str, Any]) -> list[tuple[float, float]]:
    """Full-res source-frame pixels of an on-disk small's corners.

    The on-disk small is upright. Its corners turn back into the source frame
    before the manifest scale converts them to full-resolution pixels.
    """
    w, h = (float(v) for v in info["small_size"])
    rot_back = (360 - int(info.get("rotation_applied", 0))) % 360
    scale = float(info["scale"])
    out: list[tuple[float, float]] = []
    for u, v in ((0.0, 0.0), (w, 0.0), (w, h), (0.0, h)):
        x, y = rotate_point_cw(u, v, rot_back, (w, h))
        out.append((x / scale, y / scale))
    return out


def full_px_to_small(x: float, y: float, info: Mapping[str, Any]) -> tuple[float, float]:
    """Upright-small pixel of a full-res source-frame pixel.

    Inverse of the corner mapping in :func:`small_corners_full_px`: the
    manifest scale takes source pixels into the source small, then the
    recorded quarter-turn brings them into the upright on-disk frame.
    """
    w, h = (float(v) for v in info["small_size"])
    rot = int(info.get("rotation_applied", 0))
    src = (h, w) if rot in (90, 270) else (w, h)
    scale = float(info["scale"])
    return rotate_point_cw(x * scale, y * scale, rot, src)


def rotate_direction_cw(direction: Sequence[float], deg: int) -> tuple[float, float]:
    """Map a pixel direction vector through a clockwise quarter-turn.

    The linear part of :func:`rotate_point_cw`, so it needs no frame size.
    """
    dx, dy = float(direction[0]), float(direction[1])
    if deg == 0:
        return dx, dy
    if deg == 90:
        return -dy, dx
    if deg == 180:
        return -dx, -dy
    if deg == 270:
        return dy, -dx
    raise ValueError(f"rotation must be one of 0/90/180/270, got {deg}")


def rotate_annotation(
    annotation: dict[str, Any], rotation_deg: int, small_size: tuple[float, float]
) -> dict[str, Any]:
    """Remap an annotation into its image rotated CLOCKWISE.

    Bbox-carrying streets, rail labels, and address numerals all turn. Street
    orientation flips for quarter turns that exchange its axes, and a label's
    direction vector turns with it.
    """
    if rotation_deg not in (0, 90, 180, 270):
        raise ValueError(f"rotation_deg must be one of 0/90/180/270, got {rotation_deg}")
    out = copy.deepcopy(annotation)
    if rotation_deg == 0:
        return out
    flip = {"horizontal": "vertical", "vertical": "horizontal"}
    for street in out.get("streets", []):
        street["bbox"] = list(rotate_bbox(street["bbox"], rotation_deg, small_size))
        direction = street.get("direction")
        if direction is not None:
            # an unreadable vector is DROPPED, never carried through unrotated:
            # a stale direction is a confidently wrong axis, and the matcher
            # already degrades a label that has none
            try:
                street["direction"] = list(rotate_direction_cw(direction, rotation_deg))
            except (TypeError, ValueError, IndexError, KeyError):
                del street["direction"]
        if rotation_deg in (90, 270):
            orientation = street.get("orientation")
            if orientation in flip:
                street["orientation"] = flip[orientation]
    for feature in (*(out.get("rail_labels") or ()), *(out.get("address_numerals") or ())):
        if isinstance(feature, dict) and "bbox" in feature:
            feature["bbox"] = list(rotate_bbox(feature["bbox"], rotation_deg, small_size))
    return out
