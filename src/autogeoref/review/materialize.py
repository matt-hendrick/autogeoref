"""Pure review frame, GCP, and mask materialization.

This module intentionally has no HTTP dependency. Back-half stages import it
without loading the reviewer server.
"""

from __future__ import annotations

import copy
import logging
import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..affine import (
    TO_3857,
    TO_4326,
    AffineMatrix,
    apply_affine,
    fit_affine_checked,
    gcps_from_geojson,
    invert_affine,
)
from .ops import op_linear_offset

if TYPE_CHECKING:
    from .sidecars import ReviewSidecar

#: Smallest linear scale (m/px) a recorded affine may have and still count as a placement. A
#: real sheet sits around 0.1-10 m/px; a rescue whose anchors all share one street has spread
#: PIXELS but coincident WORLD points, so the fit passes fit_affine_checked's pixel-rank test
#: yet is near-singular (~1e-12 linear part): every corner collapses onto one point, MapLibre's
#: tile math throws, and materializing ops against it would reproduce the garbage base instead
#: of the seeded placement the reviewer actually saw.
MIN_PLACEMENT_SCALE_M_PER_PX = 1e-3


def displayable_affine(m: AffineMatrix | None) -> AffineMatrix | None:
    """``m`` when its linear part is non-degenerate at sheet scale, else None."""
    if m is None:
        return None
    det = float(m[0][1] * m[1][2] - m[0][2] * m[1][1])
    if math.sqrt(abs(det)) < MIN_PLACEMENT_SCALE_M_PER_PX:
        return None
    return m


def affine_from_record(record: Mapping[str, Any]) -> AffineMatrix | None:
    """The recorded placement's affine (None without >=3 non-degenerate GCPs)."""
    feats = (record.get("gcps_geojson") or {}).get("features") or []
    if len(feats) < 3:
        return None
    return fit_affine_checked(gcps_from_geojson(dict(record["gcps_geojson"])))


def compose_ops(m: AffineMatrix, ops: Sequence[Mapping[str, Any]]) -> AffineMatrix:
    """Compose world-space ops (in order) onto a pixel->world affine."""
    out = np.asarray(m, dtype=np.float64)
    for op in ops:
        linear, offset = op_linear_offset(op)
        out = np.column_stack([linear @ out[:, 0] + offset, linear @ out[:, 1:3]])
    return np.ascontiguousarray(out)


def apply_ops_point(x: float, y: float, ops: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    """Run one 3857 point through the same op chain :func:`compose_ops` uses."""
    value = np.array([x, y], dtype=np.float64)
    for op in ops:
        linear, offset = op_linear_offset(op)
        value = linear @ value + offset
    return float(value[0]), float(value[1])


def corners_4326(m: AffineMatrix, corners_px: Sequence[tuple[float, float]]) -> list[list[float]]:
    """Pixel corners through affine + 3857->4326 (MapLibre image coordinates)."""
    out: list[list[float]] = []
    for px, py in corners_px:
        x, y = apply_affine(m, px, py)
        lng, lat = TO_4326.transform(x, y)
        out.append([lng, lat])
    return out


def seed_affine(
    scale_m_per_px: float,
    rotation_deg: float,
    center_3857: tuple[float, float],
    full_size: tuple[float, float],
) -> AffineMatrix:
    """A fresh placement at pinned constants, centered on a point.

    ``scale_m_per_px`` is EPSG:3857 meters per full-res pixel (the
    ``VolumeConfig.scale_m_per_px`` frame contract) — no cos(lat) correction
    happens here, so a true-ground-meter value renders the ghost undersized.
    """
    radians = math.radians(rotation_deg)
    linear = scale_m_per_px * np.array(
        [[math.cos(radians), math.sin(radians)], [math.sin(radians), -math.cos(radians)]]
    )
    constant = np.array(center_3857) - linear @ np.array([full_size[0] / 2, full_size[1] / 2])
    return np.ascontiguousarray(np.column_stack([constant, linear]))


def transformed_gcps_geojson(
    record: Mapping[str, Any], ops: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The recorded GCPs with world coordinates moved through the op chain."""
    collection: dict[str, Any] = copy.deepcopy(dict(record["gcps_geojson"]))
    for feature in collection.get("features") or []:
        lng, lat = feature["geometry"]["coordinates"]
        x, y = TO_3857.transform(lng, lat)
        nx, ny = apply_ops_point(x, y, ops)
        feature["geometry"]["coordinates"] = list(TO_4326.transform(nx, ny))
    return collection


def synthetic_gcps_geojson(m: AffineMatrix, full_size: tuple[float, float]) -> dict[str, Any]:
    """Corner GCPs serializing a reviewer placement with no recorded GCPs."""
    width, height = full_size
    features = []
    for fx, fy in ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)):
        px, py = width * fx, height * fy
        x, y = apply_affine(m, px, py)
        lng, lat = TO_4326.transform(x, y)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "image": [round(px), round(py)],
                    "username": "reviewer",
                    "note": "synthetic: reviewer placement corner",
                },
                "geometry": {"type": "Point", "coordinates": [lng, lat]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def final_gcps_geojson(
    record: Mapping[str, Any], side: ReviewSidecar, full_size: tuple[float, float]
) -> dict[str, Any] | None:
    """The GCP set an accept/adjusted sidecar materializes to, or None.

    A record whose GCP fit is degenerate (:func:`displayable_affine`) counts
    as UNPLACED here: the UI seeded it, the reviewer's ops composed onto the
    seed, and moving the recorded GCPs through those ops would materialize
    ``ops @ garbage`` — not the placement the reviewer saw. Synthetic corners
    of the sidecar's affine ARE that placement.
    """
    has_recorded = displayable_affine(affine_from_record(record)) is not None
    if side.ops and has_recorded:
        return transformed_gcps_geojson(record, side.ops)
    if side.ops or not has_recorded:
        if side.affine is None:
            return None
        return synthetic_gcps_geojson(np.asarray(side.affine), full_size)
    return copy.deepcopy(dict(record["gcps_geojson"]))


def mask_ring_4326(mask_px: Sequence[Sequence[float]], m: AffineMatrix) -> list[list[float]]:
    """Mask ring, full-res px -> 4326 (closed: first vertex repeated last)."""
    ring = []
    for px, py in mask_px:
        x, y = apply_affine(m, float(px), float(py))
        lng, lat = TO_4326.transform(x, y)
        ring.append([lng, lat])
    ring.append(list(ring[0]))
    return ring


def mask_px_from_ring_4326(ring: Sequence[Sequence[float]], m: AffineMatrix) -> list[list[float]]:
    """4326 exterior ring -> full-res pixel vertices (open ring) via ``m``."""
    inverse = invert_affine(m)
    points = list(ring)
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    out = []
    for lng, lat in points:
        x, y = TO_3857.transform(lng, lat)
        px, py = apply_affine(inverse, x, y)
        out.append([round(px, 1), round(py, 1)])
    return out


class _CaptureHandler(logging.Handler):
    """Collect the warp module's dry-run rejection detail for review callers."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def cutline_dryrun_checked(
    src: Path, mask_px: Sequence[Sequence[float]], m: AffineMatrix, *, timeout_s: float = 120.0
) -> tuple[bool, str]:
    """Run the accepted cutline test; return ``(ok, gdal_detail_on_failure)``."""
    from shapely.geometry import Polygon

    from ..warp import gdalwarp_cutline_dryrun

    polygon = Polygon([(point[0], point[1]) for point in mask_ring_4326(mask_px, m)])
    warp_logger = logging.getLogger("autogeoref.warp")
    capture = _CaptureHandler()
    warp_logger.addHandler(capture)
    old_level = warp_logger.level
    warp_logger.setLevel(logging.DEBUG)
    try:
        ok = gdalwarp_cutline_dryrun(src, polygon, timeout_s=timeout_s, crs_epsg=4326)
    finally:
        warp_logger.setLevel(old_level)
        warp_logger.removeHandler(capture)
    detail = "" if ok else "\n".join(capture.lines) or "gdalwarp rejected the cutline"
    return ok, detail


def _gcps4326_from_fc(collection: Mapping[str, Any]) -> list[tuple[float, float, float, float]]:
    from ..warp import gcps_from_feature_collection

    return gcps_from_feature_collection(dict(collection))


def dryrun_against_region(
    region_image: Path,
    collection: Mapping[str, Any],
    mask_px: Sequence[Sequence[float]],
    m: AffineMatrix,
    *,
    timeout_s: float = 120.0,
) -> tuple[bool, str]:
    """Dry-run an edited mask against a GCP-attached VRT of the real scan."""
    from ..warp import attach_gcps_vrt

    with tempfile.TemporaryDirectory(prefix="review-dryrun-") as tmp:
        vrt = attach_gcps_vrt(
            region_image, _gcps4326_from_fc(collection), Path(tmp) / "gcps.vrt", timeout_s=timeout_s
        )
        return cutline_dryrun_checked(vrt, mask_px, m, timeout_s=timeout_s)
