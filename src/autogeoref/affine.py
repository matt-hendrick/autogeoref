"""Affine (GDAL 'poly1') fitting and scoring.

A GDAL ``poly1`` transform is an affine least-squares fit::

    X = a0 + a1*px + a2*py
    Y = b0 + b1*px + b2*py

with ``(px, py)`` image pixels (y positive-down) and ``(X, Y)`` EPSG:3857
meters. The matrix convention throughout is a 2x3 ``M`` such that
``[X, Y] = M @ [1, px, py]``.

Distances are 3857 planar meters; ``grid_rmse_m`` optionally applies the
``cos(lat)`` Mercator correction to report true ground meters.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pyproj import Transformer

TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
TO_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

#: (px, py, X3857, Y3857)
Gcp = tuple[float, float, float, float]
AffineMatrix = NDArray[np.float64]


def gcps_from_geojson(gcps_geojson: dict[str, Any]) -> list[Gcp]:
    """``[(px, py, X3857, Y3857), ...]`` from a session-style GCP FeatureCollection."""
    out: list[Gcp] = []
    for f in gcps_geojson["features"]:
        px, py = f["properties"]["image"]
        lng, lat = f["geometry"]["coordinates"]
        x, y = TO_3857.transform(lng, lat)
        out.append((float(px), float(py), x, y))
    return out


def fit_affine(gcps: Sequence[Gcp]) -> AffineMatrix:
    """Least-squares affine fit; needs >= 3 GCPs.

    Raises ``ValueError`` on fewer than 3 points. Collinear points do not
    raise here (lstsq is minimum-norm); callers that must reject degeneracy
    check rank explicitly (see :func:`fit_affine_checked`).
    """
    if len(gcps) < 3:
        raise ValueError("need >= 3 GCPs for affine fit")
    a = np.array([[1.0, px, py] for px, py, _, _ in gcps])
    b = np.array([[x, y] for _, _, x, y in gcps])
    # one lstsq with a 2-column RHS, NOT np.linalg.solve or two per-axis
    # calls: byte-identical results to the historical per-axis lstsq were
    # validated over the full recorded corpus (a different solver would
    # break the byte-identical-replay contract the golden tests enforce)
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    return np.ascontiguousarray(coef.T, dtype=np.float64)


def fit_affine_checked(gcps: Sequence[Gcp]) -> AffineMatrix | None:
    """Like :func:`fit_affine` but returns ``None`` for degenerate (rank<3) sets."""
    if len(gcps) < 3:
        return None
    p = np.array([[1.0, px, py] for px, py, _, _ in gcps])
    if np.linalg.matrix_rank(p) < 3:
        return None
    return fit_affine(gcps)


def apply_affine(m: AffineMatrix, px: float, py: float) -> tuple[float, float]:
    v = np.array([1.0, px, py])
    return float(m[0] @ v), float(m[1] @ v)


def invert_affine(m: AffineMatrix) -> AffineMatrix:
    """World->pixel inverse, same 2x3 convention: ``[px, py] = Minv @ [1, X, Y]``.

    Raises ``numpy.linalg.LinAlgError`` for a degenerate (rank<2) linear part.
    """
    arr = np.asarray(m, dtype=np.float64)
    inv = np.linalg.inv(arr[:, 1:3])
    return np.ascontiguousarray(np.column_stack([-inv @ arr[:, 0], inv]))


def model_scales(m: AffineMatrix) -> tuple[float, float]:
    """Meters-per-pixel along the pixel x and y axes."""
    return math.hypot(m[0][1], m[1][1]), math.hypot(m[0][2], m[1][2])


def model_rotation_deg(m: AffineMatrix) -> float:
    """Angle of the pixel x-axis in world space; ~0 for a north-up sheet."""
    return math.degrees(math.atan2(m[1][1], m[0][1]))


def model_determinant(m: AffineMatrix) -> float:
    """Determinant of the 2x2 linear part — NEGATIVE for an upright placement.

    Pixel y grows DOWNWARD and EPSG:3857 y grows upward, so any rotation of a scanned page onto
    the ground reverses handedness: the determinant is negative at every quadrant. A POSITIVE
    determinant is a reflection — the sheet warped back-to-front, its labels reading backwards,
    and ``gdalwarp`` filling the reflected frame with opaque black the mosaic then paints. A
    geometric fact about a scan, not a quality bar, so there is no tolerance to tune. A
    handedness-blind angle test cannot see it, which is how a mirrored model once reached the
    mosaic: a reflection reports near 180 deg, and quadrant folding takes that to 0.
    """
    return float(m[0][1] * m[1][2] - m[0][2] * m[1][1])


def residuals_m(m: AffineMatrix, gcps: Sequence[Gcp]) -> list[float]:
    """Per-GCP residual distances in meters (3857 planar)."""
    out = []
    for px, py, x, y in gcps:
        xp, yp = apply_affine(m, px, py)
        out.append(math.hypot(xp - x, yp - y))
    return out


def grid_rmse_m(
    m_a: AffineMatrix,
    m_b: AffineMatrix,
    width: float,
    height: float,
    n: int = 5,
    mercator_correction_lat: float | None = None,
) -> float:
    """RMSE (meters) of displacement between two affines over an n x n pixel grid.

    With ``mercator_correction_lat``, 3857 meters are scaled by ``cos(lat)``
    to true ground meters.
    """
    ds = []
    for i in range(n):
        for j in range(n):
            px = width * (i + 0.5) / n
            py = height * (j + 0.5) / n
            xa, ya = apply_affine(m_a, px, py)
            xb, yb = apply_affine(m_b, px, py)
            ds.append(math.hypot(xa - xb, ya - yb))
    rmse = math.sqrt(sum(d * d for d in ds) / len(ds))
    if mercator_correction_lat is not None:
        rmse *= math.cos(math.radians(mercator_correction_lat))
    return rmse
