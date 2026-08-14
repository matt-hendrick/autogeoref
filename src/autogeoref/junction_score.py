"""Scoring primitives for the junction-snap verifier: rasters, blurs, correlation.

Everything here works in ground metres on a fixed :data:`GRID_RES_M` grid and
knows nothing about sheets, junctions or verdicts. The verifier itself, and the
constants tuned against it, live in :mod:`autogeoref.junction_snap`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .affine import AffineMatrix

FloatArray = NDArray[np.float64]

#: Scoring grid resolution.
GRID_RES_M = 4.0

_WGS84_RADIUS_M = 6378137.0


def mercator_lat_deg(y_3857: float) -> float:
    return math.degrees(math.atan(math.sinh(y_3857 / _WGS84_RADIUS_M)))


def apply_affine_pts(m: AffineMatrix, pts: FloatArray) -> FloatArray:
    """Vectorized ``[X, Y] = M @ [1, px, py]`` for an ``(N, 2)`` point array."""
    ones = np.column_stack([np.ones(len(pts)), pts])
    out: FloatArray = ones @ np.asarray(m, dtype=np.float64).T
    return out


def correlate_same(image: FloatArray, kernel: FloatArray) -> FloatArray:
    """Cross-correlation, 'same' output — equals fftconvolve(image, kernel[::-1,::-1])."""
    ih, iw = image.shape
    kh, kw = kernel.shape
    fh, fw = ih + kh - 1, iw + kw - 1
    spec = np.fft.rfft2(image, (fh, fw)) * np.fft.rfft2(kernel[::-1, ::-1], (fh, fw))
    full = np.fft.irfft2(spec, (fh, fw))
    r0, c0 = (kh - 1) // 2, (kw - 1) // 2
    out: FloatArray = full[r0 : r0 + ih, c0 : c0 + iw]
    return out


def sat_blur(img: FloatArray, sigma_px: float, probe: FloatArray) -> FloatArray:
    """Gaussian blur saturated at the single-feature peak.

    Without saturation, node clusters award wrong placements: the blurred
    raster must be clipped at what one isolated feature would score.
    """
    blur = cv2.GaussianBlur(img, (0, 0), sigma_px)
    peak = float(cv2.GaussianBlur(probe, (0, 0), sigma_px).max())
    out: FloatArray = np.minimum(blur, peak).astype(np.float64) / peak
    return out


def make_kernel(pts_m: FloatArray, weights: FloatArray, center_m: FloatArray) -> FloatArray:
    """Odd-sized correlation kernel of weighted points relative to a center."""
    d = pts_m - center_m
    r_ext = int(np.ceil(np.abs(d).max() / GRID_RES_M)) + 2
    kern = np.zeros((2 * r_ext + 1, 2 * r_ext + 1), dtype=np.float64)
    c = (d[:, 0] / GRID_RES_M + r_ext).astype(int).clip(0, 2 * r_ext)
    r = (-d[:, 1] / GRID_RES_M + r_ext).astype(int).clip(0, 2 * r_ext)
    np.add.at(kern, (r, c), weights)
    return kern


def corr_score(raster: FloatArray, kernel: FloatArray) -> FloatArray:
    return correlate_same(raster, kernel) / max(float(kernel.sum()), 1.0)


def impulse_probe() -> FloatArray:
    """One isolated feature — what :func:`sat_blur` saturates a node raster at."""
    probe = np.zeros((201, 201), dtype=np.float64)
    probe[100, 100] = 1.0
    return probe


def line_probe() -> FloatArray:
    """One isolated line — what :func:`sat_blur` saturates a geometry raster at."""
    probe = np.zeros((201, 201), dtype=np.float64)
    probe[:, 100] = 1.0
    return probe


@dataclass(frozen=True)
class ScoreWindow:
    """The ground-metre box one placement is scored over, and its grid shape."""

    minx: float
    maxx: float
    miny: float
    maxy: float
    nc: int
    nr: int


def score_window(q: FloatArray, skel_g: FloatArray, *, pad: float) -> ScoreWindow:
    """The padded bounding box of the sheet's evidence, in ground metres."""
    all_pts = np.vstack([q, skel_g]) if len(skel_g) else q
    minx = float(all_pts[:, 0].min()) - pad
    maxx = float(all_pts[:, 0].max()) + pad
    miny = float(all_pts[:, 1].min()) - pad
    maxy = float(all_pts[:, 1].max()) + pad
    return ScoreWindow(
        minx=minx,
        maxx=maxx,
        miny=miny,
        maxy=maxy,
        nc=math.ceil((maxx - minx) / GRID_RES_M),
        nr=math.ceil((maxy - miny) / GRID_RES_M),
    )


def node_raster(nodes_3857: FloatArray, k: float, window: ScoreWindow) -> FloatArray:
    """One cell set per centerline node falling inside ``window``."""
    raster = np.zeros((window.nr, window.nc), dtype=np.float64)
    sel = nodes_3857 * k
    inw = (
        (sel[:, 0] >= window.minx)
        & (sel[:, 0] <= window.maxx)
        & (sel[:, 1] >= window.miny)
        & (sel[:, 1] <= window.maxy)
    )
    sel = sel[inw]
    c = ((sel[:, 0] - window.minx) / GRID_RES_M).astype(int).clip(0, window.nc - 1)
    r = ((window.maxy - sel[:, 1]) / GRID_RES_M).astype(int).clip(0, window.nr - 1)
    raster[r, c] = 1.0
    return raster


def line_raster(
    polylines_3857: tuple[FloatArray, ...], k: float, window: ScoreWindow
) -> FloatArray:
    """Centerline geometry drawn one cell wide; polylines fully outside are skipped."""
    lr8 = np.zeros((window.nr, window.nc), dtype=np.uint8)
    for line in polylines_3857:
        lg = line * k
        if lg[:, 0].max() < window.minx or lg[:, 0].min() > window.maxx:
            continue
        if lg[:, 1].max() < window.miny or lg[:, 1].min() > window.maxy:
            continue
        px = ((lg[:, 0] - window.minx) / GRID_RES_M).astype(int)
        py = ((window.maxy - lg[:, 1]) / GRID_RES_M).astype(int)
        for i in range(len(px) - 1):
            cv2.line(lr8, (int(px[i]), int(py[i])), (int(px[i + 1]), int(py[i + 1])), 1, 1)
    out: FloatArray = lr8.astype(np.float64)
    return out
