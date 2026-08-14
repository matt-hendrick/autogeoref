"""Quarter-turn orientation normalization.

Page orientation cannot be inferred from aspect ratio or text direction; the
compass rose is the reliable marker. Detection scores three parts in turn — the
starburst hub (the only structure with strong edge energy in all four
orientation bins at once), a near-continuous needle out from it, and a compact
fleur-de-lis blob along the needle axis — and north is the direction passing
needle and fleur with a 2x margin, in the long-edge-2000 frame.

Convention: :func:`detect_quarter_turn_image` returns ``k`` in {0, 90, 180, 270}, the
degrees the image must be rotated CLOCKWISE to become north-up, so
``PIL.Image.rotate(-k, expand=True)`` is upright. Without a verified compass,
portrait returns 0 and landscape 90 — these plates are portrait-printed, so
90-vs-270 is unresolved.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float32]
MaskArray = NDArray[np.uint8]

#: All analysis constants below are calibrated in this frame (px, long edge).
ANALYSIS_LONG_EDGE = 2000
#: Box window for the orientation-diversity star score (~star diameter).
_DIVERSITY_WINDOW = 61
#: Minimum diversity score for a star candidate (real stars measure 50-65;
#: the strongest non-star structures ~30).
_MIN_STAR_SCORE = 38.0
#: Candidates tried per sheet / non-maximum-suppression radius / border kept
#: clear of candidates (margin decorations score high).
_STAR_CANDIDATES = 3
_NMS_RADIUS = 80
_BORDER_PX = 45
#: Needle: radii band sampled between star rays and fleur, lateral slack,
#: and the minimum fraction of radii with ink (a drawn line is ~continuous).
_NEEDLE_RADII = (38, 92)
_NEEDLE_LATERAL = 6
_NEEDLE_MIN_CONTINUITY = 0.85
#: Fleur-de-lis: hub distance sweep, half-window, and blob acceptance
#: (area after 5x5 opening, bbox dims, fill ratio, elongation along needle).
_FLEUR_DISTANCES = range(95, 151, 5)
_FLEUR_HALF_WINDOW = 20
_FLEUR_AREA_RANGE = (100.0, 360.0)
_FLEUR_DIM_RANGE = (10, 40)
_FLEUR_MIN_FILL = 0.35
_FLEUR_MIN_ELONGATION = 1.1
#: Direction acceptance: minimum fleur blob area and margin over runner-up.
_MIN_DIRECTION_SCORE = 100.0
_MIN_DIRECTION_MARGIN = 2.0

#: Unit steps for the four axial directions (image frame, y down).
_DIRECTIONS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
#: north direction -> degrees the image must rotate CLOCKWISE for north-up.
_NORTH_TO_CLOCKWISE: dict[str, int] = {"up": 0, "left": 90, "down": 180, "right": 270}


def _analysis_gray(im: Image.Image) -> NDArray[np.uint8]:
    """Grayscale copy of an open image, downsampled to the analysis frame."""
    gray = im.convert("L")
    w, h = gray.size
    if max(w, h) > ANALYSIS_LONG_EDGE:
        scale = ANALYSIS_LONG_EDGE / max(w, h)
        gray = gray.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
    return np.asarray(gray, dtype=np.uint8)


def _ink_mask(gray: NDArray[np.uint8]) -> MaskArray:
    """Otsu ink mask (Sanborn paper ~140-175, ink ~24-60; fixed thresholds fail)."""
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (gray < thr).astype(np.uint8)


def _diversity_map(gray: NDArray[np.uint8]) -> FloatArray:
    """Per-pixel min over four orientation channels of windowed edge energy.

    Sobel magnitude is split into four 45-degree orientation bins and
    box-filtered at star scale; taking the minimum demands strong edges in
    EVERY direction, which singles out the 16-ray starburst (buildings have
    2 wall orientations, hatching 1-2, streets 1).
    """
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = np.mod(np.degrees(np.arctan2(gy, gx)), 180.0)
    window = (_DIVERSITY_WINDOW, _DIVERSITY_WINDOW)
    channels = []
    for b in range(4):
        lo = b * 45.0 - 22.5
        chan = np.where(np.mod(ang - lo, 180.0) < 45.0, mag, 0.0).astype(np.float32)
        channels.append(cv2.boxFilter(chan, -1, window, normalize=True))
    result: FloatArray = np.stack(channels).min(axis=0).astype(np.float32)
    return result


def _star_candidates(score: FloatArray) -> list[tuple[int, int, float]]:
    """Top diversity maxima ``(x, y, score)`` with NMS, borders excluded."""
    s = score.copy()
    w = s.shape[1]
    m = _BORDER_PX
    s[:m, :] = 0.0
    s[-m:, :] = 0.0
    s[:, :m] = 0.0
    s[:, -m:] = 0.0
    out: list[tuple[int, int, float]] = []
    for _ in range(_STAR_CANDIDATES):
        idx = int(np.argmax(s))
        y, x = divmod(idx, w)
        v = float(s[y, x])
        if v <= _MIN_STAR_SCORE:
            break
        out.append((x, y, v))
        y0, x0 = max(0, y - _NMS_RADIUS), max(0, x - _NMS_RADIUS)
        s[y0 : y + _NMS_RADIUS, x0 : x + _NMS_RADIUS] = 0.0
    return out


def _refine_center(ink: MaskArray, cx: int, cy: int) -> tuple[int, int]:
    """Mean-shift the candidate onto the star hub (box-blur peaks are ~10px off).

    The hub plus rays are radially balanced, so a few centroid iterations in
    a small disc converge on the true center; the needle test needs the
    center on-axis to within its lateral slack.
    """
    h, w = ink.shape
    yy, xx = np.mgrid[-10:11, -10:11]
    disc = (xx**2 + yy**2) <= 100
    x, y = float(cx), float(cy)
    for _ in range(4):
        x0, y0 = round(x), round(y)
        if not (10 <= x0 < w - 10 and 10 <= y0 < h - 10):
            break
        win = ink[y0 - 10 : y0 + 11, x0 - 10 : x0 + 11].astype(np.float32) * disc
        total = float(win.sum())
        if total < 5.0:
            break
        x = x0 + float((win * xx).sum()) / total
        y = y0 + float((win * yy).sum()) / total
    return round(x), round(y)


def _needle_continuity(ink: MaskArray, cx: int, cy: int, direction: str) -> float:
    """Fraction of radii in the needle band with ink within lateral slack.

    Slack-per-radius (rather than one straight probe) tolerates the ~1-2 deg
    skew real scans have over the needle's length.
    """
    h, w = ink.shape
    dx, dy = _DIRECTIONS[direction]
    lo, hi = _NEEDLE_RADII
    hits = 0
    total = 0
    offsets = range(-_NEEDLE_LATERAL, _NEEDLE_LATERAL + 1)
    for r in range(lo, hi):
        total += 1
        for off in offsets:
            x = cx + dx * r if dx != 0 else cx + off
            y = cy + dy * r if dy != 0 else cy + off
            if 0 <= x < w and 0 <= y < h and ink[y, x]:
                hits += 1
                break
    return hits / total if total else 0.0


def _fleur_blob_area(ink: MaskArray, cx: int, cy: int, direction: str) -> float:
    """Best fleur-de-lis blob area along ``direction``, 0.0 if none qualifies.

    A 5x5 opening erases everything drawn as lines or small text; the fleur
    is the rare compact SOLID blob at needle-end distance, elongated along
    the needle axis (building fills are larger, digits smaller, walls thin).
    """
    h, w = ink.shape
    dx, dy = _DIRECTIONS[direction]
    kernel = np.ones((5, 5), np.uint8)
    lo_area, hi_area = _FLEUR_AREA_RANGE
    lo_dim, hi_dim = _FLEUR_DIM_RANGE
    best = 0.0
    for dist in _FLEUR_DISTANCES:
        x = cx + dx * dist
        y = cy + dy * dist
        x0, x1 = max(0, x - _FLEUR_HALF_WINDOW), min(w, x + _FLEUR_HALF_WINDOW)
        y0, y1 = max(0, y - _FLEUR_HALF_WINDOW), min(h, y + _FLEUR_HALF_WINDOW)
        if x1 - x0 < 10 or y1 - y0 < 10:
            continue
        opened = cv2.morphologyEx(ink[y0:y1, x0:x1], cv2.MORPH_OPEN, kernel)
        n, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        for i in range(1, n):
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = float(stats[i, cv2.CC_STAT_AREA])
            along, perp = (bh, bw) if dy != 0 else (bw, bh)
            if (
                lo_area <= area <= hi_area
                and lo_dim <= bw <= hi_dim
                and lo_dim <= bh <= hi_dim
                and area >= _FLEUR_MIN_FILL * bw * bh
                and along >= _FLEUR_MIN_ELONGATION * perp
            ):
                best = max(best, area)
    return best


def _compass_north(gray: NDArray[np.uint8]) -> str | None:
    """Direction of compass north ("up"/"down"/"left"/"right"), None if unverified."""
    ink = _ink_mask(gray)
    for x0, y0, score in _star_candidates(_diversity_map(gray)):
        x, y = _refine_center(ink, x0, y0)
        scores: dict[str, float] = {}
        for d in _DIRECTIONS:
            area = _fleur_blob_area(ink, x, y, d)
            continuity = _needle_continuity(ink, x, y, d)
            scores[d] = area if continuity >= _NEEDLE_MIN_CONTINUITY else 0.0
        best = max(scores, key=lambda d: scores[d])
        runner_up = sorted(scores.values())[-2]
        confident = scores[best] >= _MIN_DIRECTION_MARGIN * runner_up
        if scores[best] >= _MIN_DIRECTION_SCORE and confident:
            logger.debug(
                "compass at (%d, %d) star=%.1f north=%s scores=%s", x, y, score, best, scores
            )
            return best
    return None


def detect_quarter_turn_image(gray_or_image: NDArray[np.uint8] | Image.Image, label: str) -> int:
    """Degrees (0/90/180/270) the image must rotate CLOCKWISE to be north-up.

    Takes an already-open or already-decoded image, so a caller that must
    decode the full-res scan anyway pays for one decode; ``label`` names the
    sheet in log lines. Detection reads the compass rose (see the module
    docstring); with no verified compass, portrait falls back to 0 and
    landscape to 90.
    """
    if isinstance(gray_or_image, Image.Image):
        gray = _analysis_gray(gray_or_image)
    else:
        gray = gray_or_image
    north = _compass_north(gray)
    if north is not None:
        rotation = _NORTH_TO_CLOCKWISE[north]
        if rotation:
            logger.info("%s: compass north=%s -> rotate %d deg clockwise", label, north, rotation)
        return rotation
    h, w = gray.shape
    if w > h:
        logger.warning(
            "%s: landscape scan without a verified compass; assuming 90 deg clockwise",
            label,
        )
        return 90
    logger.debug("%s: no compass verified on portrait sheet; assuming upright", label)
    return 0
