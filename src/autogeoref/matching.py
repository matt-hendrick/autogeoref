"""The matcher core: label-axis intersections x centerline intersections ->
constrained RANSAC affine.

Pure library, no I/O: ``(annotation, index, constraints) -> model + inliers``.

The gates encoded here are contracts, not tuning knobs (see the contract
table in the project docs): every one exists because a real confidently-wrong
placement got through before it did. Never weaken one to make a test pass.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from shapely.geometry import LineString

from .affine import (
    TO_3857,
    AffineMatrix,
    apply_affine,
    fit_affine,
    model_determinant,
    model_rotation_deg,
    model_scales,
)
from .names import Aliases, normalize

if TYPE_CHECKING:
    from .centerlines import CenterlineIndex

#: Fallback plausibility range for Sanborn scan scale, meters per full-res pixel.
DEFAULT_SCALE_RANGE = (0.02, 0.5)
#: RANSAC inlier tolerance, 3857 meters.
RANSAC_TOL_M = 25.0
#: Minimum surviving inliers for a valid model.
RANSAC_MIN_INLIERS = 4
#: Fixed iteration count and seed — byte-identical re-runs are a tested contract.
RANSAC_ITERS = 5000
RANSAC_SEED = 42
#: Inlier bbox must span this share of the sheet in both axes...
SPREAD_SPAN_FRAC = 0.30
#: ...and the farthest-apart pair must have this much perpendicular spread —
#: of the full inlier set, AND of every leave-one-out subset (see loo_spread_ok).
SPREAD_PERP_FRAC = 0.05
#: A model may stretch one axis this many times the other, either way.
ASPECT_MAX = 2.0
#: Who a machine-placed GCP is attributed to. Pure metadata — nothing reads it
#: to make a decision — but it is published, so it names the placer rather than
#: a user account inherited from the schema this record shape came from.
AUTO_PLACED_BY = "autogeoref"


def perp_spread_frac(pts: np.ndarray, width: float, height: float) -> float:
    """Perpendicular spread of a pixel point set, as a fraction of the sheet diagonal.

    Spread of the set about the line through its two farthest-apart members:
    the answer to "is this constellation 2-D, or is it a line?".
    """
    if len(pts) < 2:
        return 0.0
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    a, b = np.unravel_index(np.argmax(d2), d2.shape)
    ab = pts[b] - pts[a]
    norm = float(np.linalg.norm(ab))
    if norm == 0:
        return 0.0
    ab_n = ab / norm
    rel = pts - pts[a]
    # 2-D cross product magnitude (numpy 2.x dropped 2-D np.cross)
    perp = np.abs(rel[:, 0] * ab_n[1] - rel[:, 1] * ab_n[0])
    return float(perp.max() / math.hypot(width, height))


def loo_spread_ok(pts: np.ndarray, width: float, height: float) -> bool:
    """The spread gate must survive the loss of ANY single anchor.

    A line plus one off-line anchor can pass the full-set gate while that anchor alone controls
    the fit's perpendicular dimension. Removing each anchor in turn rejects this unstable
    geometry. Fewer than 4 anchors can NEVER pass, and that is the gate's meaning: every
    leave-one-out subset of a 3-point set is 2 points, a line by definition. So lowering
    ``min_inliers`` below :data:`RANSAC_MIN_INLIERS` rejects every model while this gate is on;
    a caller must opt out to fit on 3.
    """
    if len(pts) < RANSAC_MIN_INLIERS:
        return False
    return all(
        perp_spread_frac(np.delete(pts, i, axis=0), width, height) >= SPREAD_PERP_FRAC
        for i in range(len(pts))
    )


def fold_quadrant_deg(angle_deg: float) -> float:
    """Fold an angle into [-45, 45): rotation deviation modulo quadrant turns.

    The shared orientation constraint applies after removing quadrant turns.
    """
    return ((angle_deg + 45.0) % 90.0) - 45.0


# One predicate per gate window, shared by ransac_affine's gates_ok and the
# diagnostics classifier — so the label a rejection gets can never drift from
# the test that rejected it.


def _handedness_ok(m: AffineMatrix) -> bool:
    # an upright placement of a y-down scan is negative-determinant; a
    # reflection is not a placement of this scan at all
    return model_determinant(m) < 0


def _scale_window_ok(sx: float, sy: float, scale_range: tuple[float, float]) -> bool:
    lo, hi = scale_range
    return lo < sx < hi and lo < sy < hi


def _aspect_ok(sx: float, sy: float) -> bool:
    return 1 / ASPECT_MAX < sx / sy < ASPECT_MAX


def _rotation_window_ok(
    ang_deg: float, rot_range_deg: tuple[float, float], *, quadrant_fold: bool
) -> bool:
    if quadrant_fold:
        center = (rot_range_deg[0] + rot_range_deg[1]) / 2
        half = (rot_range_deg[1] - rot_range_deg[0]) / 2
        return abs(fold_quadrant_deg(ang_deg - center)) <= half
    return rot_range_deg[0] <= ang_deg <= rot_range_deg[1]


@dataclass(frozen=True)
class Candidate:
    """One (pixel intersection <-> world intersection) correspondence.

    ``pixel`` is in FULL-resolution image space; ``world4326`` is lng/lat.
    ``streets`` keeps the raw (un-normalized) label names for audit notes.
    """

    pixel: tuple[float, float]
    world4326: tuple[float, float]
    streets: tuple[str, str]


def _usable_direction(street: dict[str, Any]) -> tuple[float, float] | None:
    """The label's unit direction vector, or ``None`` when it has no usable one."""
    raw = street.get("direction")
    if not isinstance(raw, Sequence) or isinstance(raw, str) or len(raw) != 2:
        return None
    try:
        dx, dy = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    norm = math.hypot(dx, dy)
    return None if norm == 0 else (dx / norm, dy / norm)


def label_axis(street: dict[str, Any]) -> LineString:
    """A pixel-space line along the street label's text direction.

    A non-cardinal label needs a ``direction`` vector. One that is missing or
    unreadable degrades to the axis its bbox is longer along, which is what the
    cardinal-only reads have always given — never to a dropped label.
    """
    x0, y0, x1, y1 = street["bbox"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    length = 100000
    orientation = street["orientation"]
    if orientation != "horizontal" and orientation != "vertical":
        direction = _usable_direction(street)
        if direction is not None:
            dx, dy = direction
            return LineString(
                [(cx - length * dx, cy - length * dy), (cx + length * dx, cy + length * dy)]
            )
        orientation = "horizontal" if abs(x1 - x0) >= abs(y1 - y0) else "vertical"
    if orientation == "horizontal":
        return LineString([(cx - length, cy), (cx + length, cy)])
    return LineString([(cx, cy - length), (cx, cy + length)])


def candidate_gcps(
    annotation: dict[str, Any],
    index: CenterlineIndex,
    scale: float,
    aliases: Aliases | None = None,
) -> list[Candidate]:
    """Pair up detected streets -> (pixel point, world point) candidates.

    ``scale`` is the annotation-frame/full-res ratio from the sheet manifest;
    label bboxes are annotated on the downsampled image and every downstream
    pixel coordinate is full-res.
    """
    aliases = aliases if aliases is not None else index.aliases
    streets = annotation["streets"]
    # hoist the per-street work out of the O(n^2) pair loop (pure functions)
    prepared = [(normalize(s["name"], aliases), label_axis(s), s) for s in streets]
    cands: list[Candidate] = []
    for (ka, axis_a, a), (kb, axis_b, b) in itertools.combinations(prepared, 2):
        if ka == kb:
            continue
        pix = axis_a.intersection(axis_b)
        if pix.is_empty or pix.geom_type != "Point":
            continue
        world_pts = index.intersections(ka, kb)
        # keep every world option: streets can legitimately cross twice (jogs)
        cands.extend(
            Candidate(
                pixel=(pix.x / scale, pix.y / scale),
                world4326=(lng, lat),
                streets=(a["name"], b["name"]),
            )
            for lng, lat in world_pts
        )
    return cands


@dataclass(frozen=True)
class FitGates:
    """Which of the RANSAC gates a fit must clear.

    Relaxations exist for :func:`ransac_affine_diagnostics` and for pass-1
    constant derivation, neither of which is an acceptance decision. Never
    relax a gate for one.
    """

    loo_spread: bool = True
    handedness: bool = True
    aspect: bool = True


#: Every gate on, which is what an acceptance decision must use.
STRICT_GATES = FitGates()


def ransac_affine(
    cands: list[Candidate],
    full_size: tuple[float, float],
    tol_m: float = RANSAC_TOL_M,
    min_inliers: int = RANSAC_MIN_INLIERS,
    iters: int = RANSAC_ITERS,
    seed: int = RANSAC_SEED,
    scale_range: tuple[float, float] | None = None,
    rot_range_deg: tuple[float, float] | None = None,
    rot_quadrant_fold: bool = False,
    gates: FitGates = STRICT_GATES,
) -> tuple[AffineMatrix | None, list[Candidate]]:
    """RANSAC over candidate correspondences.

    Gates on every accepted model, sampled and final refit: HANDEDNESS (the 2x2 determinant
    must be negative — a reflected model warps the sheet back-to-front and the rotation gate
    cannot see it); metres/pixel within ``scale_range``; aspect within 0.5..2.0; and rotation
    within ``rot_range_deg`` when given, quadrant-folded under ``rot_quadrant_fold``. Post
    gates: enough inliers after per-pixel dedupe, an inlier bbox spanning both axes, adequate
    perpendicular spread, and — unless ``gates.loo_spread`` is off — that spread gate again on
    every leave-one-out subset. :class:`FitGates` says who may relax what.
    """
    if len(cands) < 3:
        return None, []

    pts: list[tuple[float, float, float, float]] = [
        (c.pixel[0], c.pixel[1], *TO_3857.transform(*c.world4326)) for c in cands
    ]
    # Vectorized evaluation preserves the scalar affine operation order.
    px_arr, py_arr, wx_arr, wy_arr = (np.array(col) for col in zip(*pts, strict=True))
    w, h = full_size
    rng = random.Random(seed)
    lo, hi = scale_range or DEFAULT_SCALE_RANGE

    def gates_ok(m: AffineMatrix) -> bool:
        # handedness first: a reflection is not a placement of this scan at all,
        # and the rotation gate below is structurally blind to it (a 180 deg
        # reflection folds onto 0 deg). No threshold — see model_determinant.
        if gates.handedness and not _handedness_ok(m):
            return False
        sx, sy = model_scales(m)
        if not _scale_window_ok(sx, sy, (lo, hi)):
            return False
        if gates.aspect and not _aspect_ok(sx, sy):
            return False
        return rot_range_deg is None or _rotation_window_ok(
            model_rotation_deg(m), rot_range_deg, quadrant_fold=rot_quadrant_fold
        )

    best_model: AffineMatrix | None = None
    best_inliers: list[int] = []
    n = len(pts)
    idx = list(range(n))
    for _ in range(iters):
        sample = rng.sample(idx, 3)
        # a sample using the same world (or pixel) point twice is degenerate
        if len({pts[i][2:] for i in sample}) < 3 or len({pts[i][:2] for i in sample}) < 3:
            continue
        try:
            m = fit_affine([pts[i] for i in sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not gates_ok(m):
            continue
        dx = (m[0][0] + m[0][1] * px_arr + m[0][2] * py_arr) - wx_arr
        dy = (m[1][0] + m[1][1] * px_arr + m[1][2] * py_arr) - wy_arr
        keep = np.hypot(dx, dy) <= tol_m
        if int(np.count_nonzero(keep)) > len(best_inliers):
            best_model, best_inliers = m, np.flatnonzero(keep).tolist()

    if best_model is None or len(best_inliers) < min_inliers:
        return None, []

    # dedupe: keep one candidate per rounded pixel point (the closest-fitting)
    m = fit_affine([pts[i] for i in best_inliers])
    seen: dict[tuple[int, int], tuple[float, int]] = {}
    for i in best_inliers:
        px, py, x, y = pts[i]
        key = (round(px), round(py))
        xp, yp = apply_affine(m, px, py)
        d = math.hypot(xp - x, yp - y)
        if key not in seen or d < seen[key][0]:
            seen[key] = (d, i)
    inlier_idx = [i for _, i in seen.values()]
    if len(inlier_idx) < min_inliers:
        return None, []

    # spread gates: bbox span in both axes, then perpendicular spread of the
    # point set relative to its farthest-apart pair (near-collinear kill)
    xs = [pts[i][0] for i in inlier_idx]
    ys = [pts[i][1] for i in inlier_idx]
    if (max(xs) - min(xs)) < SPREAD_SPAN_FRAC * w or (max(ys) - min(ys)) < SPREAD_SPAN_FRAC * h:
        return None, []
    p = np.array([[pts[i][0], pts[i][1]] for i in inlier_idx])
    if perp_spread_frac(p, w, h) < SPREAD_PERP_FRAC:
        return None, []
    if gates.loo_spread and not loo_spread_ok(p, w, h):
        return None, []

    # the refit must honor the same constraints as the sampled models —
    # otherwise a weak inlier set can drag the final fit crooked again
    m = fit_affine([pts[i] for i in inlier_idx])
    if not gates_ok(m):
        return None, []
    return m, [cands[i] for i in inlier_idx]


def ransac_affine_diagnostics(
    cands: list[Candidate],
    full_size: tuple[float, float],
    *,
    scale_range: tuple[float, float] | None,
    rot_range_deg: tuple[float, float] | None,
    rot_quadrant_fold: bool = False,
    strict_result: tuple[AffineMatrix | None, list[Candidate]] | None = None,
) -> dict[str, Any]:
    """Explain an offline matcher replay without changing its acceptance path.

    ``strict_result`` lets an offline caller supply the already-computed
    production fit, avoiding a second 5,000-iteration match. The wide,
    unconstrained, and no-LOO fits exist only to name a barrier; callers must
    never serialize them as placement results.
    """

    def fit(
        *,
        scale: tuple[float, float] | None,
        rotation: tuple[float, float] | None,
        loo: bool,
        min_inliers: int,
        handedness: bool = True,
        aspect: bool = True,
    ) -> tuple[AffineMatrix | None, list[Candidate]]:
        return ransac_affine(
            cands,
            full_size,
            scale_range=scale,
            rot_range_deg=rotation,
            rot_quadrant_fold=rot_quadrant_fold,
            min_inliers=min_inliers,
            gates=FitGates(loo_spread=loo, handedness=handedness, aspect=aspect),
        )

    strict = strict_result or fit(
        scale=scale_range,
        rotation=rot_range_deg,
        loo=True,
        min_inliers=RANSAC_MIN_INLIERS,
    )
    wide: tuple[AffineMatrix | None, list[Candidate]] | None = None
    unconstrained: tuple[AffineMatrix | None, list[Candidate]] | None = None
    no_loo: tuple[AffineMatrix | None, list[Candidate]] | None = None

    if len(cands) < RANSAC_MIN_INLIERS:
        failure_class = "fewer_than_4_candidates"
    elif strict[0] is not None:
        failure_class = "strict_accept"
    else:
        # The wide fit still keeps the aspect-ratio and residual contracts. It
        # is a plausibility diagnostic, not an unconstrained placement permission.
        wide = fit(scale=(0.0, math.inf), rotation=None, loo=False, min_inliers=3)
        if wide[0] is None:
            # the wide fit enforces handedness and aspect, so when even it finds nothing, ask a
            # fit past those two gates whether one is the barrier — otherwise a reflected or
            # squashed model is mislabelled incoherent. scale=None keeps DEFAULT_SCALE_RANGE,
            # so a rank-deficient fit cannot masquerade as an aspect defect.
            unconstrained = fit(
                scale=None,
                rotation=None,
                loo=False,
                min_inliers=3,
                handedness=False,
                aspect=False,
            )
            probe = unconstrained[0]
            if probe is not None and not _handedness_ok(probe):
                failure_class = "handedness"
            else:
                failure_class = "no_coherent_wide_fit"
                if probe is not None:
                    # mirror gates_ok's order (handedness, scale, aspect,
                    # rotation): a probe whose scales already fail the page
                    # scale window is a scale rejection, and naming it
                    # "aspect" would mislabel it
                    sx, sy = model_scales(probe)
                    in_window = scale_range is None or _scale_window_ok(sx, sy, scale_range)
                    if in_window and not _aspect_ok(sx, sy):
                        failure_class = "aspect"
        else:
            sx, sy = model_scales(wide[0])
            if scale_range is not None and not _scale_window_ok(sx, sy, scale_range):
                failure_class = "scale_window"
            elif rot_range_deg is not None and not _rotation_window_ok(
                model_rotation_deg(wide[0]), rot_range_deg, quadrant_fold=rot_quadrant_fold
            ):
                failure_class = "rotation_window"
            else:
                no_loo = fit(
                    scale=scale_range,
                    rotation=rot_range_deg,
                    loo=False,
                    min_inliers=RANSAC_MIN_INLIERS,
                )
                failure_class = (
                    "leave_one_out_spread"
                    if no_loo[0] is not None
                    else "constrained_insufficient_inliers"
                )

    def summary(
        result: tuple[AffineMatrix | None, list[Candidate]] | None,
    ) -> dict[str, Any] | None:
        if result is None:
            return None
        model, inliers = result
        return {
            "has_model": model is not None,
            "n_inliers": len(inliers),
            "rotation_deg": round(model_rotation_deg(model), 4) if model is not None else None,
            "scales_m_per_px": [round(value, 6) for value in model_scales(model)]
            if model is not None
            else None,
        }

    return {
        "failure_class": failure_class,
        "n_candidates": len(cands),
        "wide_plausibility_fit": summary(wide),
        "unconstrained_fit": summary(unconstrained),
        "no_loo_fit": summary(no_loo),
        "strict_fit": summary(strict),
    }


def gcps_geojson_from(cand_list: list[Candidate], username: str = AUTO_PLACED_BY) -> dict[str, Any]:
    """Session-style GCP FeatureCollection (image px full-res, world 4326)."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "image": [round(c.pixel[0]), round(c.pixel[1])],
                    "username": username,
                    "note": f"auto: {c.streets[0]} x {c.streets[1]}",
                },
                "geometry": {"type": "Point", "coordinates": list(c.world4326)},
            }
            for c in cand_list
        ],
    }
