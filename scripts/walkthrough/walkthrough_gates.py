"""Panel 10: every check a placement must clear, shown passing and shown failing.

Each failing state is a prepared counterexample - the exemplar's own corner
guesses, moved - handed straight back to the production matcher. The refusal and
its name come from :func:`autogeoref.matching.ransac_affine_diagnostics`, so a
plate can never claim a barrier the pipeline would not raise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from walkthrough_candidates import window_text
from walkthrough_sources import FidelityError, Fit, Volume

from autogeoref.affine import model_determinant
from autogeoref.matching import (
    ASPECT_MAX,
    SPREAD_PERP_FRAC,
    SPREAD_SPAN_FRAC,
    Candidate,
    FitGates,
    fold_quadrant_deg,
    perp_spread_frac,
    ransac_affine,
    ransac_affine_diagnostics,
)

Pixel = tuple[float, float]


@dataclass(frozen=True)
class Gate:
    """One check: what it asks, and what the matcher calls a failure of it."""

    key: str
    name: str
    asks: str
    why: str
    #: what the diagnostics classifier calls this refusal, where it names it
    failure_class: str | None


GATES = (
    Gate(
        "handedness",
        "Not a mirror image",
        "Does the placement flip the page over?",
        "A reflected placement reads the sheet back to front. It is not a bad "
        "placement of this scan - it is a placement of a scan that does not "
        "exist. There is no tolerance to set: the arithmetic either flips the "
        "page or it does not.",
        "handedness",
    ),
    Gate(
        "scale_window",
        "Printed at the book's scale",
        "Does one pixel cover about as much ground as it does on every other sheet?",
        "The whole volume was printed at one scale. A placement that needs this "
        "page to be a different size from its neighbours has found a pattern "
        "somewhere else in the city.",
        "scale_window",
    ),
    Gate(
        "rotation_window",
        "Bound the same way up",
        "Is the page turned about as far from north as the rest of the book?",
        "Sheets in a bound atlas share an orientation. Quarter turns are "
        "forgiven, because a scan can be fed in sideways without the printing "
        "being crooked; anything else is not this book.",
        "rotation_window",
    ),
    Gate(
        "aspect",
        "Not stretched one way",
        "Is the placement stretching the page more across than down?",
        "A placement that squashes one direction is fitting noise. This is the "
        "backstop for a volume whose scale is not yet pinned down, where a "
        "lopsided stretch could otherwise slip through.",
        "aspect",
    ),
    Gate(
        "span",
        "Corners spread across the page",
        "Do the agreeing corners cover both directions of the sheet?",
        "Corners huddled in one part of the page pin down that part and leave "
        "the rest to swing. The far side of the sheet would then be placed by "
        "extrapolation, which is a guess wearing a placement's clothes.",
        None,
    ),
    Gate(
        "perp",
        "Corners not all in a line",
        "Do the agreeing corners form a shape, or a line?",
        "Corners strung along a single street fix the sheet along that street "
        "and say nothing about the direction across it. The sheet can slide "
        "sideways with every corner still agreeing.",
        None,
    ),
    Gate(
        "leave_one_out",
        "Not held up by one corner",
        "Does the shape survive removing any single corner?",
        "A line of corners plus one stray makes a shape on paper, and that one "
        "stray is holding the whole placement steady. Drop each corner in turn "
        "and ask again; anything that collapses was never really supported.",
        "leave_one_out_spread",
    ),
)


def _moved(fit: Fit, move: Any) -> list[Candidate]:
    """The exemplar's kept corners with their page positions moved by ``move``.

    From the kept corners, not the whole guess list: over the full list the
    search finds three other points that carry some different consensus, and the
    refusal then names a barrier the plate is not about.
    """
    return [
        Candidate(pixel=move(*c.pixel), world4326=c.world4326, streets=c.streets)
        for c in fit.inliers
    ]


def _rearranged(fit: Fit, move: Any) -> list[Candidate]:
    """Kept corners moved on the page, with the map side moved to match.

    Both sides move through the accepted placement, so the fit these produce is
    the same size and the same way up as the real one. Only where the corners
    sit on the page changes, which is what the three spread checks look at.
    """
    from autogeoref.affine import TO_4326, apply_affine

    model = fit.model
    assert model is not None  # only ever called for an accepted exemplar
    out: list[Candidate] = []
    for cand in fit.inliers:
        pixel = move(*cand.pixel)
        out.append(
            Candidate(
                pixel=pixel,
                world4326=TO_4326.transform(*apply_affine(model, *pixel)),
                streets=cand.streets,
            )
        )
    return out


def counterexamples(fit: Fit) -> dict[str, list[Candidate]]:
    """One prepared defect per check, built from the exemplar's own corners."""
    w, h = fit.sheet.full_size
    cx, cy = w / 2, h / 2
    turn = math.radians(30.0)
    px = [c.pixel[0] for c in fit.inliers]
    py = [c.pixel[1] for c in fit.inliers]
    mx, my = sum(px) / len(px), sum(py) / len(py)

    def rotate(x: float, y: float) -> Pixel:
        dx, dy = x - cx, y - cy
        return (
            cx + dx * math.cos(turn) - dy * math.sin(turn),
            cy + dx * math.sin(turn) + dy * math.cos(turn),
        )

    def crowd(x: float, y: float) -> Pixel:
        return (mx + (x - mx) * 0.22, my + (y - my) * 0.22)

    # onto the page's own diagonal, so the corners still reach both edges and
    # only the sideways spread is gone
    diagonal = (w / math.hypot(w, h), h / math.hypot(w, h))

    def flatten(x: float, y: float) -> Pixel:
        along = (x - mx) * diagonal[0] + (y - my) * diagonal[1]
        return (mx + diagonal[0] * along, my + diagonal[1] * along)

    # the one corner left where it was is the one furthest off that diagonal,
    # so the full set really does look two-dimensional until it is removed
    stray = max(
        (c.pixel for c in fit.inliers),
        key=lambda p: abs((p[0] - mx) * diagonal[1] - (p[1] - my) * diagonal[0]),
    )

    def flatten_one(x: float, y: float) -> Pixel:
        return (x, y) if (x, y) == stray else flatten(x, y)

    # the squashed set is built from the kept corners alone: with the whole
    # guess list the search finds three other points that are not squashed, and
    # the refusal then names the wrong barrier
    squashed = [
        Candidate(pixel=(c.pixel[0], c.pixel[1] * 2.6), world4326=c.world4326, streets=c.streets)
        for c in fit.inliers
    ]
    return {
        "handedness": _moved(fit, lambda x, y: (w - x, y)),
        # spread the corners rather than crowding them: shrinking far enough to
        # leave the window also leaves the diagnostics' own wide probe, and the
        # refusal comes back as "no coherent fit" instead of naming the scale
        "scale_window": _moved(fit, lambda x, y: (x * 1.5, y * 1.5)),
        "rotation_window": _moved(fit, rotate),
        "aspect": squashed,
        "span": _rearranged(fit, crowd),
        "perp": _rearranged(fit, flatten),
        "leave_one_out": _rearranged(fit, flatten_one),
    }


def fit_with(
    volume: Volume, fit: Fit, cands: list[Candidate], *, windowed: bool = True
) -> dict[str, Any]:
    scale_range = fit.constraints.scale_range if windowed else None
    rot_range = fit.constraints.rot_range_deg if windowed else None
    strict = ransac_affine(
        cands,
        fit.sheet.full_size,
        scale_range=scale_range,
        rot_range_deg=rot_range,
        rot_quadrant_fold=True,
    )
    del volume
    return ransac_affine_diagnostics(
        cands,
        fit.sheet.full_size,
        scale_range=scale_range,
        rot_range_deg=rot_range,
        rot_quadrant_fold=True,
        strict_result=strict,
    )


def measured(
    gate: Gate, fit: Fit, cands: list[Candidate], report: dict[str, Any]
) -> tuple[str, str]:
    """What this gate measured on the counterexample, and what it allows."""
    probe = report.get("unconstrained_fit") or report.get("wide_plausibility_fit") or {}
    w, h = fit.sheet.full_size
    if gate.key == "handedness":
        model = raw_model(cands)
        flip = "flips the page" if model is not None and model_determinant(model) > 0 else "upright"
        return flip, "must not flip"
    if gate.key == "scale_window":
        scales = probe.get("scales_m_per_px") or []
        got = f"{sum(scales) / len(scales):.4g} m per pixel" if scales else "-"
        return got, window_text(fit.constraints.scale_range, "{:.4g}")
    if gate.key == "rotation_window":
        angle = probe.get("rotation_deg")
        got = f"{fold_quadrant_deg(angle):.3g} deg" if angle is not None else "-"
        return got, window_text(fit.constraints.rot_range_deg, "{:.3g}") + " deg"
    if gate.key == "aspect":
        scales = probe.get("scales_m_per_px") or []
        got = f"{max(scales) / min(scales):.3g} to 1" if scales and min(scales) else "-"
        return got, f"no more than {ASPECT_MAX:g} to 1 either way"
    pts = np.array([[c.pixel[0], c.pixel[1]] for c in cands])
    if gate.key == "span":
        sx = (pts[:, 0].max() - pts[:, 0].min()) / w
        sy = (pts[:, 1].max() - pts[:, 1].min()) / h
        return f"{min(sx, sy) * 100:.0f}% of the page", f"at least {SPREAD_SPAN_FRAC:.0%} each way"
    if gate.key == "perp":
        return (
            f"{perp_spread_frac(pts, w, h) * 100:.2f}% off the line",
            f"at least {SPREAD_PERP_FRAC:.0%}",
        )
    worst = min(perp_spread_frac(np.delete(pts, i, axis=0), w, h) for i in range(len(pts)))
    return (
        f"{worst * 100:.2f}% with one corner removed",
        f"at least {SPREAD_PERP_FRAC:.0%} every time",
    )


def assert_defect(gate: Gate, fit: Fit, cands: list[Candidate]) -> None:
    """The prepared set really carries the defect its plate claims.

    The three spread checks share one refusal name, so the name alone cannot
    say which of them stopped a placement. These measure the property itself.
    """
    if gate.key not in ("span", "perp", "leave_one_out"):
        return
    w, h = fit.sheet.full_size
    pts = np.array([[c.pixel[0], c.pixel[1]] for c in cands])
    span = min(np.ptp(pts[:, 0]) / w, np.ptp(pts[:, 1]) / h)
    perp = perp_spread_frac(pts, w, h)
    worst = min(perp_spread_frac(np.delete(pts, i, axis=0), w, h) for i in range(len(pts)))
    if gate.key == "span" and not (span < SPREAD_SPAN_FRAC and perp >= SPREAD_PERP_FRAC):
        raise FidelityError(f"gate span: span {span:.3f}, perpendicular spread {perp:.3f}")
    if gate.key == "perp" and not (span >= SPREAD_SPAN_FRAC and perp < SPREAD_PERP_FRAC):
        raise FidelityError(f"gate perp: span {span:.3f}, perpendicular spread {perp:.3f}")
    if gate.key == "leave_one_out" and not (perp >= SPREAD_PERP_FRAC > worst):
        raise FidelityError(f"gate leave_one_out: full set {perp:.3f}, worst subset {worst:.3f}")


def raw_model(cands: list[Candidate]) -> Any:
    """The placement a counterexample implies, for drawing only.

    A unit page size stands in for the real one so the two spread checks, which
    have no off switch, cannot suppress the very model this is asked to draw.
    Nothing is accepted from it.
    """
    model, _ = ransac_affine(
        cands,
        (1.0, 1.0),
        scale_range=(0.0, math.inf),
        min_inliers=3,
        gates=FitGates(loo_spread=False, handedness=False, aspect=False),
    )
    return model
