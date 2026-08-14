"""Panel 10's plates: the checks passing, then each one refusing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import walkthrough_figures as figures
import walkthrough_theme as theme
from walkthrough_candidates import window_text
from walkthrough_gates import (
    GATES,
    Gate,
    assert_defect,
    counterexamples,
    fit_with,
    measured,
    raw_model,
)
from walkthrough_page import Emitter, Panel, State
from walkthrough_sources import FidelityError

from autogeoref.affine import model_rotation_deg, model_scales
from autogeoref.matching import (
    ASPECT_MAX,
    RANSAC_MIN_INLIERS,
    SPREAD_PERP_FRAC,
    SPREAD_SPAN_FRAC,
    Candidate,
    fold_quadrant_deg,
    perp_spread_frac,
)

if TYPE_CHECKING:
    from walkthrough_sources import Fit, Volume


def panel_gates(volume: Volume, ex: dict[str, Any], out: Emitter) -> Panel:
    """10 - the checks, passing, then each refusing a prepared counterexample."""
    page = ex["pages"]["hero"]
    fit = volume.match(page)
    if fit.model is None:
        raise FidelityError(f"p{page}: the gate panel needs an accepted placement")
    prepared = counterexamples(fit)
    panel = Panel(
        number=10,
        slug="the-gates",
        act="II. From a reading to a fit",
        title="Seven ways to say no",
        dek="Every check a placement must clear, shown passing and shown refusing.",
        caption=(
            "A placement can agree with itself and still be wrong. Seven checks, "
            "called the gates, stand between a fit and a committed sheet, and they "
            "were added one at a time, each after a confidently wrong placement got "
            "past the ones already there. A sheet that clears all seven is a strict "
            "accept. Four of the checks ask about the shape of the transformation: "
            "is the page flipped, resized, turned, or stretched out of true. Three "
            "ask about the corners it rests on: are they spread across the page, do "
            "they form a shape rather than a line, and does that shape survive "
            "losing any one of them. Each panel here hands the matcher a broken "
            "version of this sheet's own corners and shows how it reports the "
            "refusal."
        ),
        figures=[
            ("Checks", str(len(GATES))),
            ("Minimum corners", str(RANSAC_MIN_INLIERS)),
            ("Spread across the page", f"{SPREAD_SPAN_FRAC:.0%} each way"),
            ("Spread off the line", f"{SPREAD_PERP_FRAC:.0%}"),
        ],
        stage="match",
    )
    panel.states = [_pass_state(volume, ex, fit, panel, out)] + [
        _fail_state(volume, ex, fit, prepared[gate.key], gate, panel, out) for gate in GATES
    ]
    return panel


def _pass_state(volume: Volume, ex: dict[str, Any], fit: Fit, panel: Panel, out: Emitter) -> State:
    """The exemplar clearing all seven, with what each one measured."""
    plate = theme.plate(10, "Seven ways to say no", kicker="this sheet passes")
    left, top, right, bottom = plate.box
    y = theme.paragraph(
        plate.draw,
        (left, top),
        "The placement from the previous panel, put to every check in turn. Each "
        "line is what that check measured on this sheet, beside what it allows.",
        right - left,
        size=theme.BODY,
    )
    model = fit.model
    assert model is not None  # panel_gates refuses an exemplar without one
    report = fit.diagnostics()
    pts = np.array([[c.pixel[0], c.pixel[1]] for c in fit.inliers])
    w, h = fit.sheet.full_size
    sx, sy = model_scales(model)
    span = min(np.ptp(pts[:, 0]) / w, np.ptp(pts[:, 1]) / h)
    worst = min(perp_spread_frac(np.delete(pts, i, axis=0), w, h) for i in range(len(pts)))
    rows = [
        ("Not a mirror image", "upright", "must not flip"),
        (
            "Printed at the book's scale",
            f"{(sx + sy) / 2:.4g} m per pixel",
            window_text(fit.constraints.scale_range, "{:.4g}"),
        ),
        (
            "Bound the same way up",
            f"{fold_quadrant_deg(model_rotation_deg(model)):.3g} deg",
            window_text(fit.constraints.rot_range_deg, "{:.3g}") + " deg",
        ),
        (
            "Not stretched one way",
            f"{max(sx, sy) / min(sx, sy):.3g} to 1",
            f"no more than {ASPECT_MAX:g} to 1 either way",
        ),
        (
            "Corners spread across the page",
            f"{span * 100:.0f}% of the page",
            f"at least {SPREAD_SPAN_FRAC:.0%} each way",
        ),
        (
            "Corners not all in a line",
            f"{perp_spread_frac(pts, w, h) * 100:.1f}% off the line",
            f"at least {SPREAD_PERP_FRAC:.0%}",
        ),
        (
            "Not held up by one corner",
            f"{worst * 100:.1f}% at worst",
            f"at least {SPREAD_PERP_FRAC:.0%} every time",
        ),
    ]
    y += 18
    row_h = (bottom - y - 60) / len(rows)
    for name, got, allowed in rows:
        theme.text(plate.draw, (left + 46, y + 4), name, size=theme.BODY, bold=True)
        theme.text(plate.draw, (left + 640, y + 4), got, size=theme.BODY, fill=theme.INK)
        theme.text(
            plate.draw, (right, y + 4), allowed, size=theme.SMALL, fill=theme.INK_SOFT, anchor="ra"
        )
        _tick(plate, (left + 12, y + 14))
        y += row_h
        plate.draw.line((left, y - 12, right, y - 12), fill=theme.RULE, width=1)
    theme.text(
        plate.draw,
        (left, bottom - 34),
        f"the matcher's own verdict on this sheet: {report['failure_class'].replace('_', ' ')}",
        size=theme.SMALL,
        fill=theme.KEEP,
    )
    del volume, ex
    state = State("pass", "All seven pass", alt="Every check with what it measured on this sheet.")
    out.save(panel, state, plate.image)
    return state


def _tick(plate: theme.Plate, at: tuple[float, float]) -> None:
    x, y = at
    plate.draw.line((x, y + 2, x + 8, y + 10), fill=theme.KEEP, width=4)
    plate.draw.line((x + 8, y + 10, x + 22, y - 8), fill=theme.KEEP, width=4)


def _fail_state(
    volume: Volume,
    ex: dict[str, Any],
    fit: Fit,
    cands: list[Candidate],
    gate: Gate,
    panel: Panel,
    out: Emitter,
) -> State:
    """One gate, refusing a prepared counterexample."""
    windowed = gate.key != "aspect"
    report = fit_with(volume, fit, cands, windowed=windowed)
    if report["strict_fit"]["has_model"]:
        raise FidelityError(f"gate {gate.key}: the prepared counterexample was ACCEPTED")
    if gate.failure_class is not None and report["failure_class"] != gate.failure_class:
        raise FidelityError(
            f"gate {gate.key}: prepared counterexample was refused as "
            f"{report['failure_class']!r}, not {gate.failure_class!r}"
        )
    assert_defect(gate, fit, cands)
    plate = theme.plate(10, "Seven ways to say no", kicker=gate.name.lower())
    left, top, right, bottom = plate.box
    fig = (left, top, left + (right - left) * 0.46, bottom - 40)
    if gate.key in ("span", "perp", "leave_one_out"):
        figures.anchor_figure(
            plate,
            fig,
            [c.pixel for c in cands],
            fit.sheet.full_size,
            mark_stray=gate.key == "leave_one_out",
        )
    else:
        figures.placement_figure(
            plate,
            fig,
            volume,
            fit.page,
            figures.sheet_quad(fit.sheet.full_size, fit.model),
            raw_model(cands),
        )
    col_left = fig[2] + 44
    theme.caps(plate.draw, (col_left, top), "the check", size=theme.TINY, fill=theme.SHEET)
    theme.text(plate.draw, (col_left, top + 26), gate.name, size=theme.LEAD, bold=True)
    y = theme.paragraph(
        plate.draw, (col_left, top + 76), gate.asks, right - col_left, size=theme.BODY
    )
    got, allowed = measured(gate, fit, cands, report)
    theme.card(plate.draw, (col_left, y + 22, right, y + 170))
    theme.text(
        plate.draw, (col_left + 24, y + 40), "measured", size=theme.TINY, fill=theme.INK_SOFT
    )
    theme.text(
        plate.draw, (col_left + 24, y + 64), got, size=theme.LEAD, fill=theme.ALERT, bold=True
    )
    theme.text(
        plate.draw, (col_left + 24, y + 112), "allowed", size=theme.TINY, fill=theme.INK_SOFT
    )
    theme.text(plate.draw, (col_left + 24, y + 134), allowed, size=theme.BODY, fill=theme.INK)
    theme.text(
        plate.draw, (col_left, y + 196), "REFUSED", size=theme.TITLE, fill=theme.ALERT, bold=True
    )
    theme.text(
        plate.draw,
        (col_left, y + 244),
        f"the matcher names it: {report['failure_class'].replace('_', ' ')}",
        size=theme.SMALL,
        fill=theme.INK_SOFT,
    )
    y = theme.paragraph(
        plate.draw,
        (col_left, y + 296),
        gate.why,
        right - col_left,
        size=theme.BODY,
        fill=theme.INK_SOFT,
    )
    if not windowed:
        theme.text(
            plate.draw,
            (col_left, bottom - 12),
            "shown with the scale window off: it is the only way this check can "
            "be the first to fail",
            size=theme.TINY,
            fill=theme.INK_SOFT,
        )
    del ex
    state = State(gate.key, gate.name, alt=f"{gate.name}: a counterexample refused.")
    out.save(panel, state, plate.image)
    return state
