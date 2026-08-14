"""Panel 9: the one placement most of the corner guesses agree on."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import walkthrough_draw as sketch
import walkthrough_geo as geo
import walkthrough_theme as theme
from walkthrough_page import Emitter, Panel, State
from walkthrough_reading import column_head, split

from autogeoref.affine import TO_3857

if TYPE_CHECKING:
    from walkthrough_sources import Volume

Box = tuple[float, float, float, float]


def panel_fit(volume: Volume, ex: dict[str, Any], out: Emitter) -> Panel:
    """9 - the one placement that most of the guesses agree on."""
    from autogeoref.matching import RANSAC_ITERS, RANSAC_MIN_INLIERS, RANSAC_TOL_M

    page = ex["pages"]["hero"]
    fit = volume.match(page)
    record = fit.record
    plate = theme.plate(9, "Which corners agree", kicker="the fit")
    small = volume.small(page)
    sheet_box, col = split(plate, 0.44)
    view = sketch.place_sheet(plate, small, sheet_box, volume.scale(page))
    kept = {(round(c.pixel[0]), round(c.pixel[1])) for c in fit.inliers}
    for cand in fit.dropped:
        px, py = round(cand.pixel[0]), round(cand.pixel[1])
        if (px, py) not in kept:
            sketch.cross(plate.draw, view.full(px, py), fill=theme.DROP, radius=11)
    for cand in fit.inliers:
        sketch.ring(plate.draw, view.full(*cand.pixel), fill=theme.KEEP, radius=12)
    left, _, right, bottom = col
    y = column_head(
        plate,
        col,
        "the fit",
        "The sheet has to be moved, turned and stretched onto the map in one "
        "single motion. The question is therefore not which corner guess is right, "
        "but which set of them can all be right together. Three guesses are picked "
        "at random, a placement is worked out from them, and every other guess is "
        "asked how far off it lands. Repeat thousands of times, from the same "
        "starting point every run, and keep the placement the most guesses agree "
        "with.",
    )
    y = sketch.legend(
        plate.draw,
        (left, y + 12),
        [
            sketch.Key("ring", theme.KEEP, f"agreed within {RANSAC_TOL_M:g} m and was kept"),
            sketch.Key("cross", theme.DROP, "could not be true at the same time"),
        ],
    )
    resid = record.get("auto_residuals_m") or []
    theme.card(plate.draw, (left, y + 12, right, y + 150))
    theme.text(
        plate.draw, (left + 26, y + 32), str(len(fit.inliers)), size=theme.TITLE, fill=theme.KEEP
    )
    theme.paragraph(
        plate.draw,
        (left + 120, y + 46),
        f"of {len(fit.candidates)} guesses survived, at most one per crossing",
        right - left - 140,
        size=theme.BODY,
    )
    if resid:
        theme.text(
            plate.draw,
            (left, y + 170),
            f"How far each kept corner sits from where the placement puts it: "
            f"{min(resid):.2g} m to {max(resid):.2g} m.",
            size=theme.SMALL,
            fill=theme.INK_SOFT,
        )
    _contest_inset(plate, (left, y + 212, right, bottom - 46), fit)
    theme.text(
        plate.draw,
        (left, bottom - 36),
        f"{RANSAC_ITERS:,} tries, fixed starting point, at least "
        f"{RANSAC_MIN_INLIERS} agreeing corners",
        size=theme.SMALL,
        fill=theme.INK_SOFT,
    )
    panel = Panel(
        number=9,
        slug="the-fit",
        act="II. From a reading to a fit",
        title="Which corners agree",
        dek="One placement is found by repeatedly guessing three corners and counting who agrees.",
        caption=(
            "A placement here is one flat transformation (slide, turn, stretch) "
            "applied to the whole sheet at once. Because it is one transformation, "
            "the corner guesses have to agree with each other, and most of the "
            "wrong ones cannot. The search is simple and repeatable: "
            "take three guesses at random, work out the placement they "
            f"imply, count how many of the rest land within {RANSAC_TOL_M:g} metres "
            f"of where that placement predicts, and remember the best. The random "
            "draws start from the same fixed point every run, so the same sheet "
            "always gives the same answer."
        ),
        figures=[
            ("Corner guesses", str(len(fit.candidates))),
            ("Corners kept", str(len(fit.inliers))),
            ("Agreement tolerance", f"{RANSAC_TOL_M:g} m"),
            ("Tries", f"{RANSAC_ITERS:,}"),
            ("Worst kept corner", f"{max(resid):.3g} m" if resid else "-"),
        ],
        stage="match",
    )
    state = State("main", "The fit", alt="Kept corners ringed, dropped guesses crossed out.")
    panel.states = [state]
    out.save(panel, state, plate.image)
    return panel


def _contest_inset(plate: theme.Plate, box: Box, fit: Any) -> None:
    """One crossing whose name pair matched several places, and how it was settled."""
    import math
    from collections import Counter

    from autogeoref.affine import apply_affine
    from autogeoref.matching import RANSAC_TOL_M

    counts = Counter((round(c.pixel[0]), round(c.pixel[1])) for c in fit.candidates)
    if fit.model is None:
        return

    def spread(pixel: tuple[int, int]) -> float:
        """How far apart the modern places one crossing matched actually are."""
        pts = [
            TO_3857.transform(*c.world4326)
            for c in fit.candidates
            if (round(c.pixel[0]), round(c.pixel[1])) == pixel
        ]
        return max((math.dist(a, b) for a in pts for b in pts), default=0.0)

    # the widest-spread contested crossing, so every possibility is its own mark
    contested = max((p for p, n in counts.items() if n > 1), key=spread, default=None)
    if contested is None:
        return
    many = counts[contested]
    options = [c for c in fit.candidates if (round(c.pixel[0]), round(c.pixel[1])) == contested]
    predicted = apply_affine(fit.model, *options[0].pixel)
    world = [TO_3857.transform(*c.world4326) for c in options] + [predicted]
    left, top, right, bottom = box
    theme.text(
        plate.draw,
        (left, top),
        f"{options[0].streets[0]} x {options[0].streets[1]}: {many} places on the modern map",
        size=theme.SMALL,
        fill=theme.INK,
        bold=True,
    )
    away = [math.dist(TO_3857.transform(*c.world4326), predicted) for c in options]
    near = sum(1 for d in away if d <= RANSAC_TOL_M)
    theme.text(
        plate.draw,
        (left, top + 28),
        f"The placement predicts it inside the ring. {near} of the {many} land there, "
        f"almost on top of each other; the furthest is {max(away):.0f} m away.",
        size=theme.SMALL,
        fill=theme.INK_SOFT,
    )
    canvas = geo.canvas_for(
        (left, top + 60, right, bottom),
        geo.bounds_of(world, pad_m=RANSAC_TOL_M * 1.6),
    )
    view = canvas.view
    here = view.xy(*predicted)
    reach = RANSAC_TOL_M * view.ppm
    canvas.draw.ellipse(
        (here[0] - reach, here[1] - reach, here[0] + reach, here[1] + reach),
        outline=theme.KEEP,
        width=2,
    )
    for cand in options:
        at = view.lnglat(*cand.world4326)
        if cand in fit.inliers:
            sketch.ring(canvas.draw, at, fill=theme.KEEP, radius=9)
        else:
            sketch.cross(canvas.draw, at, fill=theme.DROP, radius=8)
    geo.scale_bar(canvas.draw, view, (24, view.size[1] - 20), metres=50)
    canvas.commit(plate)
