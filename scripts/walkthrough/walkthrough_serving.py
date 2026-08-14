"""Acts IV to VI: the volume settles, becomes a map, and is graded afterwards."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import walkthrough_draw as sketch
import walkthrough_geo as geo
import walkthrough_theme as theme
from walkthrough_page import Emitter, Panel, State
from walkthrough_reading import column_head
from walkthrough_rescue import _quad
from walkthrough_sources import FidelityError

from autogeoref.affine import apply_affine, fit_affine_checked, gcps_from_geojson
from autogeoref.seam import MIN_SHIFT_M, SheetFit, build_ties, sheet_fit_from_result

if TYPE_CHECKING:
    from walkthrough_sources import Volume

Box = tuple[float, float, float, float]


def _fits(
    volume: Volume, *, shift: dict[str, tuple[float, float]] | None = None
) -> dict[str, SheetFit]:
    """Every committed sheet's placement, optionally moved by a per-page shift."""
    out: dict[str, SheetFit] = {}
    for page, record in volume.results.items():
        if not str(record.get("status", "")).startswith("OK"):
            continue
        moved = json.loads(json.dumps(record))
        if shift and page in shift:
            from autogeoref.seam import shift_gcps_geojson

            shift_gcps_geojson(moved["gcps_geojson"], *shift[page])
        made = sheet_fit_from_result(page, moved)
        if made is not None:
            out[page] = made
    return out


@dataclass(frozen=True)
class Seam:
    """One shared corner, and how far the two sheets put it apart either side."""

    page_a: str
    page_b: str
    node: tuple[float, float]
    before: float
    after: float


def _closed_seam(volume: Volume) -> Seam:
    """The shared corner the solve closed by the most.

    Both distances are measured, never assumed: the solve is a compromise
    across the whole volume, so a corner it moved a long way is not necessarily
    a corner it closed.
    """
    deltas_raw = json.loads(volume.paths.seam_deltas.read_text())["deltas"]
    undo = {page: (-d[0], -d[1]) for page, d in deltas_raw.items()}
    before = _fits(volume, shift=undo)
    after = _fits(volume)
    best = Seam("", "", (0.0, 0.0), 0.0, 0.0)
    for pi, (pxi, pyi), pj, (pxj, pyj) in build_ties(before):
        if pi not in deltas_raw or pj not in deltas_raw or pi not in after or pj not in after:
            continue
        xi, yi = apply_affine(before[pi].coef, pxi, pyi)
        xj, yj = apply_affine(before[pj].coef, pxj, pyj)
        gap = math.hypot(xj - xi, yj - yi)
        ai, bi = apply_affine(after[pi].coef, pxi, pyi), apply_affine(after[pj].coef, pxj, pyj)
        shut = math.hypot(bi[0] - ai[0], bi[1] - ai[1])
        if gap - shut > best.before - best.after:
            best = Seam(pi, pj, ((ai[0] + bi[0]) / 2, (ai[1] + bi[1]) / 2), gap, shut)
    return best


def panel_seam(volume: Volume, ex: dict[str, Any], out: Emitter) -> Panel:
    """16 - neighbouring sheets pulled into line with each other."""
    summary = json.loads(volume.paths.seam_deltas.read_text())
    seam = _closed_seam(volume)
    page_a, page_b, node = seam.page_a, seam.page_b, seam.node
    deltas = {p: (d[0], d[1]) for p, d in summary["deltas"].items()}
    undo = {p: (-d[0], -d[1]) for p, d in deltas.items()}
    panel = Panel(
        number=16,
        slug="seams",
        act="IV. The volume settles",
        title="Closing the seams",
        dek="Neighbouring sheets are nudged until the corners they share line up.",
        caption=(
            "Each sheet is placed on its own, so two neighbours can be wrong in "
            "opposite directions and the street that runs between them arrives at "
            "the shared edge twice, in two different places. The fix treats the "
            "whole volume at once. Every corner that appears on two sheets becomes "
            "a request that those two sheets agree there, and one calculation finds "
            "the small nudge for each sheet that satisfies as many requests as "
            "possible. Only sliding is allowed, with no sheet resized or turned, "
            "and a pull towards where each sheet already was keeps the volume "
            "anchored to its own evidence rather than drifting off together."
        ),
        figures=[
            ("Shared corners", str(summary.get("ties", 0))),
            ("Typical gap before (RMS)", f"{summary.get('rms_before_m', 0):.3g} m"),
            ("Typical gap after (RMS)", f"{summary.get('rms_after_m', 0):.3g} m"),
            (
                "Sheets nudged",
                str(sum(1 for d in deltas.values() if math.hypot(*d) >= MIN_SHIFT_M)),
            ),
            ("The corner shown, before", f"{seam.before:.3g} m apart"),
            ("The same corner, after", f"{seam.after:.3g} m apart"),
        ],
        stage="seam",
        note=(
            "Order note: a run does this BEFORE the neighbour check of step 13. "
            "It is told here because a seam is easier to see once the second "
            "chances are in view, but the comparison in step 13 is made against "
            "the tidied positions this step produces."
        ),
    )
    states = []
    for key, label, shift in (("before", "Before", undo), ("after", "After", None)):
        plate = theme.plate(16, "Closing the seams", kicker=label.lower())
        left, top, right, bottom = plate.box
        fig = (left, top, left + (right - left) * 0.58, bottom)
        canvas = geo.canvas_for(
            fig, (node[0] - 90, node[1] - 90, node[0] + 90, node[1] + 90), pad=0.02
        )
        view = canvas.view
        for page in (page_a, page_b):
            record = json.loads(json.dumps(volume.results[page]))
            if shift and page in shift:
                from autogeoref.seam import shift_gcps_geojson

                shift_gcps_geojson(record["gcps_geojson"], *shift[page])
            coef = fit_affine_checked(gcps_from_geojson(record["gcps_geojson"]))
            if coef is None:
                continue
            canvas.image.alpha_composite(
                geo.warp_sheet(volume.small(page), coef, volume.scale(page), view)
            )
        sketch.ring(canvas.draw, view.xy(*node), fill=theme.SHEET, radius=16, solid=False)
        geo.scale_bar(canvas.draw, view, (26, view.size[1] - 26), metres=50)
        canvas.commit(plate)
        col = (fig[2] + 40, top, right, bottom)
        y = column_head(
            plate,
            col,
            label,
            "Where two sheets meet, at the shared corner the solve closed by the "
            "most. The ring marks the one street corner both sheets claim.",
        )
        cl, _, cr, _ = col
        theme.card(plate.draw, (cl, y + 16, cr, y + 168))
        theme.text(
            plate.draw,
            (cl + 24, y + 36),
            "gap at this corner",
            size=theme.SMALL,
            fill=theme.INK_SOFT,
        )
        theme.text(
            plate.draw,
            (cl + 24, y + 62),
            f"{seam.before:.2f} m" if key == "before" else f"{seam.after:.2f} m",
            size=theme.TITLE,
            fill=theme.ALERT if key == "before" else theme.KEEP,
        )
        theme.text(
            plate.draw,
            (cl + 24, y + 122),
            f"pages {page_a} and {page_b}",
            size=theme.BODY,
            fill=theme.INK_SOFT,
        )
        theme.paragraph(
            plate.draw,
            (cl, y + 196),
            f"Across the whole volume the typical gap at a shared corner falls "
            f"from {summary.get('rms_before_m', 0):.3g} m to "
            f"{summary.get('rms_after_m', 0):.3g} m. Sheets that would move less "
            f"than {MIN_SHIFT_M:g} m are left alone.",
            cr - cl,
            size=theme.BODY,
        )
        state = State(key, label, alt=f"The shared edge {label.lower()} the seam solve.")
        states.append(state)
        out.save(panel, state, plate.image)
    panel.states = states
    del ex
    return panel


def panel_warp(volume: Volume, ex: dict[str, Any], out: Emitter) -> Panel:
    """17 - the placement becomes an image on the ground."""
    page = ex["pages"]["hero"]
    record = volume.results[page]
    coef = fit_affine_checked(gcps_from_geojson(record["gcps_geojson"]))
    if coef is None:
        raise FidelityError(f"p{page}: the recorded control points do not fix a placement")
    plate = theme.plate(17, "Onto the ground", kicker="warp")
    left, top, right, bottom = plate.box
    fig = (left, top, left + (right - left) * 0.58, bottom)
    quad = _quad(volume, page, sheet_fit_from_result(page, record))
    canvas = geo.canvas_for(fig, geo.bounds_of(quad, pad_m=140))
    view = canvas.view
    geo.draw_streets(canvas.draw, view, volume.features, fill=(206, 196, 176), width=2)
    canvas.image.alpha_composite(geo.warp_sheet(volume.small(page), coef, volume.scale(page), view))
    for feature in record["gcps_geojson"]["features"]:
        sketch.ring(
            canvas.draw,
            view.lnglat(*feature["geometry"]["coordinates"]),
            fill=theme.KEEP,
            radius=10,
        )
    geo.scale_bar(canvas.draw, view, (26, view.size[1] - 26))
    canvas.commit(plate)
    col = (fig[2] + 40, top, right, bottom)
    y = column_head(
        plate,
        col,
        "warp",
        "The accepted corners are handed to a standard mapping tool as control "
        "points, and it stretches the whole scan so those corners land where the "
        "modern map says they are. The result is an ordinary georeferenced image "
        "file, readable by any mapping software, not a format only this project "
        "understands.",
    )
    cl, _, _cr, _ = col
    sketch.legend(
        plate.draw,
        (cl, y + 16),
        [
            sketch.Key("ring", theme.KEEP, "a control point"),
            sketch.Key("line", (206, 196, 176), "modern street centrelines"),
        ],
    )
    plate.credit(ex["centerline_credit"])
    panel = Panel(
        number=17,
        slug="warp",
        act="V. Becoming a map",
        title="Onto the ground",
        dek="The accepted corners become control points, and the scan is stretched to fit them.",
        caption=(
            "Up to here the placement has been a handful of numbers. This is where "
            "it becomes an image sitting on the ground. The corners the fit kept "
            "are written out as control points and passed to standard mapping "
            "software, which stretches the scan so that every one of them lands on "
            "its modern coordinates. What comes out is a cloud-optimised GeoTIFF, "
            "a plain, widely readable format that any mapping tool can open, and "
            "that a web viewer can read a piece of at a time instead of "
            "downloading whole."
        ),
        figures=[
            ("Sheet", f"page {page}"),
            ("Control points written", str(len(record["gcps_geojson"]["features"]))),
        ],
        stage="warp",
    )
    state = State("main", "Placed", alt="The scan stretched onto modern street centrelines.")
    panel.states = [state]
    out.save(panel, state, plate.image)
    return panel


def panel_masks(volume: Volume, ex: dict[str, Any], out: Emitter) -> Panel:
    """18 - the printed margin is cut away so it cannot cover a neighbour."""
    page = ex["pages"]["hero"]
    neighbour = ex["pages"]["mask_neighbour"]
    cut = json.loads((volume.paths.masks / f"{volume.identifier}_p{page}.geojson").read_text())[
        "features"
    ][0]["geometry"]
    quads = []
    for other in (page, neighbour):
        fit = sheet_fit_from_result(other, volume.results[other])
        quads.extend(_quad(volume, other, fit))
    panel = Panel(
        number=18,
        slug="masks",
        act="V. Becoming a map",
        title="Trimming the margins",
        dek="Each sheet is cut back to its mapped area so it cannot paint over its neighbour.",
        caption=(
            "An atlas page is not all map. There is a printed border, a page "
            "number, a compass rose, a title block and a scale bar, and once the "
            "page is stretched onto the ground all of that lies over real streets, "
            "usually the ones the next sheet along is drawing properly. So each "
            "sheet gets a cutting outline that follows the edge of its mapped area, "
            "and everything outside it is made transparent. The furniture stays in "
            "the archived scan and disappears from the map."
        ),
        figures=[("Sheet", f"page {page}"), ("Neighbour", f"page {neighbour}")],
        stage="mask",
    )
    states = []
    for key, label, mask in (("before", "Whole page", None), ("after", "Trimmed", cut)):
        plate = theme.plate(18, "Trimming the margins", kicker=label.lower())
        left, top, right, bottom = plate.box
        fig = (left, top, left + (right - left) * 0.58, bottom)
        canvas = geo.canvas_for(fig, geo.bounds_of(quads, pad_m=60))
        view = canvas.view
        for other, cutline in ((neighbour, None), (page, mask)):
            record = volume.results[other]
            coef = fit_affine_checked(gcps_from_geojson(record["gcps_geojson"]))
            if coef is None:
                continue
            own = None
            if other == page:
                own = cutline
            elif key == "after":
                own = json.loads(
                    (volume.paths.masks / f"{volume.identifier}_p{other}.geojson").read_text()
                )["features"][0]["geometry"]
            canvas.image.alpha_composite(
                geo.warp_sheet(volume.small(other), coef, volume.scale(other), view, mask=own)
            )
        geo.polygon(canvas.draw, view, cut, outline=theme.SHEET, width=3)
        geo.scale_bar(canvas.draw, view, (26, view.size[1] - 26))
        canvas.commit(plate)
        col = (fig[2] + 40, top, right, bottom)
        column_head(
            plate,
            col,
            label,
            "Two neighbouring sheets, drawn one over the other. The orange outline "
            "is the cutting line for the upper sheet."
            if key == "before"
            else "The same two sheets with every margin cut away. The street the "
            "two of them share is drawn once, by whichever sheet actually maps it.",
        )
        state = State(key, label, alt=f"Two sheets, {label.lower()}.")
        states.append(state)
        out.save(panel, state, plate.image)
    panel.states = states
    return panel


def panel_mosaic(volume: Volume, ex: dict[str, Any], out: Emitter) -> Panel:
    """19 - the volume assembled, and packed for a browser."""
    quads: list[tuple[float, float]] = []
    for page, record in volume.results.items():
        if not str(record.get("status", "")).startswith("OK"):
            continue
        fit = sheet_fit_from_result(page, record)
        if fit is not None:
            quads.extend(_quad(volume, page, fit))
    plate = theme.plate(19, "One surface, cut into tiles", kicker="mosaic and tiles")
    left, top, right, bottom = plate.box
    fig = (left, top, left + (right - left) * 0.62, bottom)
    canvas = geo.canvas_for(fig, geo.bounds_of(quads, pad_m=200))
    view = canvas.view
    geo.draw_streets(canvas.draw, view, volume.features, fill=(214, 205, 187), width=1)
    canvas.image.alpha_composite(geo.tiles_image(volume.paths.root / "tiles", view))
    geo.scale_bar(canvas.draw, view, (26, view.size[1] - 26), metres=1000)
    canvas.commit(plate)
    params = json.loads((volume.paths.root / "tiles-params.json").read_text())
    col = (fig[2] + 40, top, right, bottom)
    y = column_head(
        plate,
        col,
        "mosaic and tiles",
        "Every accepted, trimmed sheet is composited into one continuous surface, "
        "and that surface is cut into square tiles at a range of zoom levels. The "
        "tiles are packed into a single file a browser can read pieces of over the "
        "network, so a visitor downloads the few squares they are looking at "
        "rather than a whole city.",
    )
    cl, _, cr, _ = col
    theme.card(plate.draw, (cl, y + 16, cr, y + 200))
    for i, (label, value) in enumerate(
        (
            (
                "sheets in the surface",
                str(
                    sum(
                        1
                        for r in volume.results.values()
                        if str(r.get("status", "")).startswith("OK")
                    )
                ),
            ),
            ("zoom levels", f"{params.get('min_zoom')} to {params.get('max_zoom')}"),
        )
    ):
        theme.text(
            plate.draw, (cl + 24, y + 40 + i * 78), value, size=theme.TITLE, fill=theme.WORLD
        )
        theme.text(plate.draw, (cl + 220, y + 56 + i * 78), label, size=theme.BODY)
    plate.credit(ex["centerline_credit"])
    panel = Panel(
        number=19,
        slug="mosaic",
        act="V. Becoming a map",
        title="One surface, cut into tiles",
        dek="Trimmed sheets are composited into a single layer and packed for the browser.",
        caption=(
            "This is the volume, assembled. Every accepted sheet has been trimmed "
            "to its mapped area and composited into one continuous surface, and "
            "the gaps are the sheets the pipeline refused. They are not filled in "
            "or approximated. The surface is then cut into square tiles at every "
            "zoom level a reader might want and packed into a single file that a "
            "web map reads a piece at a time. No server is involved: the file sits "
            "in storage and the browser asks for the bytes it needs."
        ),
        figures=[
            (
                "Sheets in the surface",
                str(
                    sum(
                        1
                        for r in volume.results.values()
                        if str(r.get("status", "")).startswith("OK")
                    )
                ),
            ),
            ("Zoom levels", f"{params.get('min_zoom')} to {params.get('max_zoom')}"),
        ],
        stage="tile",
    )
    state = State("main", "The volume", alt="The assembled volume over modern centrelines.")
    panel.states = [state]
    out.save(panel, state, plate.image)
    return panel
