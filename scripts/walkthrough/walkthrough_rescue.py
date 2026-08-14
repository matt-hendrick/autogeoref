"""Act III plates: the second chances a refused sheet still has."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import walkthrough_charts as charts
import walkthrough_draw as sketch
import walkthrough_geo as geo
import walkthrough_theme as theme
from walkthrough_page import Emitter, Panel, State
from walkthrough_reading import column_head, split

from autogeoref.affine import TO_3857
from autogeoref.corroborate import MIN_NODES, corroborations
from autogeoref.corroborate import TOL_M as CORR_TOL_M
from autogeoref.matching import candidate_gcps
from autogeoref.rescue import MIN_AGREE, TOL_M, pinned_linear, translation_fit
from autogeoref.seam import sheet_fit_from_result
from autogeoref.vouchers import committed_vouch_nodes

if TYPE_CHECKING:
    from walkthrough_sources import Volume

Box = tuple[float, float, float, float]


def _pinned(volume: Volume) -> tuple[float, float]:
    scale = volume.constraints.scale_median
    rotation = volume.constraints.rotation_median
    assert scale is not None and rotation is not None
    return scale, rotation


def _translations(volume: Volume, page: str) -> tuple[list[tuple[float, float, bool]], list[Any]]:
    """Every guess's implied shift, and the ones that agreed."""
    sheet = volume.sheets[page]
    scale, rotation = _pinned(volume)
    linear = pinned_linear(scale * volume.vol.page_scale_multiples.get(page, 1.0), rotation)
    cands = candidate_gcps(sheet.annotation, volume.index, sheet.scale, volume.index.aliases)
    model, anchors = translation_fit(
        cands, linear, require_disjoint=False, aliases=volume.index.aliases
    )
    offsets: list[tuple[float, float, bool]] = []
    for cand in cands:
        x, y = TO_3857.transform(*cand.world4326)
        ax = linear[0][0] * cand.pixel[0] + linear[0][1] * cand.pixel[1]
        ay = linear[1][0] * cand.pixel[0] + linear[1][1] * cand.pixel[1]
        offsets.append((x - ax, y - ay, cand in anchors))
    del model
    return offsets, anchors


def panel_rescue(volume: Volume, ex: dict[str, Any], out: Emitter) -> Panel:
    """11 - with the book's scale and turn pinned, two corners can be enough."""
    page = ex["pages"]["rescued"]
    record = volume.results[page]
    offsets, anchors = _translations(volume, page)
    scale, rotation = _pinned(volume)
    plate = theme.plate(11, "A second chance, on the book's terms", kicker="rescue")
    small = volume.small(page)
    sheet_box, col = split(plate, 0.30)
    sketch.place_sheet(plate, small, sheet_box, volume.scale(page))
    left, _top, right, bottom = col
    y = column_head(
        plate,
        col,
        "rescue",
        "A refused sheet is not finished with. Its neighbours in the same book "
        "have already been placed, and they fix the two hardest numbers: how big "
        "the page is on the ground and how far it is turned. Hold both still and "
        "only one question is left - where does the page slide to? That is a far "
        "easier question, and far weaker evidence answers it.",
    )
    side = min(bottom - y - 70, (right - left) * 0.48)
    far = charts.cluster_plot(
        plate.draw,
        (left, y + 20, left + side, y + 20 + side),
        offsets,
        TOL_M,
        centre=_centre(offsets),
    )
    theme.text(
        plate.draw,
        (left, y + 34 + side),
        "each corner guess, by where it would slide the page"
        + (f" ({far} land off this plot)" if far else ""),
        size=theme.SMALL,
        fill=theme.INK_SOFT,
    )
    mid = left + side + 40
    sketch.legend(
        plate.draw,
        (mid, y + 30),
        [
            sketch.Key("ring", theme.KEEP, f"agreed within {TOL_M:g} m"),
            sketch.Key("cross", theme.DROP, "slid the page somewhere else"),
        ],
    )
    theme.paragraph(
        plate.draw,
        (mid, y + 130),
        f"Corners that agree on the same slide are pointing at the same "
        f"placement. At least {MIN_AGREE} of them are needed, from at least two "
        f"different pairs of streets and two different places on the page.",
        right - mid,
        size=theme.BODY,
        fill=theme.INK,
    )
    panel = Panel(
        number=11,
        slug="rescue",
        act="III. Second chances",
        title="A second chance, on the book's terms",
        dek="With the volume's scale and turn held fixed, only a slide is left to find.",
        caption=(
            "The strict fit has to discover six things at once, two for position "
            "and four more for size, turn and stretch, so it insists on four "
            "well-spread corners: three would pin the six down exactly, with "
            "nothing left over to contradict a mistake. Once the rest of the "
            "volume has been placed, four of those six are already known and are "
            "simply held fixed. All that remains is how far the page slides north "
            "and east. Two corners that agree on the same slide are then real "
            "evidence rather than a coincidence, and sheets that could never have "
            "cleared the strict fit come back. This second attempt is called "
            "rescue. Some sheets are not drawn the way the rest of the book is: a "
            "district printed sideways, or on a local street grid at its own "
            "angle. Holding the book's turn fixed is then holding the wrong one, "
            "and nothing can agree, so when the book's turn finds nothing the "
            "page's own turn is tried instead, read off which of its street "
            "labels it calls horizontal and vertical. That is a weaker source "
            "than the book, so it is asked for more: three agreeing corners "
            "rather than two, and only ever where the book's own turn had "
            "already come up empty."
        ),
        figures=[
            ("Sheet", f"page {page}"),
            ("Pinned scale", f"{scale:.4g} m per pixel"),
            ("Pinned turn", f"{rotation:.3g} deg"),
            ("Corners that agreed", str(len(anchors))),
            ("Agreement tolerance", f"{TOL_M:g} m"),
            ("Status", str(record.get("status"))),
        ],
        stage="rescue",
    )
    state = State("main", "The slide", alt="Each guess plotted by the slide it implies.")
    panel.states = [state]
    out.save(panel, state, plate.image)
    return panel


def _centre(offsets: list[tuple[float, float, bool]]) -> tuple[float, float]:
    agreeing = [(x, y) for x, y, ok in offsets if ok]
    if not agreeing:
        return (0.0, 0.0)
    return (
        sum(p[0] for p in agreeing) / len(agreeing),
        sum(p[1] for p in agreeing) / len(agreeing),
    )


def panel_revoked(volume: Volume, ex: dict[str, Any], out: Emitter) -> Panel:
    """12 - corners strung along one street cannot tell one street from the next."""
    page = ex["pages"]["revoked"]
    record = volume.results[page]
    anchors = [tuple(a) for a in record.get("rescue_anchors") or []]
    shared = _shared_street(anchors)
    plate = theme.plate(12, "All on one street is not enough", kicker="revoked")
    small = volume.small(page)
    sheet_box, col = split(plate, 0.34)
    view = sketch.place_sheet(plate, small, sheet_box, volume.scale(page))
    fit = sheet_fit_from_result(page, record)
    if fit is not None:
        for px, py, _x, _y, synthetic in fit.gcps:
            if not synthetic:
                sketch.ring(plate.draw, view.full(px, py), fill=theme.PROV, radius=12)
    left, _top, right, bottom = col
    y = column_head(
        plate,
        col,
        "revoked",
        "Here the corners that agreed all sit on one street. They pin the page "
        "along that street perfectly well and say nothing at all about which "
        "street it is. Slide the whole page one block sideways onto the next "
        "parallel street and every one of those corners still agrees.",
    )
    y += 6
    for a, b in anchors:
        sketch.ring(plate.draw, (left + 12, y + 12), fill=theme.PROV, radius=8)
        theme.text(plate.draw, (left + 34, y), f"{a}  x  {b}", size=theme.BODY)
        y += 34
    theme.card(plate.draw, (left, y + 16, right, y + 140))
    theme.text(
        plate.draw,
        (left + 24, y + 36),
        "every corner shares",
        size=theme.SMALL,
        fill=theme.INK_SOFT,
    )
    theme.text(
        plate.draw, (left + 24, y + 62), shared, size=theme.LEAD, fill=theme.ALERT, bold=True
    )
    theme.text(
        plate.draw,
        (left + 24, y + 104),
        "so the placement is pulled back and held aside",
        size=theme.SMALL,
        fill=theme.INK_SOFT,
    )
    theme.paragraph(
        plate.draw,
        (left, bottom - 110),
        "The rule is blunt on purpose: at least two of the agreeing corners must "
        "share no street at all. A sheet that fails it keeps its proposed "
        "placement, marked as not yet believed, and waits for something else to "
        "confirm it.",
        right - left,
        size=theme.SMALL,
        fill=theme.INK_SOFT,
    )
    panel = Panel(
        number=12,
        slug="revoked",
        act="III. Second chances",
        title="All on one street is not enough",
        dek="Corners strung along a single street cannot tell it from the street next door.",
        caption=(
            "This is the failure the rescue step is most exposed to. A city grid "
            "is full of parallel streets a block apart, and a set of corners that "
            "all lie on one of them fits the street next door exactly as well. "
            "Nothing in the arithmetic can see the difference, so the answer is a "
            "rule about the evidence rather than a cleverer measurement: unless two "
            "of the agreeing corners share no street between them, the placement is "
            "withdrawn. A sheet in that state is called revoked. It is recorded as "
            "a proposal rather than discarded, and passed to the two steps that "
            "follow, which can settle it with evidence the matcher never saw."
        ),
        figures=[
            ("Sheet", f"page {page}"),
            ("Corners that agreed", str(len(anchors))),
            ("Street they all share", shared),
            ("Status", str(record.get("status"))),
        ],
        stage="rescue",
    )
    state = State("main", "One street", alt="Every agreeing corner sitting on the same street.")
    panel.states = [state]
    out.save(panel, state, plate.image)
    return panel


def _shared_street(anchors: list[tuple[str, str]]) -> str:
    if not anchors:
        return "-"
    common = set(anchors[0])
    for pair in anchors[1:]:
        common &= set(pair)
    return sorted(common)[0] if common else "-"


def panel_corroborate(volume: Volume, ex: dict[str, Any], out: Emitter) -> Panel:
    """13 - the neighbours already placed can settle it."""
    page = ex["pages"]["corroborated"]
    record = volume.results[page]
    fit = sheet_fit_from_result(page, record)
    nodes = committed_vouch_nodes(volume.paths)
    hits = corroborations(fit, nodes) if fit is not None else []
    best: dict[tuple[float, float], tuple[str, float]] = {}
    for key, neighbour, distance in hits:
        if key not in best or distance < best[key][1]:
            best[key] = (neighbour, distance)
    agreeing = sorted((d, n, k) for k, (n, d) in best.items() if d <= CORR_TOL_M)
    plate = theme.plate(13, "The neighbours already know", kicker="corroborate")
    left, top, right, bottom = plate.box
    map_box = (left, top, left + (right - left) * 0.54, bottom)
    _neighbour_map(
        volume,
        plate,
        map_box,
        page,
        [k for _, _, k in agreeing],
        sorted({n for _, n, _ in agreeing}),
    )
    col = (map_box[2] + 44, top, right, bottom)
    y = column_head(
        plate,
        col,
        "corroborate",
        "A street corner is one physical place. If the withheld sheet and a sheet "
        "already placed both claim the same corner, the two claims can simply be "
        "compared: warp each sheet by its own placement and see how far apart the "
        "two put that one corner.",
    )
    cl, _, cr, _ = col
    theme.card(plate.draw, (cl, y + 12, cr, y + 30 + 46 * max(len(agreeing), 1)))
    ry = y + 30
    for distance, neighbour, _key in agreeing:
        sketch.ring(plate.draw, (cl + 26, ry + 10), fill=theme.KEEP, radius=9)
        theme.text(
            plate.draw, (cl + 52, ry), f"shared corner with page {neighbour}", size=theme.BODY
        )
        theme.text(
            plate.draw,
            (cr - 20, ry),
            f"{distance:.1f} m apart",
            size=theme.BODY,
            fill=theme.KEEP,
            anchor="ra",
        )
        ry += 46
    y = ry + 24
    theme.paragraph(
        plate.draw,
        (cl, y),
        f"At least {MIN_NODES} shared corners agreeing within {CORR_TOL_M:g} m "
        f"puts the sheet back. The parallel-street mistake cannot pass this: a "
        f"sheet placed one block off holds corners on the wrong street, and its "
        f"true neighbours hold the right ones, so there is nothing to compare.",
        cr - cl,
        size=theme.BODY,
    )
    panel = Panel(
        number=13,
        slug="corroborate",
        act="III. Second chances",
        title="The neighbours already know",
        dek="A withheld sheet is reinstated when placed neighbours agree about a shared corner.",
        caption=(
            "Sheets in an atlas overlap at their edges, so the same street corner "
            "often appears on two of them. That is the opening. Take a sheet whose "
            "placement was withdrawn, find corners it shares with sheets already "
            "accepted, and measure how far apart the two placements put each one. "
            "This check is called corroboration, and agreement here cannot be "
            "arranged: a sheet "
            "placed one block off along a street simply has no corners in common "
            "with the sheets around it, so it has nothing to agree with and stays "
            "withheld."
        ),
        figures=[
            ("Sheet", f"page {page}"),
            ("Shared corners agreeing", str(len(agreeing))),
            ("Needed", f"{MIN_NODES} within {CORR_TOL_M:g} m"),
            ("Status", str(record.get("status"))),
        ],
        stage="corroborate",
        note=(
            "Order note: a run does this AFTER the seam solve of step 16, not "
            "before it. The neighbours have already been nudged into line with "
            "each other by then, so what is compared here is the tidied frame, "
            "which is the whole reason the seam solve runs first."
        ),
    )
    state = State(
        "main", "Shared corners", alt="A withheld sheet beside the neighbours that vouch for it."
    )
    panel.states = [state]
    out.save(panel, state, plate.image)
    return panel


def _neighbour_map(
    volume: Volume,
    plate: theme.Plate,
    box: Box,
    page: str,
    keys: list[tuple[float, float]],
    neighbours: list[str],
) -> None:
    """The sheet's placement over its committed neighbours, with the shared corners."""
    fits = {}
    for other, record in volume.results.items():
        made = sheet_fit_from_result(other, record)
        if made is not None:
            fits[other] = made
    here = fits[page]
    quad = _quad(volume, page, here)
    neighbours = [other for other in neighbours if other in fits]
    pts = list(quad)
    for other in neighbours:
        pts.extend(_quad(volume, other, fits[other]))
    canvas = geo.canvas_for(box, geo.bounds_of(pts, pad_m=90))
    view = canvas.view
    geo.draw_streets(canvas.draw, view, volume.features, fill=(206, 196, 176), width=2)
    for other in neighbours:
        _outline(canvas.draw, view, _quad(volume, other, fits[other]), theme.KEEP)
    _outline(canvas.draw, view, quad, theme.PROV)
    for key in keys:
        sketch.ring(canvas.draw, view.xy(*key), fill=theme.SHEET, radius=11)
    geo.scale_bar(canvas.draw, view, (26, view.size[1] - 26))
    sketch.legend(
        canvas.draw,
        (20, 18),
        [
            sketch.Key("line", theme.PROV, f"page {page}, withheld"),
            sketch.Key("line", theme.KEEP, "sheets already placed: " + ", ".join(neighbours)),
            sketch.Key("ring", theme.SHEET, "a corner both of them claim"),
        ],
    )
    canvas.commit(plate)


def _quad(volume: Volume, page: str, fit: Any) -> list[tuple[float, float]]:
    entry = volume.manifest.get(f"p{page}")
    if entry is None:
        return []
    w, h = entry["full_size"]
    coef = fit.coef
    return [
        (
            float(coef[0][0] + coef[0][1] * x + coef[0][2] * y),
            float(coef[1][0] + coef[1][1] * x + coef[1][2] * y),
        )
        for x, y in ((0, 0), (w, 0), (w, h), (0, h))
    ]


def _outline(draw: Any, view: geo.MapView, quad: list[tuple[float, float]], colour: Any) -> None:
    """One sheet's footprint on a map figure."""
    if not quad:
        return
    pts = [view.xy(*p) for p in quad]
    draw.line([c for p in pts for c in p] + list(pts[0]), fill=colour, width=3)
