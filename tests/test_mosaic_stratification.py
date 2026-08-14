"""Mask extent detection, duplicate-coverage stratification, and declared hulls.

A regular sheet is bounded by the box of its printed colour; a duplicate-
coverage sheet (skeleton twin, overview) is not, must not Voronoi-compete with
the pages it duplicates — the splitter would halve both along a mid-sheet
diagonal — and must paint under them in the mosaic. A volume declaring
``content_masks`` masks each regular sheet by its colored-content hull instead.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw
from shapely.geometry import Point, Polygon, box, shape
from shapely.ops import transform as shp_transform

from autogeoref.affine import TO_3857, TO_4326
from autogeoref.bake.masks import expand_content_boxes, raw_sheet_mask, stage_masks
from autogeoref.bake.mosaic import stage_mosaic
from autogeoref.bake.warp import stage_warp
from autogeoref.config.load import load_city_config
from autogeoref.mask.geometry import (
    clip_to_gcp_hull,
    detect_content_box,
    detect_content_hull,
    detect_page_bounds,
    mask_polygon_4326,
)
from autogeoref.mask.qa import RawSheetMask
from autogeoref.paths import VolumePaths
from autogeoref.slugs import DuplicateCoverage, mosaic_paint_order
from conftest import ANTEDATED, antedated

VOL = "volY"
# Larger than the detector's 1200 work width so it downscales, as real scans do
W, H = 1800, 1200
M_PER_PX = 2.0 / 3.0
X0, Y0 = -9760000.0, 5141000.0
SHEET2_OFFSET_M = 1000.0

PAPER = (232, 224, 200)
PINK = (230, 160, 170)
BLOCK = (600, 360, 1200, 840)  # the one detailed block on the one-block sheet
DOT = (300, 900, 315, 915)
STAMP = (1500, 180, 1560, 198)
#: The LOC scans carry a solid black scanner bezel around the page; before the
#: bezel-skip fix, the page-bounds search stopped on it at index 0 on every sheet.
BEZEL_PX = 24


def _canvas(bezel: bool = False) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Blank paper, optionally inside the scanner bezel.

    NO printed frame is drawn, because no real plate carries one: the map's
    edge is the outermost block frontage line. What the search
    can still stop on is long dark LINEWORK, which is a real thing these
    plates do — see the `_broken_column` note below for which branch each
    fixture is deliberately on.
    """
    img = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(img)
    if bezel:
        draw.rectangle([0, 0, W - 1, H - 1], outline=(0, 0, 0), width=BEZEL_PX)
    return img, draw


def _one_block_sheet(path: Path) -> None:
    """One colored block, plus the junk a hull must ignore: a dot, a stamp."""
    img, draw = _canvas()
    draw.rectangle(BLOCK, fill=PINK)
    draw.ellipse(DOT, fill=(200, 60, 60))
    draw.rectangle(STAMP, fill=(200, 60, 60))
    img.save(path, "PNG")


#: Sanborn linework stops at every cross street, so on 97 of 152 censused sheets the page-bounds
#: search returns the same rect as its post-bezel fallback
#: by a census of the corpus. The fixtures below draw broken linework so
#: they sit on that majority branch; :func:`test_page_bounds_stops_on_a_long_dark_run` covers
#: the 55 where a run moves the answer.
_RUN_PX, _GAP_PX = 190, 50


def _broken_column(
    draw: ImageDraw.ImageDraw, x: int, fill: tuple[int, int, int], width: int
) -> None:
    for y in range(90, H - 90, _RUN_PX + _GAP_PX):
        draw.line([x, y, x, min(y + _RUN_PX, H - 90)], fill=fill, width=width)


def _colorless_sheet(path: Path) -> None:
    """Grayscale linework only — no colored content to hull."""
    img, draw = _canvas()
    for x in range(180, W - 120, 180):
        _broken_column(draw, x, (90, 90, 90), 6)
    img.save(path, "PNG")


def _scattered_content_sheet(path: Path) -> None:
    """Colored patches in every corner of the PAGE: the hull spans the frame."""
    img, draw = _canvas()
    for cx, cy in [
        (20, 20),
        (W - 140, 20),
        (20, H - 140),
        (W - 140, H - 140),
        (W // 2, H // 2),
    ]:
        draw.rectangle([cx, cy, cx + 120, cy + 120], fill=PINK)
    img.save(path, "PNG")


def _bezelled_sheet(path: Path) -> None:
    """One colored block on a page inside a solid black scanner bezel."""
    img, draw = _canvas(bezel=True)
    draw.rectangle(BLOCK, fill=PINK)
    img.save(path, "PNG")


def _sparse_sheet(path: Path) -> None:
    """A single small colored patch — the hull-collapse hazard, mid-page so the
    per-dimension floor can expand around it without running off the sheet."""
    img, draw = _canvas()
    draw.rectangle([840, 540, 960, 660], fill=PINK)
    img.save(path, "PNG")


# -------------------------------------------------------- page bounds unit ----


def test_page_bounds_skips_the_scanner_bezel(tmp_path: Path) -> None:
    """A bezelled sheet must return its PAGE, not the scan.

    The bezel clears every dark-run bar at index 0, so before the bezel skip
    both the hit branch and the ``else 0`` fallback returned the whole scan,
    bezel included. This sheet's own linework clears no bar, so it takes the
    post-bezel fallback — the majority branch on the corpus, and the one that
    makes the answer the page. Regular sheets are bounded by the colour box
    now, but this rectangle is still the no-colour fallback and mask QA's
    denominator.
    """
    sheet = tmp_path / "bezelled.png"
    _bezelled_sheet(sheet)
    x0, y0, x1, y1 = detect_page_bounds(sheet)
    # just inside the bezel on every side — the pad is the only inset left
    assert BEZEL_PX <= x0 < BEZEL_PX + 20
    assert BEZEL_PX <= y0 < BEZEL_PX + 20
    assert W - BEZEL_PX - 20 < x1 <= W - BEZEL_PX
    assert H - BEZEL_PX - 20 < y1 <= H - BEZEL_PX
    # and what is left is the page: the bezel is all that was dropped
    assert (x1 - x0) * (y1 - y0) > 0.90 * W * H


def test_page_bounds_stops_on_a_long_dark_run(tmp_path: Path) -> None:
    """The other branch: a dark run over ``run_frac`` of the cross dimension
    stops the search, and the rectangle is then a TRUNCATED page.

    This is not a hypothetical path and not a printed frame — it moves the answer on 55 of 152
    censused sheets, driven by long dark linework, a rail corridor or a solid block edge, and
    takes 18% off `_004` p118's left side. It is
    pinned here because the shipped rectangle is mask QA's denominator, so a regression that
    widened the bar's reach would quietly shorten that denominator, and because no other fixture
    in this file exercises the branch at all.
    """
    sheet = tmp_path / "long_run.png"
    img, draw = _canvas()
    bar_x = 300
    draw.line([bar_x, 40, bar_x, H - 40], fill=(20, 20, 20), width=10)
    img.save(sheet, "PNG")
    x0, y0, x1, y1 = detect_page_bounds(sheet)
    # the left search stopped ON the bar, well inside the page
    assert bar_x - 20 < x0 < bar_x + 20
    # the bar is a single column, so no other side has anything to stop on
    assert x1 > W - 20
    assert y0 < 20 and y1 > H - 20
    # a control: break the same bar and the left edge returns to the page
    broken = tmp_path / "broken_run.png"
    img, draw = _canvas()
    _broken_column(draw, bar_x, (20, 20, 20), 10)
    img.save(broken, "PNG")
    assert detect_page_bounds(broken)[0] < 20


# ------------------------------------------------------- content box unit ----


def test_content_box_bounds_the_printed_colour_with_a_pad(tmp_path: Path) -> None:
    sheet = tmp_path / "one_block.png"
    _one_block_sheet(sheet)
    rect = detect_page_bounds(sheet)
    found = detect_content_box(sheet, rect)
    assert found is not None
    x0, y0, x1, y1 = found
    bx0, by0, bx1, by1 = BLOCK
    # the colored block is inside, padded outward
    assert x0 < bx0 and y0 < by0 and x1 > bx1 and y1 > by1
    # vertically the pad is the whole story (the floor does not bind there):
    # a pad measured as a fraction of the rectangle's WIDTH, both sides
    pad = 0.08 * (rect[2] - rect[0])
    assert by0 - y0 == pytest.approx(pad, rel=0.1)
    assert y1 - by1 == pytest.approx(pad, rel=0.1)
    # horizontally the 60% floor binds and widens it further
    assert bx0 - x0 > pad
    assert x1 - x0 == pytest.approx(0.60 * (rect[2] - rect[0]), rel=0.02)
    # and it is a real bound, not the page
    assert (x1 - x0) * (y1 - y0) < 0.6 * (rect[2] - rect[0]) * (rect[3] - rect[1])


def test_content_box_floor_stops_a_sparse_sheet_collapsing(tmp_path: Path) -> None:
    """A bbox of one small patch is expanded to the per-dimension floor.

    Without the floor a sheet drawn sparsely — one coloured cluster in an otherwise mapped frame
    — would mask down to that cluster and stop serving the rest. The floor is expanded about the
    box centre and then clipped to the page, so it delivers less than 60% only where the page
    runs out. ``min_ink_kept=0`` isolates the floor from the own-ink guard, so no change to the
    guard can quietly turn these into assertions about the guard. The patch is this sheet's only
    ink and the floor contains it, so the guard passes here either way.
    """
    sheet = tmp_path / "sparse.png"
    _sparse_sheet(sheet)
    rect = detect_page_bounds(sheet)
    found = detect_content_box(sheet, rect, min_ink_kept=0.0)
    assert found is not None
    x0, y0, x1, y1 = found
    assert x1 - x0 >= 0.60 * (rect[2] - rect[0]) - 2
    assert y1 - y0 >= 0.60 * (rect[3] - rect[1]) - 2


def test_content_box_none_without_colored_content(tmp_path: Path) -> None:
    """No colour, no box — the caller keeps the page rectangle."""
    sheet = tmp_path / "colorless.png"
    _colorless_sheet(sheet)
    assert detect_content_box(sheet, detect_page_bounds(sheet)) is None


def test_content_box_refused_when_it_would_cut_the_sheets_own_ink(tmp_path: Path) -> None:
    """Colour is not where the map is on every era.

    A sheet whose colour sits in one corner but whose LINEWORK covers the page
    — the later Sanborn eras — must keep its rectangle: a colour box there
    serves less of the drawn map than the page does, and nothing downstream
    puts it back. Same sheet with the guard relaxed still yields a box, so the
    refusal is the guard and not a missing colour signal.
    """
    sheet = tmp_path / "line_drawn.png"
    img, draw = _canvas()
    draw.rectangle([840, 540, 960, 660], fill=PINK)
    for x in range(120, W - 100, 60):
        _broken_column(draw, x, (40, 40, 40), 10)
    img.save(sheet, "PNG")
    rect = detect_page_bounds(sheet)
    assert detect_content_box(sheet, rect) is None
    assert detect_content_box(sheet, rect, min_ink_kept=0.0) is not None


def test_content_box_never_exceeds_the_page_rect(tmp_path: Path) -> None:
    """The pad is clipped to the rectangle, however generous it is."""
    sheet = tmp_path / "one_block.png"
    _one_block_sheet(sheet)
    rect = detect_page_bounds(sheet)
    found = detect_content_box(sheet, rect, pad_frac=1.0)
    assert found == rect


# ------------------------------------------------------ content hull unit ----


def test_content_hull_hugs_the_block_and_ignores_junk(tmp_path: Path) -> None:
    sheet = tmp_path / "one_block.png"
    _one_block_sheet(sheet)
    rect = detect_page_bounds(sheet)
    ring = detect_content_hull(sheet, rect)
    assert ring is not None
    hull = Polygon(ring)
    x0, y0, x1, y1 = BLOCK
    assert hull.covers(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
    rx0, ry0, rx1, ry1 = rect
    assert hull.area < 0.38 * (rx1 - rx0) * (ry1 - ry0)
    assert not hull.covers(Point(307, 907))  # the dot
    assert not hull.covers(Point(1530, 189))  # the stamp


def test_content_hull_none_without_colored_content(tmp_path: Path) -> None:
    sheet = tmp_path / "colorless.png"
    _colorless_sheet(sheet)
    assert detect_content_hull(sheet, detect_page_bounds(sheet)) is None


def test_content_hull_none_when_content_spans_the_frame(tmp_path: Path) -> None:
    sheet = tmp_path / "scattered.png"
    _scattered_content_sheet(sheet)
    assert detect_content_hull(sheet, detect_page_bounds(sheet)) is None


# ------------------------------------------------------------- gcp hull clip ----


def test_clip_to_gcp_hull_keeps_constrained_ground_only() -> None:
    mask = box(0.0, 0.0, 1000.0, 1000.0)
    pts = [(100.0, 100.0), (400.0, 100.0), (400.0, 400.0), (100.0, 400.0)]
    clipped = clip_to_gcp_hull(mask, pts, margin_m=10.0)
    assert clipped.covers(Point(250, 250))
    assert not clipped.covers(Point(600, 600))
    assert clipped.area < 0.2 * mask.area


def test_clip_to_gcp_hull_keeps_mask_whole_on_degenerate_hull() -> None:
    mask = box(0.0, 0.0, 1000.0, 1000.0)
    assert clip_to_gcp_hull(mask, [(1.0, 1.0), (2.0, 2.0)]).equals(mask)
    assert clip_to_gcp_hull(mask, [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]).equals(mask)


def test_overview_hull_ignores_the_rescue_model_corners(tmp_path: Path) -> None:
    """The hull is the ground the EVIDENCE earned, so synthetics cannot vote.

    Since the always-corners change every rescue record carries three model corners at
    10%/90% of the page. Counted as anchors they would define the hull for a
    rescued overview sheet — a fixed ~32%-of-page triangle bearing no relation
    to what the sheet actually matched, and wider than a tight real hull.
    """
    from autogeoref.rescue import SYNTHETIC_STREETS

    page_w, page_h = 1800.0, 1200.0

    def feature(px: float, py: float, note: str | None) -> dict[str, Any]:
        lng, lat = TO_4326.transform(X0 + px * M_PER_PX, Y0 - py * M_PER_PX)
        props: dict[str, Any] = {"image": [px, py]}
        if note is not None:
            props["note"] = note
        return {
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Point", "coordinates": [lng, lat]},
        }

    # four real anchors in one corner of the page...
    anchors = [
        feature(200, 200, "auto: A x B"),
        feature(500, 200, "auto: C x D"),
        feature(500, 500, "auto: E x F"),
        feature(200, 500, "auto: G x H"),
    ]
    corner_note = f"auto: {SYNTHETIC_STREETS[0]} x {SYNTHETIC_STREETS[1]}"
    corners = [
        feature(page_w * 0.1, page_h * 0.1, corner_note),
        feature(page_w * 0.9, page_h * 0.1, corner_note),
        feature(page_w * 0.1, page_h * 0.9, corner_note),
    ]

    image = tmp_path / f"{VOL}_pcbd1.jpg"
    _one_block_sheet(image)

    def masked(features: list[dict[str, Any]]) -> float:
        record = {
            "page": "cbd1",
            "status": "OK (rescued)",
            "gcps_geojson": {"type": "FeatureCollection", "features": features},
        }
        return raw_sheet_mask(
            image,
            f"{VOL}_pcbd1",
            "cbd1",
            record,
            content_masks=False,
            content_mask_exempt=(),
            duplicates=DuplicateCoverage(frozenset({"cbd1"})),
        ).poly_3857.area

    anchors_only = masked(anchors)
    with_corners = masked(anchors + corners)
    assert with_corners == pytest.approx(anchors_only), (
        "the model corners widened the overview hull — they are not evidence"
    )


# -------------------------------------------------------------- paint order ----


def test_mosaic_paint_order_puts_duplicate_coverage_first() -> None:
    slugs = ["v_p13S", "v_p2", "v_pcbd1", "v_p13", "v_p1"]
    declared = ("cbd1",)
    twins = DuplicateCoverage.resolve(("1", "2", "13", "13S", "cbd1"), declared)
    regular = ["v_p1", "v_p2", "v_p13"]
    assert mosaic_paint_order(slugs, twins) == ["v_p13S", "v_pcbd1", *regular]
    # an undeclared named page is a regular sheet and keeps page order
    undeclared = DuplicateCoverage(skeletons=twins.skeletons)
    assert mosaic_paint_order(slugs, undeclared) == ["v_p13S", *regular, "v_pcbd1"]
    # and with no numeric twin in the volume, p13S is a regular sheet too: it
    # sorts with the rest of the tail instead of painting underneath
    assert mosaic_paint_order(slugs, DuplicateCoverage(frozenset(declared))) == [
        "v_pcbd1",
        *regular,
        "v_p13S",
    ]


def test_content_masks_is_a_per_volume_declaration() -> None:

    cfg = load_city_config(
        Path(__file__).resolve().parent.parent / "configs" / "chicago" / "chicago.toml"
    )
    assert cfg.volume("sanborn01790_017").content_masks is True
    assert cfg.volume("sanborn01790_018").content_masks is True
    assert cfg.volume("sanborn01790_034").content_masks is False
    assert cfg.volume("sanborn01790_999").content_masks is False  # undeclared default


def test_content_mask_exempt_is_a_per_page_declaration() -> None:

    cfg = load_city_config(
        Path(__file__).resolve().parent.parent / "configs" / "chicago" / "chicago.toml"
    )
    # the sparse rail-yard sheets keep page rectangles
    assert cfg.volume("sanborn01790_018").content_mask_exempt == ("57", "59", "63")
    assert cfg.volume("sanborn01790_017").content_mask_exempt == ()


# ------------------------------------------------------------ staged (gdal) ----


def _gcps_fc(x0: float, y0: float) -> dict[str, Any]:
    feats = []
    for px, py in [(0, 0), (W, 0), (W, H), (0, H)]:
        lng, lat = TO_4326.transform(x0 + px * M_PER_PX, y0 - py * M_PER_PX)
        feats.append(
            {
                "type": "Feature",
                "properties": {"image": [px, py]},
                "geometry": {"type": "Point", "coordinates": [lng, lat]},
            }
        )
    return {"type": "FeatureCollection", "features": feats}


def _result(page: str, x0: float, y0: float) -> dict[str, Any]:
    return {"page": page, "status": "OK", "gcps_geojson": _gcps_fc(x0, y0)}


@pytest.fixture(scope="module")
def volume(tmp_path_factory: pytest.TempPathFactory) -> VolumePaths:
    """p1 one-block sheet, p2 colorless neighbour, p1S skeleton twin of p1."""
    root = tmp_path_factory.mktemp("stratified") / VOL
    paths = VolumePaths(root=root)
    paths.regions.mkdir(parents=True)
    paths.results.mkdir(parents=True)
    _one_block_sheet(paths.regions / f"{VOL}_p1.png")
    _colorless_sheet(paths.regions / f"{VOL}_p2.png")
    _colorless_sheet(paths.regions / f"{VOL}_p1S.png")
    (paths.results / "p1.json").write_text(json.dumps(_result("1", X0, Y0)))
    (paths.results / "p2.json").write_text(json.dumps(_result("2", X0 + SHEET2_OFFSET_M, Y0)))
    (paths.results / "p1S.json").write_text(json.dumps(_result("1S", X0, Y0)))
    stage_warp(paths, VOL)
    return paths


@pytest.mark.gdal
def test_duplicate_coverage_sheet_is_not_bisector_halved(volume: VolumePaths) -> None:
    fc = json.loads(stage_masks(volume, VOL).read_text())
    masks = {
        f["properties"]["slug"]: shp_transform(TO_3857.transform, shape(f["geometry"]))
        for f in fc["features"]
    }
    p1, p1s, p2 = masks[f"{VOL}_p1"], masks[f"{VOL}_p1S"], masks[f"{VOL}_p2"]
    # the twins coincide: neither may lose its half to the other
    assert p1.intersection(p1s).area > 0.9 * min(p1.area, p1s.area)
    # regular neighbours still resolve their seam
    assert p1.intersection(p2).area < 1.0


@pytest.mark.gdal
def test_mosaic_paints_duplicate_coverage_under_regular_sheets(volume: VolumePaths) -> None:
    stage_masks(volume, VOL)
    stage_mosaic(volume)
    parts = json.loads((volume.root / "mosaic-parts" / "parts.json").read_text())
    assert parts == [f"{VOL}_p1S.vrt", f"{VOL}_p1.vrt", f"{VOL}_p2.vrt"]


def _mask_areas(fc: dict[str, Any]) -> dict[str, float]:
    return {
        f["properties"]["slug"]: shp_transform(TO_3857.transform, shape(f["geometry"])).area
        for f in fc["features"]
    }


@pytest.mark.gdal
def test_regular_sheets_default_to_their_colour_box(volume: VolumePaths) -> None:
    """The default mask is the printed-colour box, not the scanned page.

    A colorless regular sheet has no box to take and keeps its rectangle; a
    duplicate-coverage sheet is fallback coverage and is never colour-bounded.
    """
    areas = _mask_areas(json.loads(stage_masks(volume, VOL).read_text()))
    # p1S is drawn on the same canvas and keeps its whole rectangle, so it is
    # the page-rectangle yardstick p1's colour box is measured against

    assert areas[f"{VOL}_p1"] < 0.6 * areas[f"{VOL}_p1S"]
    assert areas[f"{VOL}_p2"] > 0.9 * areas[f"{VOL}_p1S"]


@pytest.mark.gdal
def test_declared_content_masks_trim_only_regular_colored_sheets(volume: VolumePaths) -> None:
    before = _mask_areas(json.loads(stage_masks(volume, VOL).read_text()))
    after = _mask_areas(json.loads(stage_masks(volume, VOL, content_masks=True).read_text()))

    # the declared hull is tighter still than the one-block sheet's colour box
    assert after[f"{VOL}_p1"] < 0.6 * before[f"{VOL}_p1"]
    # a colorless regular sheet and a duplicate-coverage sheet keep page rects
    assert after[f"{VOL}_p2"] > 0.9 * before[f"{VOL}_p2"]
    assert after[f"{VOL}_p1S"] > 0.9 * before[f"{VOL}_p1S"]


@pytest.mark.gdal
def test_content_mask_exempt_page_keeps_its_page_rect(
    volume: VolumePaths, caplog: pytest.LogCaptureFixture
) -> None:
    """An exempt page drops EVERY colour bound, the box as well as the hull.

    The declaration names sheets whose drawn ground a colour bound chops, so
    honouring it by substituting a narrower colour bound would be no remedy.
    """
    hulled = _mask_areas(json.loads(stage_masks(volume, VOL, content_masks=True).read_text()))
    plain = _mask_areas(json.loads(stage_masks(volume, VOL).read_text()))
    with caplog.at_level(logging.WARNING, logger="autogeoref.bake.masks"):
        exempt = _mask_areas(
            json.loads(
                stage_masks(
                    volume, VOL, content_masks=True, content_mask_exempt=("1", "999")
                ).read_text()
            )
        )

    # p1S is the same canvas, never colour-bounded: the page-rect yardstick
    assert exempt[f"{VOL}_p1"] > 0.9 * exempt[f"{VOL}_p1S"]
    assert exempt[f"{VOL}_p1"] > 1.5 * hulled[f"{VOL}_p1"]
    assert exempt[f"{VOL}_p1"] > 1.5 * plain[f"{VOL}_p1"]
    # an exempt page matching no committed sheet is named out loud: silence
    # here is the hull collapse the exemption exists to prevent
    assert any("999" in record.message for record in caplog.records)


def test_stage_masks_rejects_a_bare_string_exempt(tmp_path: Path) -> None:
    # a str is a Collection[str] of its characters — "57" must not silently
    # exempt pages 5 and 7
    with pytest.raises(TypeError):
        stage_masks(VolumePaths(root=tmp_path), VOL, content_mask_exempt="57")


# ---------------------------------------------------- staged overview clip ----

OVOL = "volO"
# GCPs constrain only the left half of the overview sheet; the right half is
# extrapolation the clipped mask must drop.
OVERVIEW_GCP_PX = [(0, 0), (W // 2, 0), (W // 2, H), (0, H)]


def _overview_gcps_fc(x0: float, y0: float) -> dict[str, Any]:
    feats = []
    for px, py in OVERVIEW_GCP_PX:
        lng, lat = TO_4326.transform(x0 + px * M_PER_PX, y0 - py * M_PER_PX)
        feats.append(
            {
                "type": "Feature",
                "properties": {"image": [px, py]},
                "geometry": {"type": "Point", "coordinates": [lng, lat]},
            }
        )
    return {"type": "FeatureCollection", "features": feats}


@pytest.fixture(scope="module")
def overview_volume(tmp_path_factory: pytest.TempPathFactory) -> VolumePaths:
    """p1 detail sheet plus a pcbd1 overview constrained only on its left half.

    p1 is drawn with content in every corner so its colour box spans the frame
    — the clip under test is the OVERVIEW's, and a regular sheet's own extent
    must not be what makes the assertions pass.
    """
    root = tmp_path_factory.mktemp("overview") / OVOL
    paths = VolumePaths(root=root)
    paths.regions.mkdir(parents=True)
    paths.results.mkdir(parents=True)
    _scattered_content_sheet(paths.regions / f"{OVOL}_p1.png")
    _colorless_sheet(paths.regions / f"{OVOL}_pcbd1.png")
    (paths.results / "p1.json").write_text(json.dumps(_result("1", X0, Y0)))
    (paths.results / "pcbd1.json").write_text(
        json.dumps({"page": "cbd1", "status": "OK", "gcps_geojson": _overview_gcps_fc(X0, Y0)})
    )
    stage_warp(paths, OVOL)
    return paths


def _masks_3857(collection_path_text: str) -> dict[str, Any]:
    fc = json.loads(collection_path_text)
    return {
        f["properties"]["slug"]: shp_transform(TO_3857.transform, shape(f["geometry"]))
        for f in fc["features"]
    }


INSIDE_HULL = Point(X0 + (W // 4) * M_PER_PX, Y0 - (H // 2) * M_PER_PX)
BEYOND_HULL = Point(X0 + (3 * W // 4) * M_PER_PX, Y0 - (H // 2) * M_PER_PX)


@pytest.mark.gdal
def test_overview_mask_is_clipped_to_its_inlier_gcp_hull(overview_volume: VolumePaths) -> None:
    masks = _masks_3857(stage_masks(overview_volume, OVOL, overview_pages=("cbd1",)).read_text())
    cbd = masks[f"{OVOL}_pcbd1"]
    # constrained ground stays; ground beyond the GCP hull is dropped
    assert cbd.covers(INSIDE_HULL)
    assert not cbd.covers(BEYOND_HULL)
    # the regular sheet is not clipped: its page rect still spans the frame
    assert masks[f"{OVOL}_p1"].covers(BEYOND_HULL)
    # the production combination — content_masks declared — clips the same way
    hulled = _masks_3857(
        stage_masks(overview_volume, OVOL, content_masks=True, overview_pages=("cbd1",)).read_text()
    )
    assert hulled[f"{OVOL}_pcbd1"].covers(INSIDE_HULL)
    assert not hulled[f"{OVOL}_pcbd1"].covers(BEYOND_HULL)


@pytest.mark.gdal
def test_undeclared_named_page_is_a_regular_sheet(overview_volume: VolumePaths) -> None:
    # the overview class is declared per volume, never derived from the id:
    # without the declaration the named sheet is a regular sheet — no hull
    # clip, no "overview" style (raw-mask level, where the class decision
    # lives; the split downstream is a regular-sheet competition either way)
    from autogeoref.paths import regions_by_page

    image = regions_by_page(overview_volume.regions)["cbd1"]
    record = json.loads((overview_volume.results / "pcbd1.json").read_text())

    def raw(declared: tuple[str, ...]) -> Any:
        return raw_sheet_mask(
            image,
            f"{OVOL}_pcbd1",
            "cbd1",
            record,
            content_masks=False,
            content_mask_exempt=(),
            duplicates=DuplicateCoverage(frozenset(declared)),
        )

    declared = raw(("cbd1",))
    assert declared.style == "overview"
    assert not declared.poly_3857.covers(BEYOND_HULL)
    undeclared = raw(())
    assert undeclared.style == "page"  # colorless: the regular fallback
    assert undeclared.poly_3857.covers(BEYOND_HULL)


@pytest.mark.gdal
def test_mosaic_separates_declared_overview_paint(overview_volume: VolumePaths) -> None:
    stage_masks(overview_volume, OVOL, overview_pages=("cbd1",))
    stage_mosaic(overview_volume, overview_pages=("cbd1",))
    mosaic = overview_volume.root / "mosaic.tif"
    assert mosaic.is_file()
    assert (overview_volume.root / "mosaic-overview.tif").is_file()
    # a declaration change is CONFIG — no mask, part, or COG mtime moves — yet
    # the detail mosaic's CONTENT is wrong until rebuilt. The persisted
    # partition manifest is what makes the freshness check see it: the same
    # masks with the declaration withdrawn must rebuild mosaic.tif (overview
    # paint folded back in) and remove the stale underlay mosaic. Pinning the
    # tree leaves that manifest the only input a run can move forward.
    stage_mosaic(antedated(overview_volume))
    assert not (overview_volume.root / "mosaic-overview.tif").exists()
    assert mosaic.stat().st_mtime > ANTEDATED  # the declaration change rebuilt it
    # unchanged declaration on a second run stays fresh
    stage_mosaic(antedated(overview_volume))
    assert mosaic.stat().st_mtime == ANTEDATED


RVOL = "volR"


@pytest.fixture(scope="module")
def reviewer_overview_volume(tmp_path_factory: pytest.TempPathFactory) -> VolumePaths:
    """Overviews under reviewer evidence: a verified placement and a drawn ring."""
    from autogeoref.volume import STATUS_REVIEWER_VERIFIED

    root = tmp_path_factory.mktemp("reviewer-overview") / RVOL
    paths = VolumePaths(root=root)
    paths.regions.mkdir(parents=True)
    paths.results.mkdir(parents=True)
    _colorless_sheet(paths.regions / f"{RVOL}_pcbd1.png")
    _colorless_sheet(paths.regions / f"{RVOL}_pcbd2.png")
    verified = {
        "page": "cbd1",
        "status": STATUS_REVIEWER_VERIFIED,
        "gcps_geojson": _overview_gcps_fc(X0, Y0),
    }
    ringed = {
        "page": "cbd2",
        "status": "OK",
        "gcps_geojson": _overview_gcps_fc(X0, Y0),
        "reviewer_mask_px": [[0, 0], [W, 0], [W, H], [0, H]],
    }
    (paths.results / "pcbd1.json").write_text(json.dumps(verified))
    (paths.results / "pcbd2.json").write_text(json.dumps(ringed))
    stage_warp(paths, RVOL)
    return paths


@pytest.mark.gdal
def test_reviewer_evidence_bypasses_the_overview_clip(
    reviewer_overview_volume: VolumePaths,
) -> None:
    masks = _masks_3857(
        stage_masks(reviewer_overview_volume, RVOL, overview_pages=("cbd1", "cbd2")).read_text()
    )
    # a reviewer-verified placement's GCPs may be synthetic corners, not fit
    # inliers — its mask stays whole
    assert masks[f"{RVOL}_pcbd1"].covers(BEYOND_HULL)
    # a drawn mask_px ring wins outright; no clip applies on top of it
    assert masks[f"{RVOL}_pcbd2"].covers(BEYOND_HULL)


@pytest.mark.gdal
def test_overview_only_volume_keeps_a_single_mosaic(
    reviewer_overview_volume: VolumePaths,
) -> None:
    # a volume whose committed sheets are ALL overview pages (the standalone
    # CBD volumes) has no detail paint to protect: its archive IS the overview
    stage_masks(reviewer_overview_volume, RVOL, overview_pages=("cbd1", "cbd2"))
    stage_mosaic(reviewer_overview_volume, overview_pages=("cbd1", "cbd2"))
    assert (reviewer_overview_volume.root / "mosaic.tif").is_file()
    assert not (reviewer_overview_volume.root / "mosaic-overview.tif").exists()


# ------------------------------------------------------- pre-split expansion ----

#: These are synthetic ``RawSheetMask``es built without an image, so the page
#: rect is declared rather than detected; the inset only keeps it visibly
#: distinct from the full pixel frame in the assertions below.
PAGE_RECT_INSET = 60
RECT = (PAGE_RECT_INSET, PAGE_RECT_INSET, W - PAGE_RECT_INSET, H - PAGE_RECT_INSET)


def _raw_at(
    slug: str,
    x0: float,
    box_px: tuple[float, float, float, float] | None,
    style: str = "content_box",
) -> Any:
    """A raw mask placed at ``x0``: a colour box when ``box_px`` is given,
    else the whole page rectangle."""

    matrix = ((x0, M_PER_PX, 0.0), (Y0, 0.0, -M_PER_PX))
    extent = box_px if box_px is not None else RECT
    bx0, by0, bx1, by1 = extent
    return RawSheetMask(
        slug=slug,
        image=Path("unused.png"),
        matrix=matrix,
        style=style,
        rect=RECT,
        ring_px=(((bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)) if box_px is not None else None),
        poly_3857=shp_transform(TO_3857.transform, mask_polygon_4326(extent, matrix)),
    )


#: Boxes retreated 200 px from the shared page edge; with the second sheet's
#: page 1000 m over, the channel between the boxes is 70 m — a slot.
BOX_A = (300.0, 150.0, 1575.0, 1050.0)
BOX_B = (180.0, 150.0, 1500.0, 1050.0)
SLOT_NEIGHBOUR_X0 = X0 + 1000.0


def test_expansion_closes_a_narrow_slot_and_stops_at_the_page_rect() -> None:
    """Both boxes grow across the 70 m channel their pages serve; the grown
    rectangles stay inside their page rectangles."""

    raws = [
        _raw_at("volY_p1", X0, BOX_A),
        _raw_at("volY_p2", SLOT_NEIGHBOUR_X0, BOX_B),
    ]
    grown = expand_content_boxes(raws, DuplicateCoverage())
    mid_channel = Point(X0 + 1085.0, Y0 - 400.0)
    assert not any(r.poly_3857.covers(mid_channel) for r in raws)
    union = grown[0].poly_3857.union(grown[1].poly_3857)
    assert union.covers(mid_channel)
    for raw in grown:
        xs = [p[0] for p in raw.ring_px]
        ys = [p[1] for p in raw.ring_px]
        assert min(xs) >= RECT[0] and max(xs) <= RECT[2]
        assert min(ys) >= RECT[1] and max(ys) <= RECT[3]
    # the slot ground belongs to a sheet's own page only: p1 may not reach
    # past its page rectangle's right edge however wide the slot
    p1_page_right = X0 + RECT[2] * M_PER_PX
    assert grown[0].poly_3857.bounds[2] <= p1_page_right + 1.0


def test_expansion_ignores_a_channel_wider_than_the_slot_criterion() -> None:
    """A 370 m channel is the volume's own uncovered ground, not a seam slot."""

    raws = [
        _raw_at("volY_p1", X0, BOX_A),
        _raw_at("volY_p2", X0 + 1300.0, BOX_B),
    ]
    grown = expand_content_boxes(raws, DuplicateCoverage())
    for before, after in zip(raws, grown, strict=True):
        assert after.ring_px == before.ring_px
        assert after.poly_3857.equals(before.poly_3857)


def test_expansion_moves_only_content_box_sheets() -> None:
    """A hull sheet's retreat is the volume's declaration and a ``page`` sheet
    already covers its page: neither grows, even beside a slot."""

    hull = _raw_at("volY_p1", X0, BOX_A, style="hull")
    box = _raw_at("volY_p2", SLOT_NEIGHBOUR_X0, BOX_B)
    page_rect = _raw_at("volY_p3", X0 + 2000.0, None, style="page")
    grown = expand_content_boxes([hull, box, page_rect], DuplicateCoverage())
    assert grown[0].ring_px == hull.ring_px
    assert grown[0].poly_3857.equals(hull.poly_3857)
    assert grown[2].poly_3857.equals(page_rect.poly_3857)
    # the box beside the hull still recovers the slot share its own page serves
    assert grown[1].poly_3857.area > box.poly_3857.area


# ------------------------------------------- slot volume, bake end to end ----

SLOTVOL = "volS"
#: Two identical sheets whose colour box retreats ~150 m from each page edge;
#: with the pages placed 900 m apart the two boxes leave a ~80 m channel that
#: neither serves — the seam-gap defect in miniature. The block stays under
#: half the frame so the median-saturation threshold still sees it as a
#: distinct coloured region.
SLOT_MID = (X0 + 1050.0, Y0 - 400.0)


def _wide_block_sheet(path: Path) -> None:
    img, draw = _canvas()
    draw.rectangle([420, 260, 1380, 940], fill=PINK)
    img.save(path, "PNG")


@pytest.fixture(scope="module")
def slot_volume(tmp_path_factory: pytest.TempPathFactory) -> VolumePaths:
    root = tmp_path_factory.mktemp("slotvol") / SLOTVOL
    paths = VolumePaths(root=root)
    paths.regions.mkdir(parents=True)
    paths.results.mkdir(parents=True)
    _wide_block_sheet(paths.regions / f"{SLOTVOL}_p1.png")
    _wide_block_sheet(paths.regions / f"{SLOTVOL}_p2.png")
    (paths.results / "p1.json").write_text(json.dumps(_result("1", X0, Y0)))
    (paths.results / "p2.json").write_text(json.dumps(_result("2", X0 + 900.0, Y0)))
    stage_warp(paths, SLOTVOL)
    return paths


@pytest.mark.gdal
def test_bake_closes_the_inter_sheet_slot_end_to_end(slot_volume: VolumePaths) -> None:
    """The default chain serves the channel between two retreating boxes, and
    the volume-level coverage metric reports the volume clean."""
    masks = _masks_3857(stage_masks(slot_volume, SLOTVOL).read_text())
    union = masks[f"{SLOTVOL}_p1"].union(masks[f"{SLOTVOL}_p2"])
    assert union.covers(Point(*SLOT_MID))
    qa = json.loads((slot_volume.masks / "masks-qa.json").read_text())
    assert qa["sheets"][f"{SLOTVOL}_p1"]["style"] == "content_box"
    assert qa["coverage"]["slot_per_1k"] <= 0.5
    assert qa["volume_flags"] == []


@pytest.mark.gdal
def test_coverage_flag_catches_the_slot_without_the_expansion(
    slot_volume: VolumePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the pre-split expansion disabled the channel stays unserved — and
    the coverage metric is the instrument that says so, where every per-sheet
    flag stays silent (the defect is between sheets and blank-weighted)."""
    from autogeoref.bake import masks as bake_masks

    monkeypatch.setattr(bake_masks, "expand_content_boxes", lambda raws, *_args: raws)
    masks = _masks_3857(stage_masks(slot_volume, SLOTVOL).read_text())
    union = masks[f"{SLOTVOL}_p1"].union(masks[f"{SLOTVOL}_p2"])
    assert not union.covers(Point(*SLOT_MID))
    qa = json.loads((slot_volume.masks / "masks-qa.json").read_text())
    assert qa["coverage"]["slot_per_1k"] > 0.5
    assert qa["volume_flags"] == ["coverage_gaps"]
    assert qa["flagged"] == {}


def test_expansion_hands_back_an_enclosed_hole() -> None:
    """A hole fully enclosed by the boxes is handed back even when it is too
    wide for the closing to fill — the interior-ring branch, not the notch."""

    def bar(slug: str, page_m: tuple[float, ...], box_m: tuple[float, ...]) -> RawSheetMask:
        """A sheet whose page and colour box are given in metres from X0/Y0."""
        px0, py0, px1, py1 = page_m
        bx0, by0, bx1, by1 = box_m
        matrix = ((X0 + px0, M_PER_PX, 0.0), (Y0 - py0, 0.0, -M_PER_PX))
        rect = (0, 0, round((px1 - px0) / M_PER_PX), round((py1 - py0) / M_PER_PX))
        box_px = (
            (bx0 - px0) / M_PER_PX,
            (by0 - py0) / M_PER_PX,
            (bx1 - px0) / M_PER_PX,
            (by1 - py0) / M_PER_PX,
        )
        return RawSheetMask(
            slug=slug,
            image=Path("unused.png"),
            matrix=matrix,
            style="content_box",
            rect=rect,
            ring_px=(
                (box_px[0], box_px[1]),
                (box_px[2], box_px[1]),
                (box_px[2], box_px[3]),
                (box_px[0], box_px[3]),
            ),
            poly_3857=shp_transform(TO_3857.transform, mask_polygon_4326(box_px, matrix)),
        )

    # four bars whose boxes frame a 1200 x 200 m hole: too wide for the 50 m
    # closing, enclosed on all sides, and inside the top and bottom pages
    raws = [
        bar("volY_p1", (0, 0, 2000, 600), (0, 0, 2000, 400)),
        bar("volY_p2", (0, 400, 2000, 1000), (0, 600, 2000, 1000)),
        bar("volY_p3", (0, 0, 400, 1000), (0, 0, 400, 1000)),
        bar("volY_p4", (1600, 0, 2000, 1000), (1600, 0, 2000, 1000)),
    ]
    hole_centre = Point(X0 + 1000.0, Y0 - 500.0)
    assert not any(r.poly_3857.covers(hole_centre) for r in raws)
    grown = expand_content_boxes(raws, DuplicateCoverage())
    union = grown[0].poly_3857.union(grown[1].poly_3857)
    assert union.covers(hole_centre)
