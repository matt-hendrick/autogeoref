"""Bake-time mask QA: the two measured failure classes must flag, and only they.

A synthetic five-sheet hull-style volume mirroring the defects the calibration record
measured, plus the auto-exemption remedy qualified on the labelled triage corpus:

- p1 genuine one-block sheet, its hull small ON PURPOSE and capturing all the drawn
  content — must NOT flag.
- p2 sparse-but-fully-drawn, where the saturation hull collapses onto one cluster and
  chops the linework; nothing sits under its rectangle, so auto-exemption swaps it in.
- p3 drawn at BOTH ends with a wide blank band between, painting it over the overview
  below — must flag ``blank_overpaint``.
- p4 like p2, but its blank half sits OVER the overview, so the swap would add
  ``blank_overpaint`` and the guard must abstain, keeping both hull and flag.
- pcbd1 the overview underneath, which must not flag.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from autogeoref.affine import TO_4326
from autogeoref.bake.layers import committed_layers
from autogeoref.bake.masks import raw_sheet_mask, stage_masks
from autogeoref.bake.warp import stage_warp
from autogeoref.mask.move import move_blank_cores
from autogeoref.mask.qa import (
    BLANK_OVERPAINT_MIN_CORE_M2,
    RawSheetMask,
    load_masks_qa,
    qa_masks,
    qa_note,
)
from autogeoref.paths import VolumePaths, regions_by_page
from autogeoref.report import build_report, report_markdown
from autogeoref.slugs import DuplicateCoverage

pytestmark = pytest.mark.gdal

#: The declared-overview class the CBD tests measure against, and the empty one.
CBD1, NONE = DuplicateCoverage(frozenset({"cbd1"})), DuplicateCoverage()

VOL = "volq"
W, H = 600, 400
M_PER_PX = 2.0
X0, Y0 = -9760000.0, 5141000.0
#: p3's right half is blank; the overview sheet sits under exactly that half.
P3_X0 = X0 + 2 * 1300 * M_PER_PX
CBD_X0 = P3_X0 + (W // 2) * M_PER_PX


# These sheets carry NO printed frame, because no real plate does: the map's edge is the
# outermost block frontage line. Their linework is drawn broken at every crossing so
# `mask.geometry.detect_page_bounds` returns the whole page on all five — this file's
# contract is a calibrated set of QA FLAGS, and a truncated rect would move the ink
# raster's frame and the ratio denominator under them. The search's other branch is
# pinned in `test_mosaic_stratification.test_page_bounds_stops_on_a_long_dark_run`.

#: Segment length and gap of the broken linework, in full-res px.
_RUN, _GAP = 70, 22


def _broken_v(draw: ImageDraw.ImageDraw, x: int, y0: int, y1: int) -> None:
    for y in range(y0, y1, _RUN + _GAP):
        draw.line([(x, y), (x, min(y + _RUN, y1))], fill=(30, 30, 30), width=8)


def _broken_h(draw: ImageDraw.ImageDraw, y: int, x0: int, x1: int) -> None:
    for x in range(x0, x1, _RUN + _GAP):
        draw.line([(x, y), (min(x + _RUN, x1), y)], fill=(30, 30, 30), width=8)


def _one_block_sheet(path: Path) -> None:
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([200, 120, 400, 280], fill=(200, 60, 60))
    img.save(path, "JPEG", quality=90)


def _sparse_drawn_sheet(path: Path) -> None:
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 30, 150, 150], fill=(200, 60, 60))
    for x in range(40, W - 24, 16):
        _broken_v(draw, x, 24, H - 24)
    for y in range(40, H - 24, 16):
        _broken_h(draw, y, 24, W - 24)
    img.save(path, "JPEG", quality=90)


def _blank_interior_sheet(path: Path) -> None:
    # drawn at both ends, blank between: the colour box spans the page, so the
    # blank band is INTERIOR and no bbox can exclude it. Each end is two
    # stacked blocks rather than one tall one, so no column of this sheet is
    # dark over 55% of the page and the page-bounds search still returns the
    # page; the 10 px seam is well under BLANK_CORE_M and forms no blank core.
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    for x0, x1 in ((40, 220), (520, 570)):
        draw.rectangle([x0, 60, x1, 195], fill=(200, 60, 60))
        draw.rectangle([x0, 205, x1, 340], fill=(200, 60, 60))
    img.save(path, "JPEG", quality=90)


def _sparse_half_blank_sheet(path: Path) -> None:
    # right half: p2's collapse pattern (colored cluster + monochrome
    # linework); left half: blank paper — the half a rectangle would paint
    # over the overview sheet beneath
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([W - 150, 30, W - 30, 150], fill=(200, 60, 60))
    for x in range(W // 2 + 8, W - 24, 16):
        _broken_v(draw, x, 24, H - 24)
    for y in range(40, H - 24, 16):
        _broken_h(draw, y, W // 2 + 8, W - 24)
    img.save(path, "JPEG", quality=90)


def _dense_sheet(path: Path) -> None:
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    for x in range(40, W - 60, 90):
        for y in range(40, H - 60, 70):
            draw.rectangle([x, y, x + 60, y + 45], fill=(200, 60, 60))
    img.save(path, "JPEG", quality=90)


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
    root = tmp_path_factory.mktemp("mask_qa") / VOL
    paths = VolumePaths(root=root)
    paths.regions.mkdir(parents=True)
    paths.results.mkdir(parents=True)
    _one_block_sheet(paths.regions / f"{VOL}_p1.jpg")
    _sparse_drawn_sheet(paths.regions / f"{VOL}_p2.jpg")
    _blank_interior_sheet(paths.regions / f"{VOL}_p3.jpg")
    _sparse_half_blank_sheet(paths.regions / f"{VOL}_p4.jpg")
    _dense_sheet(paths.regions / f"{VOL}_pcbd1.jpg")
    (paths.results / "p1.json").write_text(json.dumps(_result("1", X0, Y0)))
    (paths.results / "p2.json").write_text(json.dumps(_result("2", X0 + 1300 * M_PER_PX, Y0)))
    (paths.results / "p3.json").write_text(json.dumps(_result("3", P3_X0, Y0)))
    # p4's blank left half sits exactly over pcbd1's right half
    (paths.results / "p4.json").write_text(
        json.dumps(_result("4", CBD_X0 + (W // 2) * M_PER_PX, Y0))
    )
    (paths.results / "pcbd1.json").write_text(json.dumps(_result("cbd1", CBD_X0, Y0)))
    return paths


@pytest.fixture(scope="module")
def qa_doc(volume: VolumePaths) -> dict[str, Any]:
    stage_warp(volume, VOL)
    stage_masks(
        volume, VOL, content_masks=True, content_mask_exempt=["3"], overview_pages=("cbd1",)
    )
    doc = load_masks_qa(volume.masks)
    assert doc is not None, "stage_masks must persist masks/masks-qa.json"
    return doc


def test_known_failure_classes_flag_and_healthy_sheets_do_not(
    qa_doc: dict[str, Any],
) -> None:
    assert qa_doc["flagged"] == {
        f"{VOL}_p3": ["blank_overpaint"],
        f"{VOL}_p4": ["hull_collapse"],
    }


def test_auto_exemption_swaps_the_rescuable_collapse(qa_doc: dict[str, Any]) -> None:
    """p2's rectangle passes the QA re-check (nothing sits under its blank
    margins), so the loop swaps it in, the flag clears, and the swap is
    recorded for audit."""
    assert qa_doc["auto_exempted"] == [f"{VOL}_p2"]
    p2 = qa_doc["sheets"][f"{VOL}_p2"]
    assert p2["style"] == "page"
    assert p2["flags"] == []
    assert p2["ink_uncovered_frac"] < 0.05


def test_auto_exemption_abstains_when_rectangle_would_overpaint(
    qa_doc: dict[str, Any],
) -> None:
    """p4 collapses exactly like p2, but its blank half sits over the overview
    sheet: the rectangle swap would add blank_overpaint, so the guard keeps
    the hull and the flag."""
    p4 = qa_doc["sheets"][f"{VOL}_p4"]
    assert p4["style"] == "hull"
    assert p4["flags"] == ["hull_collapse"]


def test_auto_exempt_verdict_guards() -> None:
    """The two verdict guards the synthetic bake cannot reach: the severity
    bar and the blank-core worsening veto (each was forced by a measured
    wrong accept)."""
    from autogeoref.bake.masks import AUTO_EXEMPT_MIN_UNCOVERED, auto_exempt_verdict

    base = {
        "sheets": {
            "v_p1": {"ink_uncovered_frac": 0.94, "blank_core_m2": 0.0},
            "v_p2": {"ink_uncovered_frac": 0.0, "blank_core_m2": 8000.0},
        },
        "flagged": {"v_p1": ["hull_collapse"], "v_p2": ["blank_overpaint"]},
    }
    clean_after = {
        "sheets": {"v_p1": {"blank_core_m2": 0.0}, "v_p2": {"blank_core_m2": 8000.0}},
        "flagged": {"v_p2": ["blank_overpaint"]},
    }
    accept, _ = auto_exempt_verdict("v_p1", base, clean_after)
    assert accept

    # severity: a sub-majority collapse is never a candidate, whatever the
    # candidate QA says
    mild = json.loads(json.dumps(base))
    mild["sheets"]["v_p1"]["ink_uncovered_frac"] = AUTO_EXEMPT_MIN_UNCOVERED
    accept, reason = auto_exempt_verdict("v_p1", mild, clean_after)
    assert not accept and "catastrophic bar" in reason

    # worsening: a sheet that already flags blank_overpaint must not get
    # worse — "no NEW flag" alone is blind to that
    worse = json.loads(json.dumps(clean_after))
    worse["sheets"]["v_p2"]["blank_core_m2"] = 42000.0
    accept, reason = auto_exempt_verdict("v_p1", base, worse)
    assert not accept and "blank core grows" in reason

    # a new flag on ANY sheet vetoes
    new_flag = json.loads(json.dumps(clean_after))
    new_flag["flagged"]["v_p1"] = ["blank_overpaint"]
    accept, reason = auto_exempt_verdict("v_p1", base, new_flag)
    assert not accept and "introduces" in reason

    # a rectangle that somehow fails to clear the collapse vetoes
    uncleared = json.loads(json.dumps(clean_after))
    uncleared["flagged"]["v_p1"] = ["hull_collapse"]
    accept, reason = auto_exempt_verdict("v_p1", base, uncleared)
    assert not accept and "does not clear" in reason

    # measurement failure defaults to abstain, never accept: a neighbour
    # pushed into heal exhaustion, or a lost blank-core measurement, cannot
    # demonstrate the swap is safe
    exhausted = json.loads(json.dumps(clean_after))
    exhausted["sheets"]["v_p2"] = {"unmasked": True}
    exhausted["flagged"] = {}
    accept, reason = auto_exempt_verdict("v_p1", base, exhausted)
    assert not accept and "unmasked" in reason
    unmeasured = json.loads(json.dumps(clean_after))
    del unmeasured["sheets"]["v_p2"]["blank_core_m2"]
    accept, reason = auto_exempt_verdict("v_p1", base, unmeasured)
    assert not accept and "unmeasurable" in reason


def test_colour_box_is_refused_when_it_would_cut_the_sheets_own_ink(
    volume: VolumePaths,
) -> None:
    """p4's colour box keeps its coloured corner and drops its linework.

    That guard runs BEFORE the mask is built, so the sheet falls back to its
    page rectangle: it stays untidy, never truncated. p2 is the same
    story. p1 and p3 are genuinely bounded by colour and do take boxes.
    """

    images = regions_by_page(volume.regions)
    styles = {
        slug: raw_sheet_mask(
            images[page],
            slug,
            page,
            record,
            content_masks=False,
            content_mask_exempt=(),
            duplicates=CBD1,
        ).style
        for page, slug, record in committed_layers(volume, VOL)
    }
    assert styles[f"{VOL}_p4"] == "page"
    assert styles[f"{VOL}_p2"] == "page"
    assert styles[f"{VOL}_p1"] == "content_box"
    assert styles[f"{VOL}_p3"] == "content_box"  # undeclared here, so not exempt


def test_uncovered_ink_flags_a_box_the_split_then_cut(volume: VolumePaths) -> None:
    """The post-split backstop to the pre-split own-ink guard.

    ``detect_content_box``'s guard bounds what the BOX drops; the overlap
    split can take more away afterwards, and this flag is what notices. It is
    deliberately not ``hull_collapse``: the bake's rectangle swap is a remedy
    for a collapsed HULL, and ``_auto_exempt_collapsed`` keys on that name, so
    a box's uncovered ink must never enter the loop as a swap candidate.
    """
    from shapely.geometry import box as shp_box
    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform

    from autogeoref.affine import TO_3857, TO_4326

    fc = json.loads((volume.masks / "masks.geojson").read_text())
    finals: dict[str, Any] = {f["properties"]["slug"]: shape(f["geometry"]) for f in fc["features"]}
    images = regions_by_page(volume.regions)
    raws = [
        raw_sheet_mask(
            images[page],
            slug,
            page,
            record,
            content_masks=False,
            content_mask_exempt=(),
            duplicates=CBD1,
        )
        for page, slug, record in committed_layers(volume, VOL)
    ]
    p1 = next(r for r in raws if r.slug == f"{VOL}_p1")
    assert p1.style == "content_box"
    # stand in for a split that cut most of p1's box away
    minx, miny, maxx, maxy = p1.poly_3857.bounds
    trimmed = p1.poly_3857.intersection(shp_box(minx, miny, minx + (maxx - minx) * 0.25, maxy))
    finals[f"{VOL}_p1"] = shp_transform(TO_4326.transform, trimmed)
    assert shp_transform(TO_3857.transform, finals[f"{VOL}_p1"]).area < p1.poly_3857.area

    doc = qa_masks(VOL, raws, finals, content_masks=False, duplicates=CBD1)
    entry = doc["sheets"][f"{VOL}_p1"]
    assert entry["ink_uncovered_frac"] > doc["thresholds"]["uncovered_ink_min_frac"]
    assert entry["flags"] == ["uncovered_ink"]
    assert all("hull_collapse" not in flags for flags in doc["flagged"].values())


def test_genuine_one_block_sheet_keeps_its_small_hull(qa_doc: dict[str, Any]) -> None:
    """A small hull alone is NOT collapse: p1's hull covers all its content."""
    p1 = qa_doc["sheets"][f"{VOL}_p1"]
    assert p1["style"] == "hull"
    assert p1["hull_page_ratio"] < 0.6
    assert p1["ink_uncovered_frac"] < 0.05
    assert p1["flags"] == []


def test_collapse_is_measured_as_content_no_sheet_covers(
    qa_doc: dict[str, Any],
) -> None:
    p4 = qa_doc["sheets"][f"{VOL}_p4"]
    assert p4["style"] == "hull"
    assert p4["ink_uncovered_frac"] > qa_doc["thresholds"]["hull_collapse_min_uncovered_ink"]


def test_blank_overpaint_names_the_covered_sheet(qa_doc: dict[str, Any]) -> None:
    p3 = qa_doc["sheets"][f"{VOL}_p3"]
    # exempt, so no colour bound at all — see the exemption test in
    # tests/test_mosaic_stratification.py
    assert p3["style"] == "page"
    assert p3["blank_core_m2"] > qa_doc["thresholds"]["blank_overpaint_min_core_m2"]
    assert p3["blank_over"] == [f"{VOL}_pcbd1"]


def test_overview_sheet_is_classified_and_unflagged(qa_doc: dict[str, Any]) -> None:
    cbd = qa_doc["sheets"][f"{VOL}_pcbd1"]
    assert cbd["style"] == "overview"
    assert cbd["flags"] == []
    # duplicate-coverage sheets stay out of the regular-overlap signal
    assert "raw_overlap_frac_max" not in cbd


def test_regular_raw_overlap_signal_is_recorded(qa_doc: dict[str, Any]) -> None:
    """The pre-split overlap metric (the undeclared-style signal) exists for
    every regular sheet and is small here: the regular sheets were laid out
    apart on purpose."""
    for slug in (f"{VOL}_p1", f"{VOL}_p2", f"{VOL}_p3"):
        entry = qa_doc["sheets"][slug]
        assert entry.get("raw_overlap_frac_max", 0.0) < 0.05


def test_qa_note_and_report_surface_the_flags(qa_doc: dict[str, Any]) -> None:
    note = qa_note(qa_doc)
    assert note is not None
    assert "2 sheet(s)" in note
    assert "p4 (hull_collapse)" in note
    assert "p3 (blank_overpaint)" in note
    assert "auto-exempted 1 collapsed hull(s): p2" in note
    report = build_report(VOL, {}, notes=[note])
    assert f"NOTE: {note}" in report_markdown(report)


def test_qa_note_is_none_when_clean() -> None:
    assert qa_note({"flagged": {}}) is None


def test_undeclared_volume_records_blank_metric_but_never_flags(
    volume: VolumePaths, qa_doc: dict[str, Any]
) -> None:
    """On a volume with NO declared mask style, a page rectangle painting
    over the duplicate-coverage sheet beneath it is the design (fallback
    layering), not overpaint — the metric is recorded, the flag is not."""
    from shapely.geometry import shape

    fc = json.loads((volume.masks / "masks.geojson").read_text())
    finals = {f["properties"]["slug"]: shape(f["geometry"]) for f in fc["features"]}
    images = regions_by_page(volume.regions)
    raws = [
        raw_sheet_mask(
            images[page],
            slug,
            page,
            record,
            content_masks=False,
            content_mask_exempt=(),
            duplicates=CBD1,
        )
        for page, slug, record in committed_layers(volume, VOL)
    ]
    doc = qa_masks(VOL, raws, finals, content_masks=False, duplicates=CBD1)
    assert doc["sheets"][f"{VOL}_p3"]["blank_core_m2"] > 4000.0
    # nothing at all flags here: `blank_overpaint` is gated on the declaration
    # and `uncovered_ink`, which is not, has nothing to fire on
    assert doc["flagged"] == {}


def _sheet_at(matrix: Any, image: Path, slug: str) -> Any:
    """A one-sheet raw mask whose ground footprint follows ``matrix``."""
    from shapely.geometry import Polygon

    corners = [
        (
            matrix[0][0] + matrix[0][1] * px + matrix[0][2] * py,
            matrix[1][0] + matrix[1][1] * px + matrix[1][2] * py,
        )
        for px, py in ((0, 0), (W, 0), (W, H), (0, H))
    ]
    return RawSheetMask(
        slug=slug,
        image=image,
        matrix=matrix,
        style="content_box",
        rect=(20, 20, W - 20, H - 20),
        ring_px=None,
        poly_3857=Polygon(corners),
    )


def test_mirrored_placement_is_flagged(tmp_path: Path) -> None:
    """The handedness backstop. `matching.gates_ok` rejects a reflected model
    at accept time, so this must never fire on a matched sheet — it is here
    for the paths that skip that gate, because the damage is delivered THROUGH
    the mask: gdalwarp fills the reflected frame with opaque black and the
    sheet's own mask paints it over the basemap
    (`_024` p98).
    """
    from shapely.ops import transform as shp_transform

    _one_block_sheet(tmp_path / "sheet.jpg")
    # upright: pixel y grows DOWN, 3857 y grows UP -> negative determinant
    upright = ((X0, M_PER_PX, 0.0), (Y0, 0.0, -M_PER_PX))
    # mirrored: the y flip dropped. Same scale, same aspect, same rotation.
    mirrored = ((X0, M_PER_PX, 0.0), (Y0, 0.0, M_PER_PX))

    docs = {}
    for name, matrix in (("upright", upright), ("mirrored", mirrored)):
        sheet = _sheet_at(matrix, tmp_path / "sheet.jpg", f"{VOL}_p1")
        finals = {sheet.slug: shp_transform(TO_4326.transform, sheet.poly_3857)}
        docs[name] = qa_masks(VOL, [sheet], finals, content_masks=True, duplicates=NONE)

    assert docs["upright"]["sheets"][f"{VOL}_p1"]["flags"] == []
    assert "mirrored" not in docs["upright"]["sheets"][f"{VOL}_p1"]
    entry = docs["mirrored"]["sheets"][f"{VOL}_p1"]
    assert entry["flags"] == ["mirrored"]
    assert entry["mirrored"] is True
    # the rest of the instrument still measures: a mirrored sheet must not
    # short-circuit the metrics, or a real mask defect would hide behind it.
    # m_per_px comes off |det|, so it survives the sign.
    assert entry["mask_area_m2"] > 0
    assert "ink_captured_frac" in entry
    assert docs["mirrored"]["flagged"] == {f"{VOL}_p1": ["mirrored"]}
    assert qa_note(docs["mirrored"]) == (
        "mask QA flags on 1 sheet(s): p1 (mirrored) — see masks/masks-qa.json"
    )


def test_recorded_model_window_flag(tmp_path: Path) -> None:
    """The accept-time scale/rotation window, re-checked on the model that
    warps. It exists for the record-vs-placing-model divergence class
    a rescue passes on a pinned model but
    serves an unconstrained refit nobody re-tested. Quadrant-rotated scans and
    declared page-scale multiples pass exactly as they do at accept time.
    """
    import math

    from shapely.ops import transform as shp_transform

    _one_block_sheet(tmp_path / "sheet.jpg")
    window = (M_PER_PX, 0.0)

    def linear(scale: float, rot_deg: float) -> Any:
        s, c = scale * math.sin(math.radians(rot_deg)), scale * math.cos(math.radians(rot_deg))
        return ((X0, c, s), (Y0, s, -c))

    # anisotropic: x-axis +5% (inside), y-axis -15% (outside) — the flag must
    # test both axes and the recorded deviation must be the worst BY MAGNITUDE
    aniso = ((X0, M_PER_PX * 1.05, 0.0), (Y0, 0.0, -M_PER_PX * 0.85))

    cases = {
        "inside": (linear(M_PER_PX, 0.0), None),
        "scale_out": (linear(M_PER_PX * 1.2, 0.0), None),  # +20% > ±10%
        "aniso_out": (aniso, None),
        "rot_out": (linear(M_PER_PX, 2.0), None),  # 2.0° > ±1.5°
        "quadrant": (linear(M_PER_PX, 90.0), None),  # folds to 0° — rotated format
        "multiple": (linear(M_PER_PX * 3.0, 0.0), {"1": 3.0}),  # declared CBD scale
    }
    entries = {}
    for name, (matrix, mult) in cases.items():
        sheet = _sheet_at(matrix, tmp_path / "sheet.jpg", f"{VOL}_p1")
        finals = {sheet.slug: shp_transform(TO_4326.transform, sheet.poly_3857)}
        doc = qa_masks(
            VOL,
            [sheet],
            finals,
            content_masks=True,
            duplicates=NONE,
            placement_window=window,
            page_scale_multiples=mult,
        )
        entries[name] = doc["sheets"][f"{VOL}_p1"]

    assert entries["inside"]["flags"] == []
    assert entries["inside"]["window_scale_dev_frac"] == 0.0
    assert entries["inside"]["window_rot_dev_deg"] == 0.0
    assert entries["scale_out"]["flags"] == ["outside_window"]
    assert entries["scale_out"]["window_scale_dev_frac"] == pytest.approx(0.2)
    assert entries["aniso_out"]["flags"] == ["outside_window"]
    assert entries["aniso_out"]["window_scale_dev_frac"] == pytest.approx(-0.15)
    assert entries["rot_out"]["flags"] == ["outside_window"]
    assert entries["rot_out"]["window_rot_dev_deg"] == pytest.approx(2.0)
    assert entries["quadrant"]["flags"] == []
    assert entries["quadrant"]["window_rot_dev_deg"] == pytest.approx(0.0)
    assert entries["multiple"]["flags"] == []

    # not measured without the volume constants — no metrics, no flag
    sheet = _sheet_at(linear(M_PER_PX * 1.2, 0.0), tmp_path / "sheet.jpg", f"{VOL}_p1")
    finals = {sheet.slug: shp_transform(TO_4326.transform, sheet.poly_3857)}
    doc = qa_masks(VOL, [sheet], finals, content_masks=True, duplicates=NONE)
    assert "window_scale_dev_frac" not in doc["sheets"][f"{VOL}_p1"]
    assert doc["sheets"][f"{VOL}_p1"]["flags"] == []


def test_handedness_helper_measures_without_a_volume(tmp_path: Path) -> None:
    """The extracted helper's contract: mutate the entry, write only measured
    keys. The singular-vs-mirrored specimen pair (module docstring) is the
    case a whole-volume fixture cannot cheaply pin."""
    from autogeoref.mask.qa import _handedness

    def measure(matrix: Any) -> tuple[dict[str, Any], tuple[float, float, float]]:
        entry: dict[str, Any] = {"style": "content_box", "flags": []}
        sheet = _sheet_at(matrix, tmp_path / "absent.jpg", f"{VOL}_p1")
        return entry, _handedness(entry, sheet)

    upright, (det, sx, sy) = measure(((X0, M_PER_PX, 0.0), (Y0, 0.0, -M_PER_PX)))
    assert det < 0 and (sx, sy) == (M_PER_PX, M_PER_PX)
    assert upright["axis_perpendicularity"] == 1.0
    assert upright["flags"] == [] and "mirrored" not in upright

    mirrored, (det, _, _) = measure(((X0, M_PER_PX, 0.0), (Y0, 0.0, M_PER_PX)))
    assert det > 0
    assert mirrored["mirrored"] is True and mirrored["flags"] == ["mirrored"]

    # collapsed axes: the flag fires (both classes want an operator) and the
    # recorded perpendicularity is what tells them apart by eye
    singular, _ = measure(((X0, M_PER_PX, 0.0), (Y0, M_PER_PX, 0.0)))
    assert singular["axis_perpendicularity"] == 0.0
    assert singular["flags"] == ["mirrored"]


def test_core_blocks_reduction_owns_both_predicates() -> None:
    """One reduction serves the blank-margin and contested-ground metrics;
    only the loser-ink predicate differs, and an unmeasurably small frame
    reduces to zero blocks, never to a guess."""
    import numpy as np

    from autogeoref.mask.qa import _core_blocks

    mask = np.zeros((8, 8), dtype=bool)
    mask[0:4, 0:4] = True
    mask[4:8, 4:8] = True
    mask[7, 7] = False  # one missing pixel breaks the second block
    assert _core_blocks(mask, 4) == 1

    # a frame smaller than one block holds no core
    assert _core_blocks(mask, 9) == 0

    # contested predicate: a full blank block counts only when the
    # neighbour's ink there clears CONTESTED_INK_FLOOR
    ink = np.zeros((8, 8), dtype=bool)
    assert _core_blocks(mask, 4, extra=ink) == 0
    ink[1, 1] = True  # 1/16 = 0.0625 >= 0.05
    assert _core_blocks(mask, 4, extra=ink) == 1


def test_qa_masks_omits_deep_metrics_for_unmeasurable_sheets(tmp_path: Path) -> None:
    """The driver's bail-outs are the omitted-vs-zero contract: a sheet with
    no final publishes only ``unmasked``, and a sheet whose image is missing
    stops at its area — neither publishes a deep metric as measured-clean."""
    from shapely.geometry import Polygon

    matrix = ((X0, M_PER_PX, 0.0), (Y0, 0.0, -M_PER_PX))
    no_final = _sheet_at(matrix, tmp_path / "absent1.jpg", f"{VOL}_p1")
    no_image = _sheet_at(matrix, tmp_path / "absent2.jpg", f"{VOL}_p2")
    square_4326 = Polygon([(-87.7, 41.8), (-87.69, 41.8), (-87.69, 41.81), (-87.7, 41.81)])

    finals = {f"{VOL}_p1": None, f"{VOL}_p2": square_4326}
    doc = qa_masks(VOL, [no_final, no_image], finals, duplicates=NONE)

    p1 = doc["sheets"][f"{VOL}_p1"]
    assert p1["unmasked"] is True
    assert "mask_area_m2" not in p1

    p2 = doc["sheets"][f"{VOL}_p2"]
    assert "unmasked" not in p2
    assert "mask_area_m2" in p2

    deep = {
        "ink_frac_in_mask",
        "ink_captured_frac",
        "ink_uncovered_frac",
        "blank_core_m2",
        "blank_over_neighbor_m2",
        "blank_over_neighbor_core_m2",
    }
    for entry in (p1, p2):
        assert deep.isdisjoint(entry)


#: The `_020` p10/p11 miniature's page rect and raw-overlap offset, shared by
#: the metric test and the blank-core-move regression tests. DECLARED, not
#: detected — these ``RawSheetMask``es are built by hand — so the inset is
#: free; it only keeps the rect visibly distinct from the full pixel frame.
_PAIR_RECT = (20, 20, 580, 380)
_PAIR_LOSING_X0 = X0 + 500 * M_PER_PX  # raw rects overlap by 60 px


def _pair_sheet_poly(x0: float) -> Any:
    from shapely.geometry import Polygon

    rect = _PAIR_RECT
    return Polygon(
        [
            (x0 + rect[0] * M_PER_PX, Y0 - rect[1] * M_PER_PX),
            (x0 + rect[2] * M_PER_PX, Y0 - rect[1] * M_PER_PX),
            (x0 + rect[2] * M_PER_PX, Y0 - rect[3] * M_PER_PX),
            (x0 + rect[0] * M_PER_PX, Y0 - rect[3] * M_PER_PX),
        ]
    )


def _contested_blank_pair(tmp_path: Path) -> tuple[list[Any], dict[str, Any]]:
    """The `_020` p10/p11 defect in miniature (
    section 5): p10 draws the rail yard, p11 leaves the same ground blank but
    frames it with its own colour, and the centroid split awards it to p11.

    Returns the two raw sheets and the shipped split of their masks.
    """
    from autogeoref.mask.geometry import split_overlaps

    # p11 analog: coloured at both ends, blank between — its box must contain
    # the contested ground, and the nearer centroid wins it in the split
    win_img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(win_img)
    draw.rectangle([30, 30, 250, 370], fill=(200, 60, 60))
    draw.rectangle([560, 30, 575, 370], fill=(200, 60, 60))
    win_img.save(tmp_path / "win.jpg", "JPEG", quality=90)

    # p10 analog: solid drawn content across the whole contested ground
    lose_img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(lose_img)
    draw.rectangle([20, 20, 250, 380], fill=(200, 60, 60))
    lose_img.save(tmp_path / "lose.jpg", "JPEG", quality=90)

    def raw(slug: str, image: Path, x0: float) -> RawSheetMask:
        return RawSheetMask(
            slug=slug,
            image=image,
            matrix=((x0, M_PER_PX, 0.0), (Y0, 0.0, -M_PER_PX)),
            style="content_box",
            rect=_PAIR_RECT,
            ring_px=None,
            poly_3857=_pair_sheet_poly(x0),
        )

    sheets = [
        raw("volc_p11", tmp_path / "win.jpg", X0),
        raw("volc_p10", tmp_path / "lose.jpg", _PAIR_LOSING_X0),
    ]
    split = split_overlaps({s.slug: s.poly_3857 for s in sheets})
    return sheets, split


def test_blank_interior_over_neighbor_ink_is_measured_on_contested_ground(
    tmp_path: Path,
) -> None:
    """Post-split the regular masks are a zero-overlap partition, so any
    final-vs-final measure is identically zero — the metric must measure the
    split-CONTESTED ground (final winner cell vs raw loser claim) instead,
    and must name the sheet whose ink was painted over, not the blank one.
    """
    from shapely.ops import transform as shp_transform

    sheets, split = _contested_blank_pair(tmp_path)
    # the blank sheet's centroid is nearer the contested ground: it wins some
    assert split["volc_p11"].area < _pair_sheet_poly(X0).area
    finals = {slug: shp_transform(TO_4326.transform, poly) for slug, poly in split.items()}

    doc = qa_masks("volc", sheets, finals, content_masks=False, duplicates=NONE)
    win = doc["sheets"]["volc_p11"]
    lose = doc["sheets"]["volc_p10"]
    # the winner's blank interior over the loser's ink is now non-zero and
    # large enough that no fringe-noise reading explains it
    assert win["blank_over_neighbor_m2"] > 5000.0
    assert win["blank_over_neighbor_core_m2"] > BLANK_OVERPAINT_MIN_CORE_M2
    assert win["blank_over_neighbor"] == ["volc_p10"]
    # the drawn sheet is not blamed for the ground it kept
    assert lose["blank_over_neighbor_m2"] == 0.0
    assert "blank_over_neighbor" not in lose
    # a partition has no paint-order overlap: the blank-margin signal stays
    # silent, and the defect metric is diagnostics only — nothing flags
    assert win["blank_core_m2"] == 0.0
    assert doc["flagged"] == {}


def test_resolve_masks_consults_the_dryrun_once_and_reuses_the_heal_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The promoted resolve ladder (expand -> split -> move -> heal) in
    isolation — untestable while it was a closure inside stage_masks. The
    heal cache is its contract with the auto-exemption loop: a candidate
    re-score must not re-run gdalwarp on polygons it left alone, and an
    exhausted ladder yields None finals (unmaskable), never a crash."""
    from autogeoref import warp
    from autogeoref.bake.masks import _resolve_masks
    from autogeoref.paths import VolumePaths

    calls: list[str] = []

    def accept_all(cog: Path, candidate: Any, *, timeout_s: float, crs_epsg: int) -> bool:
        calls.append(cog.stem)
        return True

    monkeypatch.setattr(warp, "gdalwarp_cutline_dryrun", accept_all)
    sheets, _ = _contested_blank_pair(tmp_path)
    paths = VolumePaths(root=tmp_path / "volc")
    heal_cache: dict[Any, Any] = {}
    raws, finals = _resolve_masks(
        sheets, paths, duplicates=NONE, heal_cache=heal_cache, ink_rasters={}, timeout_s=5.0
    )
    assert [r.slug for r in raws] == [s.slug for s in sheets]
    assert set(finals) == {"volc_p11", "volc_p10"}
    assert all(poly is not None for poly in finals.values())
    # the split partitions the pair: no served overlap survives (finals are
    # EPSG:4326, so any real overlap is ~1e-5 deg² — assert exact partition)
    assert finals["volc_p11"].intersection(finals["volc_p10"]).area == 0
    assert sorted(calls) == ["volc_p10", "volc_p11"]

    # a re-resolve of the same raws (an auto-exemption candidate re-score)
    # answers from the cache: no new dry-run
    _, again = _resolve_masks(
        sheets, paths, duplicates=NONE, heal_cache=heal_cache, ink_rasters={}, timeout_s=5.0
    )
    assert len(calls) == 2
    assert {slug: poly.wkb for slug, poly in again.items()} == {
        slug: poly.wkb for slug, poly in finals.items()
    }

    # a ladder nothing accepts exhausts to None finals, never a failure
    monkeypatch.setattr(warp, "gdalwarp_cutline_dryrun", lambda *_a, **_k: False)
    _, exhausted = _resolve_masks(
        sheets, paths, duplicates=NONE, heal_cache={}, ink_rasters={}, timeout_s=5.0
    )
    assert exhausted == {"volc_p11": None, "volc_p10": None}


def test_blank_core_move_returns_contested_ground_to_the_drawn_sheet(
    tmp_path: Path,
) -> None:
    """The blank-core remedy on the same miniature: the winner's fully-blank
    contested cells move to the drawn loser, the coverage union survives, the
    pair stays disjoint, and the defect metric drops to (near) zero — the
    moved-cell regression the handoff asks for beside the metric test."""
    from shapely.ops import transform as shp_transform
    from shapely.ops import unary_union

    sheets, split = _contested_blank_pair(tmp_path)
    moved = move_blank_cores(sheets, split)

    # ground moved: the blank winner shed, the drawn loser gained
    assert moved["volc_p11"].area < split["volc_p11"].area - 1000.0
    assert moved["volc_p10"].area > split["volc_p10"].area + 1000.0
    # no double coverage, and the union invariant holds on the solid measure
    assert moved["volc_p11"].intersection(moved["volc_p10"]).area < 1.0
    before = unary_union(list(split.values()))
    after = unary_union(list(moved.values()))
    assert before.difference(after).buffer(-0.5).area < 1.0

    finals = {slug: shp_transform(TO_4326.transform, g) for slug, g in moved.items()}
    doc = qa_masks("volc", sheets, finals, content_masks=False, duplicates=NONE)
    win = doc["sheets"]["volc_p11"]
    # the metric measured > BLANK_OVERPAINT_MIN_CORE_M2 on the shipped split
    # (previous test); after the move only sub-core residue may remain
    assert win["blank_over_neighbor_core_m2"] < BLANK_OVERPAINT_MIN_CORE_M2 / 4
    assert doc["flagged"] == {}


def test_blank_core_move_refuses_to_hollow_a_sheet_past_half(tmp_path: Path) -> None:
    """A fully-blank winner whose every contested cell would move must keep
    its mask: the move holds cuts to the same more-than-half bar as the
    split's ``_usable`` guard, and an unusable removal moves nothing (the
    cells stay with the winner and the defect stays measured)."""
    from shapely.geometry import box as shp_box

    blank_img = Image.new("RGB", (W, H), "white")
    blank_img.save(tmp_path / "blank.jpg", "JPEG", quality=90)
    inked_img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(inked_img)
    draw.rectangle([20, 20, 580, 380], fill=(200, 60, 60))
    inked_img.save(tmp_path / "inked.jpg", "JPEG", quality=90)

    def raw(slug: str, image: Path, x0: float) -> RawSheetMask:
        return RawSheetMask(
            slug=slug,
            image=image,
            matrix=((x0, M_PER_PX, 0.0), (Y0, 0.0, -M_PER_PX)),
            style="content_box",
            rect=_PAIR_RECT,
            ring_px=None,
            poly_3857=_pair_sheet_poly(x0),
        )

    # the blank sheet's raw mask sits INSIDE the drawn sheet's raw claim, so
    # everything the split leaves it is contested blank-over-ink
    sheets = [
        raw("volh_p1", tmp_path / "blank.jpg", X0),
        raw("volh_p2", tmp_path / "inked.jpg", X0 + 100 * M_PER_PX),
    ]
    bounds = _pair_sheet_poly(X0).bounds
    split = {
        "volh_p1": shp_box(bounds[0], bounds[1], bounds[0] + 560.0, bounds[3]),
        "volh_p2": shp_box(bounds[0] + 560.0, bounds[1], bounds[2] + 200.0, bounds[3]),
    }
    moved = move_blank_cores(sheets, split)
    assert moved["volh_p1"].equals(split["volh_p1"])
    assert moved["volh_p2"].equals(split["volh_p2"])


def test_blank_core_move_refuses_cells_the_receiving_sheet_cannot_reach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A moved cell must REACH the sheet it moves to.

    The reported artifact is an island of the receiving sheet stranded inside its neighbour — it
    renders that sheet's own blank margin and imprint over the neighbour's drawn ground, and
    takes a hole out of the neighbour. The miniature gives the loser ink in two separate places
    inside the winner's contested band: one against the split cut, one deep inside with a blank
    gap between. Only the first may move, and both finals must stay simple Polygons. Deleting
    the guard fails this test — the monkeypatched half shows exactly what it prevents.
    """
    from shapely.geometry import Point

    from autogeoref.mask import move as mask_move
    from autogeoref.mask.geometry import split_overlaps

    # the winner draws its left half and leaves the whole contested band blank
    win_img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(win_img)
    draw.rectangle([30, 30, 300, 370], fill=(200, 60, 60))
    win_img.save(tmp_path / "reach_win.jpg", "JPEG", quality=90)

    # the loser draws its own ground (px >= 155) plus a band against the cut
    # (px 110-150) and an ISOLATED patch far inside the winner (px 20-55)
    lose_img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(lose_img)
    draw.rectangle([155, 30, 570, 370], fill=(200, 60, 60))
    draw.rectangle([110, 30, 150, 370], fill=(200, 60, 60))
    draw.rectangle([20, 100, 55, 300], fill=(200, 60, 60))
    lose_img.save(tmp_path / "reach_lose.jpg", "JPEG", quality=90)

    def raw(slug: str, image: Path, x0: float) -> RawSheetMask:
        return RawSheetMask(
            slug=slug,
            image=image,
            matrix=((x0, M_PER_PX, 0.0), (Y0, 0.0, -M_PER_PX)),
            style="content_box",
            rect=_PAIR_RECT,
            ring_px=None,
            poly_3857=_pair_sheet_poly(x0),
        )

    # a 520 m raw overlap, so the contested band is deep enough to hold both
    sheets = [
        raw("volr_p1", tmp_path / "reach_win.jpg", X0),
        raw("volr_p2", tmp_path / "reach_lose.jpg", X0 + 300 * M_PER_PX),
    ]
    split = split_overlaps({s.slug: s.poly_3857 for s in sheets})
    moved = move_blank_cores(sheets, split)

    # the band against the cut moved: the drawn sheet gained real ground
    assert moved["volr_p2"].area > split["volr_p2"].area + 1000.0
    # the isolated patch did NOT: its ground stays with the blank winner
    assert moved["volr_p1"].contains(Point(X0 + 675.0, Y0 - 400.0))
    # and both finals stay the shipped Polygon-only convention
    for slug in ("volr_p1", "volr_p2"):
        assert moved[slug].geom_type == "Polygon", slug
        assert not moved[slug].interiors, slug

    # without the guard the same inputs strand the patch and punch the hole
    monkeypatch.setattr(mask_move, "_reachable", lambda claims, _split: dict(claims))
    unguarded = move_blank_cores(sheets, split)
    assert not unguarded["volr_p1"].contains(Point(X0 + 675.0, Y0 - 400.0))
    assert unguarded["volr_p2"].geom_type == "MultiPolygon"
    stranded = unguarded["volr_p1"]
    parts = stranded.geoms if stranded.geom_type == "MultiPolygon" else [stranded]
    assert any(p.interiors for p in parts), "the stranded island must hole its neighbour"


def test_blank_core_move_reverts_wholesale_when_the_union_check_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coverage-union invariant is a hard requirement of the blank-core remedy
    record: the check must exist AND revert. Forcing the tolerance below zero
    makes any move fail it, so a scenario that otherwise moves ground (the
    contested-pair miniature) must come back untouched — deleting the check
    or its revert path fails this test."""
    from autogeoref.mask import move as mask_move

    sheets, split = _contested_blank_pair(tmp_path)
    assert not move_blank_cores(sheets, split)["volc_p11"].equals(split["volc_p11"])

    monkeypatch.setattr(mask_move, "_MAX_UNCOVERED_SOLID_M2", -1.0)
    reverted = move_blank_cores(sheets, split)
    assert reverted["volc_p11"].equals(split["volc_p11"])
    assert reverted["volc_p10"].equals(split["volc_p10"])


def test_stage_masks_runs_the_blank_core_move_end_to_end(tmp_path: Path) -> None:
    """The full detect -> split -> move -> heal chain on the miniature defect:
    the served winner mask no longer covers the blank band over the drawn
    neighbour, the neighbour's mask does, and ``masks-qa.json`` records only
    sub-core residue on contested ground."""
    from shapely.geometry import Point, shape

    vol = "volm"
    root = tmp_path / vol
    paths = VolumePaths(root=root)
    paths.regions.mkdir(parents=True)
    paths.results.mkdir(parents=True)

    win_img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(win_img)
    draw.rectangle([30, 30, 250, 370], fill=(200, 60, 60))
    draw.rectangle([560, 30, 575, 370], fill=(200, 60, 60))
    win_img.save(paths.regions / f"{vol}_p11.jpg", "JPEG", quality=90)
    lose_img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(lose_img)
    draw.rectangle([20, 20, 580, 380], fill=(200, 60, 60))
    lose_img.save(paths.regions / f"{vol}_p10.jpg", "JPEG", quality=90)
    (paths.results / "p11.json").write_text(json.dumps(_result("11", X0, Y0)))
    (paths.results / "p10.json").write_text(json.dumps(_result("10", X0 + 500 * M_PER_PX, Y0)))

    stage_warp(paths, vol)
    stage_masks(paths, vol)

    fc = json.loads((paths.masks / "masks.geojson").read_text())
    finals = {f["properties"]["slug"]: shape(f["geometry"]) for f in fc["features"]}
    # a point in the blank band the bisector had awarded to the blank sheet
    probe = Point(TO_4326.transform(X0 + 1070.0, Y0 - 400.0))
    assert not finals[f"{vol}_p11"].contains(probe), "blank winner still serves the band"
    assert finals[f"{vol}_p10"].contains(probe), "drawn neighbour does not serve the band"

    doc = load_masks_qa(paths.masks)
    assert doc is not None
    win = doc["sheets"][f"{vol}_p11"]
    assert win["blank_over_neighbor_core_m2"] < BLANK_OVERPAINT_MIN_CORE_M2 / 4


def test_load_masks_qa_tolerates_absence_and_damage(tmp_path: Path) -> None:
    assert load_masks_qa(tmp_path) is None
    (tmp_path / "masks-qa.json").write_text("{trunc")
    assert load_masks_qa(tmp_path) is None


def test_status_reads_the_flags(volume: VolumePaths, qa_doc: dict[str, Any]) -> None:
    from autogeoref.status import build_status

    work = volume.root.parent
    rows = build_status(work=work, fixtures=work / "no-fixtures", tiles=work / "no-tiles")
    row = next(r for r in rows if r.volume == VOL)
    assert row.mask_qa_flagged == 2
    assert "mask QA flags on 2 sheet(s)" in row.note


# ---------------------------------------------------- volume coverage gaps ----


def _coverage_doc(tmp_path: Path, gap: bool, twin: bool = False) -> dict[str, Any]:
    """Two side-by-side pages; finals either leave an 80 m channel between the
    masks (the seam-gap defect in miniature) or cover both pages whole. The
    images do not exist: the volume metric is pure geometry and must not
    depend on the per-sheet ink metrics."""
    from shapely.geometry import Polygon
    from shapely.ops import transform as shp_transform

    def strip(x0: float, x1: float) -> Any:
        poly = Polygon([(x0, Y0 - 40), (x1, Y0 - 40), (x1, Y0 - 760), (x0, Y0 - 760)])
        return shp_transform(TO_4326.transform, poly)

    def raw(slug: str, x0: float) -> Any:
        return RawSheetMask(
            slug=slug,
            image=tmp_path / "missing.jpg",
            matrix=((x0, M_PER_PX, 0.0), (Y0, 0.0, -M_PER_PX)),
            style="content_box",
            rect=(20, 20, W - 20, H - 20),
            ring_px=None,
            poly_3857=Polygon(),
        )

    sheets = [raw("volv_p1", X0), raw("volv_p2", X0 + 600.0)]
    finals = {
        "volv_p1": strip(X0 + 40, X0 + (560 if gap else 640)),
        "volv_p2": strip(X0 + 640, X0 + 1760),
    }
    pages = ["1", "2"]
    if twin:
        # a skeleton twin whose served fallback paint covers the channel. Its
        # page rectangle is pushed east so that misclassifying it as a regular
        # sheet would GROW the footprint the slot ratio divides by.
        sheets.append(raw("volv_p1S", X0 + 2000.0))
        finals["volv_p1S"] = strip(X0 + 400, X0 + 800)
        pages.append("1S")
    return qa_masks("volv", sheets, finals, duplicates=DuplicateCoverage.resolve(pages))


def test_volume_coverage_flags_an_inter_sheet_slot(tmp_path: Path) -> None:
    doc = _coverage_doc(tmp_path, gap=True)
    cov = doc["coverage"]
    assert cov["footprint_m2"] > 1_000_000
    assert cov["slot_m2"] > 30_000
    assert cov["slot_per_1k"] > 0.5
    assert doc["volume_flags"] == ["coverage_gaps"]
    # the defect is between sheets: no per-sheet flag fires for it
    assert doc["flagged"] == {}
    assert doc["thresholds"]["coverage_gaps_max_slot_per_1k"] == 0.5
    note = qa_note(doc)
    assert note is not None
    assert "coverage_gaps" in note


def test_volume_coverage_clean_when_the_pages_are_served(tmp_path: Path) -> None:
    doc = _coverage_doc(tmp_path, gap=False)
    assert doc["coverage"]["slot_per_1k"] <= 0.5
    assert doc["volume_flags"] == []
    assert qa_note(doc) is None


def test_volume_coverage_counts_skeleton_twin_paint_as_served(tmp_path: Path) -> None:
    """A skeleton twin paints UNDER the regular sheets in the detail mosaic,
    so a channel its mask covers shows the reader map, not a hole — and it
    contributes no page rectangle of its own to the footprint that divides it."""
    doc = _coverage_doc(tmp_path, gap=True, twin=True)
    assert doc["coverage"]["slot_per_1k"] <= 0.5
    assert doc["volume_flags"] == []
    without = _coverage_doc(tmp_path, gap=True)["coverage"]["footprint_m2"]
    assert doc["coverage"]["footprint_m2"] == without


def test_volume_flags_absent_when_coverage_unmeasurable(tmp_path: Path) -> None:
    """A volume with no page rectangle (all reviewer ``mask_px`` crops) cannot
    measure coverage — the document must say "not measured", never publish
    "measured clean"."""
    from shapely.geometry import Polygon
    from shapely.ops import transform as shp_transform

    poly = Polygon([(X0, Y0), (X0 + 500, Y0), (X0 + 500, Y0 - 500), (X0, Y0 - 500)])
    sheet = RawSheetMask(
        slug="volv_p1",
        image=tmp_path / "missing.jpg",
        matrix=((X0, M_PER_PX, 0.0), (Y0, 0.0, -M_PER_PX)),
        style="mask_px",
        rect=None,
        ring_px=((0.0, 0.0), (250.0, 0.0), (250.0, 250.0), (0.0, 250.0)),
        poly_3857=poly,
    )
    finals = {"volv_p1": shp_transform(TO_4326.transform, poly)}
    doc = qa_masks("volv", [sheet], finals, duplicates=NONE)
    assert "coverage" not in doc
    assert "volume_flags" not in doc


def test_status_distinguishes_unmeasured_from_clean_volume_flags(tmp_path: Path) -> None:
    """A masks-qa.json from before the volume metric has no ``volume_flags``
    key: that is "not measured", never "measured clean"."""
    from autogeoref.status import build_status

    for vol, doc in {
        "volflag": {"flagged": {}, "volume_flags": ["coverage_gaps"]},
        "volclean": {"flagged": {}, "volume_flags": []},
        "vollegacy": {"flagged": {}},
    }.items():
        masks = tmp_path / "work" / vol / "masks"
        masks.mkdir(parents=True)
        (masks / "masks-qa.json").write_text(json.dumps(doc))
    rows = {
        r.volume: r
        for r in build_status(
            work=tmp_path / "work",
            fixtures=tmp_path / "no-fixtures",
            tiles=tmp_path / "no-tiles",
        )
    }
    assert rows["volflag"].mask_qa_volume_flags == ("coverage_gaps",)
    assert "mask QA volume flag(s): coverage_gaps" in rows["volflag"].note
    assert rows["volclean"].mask_qa_volume_flags == ()
    assert "volume flag" not in rows["volclean"].note
    assert rows["vollegacy"].mask_qa_volume_flags is None
    assert "volume flag" not in rows["vollegacy"].note
