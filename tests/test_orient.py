"""Quarter-turn orientation: compass detection, normalization, annotation remap.

GOLDEN NOTE (deviation from ideas-analysis.md §3.1 /
REPORT.md): the docs name p16-p22 + p64 as the eight quarter-turned sheets,
but the recorded fits and human ground truth say otherwise. The eight
print-rotated sheets are p17-p23 + p99 (recorded/GT affine fits ~= -89 deg;
p23 and p99 have no recorded auto fit but their human GT frames are rotated).
p16 and p64 are UPRIGHT-but-slightly-crooked (recorded fits +2.5 / +4.4 deg
— they are precisely the near-window rejects §3.2 describes: "p16 misses the
rotation window by 0.09 deg", "p64 borderline... outside scale/rotation
windows"). The golden tests below assert the measured truth.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from autogeoref.frames import rotate_annotation
from autogeoref.orient import detect_quarter_turn_image


def detect_quarter_turn(image_path: Path) -> int:
    """Detect from a path, the way these cases read. The product decodes first."""
    with Image.open(image_path) as im:
        return detect_quarter_turn_image(im, image_path.name)


# print-rotated in the corpus: content a quarter turn from north-up (north=left)
ROTATED_PAGES = ["p17", "p18", "p19", "p20", "p21", "p22", "p23", "p99"]
UPRIGHT_PAGES = ["p2", "p6", "p8", "p37", "p53", "p1", "p3", "p9"]
# named as rotated by the project docs but measured upright (see module docstring)
DOC_LISTED_BUT_UPRIGHT = ["p16", "p64"]


def _synthetic_sheet(w: int = 1326, h: int = 2000, compass: bool = True) -> Image.Image:
    """Portrait Sanborn-like sheet: horizontal text bars + optional compass.

    The compass (16-ray starburst, hub, needle, solid fleur-de-lis at the
    north end, small S mark at the south end) is drawn north-up.
    """
    im = Image.new("L", (w, h), 235)
    d = ImageDraw.Draw(im)
    cx, cy = int(w * 0.53), int(h * 0.45)
    rng = np.random.default_rng(7)
    for _ in range(120):
        x = int(rng.integers(80, w - 140))
        y = int(rng.integers(80, h - 100))
        if abs(x - cx) < 220 and abs(y - cy) < 260:
            continue  # keep the compass area clear, like a real street/park
        d.rectangle([x, y, x + int(rng.integers(30, 90)), y + 8], fill=40)
    if compass:
        for i in range(16):
            a = math.radians(i * 22.5)
            r = 30 if i % 2 == 0 else 22
            d.line([cx, cy, cx + r * math.cos(a), cy + r * math.sin(a)], fill=30, width=2)
        d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=30)
        d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], outline=30, width=2)
        d.line([cx, cy - 112, cx, cy + 112], fill=30, width=2)  # needle
        d.ellipse([cx - 7, cy - 137, cx + 7, cy - 113], fill=30)  # fleur (north)
        d.rectangle([cx - 3, cy + 112, cx + 3, cy + 120], fill=30)  # S (south)
    return im


# ---------------------------------------------------------------- detection


@pytest.mark.parametrize(("ccw_deg", "expected"), [(0, 0), (90, 90), (180, 180), (270, 270)])
def test_detect_synthetic_all_rotations(tmp_path: Path, ccw_deg: int, expected: int) -> None:
    """PIL-rotating the upright sheet CCW by k means k clockwise undoes it."""
    im = _synthetic_sheet()
    if ccw_deg:
        im = im.rotate(ccw_deg, expand=True)
    p = tmp_path / f"synth_{ccw_deg}.png"
    im.save(p)
    assert detect_quarter_turn(p) == expected


def test_detect_portrait_without_compass_falls_back_to_zero(tmp_path: Path) -> None:
    """Documented behavior: portrait 180-flips are NOT detected (none in corpus)."""
    p = tmp_path / "no_compass.png"
    _synthetic_sheet(compass=False).save(p)
    assert detect_quarter_turn(p) == 0


def test_detect_landscape_without_compass_falls_back_to_ninety(tmp_path: Path) -> None:
    """Sanborn sheets are portrait-printed; a landscape scan is quarter-turned."""
    p = tmp_path / "landscape.png"
    _synthetic_sheet(compass=False).rotate(90, expand=True).save(p)
    assert detect_quarter_turn(p) == 90


# -------------------------------------------------------- annotation remap


def _ann(bbox: list[int], orientation: str) -> dict[str, Any]:
    return {
        "streets": [{"name": "61ST", "bbox": list(bbox), "orientation": orientation}],
        "page_number_seen": "17",
    }


def test_rotate_annotation_90() -> None:
    # source image (W, H) = (100, 200); cw 90: (x, y) -> (H - y, x)
    out = rotate_annotation(_ann([10, 20, 30, 60], "horizontal"), 90, (100, 200))
    assert out["streets"][0]["bbox"] == [140, 10, 180, 30]
    assert out["streets"][0]["orientation"] == "vertical"
    assert out["page_number_seen"] == "17"


def test_rotate_annotation_180() -> None:
    out = rotate_annotation(_ann([10, 20, 30, 60], "horizontal"), 180, (100, 200))
    assert out["streets"][0]["bbox"] == [70, 140, 90, 180]
    assert out["streets"][0]["orientation"] == "horizontal"  # unchanged


def test_rotate_annotation_270() -> None:
    out = rotate_annotation(_ann([10, 20, 30, 60], "vertical"), 270, (100, 200))
    assert out["streets"][0]["bbox"] == [20, 70, 60, 90]
    assert out["streets"][0]["orientation"] == "horizontal"


def test_rotate_annotation_zero_is_copy() -> None:
    ann = _ann([1, 2, 3, 4], "vertical")
    out = rotate_annotation(ann, 0, (100, 200))
    assert out == ann
    assert out is not ann and out["streets"][0] is not ann["streets"][0]


def test_rotate_annotation_rejects_bad_angle() -> None:
    with pytest.raises(ValueError, match="rotation_deg must be one of"):
        rotate_annotation(_ann([1, 2, 3, 4], "vertical"), 45, (100, 200))


def test_rotate_annotation_round_trip() -> None:
    """Annotating on the rotated image then correcting recovers the original.

    Upright frame U is (W, H) = (120, 200). The print-rotated original R is
    U rotated 90 ccw, i.e. U rotated 270 cw, so an annotation made on R is
    ``rotate_annotation(ann_U, 270, (W, H))`` in R's (200, 120) frame.
    Correcting R by its detected 90 cw must recover ann_U exactly (integer
    bboxes -> no rounding slack needed).
    """
    ann_u = {
        "streets": [
            {"name": "61ST", "bbox": [8, 30, 48, 65], "orientation": "horizontal"},
            {"name": "PRAIRIE AVE.", "bbox": [70, 10, 100, 180], "orientation": "vertical"},
        ],
        "page_number_seen": "2",
    }
    ann_r = rotate_annotation(ann_u, 270, (120, 200))
    assert rotate_annotation(ann_r, 90, (200, 120)) == ann_u
    # and the 180 involution
    assert rotate_annotation(rotate_annotation(ann_u, 180, (120, 200)), 180, (120, 200)) == ann_u


# ------------------------------------------------------------------ golden


@pytest.fixture(scope="session")
def ref_sheets_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / "ref-volume" / "sheets"


@pytest.mark.golden
def test_golden_rotated_sheets_detect_quarter_turn(ref_sheets_dir: Path) -> None:
    """All eight print-rotated sheets detect the SAME quarter turn: 90 cw.

    90 (north=left on the page) matches the recorded ~ -89 deg affine fits:
    rotating these sheets 90 deg clockwise makes them north-up, which puts
    their (currently page-horizontal) N-S avenue labels vertical — the
    orientation the annotator prompt assumes. Asserting one common value is
    stronger than the "at least 6 of 8 agree" floor.
    """
    got = {p: detect_quarter_turn(ref_sheets_dir / f"{p}_small.jpg") for p in ROTATED_PAGES}
    assert all(v in (90, 270) for v in got.values()), got
    assert set(got.values()) == {90}, got


@pytest.mark.golden
def test_golden_upright_sheets_detect_zero(ref_sheets_dir: Path) -> None:
    got = {p: detect_quarter_turn(ref_sheets_dir / f"{p}_small.jpg") for p in UPRIGHT_PAGES}
    assert got == dict.fromkeys(UPRIGHT_PAGES, 0)


@pytest.mark.golden
def test_golden_doc_listed_sheets_are_actually_upright(ref_sheets_dir: Path) -> None:
    """p16/p64 are named rotated by the docs but are measured upright (crooked).

    Recorded fits: p16 +2.5 deg, p64 +4.4 deg — within-quadrant crooked
    sheets, not quarter turns (module docstring has the full accounting).
    Rotating them would BREAK two good sheets, so 0 is asserted.
    """
    pages = DOC_LISTED_BUT_UPRIGHT
    got = {p: detect_quarter_turn(ref_sheets_dir / f"{p}_small.jpg") for p in pages}
    assert got == dict.fromkeys(pages, 0)


# ------------------------------------------------------------------- prep


def test_prep_sheet_normalize_orientation(tmp_path: Path) -> None:
    from autogeoref.prep import prep_sheet

    upright = _synthetic_sheet()  # (1326, 2000)
    src_dir = tmp_path / "regions"
    src_dir.mkdir()
    src = src_dir / "chicago_ill_1895_vol_16_p17.jpg"
    upright.rotate(90, expand=True).convert("RGB").save(src)  # (2000, 1326) north=left
    out_dir = tmp_path / "sheets"

    entry = prep_sheet(src, out_dir, "17", normalize_orientation=True)
    assert entry["rotation_applied"] == 90
    assert entry["full_size"] == [2000, 1326]  # source scan frame, unrotated
    assert entry["small_size"] == [1326, 2000]  # written (upright) frame
    with Image.open(out_dir / "p17_small.jpg") as im:
        assert im.size == (1326, 2000)
    assert detect_quarter_turn(out_dir / "p17_small.jpg") == 0

    # normalization is the default;
    # the pre-flip behavior stays available via explicit opt-out
    default = prep_sheet(src, out_dir / "default", "17")
    assert default["rotation_applied"] == 90
    plain = prep_sheet(src, out_dir / "plain", "17", normalize_orientation=False)
    assert "rotation_applied" not in plain
    with Image.open(out_dir / "plain" / "p17_small.jpg") as im:
        assert im.size == (2000, 1326)


def test_prep_sheet_normalize_orientation_upright_noop(tmp_path: Path) -> None:
    from autogeoref.prep import prep_sheet

    src_dir = tmp_path / "regions"
    src_dir.mkdir()
    src = src_dir / "p2.jpg"
    _synthetic_sheet().convert("RGB").save(src)
    entry = prep_sheet(src, tmp_path / "sheets", "2", normalize_orientation=True)
    assert "rotation_applied" not in entry
    assert entry["small_size"] == [1326, 2000]
