"""Gate contracts for the per-page scale override.

A book can bind a sheet printed at another scale — a district plate at several times the
book's metres-per-pixel — and the volume scale window rejects it however well it fits. The
override RE-CENTERS that window on a NAMED page.

These are contract tests in the sense of `test_gate_contracts.py`: an exemption from a
gate is one screw-up away from being an escape hatch, and each test here pins one of the
properties that keeps it an exemption. They live in their own file only for historical
reasons and belong to the same contract family.
"""

import math
from pathlib import Path
from typing import Any

import pytest

import autogeoref.volume as volume_mod
from autogeoref.affine import TO_4326
from autogeoref.config.load import load_city_config
from autogeoref.config.model import ConfigError
from autogeoref.matching import Candidate, ransac_affine
from autogeoref.slugs import page_from_slug
from autogeoref.volume import SheetInput, constraints_for_page, constraints_from_constants

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

# the same synthetic volume as test_gate_contracts: 0.067 m/px, ~1 deg, 5900x7300
SCALE = 0.067
ROT = math.radians(1.0)
FULL_SIZE = (5900.0, 7300.0)
ORIGIN_3857 = (-9760000.0, 5140000.0)


def world_of(px: float, py: float, scale: float = SCALE, rot: float = ROT) -> tuple[float, float]:
    x = ORIGIN_3857[0] + scale * (math.cos(rot) * px + math.sin(rot) * py)
    y = ORIGIN_3857[1] + scale * (math.sin(rot) * px - math.cos(rot) * py)
    return TO_4326.transform(x, y)


def grid_candidates(scale: float = SCALE) -> list[Candidate]:
    """A well-spread 3x3 of exact correspondences at the given scale."""
    cands = []
    for i in range(3):
        for j in range(3):
            px = FULL_SIZE[0] * (0.1 + 0.4 * i)
            py = FULL_SIZE[1] * (0.1 + 0.4 * j)
            cands.append(
                Candidate(
                    pixel=(px, py),
                    world4326=world_of(px, py, scale),
                    streets=(f"A{i}", f"B{j}"),
                )
            )
    return cands


def test_override_touches_only_the_pages_that_declare_it() -> None:
    """The blast radius is exactly the named pages."""
    vc = constraints_from_constants(SCALE, 1.0)
    assert constraints_for_page("7", vc, {"cbd1": 4.0}) is vc  # untouched, not merely equal
    assert constraints_for_page("7", vc, {}) is vc
    assert constraints_for_page("cbd1", vc, {}) is vc  # no declaration, no exemption


def test_override_recenters_the_window_and_does_not_widen_it() -> None:
    """The SAME +/-10% tolerance, around a different center."""
    vc = constraints_from_constants(SCALE, 1.0)
    over = constraints_for_page("cbd1", vc, {"cbd1": 4.0})
    assert over.scale_median == 4.0 * SCALE
    assert over.scale_range == (0.9 * 4.0 * SCALE, 1.1 * 4.0 * SCALE)
    # THE invariant: window width as a FRACTION of its center is unchanged
    assert vc.scale_range is not None and over.scale_range is not None
    width = lambda r, c: (r[1] - r[0]) / c  # noqa: E731
    assert width(over.scale_range, over.scale_median) == pytest.approx(
        width(vc.scale_range, vc.scale_median)
    )
    assert over.rot_range_deg == vc.rot_range_deg  # a bound sheet shares the orientation


def test_override_still_rejects_a_sheet_off_the_recentered_scale() -> None:
    """An exemption, not a free pass: 12% off the RE-CENTERED median still fails.

    This is the property that killed a hand-computed 0.23 m/px prior for the real
    sheets (its assumed scan dpi was 18% wrong): a declared window that is itself
    wrong yields NO model, rather than quietly accepting whatever fits.
    """
    vc = constraints_from_constants(SCALE, 1.0)
    over = constraints_for_page("cbd1", vc, {"cbd1": 4.0})
    m, _ = ransac_affine(
        grid_candidates(scale=4.0 * SCALE * 1.12),
        FULL_SIZE,
        scale_range=over.scale_range,
        rot_range_deg=over.rot_range_deg,
    )
    assert m is None


def test_override_accepts_the_off_scale_sheet_it_declares() -> None:
    """Positive control: a true 4x sheet passes, and the volume's own window rejects it."""
    vc = constraints_from_constants(SCALE, 1.0)
    over = constraints_for_page("cbd1", vc, {"cbd1": 4.0})
    cands = grid_candidates(scale=4.0 * SCALE)

    m, inl = ransac_affine(
        cands, FULL_SIZE, scale_range=over.scale_range, rot_range_deg=over.rot_range_deg
    )
    assert m is not None
    assert len(inl) == 9

    m_vol, _ = ransac_affine(
        cands, FULL_SIZE, scale_range=vc.scale_range, rot_range_deg=vc.rot_range_deg
    )
    assert m_vol is None  # the override is doing the work, not a loosened gate


def test_overridden_pages_do_not_vote_on_the_volume_median() -> None:
    """A 4x sheet must not drag the constant its own multiple is measured against."""
    matched: list[str] = []

    def spy(
        annotation: dict[str, Any], index: object, scale: float, aliases: object = None
    ) -> list[Any]:
        matched.append(annotation["page"])
        return []

    sheets = [
        SheetInput(page=p, annotation={"streets": [], "page": p}, full_size=FULL_SIZE, scale=1.0)
        for p in ("1", "cbd1")
    ]
    real = volume_mod.candidate_gcps
    volume_mod.candidate_gcps = spy
    try:
        volume_mod.derive_constraints(sheets, None, None, page_scale_multiples={"cbd1": 4.0})
    finally:
        volume_mod.candidate_gcps = real

    assert matched == ["1"], f"pass 1 matched an overridden page: {matched}"


def test_the_exclusion_moves_the_median_and_that_is_the_point() -> None:
    """The exclusion is not cosmetic: an off-scale sheet really does pollute pass 1.

    Pass 1 is UNCONSTRAINED — that is how the volume constant gets discovered — so an off-scale
    sheet fits happily at its own scale and VOTES. It does not drag the median, which is robust
    to a couple of outliers; it pushes extra values onto the END of the sorted list, which
    SHIFTS THE MEDIAN'S INDEX onto a neighbouring sheet's value. That is a small move, and a
    small move is all it takes: on the real fixtures the shift was enough to reject a sheet that
    passes against the honest window. Reproduced here with three in-book sheets at distinct
    scales, so the index shift is visible.
    """
    scales = {"1": 0.0665, "2": 0.0670, "3": 0.0675, "cbd1": 4.0 * SCALE}
    sheets = [
        SheetInput(page=p, annotation={"streets": [], "page": p}, full_size=FULL_SIZE, scale=1.0)
        for p in scales
    ]

    def spy(
        annotation: dict[str, Any], index: object, scale: float, aliases: object = None
    ) -> list[Any]:
        return grid_candidates(scale=scales[annotation["page"]])

    real = volume_mod.candidate_gcps
    volume_mod.candidate_gcps = spy
    try:
        polluted = volume_mod.derive_constraints(sheets, None, None)
        honest = volume_mod.derive_constraints(
            sheets, None, None, page_scale_multiples={"cbd1": 4.0}
        )
    finally:
        volume_mod.candidate_gcps = real

    assert polluted.scale_median is not None and honest.scale_median is not None
    # the 4x sheet's two values shift the median one position up the sorted list
    assert honest.scale_median == pytest.approx(0.0670, rel=1e-3)
    assert polluted.scale_median == pytest.approx(0.0675, rel=1e-3)
    assert polluted.scale_median > honest.scale_median


def test_named_page_ids_do_not_reopen_the_crop_layer_door() -> None:
    """`pcbd1` parses; `_p10_1` must still NOT.

    The allow-list is literal ids, never a relaxation of the numeric form. A
    general "letters allowed" rule would also admit the volunteer crop layers,
    whose GCP pixels are in a CROP's frame with no offset back to the page —
    binding those to a full page fabricates a placement that fits cleanly and is
    entirely wrong (see the slugs module docstring).
    """
    assert page_from_slug("chicago_ill_1906_vol_1_pcbd1") == "cbd1"
    assert page_from_slug("chicago_ill_1906_vol_1_pcbd2") == "cbd2"
    assert page_from_slug("chicago_ill_1906_vol_1_p10") == "10"
    assert page_from_slug("chicago_ill_1906_vol_1_p10_1") is None
    assert page_from_slug("chicago_ill_1906_vol_1_p10_2") is None


def test_shipped_config_declares_the_cbd_sheets_and_nothing_else() -> None:
    """The exemption is opt-in per page, and only the two Loop books have it."""
    cfg = load_city_config(CONFIGS / "chicago" / "chicago.toml")
    for vid in ("sanborn01790_017", "sanborn01790_018"):
        assert cfg.volume(vid).page_scale_multiples == {"cbd1": 4.0, "cbd2": 4.0}
    assert cfg.volume("sanborn01790_024").page_scale_multiples == {}


def test_config_rejects_a_multiple_that_is_really_a_knob(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """m/px values, no-ops and implausible ratios are all refused at load."""
    head = '[city]\nname = "X"\naliases_dir = "a"\n[volumes.v1]\n'
    for bad in ('{ cbd1 = "4.0" }', "{ cbd1 = 0.0 }", "{ cbd1 = 40.0 }", "4.0", "{ cbd1 = 0.27 }"):
        path = tmp_path / "bad.toml"
        path.write_text(f"{head}page_scale_multiples = {bad}\n")
        with pytest.raises(ConfigError):
            load_city_config(path)
