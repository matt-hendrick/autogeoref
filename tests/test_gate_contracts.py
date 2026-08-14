"""Gate contracts: synthetic candidate sets that MUST flag.

(a) collinear inliers; (b) scale 12% off the volume median; (c) rotation 2 deg
off median; (d) a consistent-but-shifted model (all intersections displaced
one block, which implies an off-window scale); (e) a CHAIN — collinear anchors
plus one off-line witness, which clears the full-set spread gate and fails it
on leave-one-out; (f) a MIRRORED model, whose scale, aspect and rotation are
all indistinguishable from the upright twin's. Each must reject, never accept.
A well-spread compliant set is the positive control, and so is an upright scan
at each of the four quadrants — the handedness gate must cost none of them.
"""

import math
from typing import Any

import numpy as np

from autogeoref.affine import (
    TO_3857,
    TO_4326,
    fit_affine,
    model_determinant,
    model_rotation_deg,
    model_scales,
)
from autogeoref.junction_snap import (
    CenterlineWorld,
    JunctionExtraction,
    verify_placement,
    world_from_centerlines,
)
from autogeoref.matching import (
    SPREAD_PERP_FRAC,
    SPREAD_SPAN_FRAC,
    Candidate,
    FitGates,
    fold_quadrant_deg,
    loo_spread_ok,
    perp_spread_frac,
    ransac_affine,
    ransac_affine_diagnostics,
)

# a plausible volume: 0.067 m/px, ~1 deg rotation, sheet 5900 x 7300 px
SCALE = 0.067
ROT = math.radians(1.0)
FULL_SIZE = (5900.0, 7300.0)
ORIGIN_3857 = (-9760000.0, 5140000.0)  # Chicago-ish
SCALE_RANGE = (0.9 * SCALE, 1.1 * SCALE)
ROT_RANGE = (1.0 - 1.5, 1.0 + 1.5)


def world_of(
    px: float,
    py: float,
    scale: float = SCALE,
    rot: float = ROT,
    mirror: bool = False,
) -> tuple[float, float]:
    """Ground-truth affine: pixel -> 3857 -> 4326 (y-down pixels, y-up meters).

    The y flip is what makes an upright placement's determinant negative. With
    ``mirror`` the flip is dropped — a pure rotation of y-down pixels onto
    y-up meters, i.e. the sheet warped back-to-front. Every scale, aspect and
    rotation the gates measure is identical to the upright model's; only the
    handedness differs, which is precisely why no angle test can see it.
    """
    flip = 1.0 if mirror else -1.0
    x = ORIGIN_3857[0] + scale * (math.cos(rot) * px - flip * math.sin(rot) * py)
    y = ORIGIN_3857[1] + scale * (math.sin(rot) * px + flip * math.cos(rot) * py)
    return TO_4326.transform(x, y)


def grid_candidates(
    nx: int = 3,
    ny: int = 3,
    scale: float = SCALE,
    rot: float = ROT,
    mirror: bool = False,
) -> list[Candidate]:
    """A well-spread grid of exact correspondences under the ground-truth affine."""
    cands = []
    for i in range(nx):
        for j in range(ny):
            px = FULL_SIZE[0] * (0.1 + 0.8 * i / max(nx - 1, 1))
            py = FULL_SIZE[1] * (0.1 + 0.8 * j / max(ny - 1, 1))
            cands.append(
                Candidate(
                    pixel=(px, py),
                    world4326=world_of(px, py, scale, rot, mirror),
                    streets=(f"A{i}", f"B{j}"),
                )
            )
    return cands


def _anisotropic_candidates(sx: float, sy: float) -> list[Candidate]:
    """A well-spread grid under an axis-anisotropic (but upright) affine."""
    cands = []
    for i in range(3):
        for j in range(3):
            px = FULL_SIZE[0] * (0.1 + 0.4 * i)
            py = FULL_SIZE[1] * (0.1 + 0.4 * j)
            x = ORIGIN_3857[0] + sx * px
            y = ORIGIN_3857[1] - sy * py
            cands.append(
                Candidate(
                    pixel=(px, py),
                    world4326=TO_4326.transform(x, y),
                    streets=(f"A{i}", f"B{j}"),
                )
            )
    return cands


def test_contract_values() -> None:
    """The matcher spread constants are contracts. Never tune them casually."""
    assert SPREAD_SPAN_FRAC == 0.30
    assert SPREAD_PERP_FRAC == 0.05


def test_positive_control_accepts() -> None:
    """A 3x3 grid: 2-D with any one anchor removed, so the LOO gate is a no-op."""
    m, inl = ransac_affine(
        grid_candidates(), FULL_SIZE, scale_range=SCALE_RANGE, rot_range_deg=ROT_RANGE
    )
    assert m is not None
    assert len(inl) == 9
    assert loo_spread_ok(np.array([c.pixel for c in inl]), *FULL_SIZE)


def test_diagnostic_trace_preserves_strict_matcher_result() -> None:
    trace = ransac_affine_diagnostics(
        grid_candidates(),
        FULL_SIZE,
        scale_range=SCALE_RANGE,
        rot_range_deg=ROT_RANGE,
    )
    assert trace["failure_class"] == "strict_accept"
    assert trace["strict_fit"]["n_inliers"] == 9

    sparse = ransac_affine_diagnostics(
        grid_candidates(nx=1, ny=3),
        FULL_SIZE,
        scale_range=SCALE_RANGE,
        rot_range_deg=ROT_RANGE,
    )
    assert sparse["failure_class"] == "fewer_than_4_candidates"


def _diagnose(cands: list[Candidate]) -> dict[str, Any]:
    return ransac_affine_diagnostics(
        cands, FULL_SIZE, scale_range=SCALE_RANGE, rot_range_deg=ROT_RANGE
    )


def test_failure_class_scale_window() -> None:
    """A coherent fit whose implied scale leaves the volume window is named as such."""
    trace = _diagnose(grid_candidates(scale=SCALE * 1.12))
    assert trace["failure_class"] == "scale_window"
    assert trace["wide_plausibility_fit"]["has_model"] is True


def test_failure_class_rotation_window() -> None:
    trace = _diagnose(grid_candidates(rot=math.radians(3.0)))
    assert trace["failure_class"] == "rotation_window"
    assert trace["wide_plausibility_fit"]["has_model"] is True


def test_failure_class_no_coherent_wide_fit() -> None:
    """Collinear worlds under 2-D pixels: every sampled affine is degenerate."""
    pixels = [
        (FULL_SIZE[0] * 0.1, FULL_SIZE[1] * 0.1),
        (FULL_SIZE[0] * 0.35, FULL_SIZE[1] * 0.9),
        (FULL_SIZE[0] * 0.65, FULL_SIZE[1] * 0.1),
        (FULL_SIZE[0] * 0.9, FULL_SIZE[1] * 0.9),
    ]
    cands = [
        Candidate(pixel=(px, py), world4326=world_of(px, 0.0), streets=(f"A{k}", f"B{k}"))
        for k, (px, py) in enumerate(pixels)
    ]
    trace = _diagnose(cands)
    assert trace["failure_class"] == "no_coherent_wide_fit"
    assert trace["wide_plausibility_fit"]["has_model"] is False


def test_failure_class_handedness() -> None:
    """A mirrored constellation is named by the gate that refuses it — not
    mislabelled as incoherent. Same handedness defect as ``_024`` p98."""
    trace = _diagnose(grid_candidates(mirror=True))
    assert trace["failure_class"] == "handedness"
    # the wide fit keeps the handedness contract, so it finds nothing...
    assert trace["wide_plausibility_fit"]["has_model"] is False
    # ...and only the unconstrained probe can hold the reflection up to name it
    assert trace["unconstrained_fit"]["has_model"] is True


def test_failure_class_aspect() -> None:
    """A squashed model (axis scales 1:2.5) is named by the aspect gate.

    No page scale window here: a +/-10% window can never hold both axes of a
    ratio-2.5 model, so under the gate order (scale before aspect) the aspect
    name is reachable only when ``scale_range`` is None.
    """
    trace = ransac_affine_diagnostics(
        _anisotropic_candidates(SCALE, SCALE * 2.5),
        FULL_SIZE,
        scale_range=None,
        rot_range_deg=None,
    )
    assert trace["failure_class"] == "aspect"
    assert trace["wide_plausibility_fit"]["has_model"] is False
    assert trace["unconstrained_fit"]["has_model"] is True


def test_anisotropic_probe_outside_scale_window_is_not_named_aspect() -> None:
    """A probe whose scales fail the page window is a scale rejection: the
    classifier must mirror gates_ok's order (scale before aspect) and fall
    through to the pre-existing label, never mislabel it as an aspect defect."""
    trace = _diagnose(_anisotropic_candidates(SCALE * 2, SCALE * 5))
    assert trace["failure_class"] == "no_coherent_wide_fit"
    # the probe DID find the squashed model — the label falls through because
    # its scales fail the page window, not because the probe found nothing
    assert trace["unconstrained_fit"]["has_model"] is True


def test_failure_class_leave_one_out_spread() -> None:
    """The chain constellation is named by the gate that refuses it."""
    trace = _diagnose(chain_candidates())
    assert trace["failure_class"] == "leave_one_out_spread"
    assert trace["no_loo_fit"]["has_model"] is True
    assert trace["no_loo_fit"]["n_inliers"] == 4


def test_failure_class_constrained_insufficient_inliers() -> None:
    """A wide-plausible triangle plus one impossible correspondence: the
    constrained fit cannot reach four inliers and says so."""
    good = [
        (FULL_SIZE[0] * 0.1, FULL_SIZE[1] * 0.1),
        (FULL_SIZE[0] * 0.9, FULL_SIZE[1] * 0.15),
        (FULL_SIZE[0] * 0.15, FULL_SIZE[1] * 0.9),
    ]
    cands = [
        Candidate(pixel=(px, py), world4326=world_of(px, py), streets=(f"A{k}", f"B{k}"))
        for k, (px, py) in enumerate(good)
    ]
    # kilometres off: any sample containing it implies a wildly anisotropic
    # affine, so it can neither join nor win the wide fit
    px, py = FULL_SIZE[0] * 0.9, FULL_SIZE[1] * 0.9
    far = world_of(px + 400000, py - 300000)
    cands.append(Candidate(pixel=(px, py), world4326=far, streets=("A9", "B9")))
    trace = _diagnose(cands)
    assert trace["failure_class"] == "constrained_insufficient_inliers"
    assert trace["no_loo_fit"]["has_model"] is False


def test_collinear_inliers_reject() -> None:
    # six exact correspondences, all on one diagonal line across the sheet
    cands = []
    for k in range(6):
        px = FULL_SIZE[0] * (0.1 + 0.16 * k)
        py = FULL_SIZE[1] * (0.1 + 0.16 * k)
        cands.append(
            Candidate(pixel=(px, py), world4326=world_of(px, py), streets=(f"A{k}", f"B{k}"))
        )
    m, inl = ransac_affine(cands, FULL_SIZE, scale_range=SCALE_RANGE, rot_range_deg=ROT_RANGE)
    assert m is None
    assert inl == []


def test_scale_12pct_off_rejects() -> None:
    cands = grid_candidates(scale=SCALE * 1.12)
    m, _ = ransac_affine(cands, FULL_SIZE, scale_range=SCALE_RANGE, rot_range_deg=ROT_RANGE)
    assert m is None


def test_rotation_2deg_off_rejects() -> None:
    cands = grid_candidates(rot=math.radians(3.0))  # median 1 deg, window +/-1.5
    m, _ = ransac_affine(cands, FULL_SIZE, scale_range=SCALE_RANGE, rot_range_deg=ROT_RANGE)
    assert m is None


def test_rotation_quadrant_scan_is_not_crooked() -> None:
    # a 90deg-rotated SCAN of a straight sheet must pass with quadrant folding
    cands = grid_candidates(rot=math.radians(91.0))
    m, _ = ransac_affine(
        cands,
        FULL_SIZE,
        scale_range=SCALE_RANGE,
        rot_range_deg=ROT_RANGE,
        rot_quadrant_fold=True,
    )
    assert m is not None
    # ...but a crooked sheet (2 deg off within the quadrant) still rejects
    cands = grid_candidates(rot=math.radians(93.0))
    m, _ = ransac_affine(
        cands,
        FULL_SIZE,
        scale_range=SCALE_RANGE,
        rot_range_deg=ROT_RANGE,
        rot_quadrant_fold=True,
    )
    assert m is None


def test_mirrored_model_rejects() -> None:
    """A reflected placement is not a placement of this scan.

    ``_024`` p98 (det +4.387e-03, rotation -178.856 deg) was a plain RANSAC
    accept, served since July, painting gdalwarp's opaque black fill over 4.6%
    of the points sampled inside its own mask.
    Its correspondences are internally flawless; the handedness is the defect.
    """
    cands = grid_candidates(mirror=True)
    m, inl = ransac_affine(cands, FULL_SIZE, scale_range=SCALE_RANGE, rot_range_deg=ROT_RANGE)
    assert m is None
    assert inl == []

    # It is the DETERMINANT that rejects it, not scale, aspect or rotation:
    # the upright twin of the same grid is a clean accept, and the mirrored
    # model's own measurements sit inside every other window.
    upright, upright_inl = ransac_affine(
        grid_candidates(), FULL_SIZE, scale_range=SCALE_RANGE, rot_range_deg=ROT_RANGE
    )
    assert upright is not None
    assert len(upright_inl) == 9
    mirrored_fit = fit_affine(
        [(c.pixel[0], c.pixel[1], *TO_3857.transform(*c.world4326)) for c in cands]
    )
    assert model_determinant(mirrored_fit) > 0
    assert model_determinant(upright) < 0
    sx, sy = model_scales(mirrored_fit)
    assert SCALE_RANGE[0] < sx < SCALE_RANGE[1] and SCALE_RANGE[0] < sy < SCALE_RANGE[1]
    assert 0.5 < sx / sy < 2.0
    assert ROT_RANGE[0] <= model_rotation_deg(mirrored_fit) <= ROT_RANGE[1]

    # ...and no widening of the plausibility windows recovers it either
    m, _ = ransac_affine(cands, FULL_SIZE)
    assert m is None


def test_mirrored_model_at_180_rejects_despite_the_quadrant_fold() -> None:
    """THE `_024` p98 hole: the fold cannot distinguish 180 deg from a reflection.

    ``fold_quadrant_deg`` exists so a quadrant-rotated SCAN of a straight sheet
    is not called crooked, and it is why p98 sailed through: a reflection near
    180 deg folds onto ~0 deg and lands dead centre of the rotation window. The
    fold stays; the determinant is what refuses the sheet.
    """
    cands = grid_candidates(rot=math.radians(181.0), mirror=True)
    mirrored_fit = fit_affine(
        [(c.pixel[0], c.pixel[1], *TO_3857.transform(*c.world4326)) for c in cands]
    )
    # the rotation gate is genuinely satisfied — folded deviation is ~0
    assert abs(fold_quadrant_deg(model_rotation_deg(mirrored_fit) - 1.0)) < 0.5
    m, _ = ransac_affine(
        cands,
        FULL_SIZE,
        scale_range=SCALE_RANGE,
        rot_range_deg=ROT_RANGE,
        rot_quadrant_fold=True,
    )
    assert m is None


def test_upright_scans_at_every_quadrant_still_accept() -> None:
    """The handedness gate must not cost the rotated-format sheets.

    An upright placement is negative-determinant at EVERY quadrant — that is
    the whole reason the sign is a safe test — so all four quadrant-rotated
    scans must keep accepting under the fold.
    """
    for quadrant_deg in (1.0, 91.0, 181.0, 271.0):
        cands = grid_candidates(rot=math.radians(quadrant_deg))
        m, inl = ransac_affine(
            cands,
            FULL_SIZE,
            scale_range=SCALE_RANGE,
            rot_range_deg=ROT_RANGE,
            rot_quadrant_fold=True,
        )
        assert m is not None, f"{quadrant_deg} deg scan rejected"
        assert len(inl) == 9
        assert model_determinant(m) < 0


def test_consistent_but_shifted_model_rejects() -> None:
    """All intersections displaced one block: matched to the wrong parallel
    street grid whose spacing differs, so the implied scale leaves the
    volume window. The model is internally consistent — only the volume
    scale contract kills it."""
    # wrong-grid spacing: pixel grid says blocks are 1000 px apart, but the
    # matched (wrong) world grid has 0.85x the true spacing
    cands = grid_candidates(scale=SCALE * 0.85)
    m, _ = ransac_affine(cands, FULL_SIZE, scale_range=SCALE_RANGE, rot_range_deg=ROT_RANGE)
    assert m is None
    # sanity: without the volume constraint the same set fits happily —
    # proving the contract, not RANSAC, is what rejects it
    m, _ = ransac_affine(cands, FULL_SIZE)
    assert m is not None


def chain_candidates() -> list[Candidate]:
    """THE _089 p15 shape: three anchors on one line across the sheet, plus a
    single off-line witness. Exact correspondences — internally flawless."""
    pixels = [
        (FULL_SIZE[0] * 0.1, FULL_SIZE[1] * 0.1),  # )
        (FULL_SIZE[0] * 0.5, FULL_SIZE[1] * 0.5),  # )- one line, spanning the sheet
        (FULL_SIZE[0] * 0.9, FULL_SIZE[1] * 0.9),  # )
        (FULL_SIZE[0] * 0.9, FULL_SIZE[1] * 0.3),  # the lone witness
    ]
    return [
        Candidate(pixel=(px, py), world4326=world_of(px, py), streets=(f"A{k}", f"B{k}"))
        for k, (px, py) in enumerate(pixels)
    ]


def test_chain_inliers_reject() -> None:
    """A line plus one witness is not a 2-D fit.

    This constellation passes the ordinary gates — it
    spans the sheet in both axes and its full-set perpendicular spread clears
    the 5% gate several times over — yet the entire perpendicular dimension of
    the affine rests on ONE anchor. That is _089 p15, which was strict-accepted
    with residuals of 0.01-5.3 m and sat 217.6 m from the human GCPs. The
    leave-one-out gate is what refuses it.
    """
    cands = chain_candidates()
    pts = np.array([c.pixel for c in cands])

    # the OLD gates all pass — otherwise this test would not be exercising the new one
    assert (max(p[0] for p in pts) - min(p[0] for p in pts)) >= SPREAD_SPAN_FRAC * FULL_SIZE[0]
    assert (max(p[1] for p in pts) - min(p[1] for p in pts)) >= SPREAD_SPAN_FRAC * FULL_SIZE[1]
    assert perp_spread_frac(pts, *FULL_SIZE) > SPREAD_PERP_FRAC

    # ...and drop the witness and the spread collapses to nothing
    assert not loo_spread_ok(pts, *FULL_SIZE)

    m, inl = ransac_affine(cands, FULL_SIZE, scale_range=SCALE_RANGE, rot_range_deg=ROT_RANGE)
    assert m is None
    assert inl == []

    # the gate, and nothing else, is what rejected it: opted out, the same set fits
    m, inl = ransac_affine(
        cands,
        FULL_SIZE,
        scale_range=SCALE_RANGE,
        rot_range_deg=ROT_RANGE,
        gates=FitGates(loo_spread=False),
    )
    assert m is not None
    assert len(inl) == 4


def test_sparse_corner_cluster_rejects() -> None:
    # spread gate: inliers confined to one corner (<30% span) must reject
    cands = []
    for i in range(3):
        for j in range(3):
            px = FULL_SIZE[0] * (0.05 + 0.08 * i)
            py = FULL_SIZE[1] * (0.05 + 0.08 * j)
            cands.append(
                Candidate(pixel=(px, py), world4326=world_of(px, py), streets=(f"A{i}", f"B{j}"))
            )
    m, _ = ransac_affine(cands, FULL_SIZE, scale_range=SCALE_RANGE, rot_range_deg=ROT_RANGE)
    assert m is None


# ---------------------------------------------------------------------------
# Junction verifier: supports or abstains, never refutes.
#
# Regression fixture for the contract below.

JUNCTION_BLOCK_M = 120.0  # synthetic city block, metres
JUNCTION_M_PER_PX = 0.5  # small-frame scale for the synthetic sheet


def _seg(
    a: tuple[float, float],
    b: tuple[float, float],
    name: str,
    fnode: int,
    tnode: int,
) -> dict[str, Any]:
    return {
        "properties": {"street_nam": name, "fnode_id": fnode, "tnode_id": tnode},
        "geometry": {
            "type": "LineString",
            "coordinates": [list(TO_4326.transform(*a)), list(TO_4326.transform(*b))],
        },
    }


def _grid_centerlines(n: int = 7) -> list[dict[str, Any]]:
    """A synthetic street grid in WGS84, with the node ids the scorer needs."""
    ox, oy = ORIGIN_3857
    feats: list[dict[str, Any]] = []
    for i in range(n):  # east-west streets
        y = oy - i * JUNCTION_BLOCK_M
        for j in range(n - 1):
            x0 = ox + j * JUNCTION_BLOCK_M
            x1 = ox + (j + 1) * JUNCTION_BLOCK_M
            feats.append(_seg((x0, y), (x1, y), f"EW{i}", int(i * n + j), int(i * n + j + 1)))
    for j in range(n):  # north-south streets
        x = ox + j * JUNCTION_BLOCK_M
        for i in range(n - 1):
            y0 = oy - i * JUNCTION_BLOCK_M
            y1 = oy - (i + 1) * JUNCTION_BLOCK_M
            feats.append(_seg((x, y0), (x, y1), f"NS{j}", int(i * n + j), int((i + 1) * n + j)))
    return feats


def _synthetic_sheet() -> tuple[JunctionExtraction, list[list[float]]]:
    """Drawn junctions + skeleton on a synthetic sheet, and its TRUE affine."""
    n = 7
    world = world_from_centerlines(_grid_centerlines(n))
    assert world is not None
    px_per_block = JUNCTION_BLOCK_M / JUNCTION_M_PER_PX
    size = int((n - 1) * px_per_block) + 40
    skeleton = np.zeros((size, size), dtype=bool)
    junctions: list[tuple[float, float]] = []
    ox, oy = ORIGIN_3857
    for i in range(n):
        for j in range(n):
            px = 20.0 + j * px_per_block
            py = 20.0 + i * px_per_block
            junctions.append((px, py))
            skeleton[int(py), :] = True
            skeleton[:, int(px)] = True
    extraction = JunctionExtraction(
        junctions_px=np.array(junctions, dtype=float),
        junction_types=["X"] * len(junctions),
        skeleton=skeleton,
        diagnostics={"n_junctions": len(junctions)},
    )
    # small-frame pixel -> 3857: y-down pixels, y-up metres
    true_affine = [
        [ox - 20.0 * JUNCTION_M_PER_PX, JUNCTION_M_PER_PX, 0.0],
        [oy + 20.0 * JUNCTION_M_PER_PX, 0.0, -JUNCTION_M_PER_PX],
    ]
    return extraction, true_affine


def _shift_east(m: list[list[float]], metres: float) -> list[list[float]]:
    out = [list(m[0]), list(m[1])]
    out[0][0] += metres
    return out


def test_junction_verifier_supports_or_abstains_never_refutes() -> None:
    """CONTRACT: ``SnapVerdict.supports`` is True or None. Never False.

    A refute is the SOLE blocker in verified_accept, and the old binary rule emitted one for
    anything its support clause did not catch — a BLOCKING vote on sheets the channel had no
    evidence about. Measured over the ground-truth volumes, that veto never changed an outcome
    on a single page, while it falsely vetoed a fifth of the correctly placed human sheets. So
    the channel abstains where it used to refute, and restoring `else False` here re-arms a veto
    with no measured benefit.
    """
    extraction, true_affine = _synthetic_sheet()
    world: CenterlineWorld = world_from_centerlines(_grid_centerlines(7))
    assert world is not None

    seen = set()
    for metres in (0.0, 2.0, 90.0, JUNCTION_BLOCK_M, 1.0):
        verdict = verify_placement(
            extraction,
            _shift_east(true_affine, metres),
            world,
            small_to_full=1.0,
        )
        seen.add(verdict.supports)
        assert verdict.supports is not False, (
            f"junction channel REFUTED at {metres:.0f} m east — it may only "
            f"support or abstain (verdict: {verdict.supports})"
        )
    assert True in seen or None in seen
