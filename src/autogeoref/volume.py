"""Volume-level match driver: two-pass constraint derivation + per-sheet gates.

The volume is the irreducible unit of work: the correctness-critical
constraints (scale median +/-10%, rotation median +/-1.5 deg) are computed
ACROSS a volume's sheets, so single sheets cannot be meaningfully gated in
isolation.

Pass 1 runs unconstrained RANSAC over every annotated sheet to establish the
volume's scan scale and print orientation (medians). Pass 2 re-runs with both
pinned to median +/- tolerance and produces the final per-sheet records.

Status vocabulary is part of the recorded contract — downstream stages key on
these exact strings (and prefixes).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .affine import (
    gcps_from_geojson,
    model_rotation_deg,
    model_scales,
    residuals_m,
)
from .matching import (
    FitGates,
    candidate_gcps,
    fold_quadrant_deg,
    gcps_geojson_from,
    ransac_affine,
)

if TYPE_CHECKING:
    from .centerlines import CenterlineIndex
    from .names import Aliases

STATUS_OK = "OK"
STATUS_REJECTED = "REJECTED (no valid RANSAC model)"
STATUS_RESCUED = "OK (rescued)"
STATUS_RESCUE_REVOKED = "REJECTED (rescue revoked: anchors share one street)"
STATUS_CORROBORATED = "OK (rescued, neighbor-corroborated)"
#: Prefix of the >=2-independent-verifiers acceptance status, e.g.
#: ``OK (verified: junction+addresses)``. OK-prefixed deliberately: a
#: verified placement inherits committed semantics everywhere statuses are
#: prefix-checked (match skip, is_committed) — but like corroborated sheets
#: it is pinned out of the seam solve and the corroboration voucher pool.
STATUS_VERIFIED_PREFIX = "OK (verified: "
#: A placement the reviewer confirmed or adjusted in the review UI.
#: OK-prefixed so it inherits committed semantics, and it DOES vouch in
#: corroboration — a human placement is the strongest evidence here — but it is
#: pinned out of the seam solve and excluded from every auto-acceptance
#: statistic, since a manual sheet inflating the auto-accept rate is exactly
#: the red flag the contracts warn about.
STATUS_REVIEWER_VERIFIED = "OK (reviewer-verified)"
#: Pre-rename spelling. It survives only in locally applied review results
#: under gitignored ``work/`` trees (frozen fixtures never carried it);
#: readers accept it via :func:`is_reviewer_verified` so no local state is
#: invalidated. Never written anymore.
STATUS_REVIEWER_VERIFIED_LEGACY = "OK (owner-verified)"

REJECTED_PREFIX = "REJECTED"
REVOKED_PREFIX = "REJECTED (rescue revoked"


def status_verified(channels: list[str]) -> str:
    """Status string for a placement accepted by the named evidence channels."""
    return STATUS_VERIFIED_PREFIX + "+".join(channels) + ")"


def status_ok(status: str) -> bool:
    """Any OK variant — THE status-prefix convention lives in this module.

    Prefix-check outside this file through this predicate (or :func:`is_committed`, which
    names the same test where committed EVIDENCE is meant), never a raw ``startswith("OK")``.
    """
    return status.startswith("OK")


def is_reviewer_verified(status: str) -> bool:
    """True for a reviewer-verified placement, either spelling.

    Every reader of recorded statuses goes through this predicate so
    results applied before the owner->reviewer rename keep their exact
    gate semantics (seam pinning, report accounting).
    """
    return status in (STATUS_REVIEWER_VERIFIED, STATUS_REVIEWER_VERIFIED_LEGACY)


def reviewer_result_key(record: Mapping[str, Any], suffix: str) -> Any:
    """Read ``reviewer_<suffix>`` from a result record, either spelling.

    Results applied before the owner->reviewer rename carry
    ``owner_review``/``owner_mask_px``; the current spelling wins when both
    are present. All result-JSON readers of these keys go through here.
    """
    value = record.get(f"reviewer_{suffix}")
    return value if value is not None else record.get(f"owner_{suffix}")


#: Sheets share a printed orientation; rotation constraint half-width (deg).
ROT_TOL_DEG = 1.5
#: Scale constraint: volume median +/- 10%.
SCALE_TOL_FRAC = 0.10


@dataclass(frozen=True)
class VolumeConstraints:
    """Scale/rotation windows derived from the volume's pass-1 medians."""

    scale_range: tuple[float, float] | None
    rot_range_deg: tuple[float, float] | None
    scale_median: float | None = None
    rotation_median: float | None = None


@dataclass(frozen=True)
class SheetInput:
    """One sheet's matcher inputs: annotation + manifest geometry."""

    page: str
    annotation: dict[str, Any]
    full_size: tuple[float, float]
    scale: float  # annotation-frame / full-res ratio


def _median(values: list[float]) -> float:
    """The production pipeline's median: sorted()[n//2] (upper median)."""
    return sorted(values)[len(values) // 2]


def derive_constraints(
    sheets: list[SheetInput],
    index: CenterlineIndex,
    aliases: Aliases | None = None,
    rot_tol_deg: float = ROT_TOL_DEG,
    scale_tol_frac: float = SCALE_TOL_FRAC,
    page_scale_multiples: Mapping[str, float] | None = None,
) -> VolumeConstraints:
    """Pass 1: unconstrained RANSAC medians -> the pass-2 constraint windows.

    Rotations are folded modulo quadrant turns before the median: sheets share a printed
    orientation, but individual scans can be rotated by 90/180/270 degrees without being crooked
    (see :func:`matching.fold_quadrant_deg`). Pages carrying a scale override
    (``page_scale_multiples``) contribute NEITHER their scale nor their rotation to the medians:
    the override is expressed as a multiple OF those medians, so letting a 4x sheet vote on the
    constant it is measured against would move the target it aims at (and, unfolded, drag every
    other sheet's window with it).
    """
    off_scale = set(page_scale_multiples or {})
    scales: list[float] = []
    rotations: list[float] = []
    for sheet in sheets:
        if sheet.page in off_scale:
            continue
        cands = candidate_gcps(sheet.annotation, index, sheet.scale, aliases)
        # Pass 1 estimates scale and rotation only; positional spread is an
        # acceptance constraint applied in pass 2.
        m, _ = ransac_affine(cands, sheet.full_size, gates=FitGates(loo_spread=False))
        if m is not None:
            sx, sy = model_scales(m)
            scales.extend((sx, sy))
            rotations.append(fold_quadrant_deg(model_rotation_deg(m)))
    scale_range = None
    rot_range = None
    smed: float | None = None
    rmed: float | None = None
    if scales:
        smed = _median(scales)
        scale_range = ((1 - scale_tol_frac) * smed, (1 + scale_tol_frac) * smed)
    if rotations:
        rmed = _median(rotations)
        rot_range = (rmed - rot_tol_deg, rmed + rot_tol_deg)
    return VolumeConstraints(
        scale_range=scale_range,
        rot_range_deg=rot_range,
        scale_median=smed,
        rotation_median=rmed,
    )


def constraints_from_constants(
    scale_m_per_px: float,
    rotation_deg: float,
    rot_tol_deg: float = ROT_TOL_DEG,
    scale_tol_frac: float = SCALE_TOL_FRAC,
) -> VolumeConstraints:
    """Constraint windows from known volume constants (recorded runs, configs)."""
    return VolumeConstraints(
        scale_range=((1 - scale_tol_frac) * scale_m_per_px, (1 + scale_tol_frac) * scale_m_per_px),
        rot_range_deg=(rotation_deg - rot_tol_deg, rotation_deg + rot_tol_deg),
        scale_median=scale_m_per_px,
        rotation_median=rotation_deg,
    )


def constraints_for_page(
    page: str,
    constraints: VolumeConstraints,
    page_scale_multiples: Mapping[str, float] | None = None,
    scale_tol_frac: float = SCALE_TOL_FRAC,
) -> VolumeConstraints:
    """The volume's constraints, RE-CENTERED for a page printed at another scale.

    A book can bind a page printed at a different scale, and the declared multiple re-centers
    that page's scale window without weakening any gate. Three properties stop the exemption
    becoming an escape hatch: **named pages only**, so a page absent from the mapping gets the
    volume's own constraints untouched; **re-centered, NOT widened**, so the same tolerance
    applies around ``multiple x scale_median`` and a fit landing elsewhere still fails; and **a
    declaration, not a search**, so ONE window is tried and nothing iterates over scales looking
    for a fit. The rotation window is untouched — a bound sheet shares the book's orientation.
    """
    multiple = (page_scale_multiples or {}).get(page)
    if multiple is None or constraints.scale_median is None:
        return constraints
    scale = multiple * constraints.scale_median
    return replace(
        constraints,
        scale_range=((1 - scale_tol_frac) * scale, (1 + scale_tol_frac) * scale),
        scale_median=scale,
    )


def match_sheet(
    sheet: SheetInput,
    index: CenterlineIndex,
    constraints: VolumeConstraints,
    aliases: Aliases | None = None,
) -> dict[str, Any]:
    """Pass 2 for one sheet: constrained RANSAC -> a result record.

    The record schema matches the recorded fixtures exactly (same keys, same
    status strings, same rounding). Human pins are not an input here and the
    record carries no score: grading a placement is a separate pass
    (:mod:`autogeoref.score_pass`).
    """
    cands = candidate_gcps(sheet.annotation, index, sheet.scale, aliases)
    m_auto, inliers = ransac_affine(
        cands,
        sheet.full_size,
        scale_range=constraints.scale_range,
        rot_range_deg=constraints.rot_range_deg,
        rot_quadrant_fold=True,
    )
    result: dict[str, Any] = {
        "page": sheet.page,
        "n_streets": len(sheet.annotation["streets"]),
        "n_candidates": len(cands),
        "n_inliers": len(inliers),
    }
    if m_auto is None:
        result["status"] = STATUS_REJECTED
        return result

    auto_pts = gcps_from_geojson(gcps_geojson_from(inliers))
    result["rotation_deg"] = round(model_rotation_deg(m_auto), 2)
    result["auto_residuals_m"] = [round(r, 2) for r in residuals_m(m_auto, auto_pts)]
    result["inlier_streets"] = [list(c.streets) for c in inliers]
    result["gcps_geojson"] = gcps_geojson_from(inliers)
    result["status"] = STATUS_OK
    return result


def is_committed(record: dict[str, Any]) -> bool:
    """True when a record's placement may act as committed evidence.

    The volume's own funnel decides, and only it: any OK status commits. A
    human score can still take a placement out of served evidence, but it does
    so through the scoring pass afterwards, never from inside a run.
    """
    return str(record.get("status", "")).startswith("OK")
