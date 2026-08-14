"""The rescue stage: translation-only placement of REJECTED pages.

The linear part is pinned to the volume constants and only the translation is
solved. A cluster with a disjoint pair commits; one whose anchors all share a
street is recorded revoked, for corroboration to answer.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from ..affine import TO_3857, fit_affine_checked, model_determinant, model_rotation_deg
from ..errors import PipelineError
from ..matching import Candidate, candidate_gcps, gcps_geojson_from
from ..own_grid import OwnGridEstimator
from ..paths import iter_results, write_result
from ..rescue import has_disjoint_pair, pinned_linear, translation_fit, with_synthetic_corners
from ..sheet_inputs import sheet_input_from
from ..volume import (
    REJECTED_PREFIX,
    REVOKED_PREFIX,
    STATUS_RESCUE_REVOKED,
    STATUS_RESCUED,
)
from ..volume_constants import resolve_constants

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from ..centerlines import CenterlineIndex
    from ..config.model import VolumeConfig
    from ..paths import VolumePaths

logger = logging.getLogger(__name__)

#: Agreeing anchors a cluster found on the sheet's own grid must have. Higher
#: than ``rescue.MIN_AGREE`` on purpose: see :func:`_fit_on_own_grid`.
OWN_GRID_MIN_AGREE = 3
#: And what one whose anchors all share a street must have. That cluster cannot
#: commit on its own and goes to corroboration, which can and has vouched for a
#: parallel-street slide hundreds of metres long.
OWN_GRID_SHARED_STREET_MIN_AGREE = 4
#: How far mod 90 the sheet's own rotation must sit from the volume's before its
#: own grid is a distinct hypothesis.
OWN_GRID_DEV_GATE = 2.0
#: A mod-180 turn beyond this is a quarter turn the volume's pin cannot reach at
#: any deviation.
OWN_GRID_TURN_GATE = 15.0


def _translation_fit_any_quadrant(
    cands: list[Candidate],
    page: str,
    page_scale: float,
    pin_rot: float,
    aliases: Mapping[str, str] | None,
) -> tuple[list[list[float]] | None, list[Candidate]]:
    """Retry the pinned-translation fit at the three other quadrants.

    OPT-IN (``VolumeConfig.quadrant_rescue``): a quadrant-rotated scan of a
    straight sheet fails the base-orientation fit outright (every implied
    translation is scattered by the ~90 deg orientation error). Try the other
    quadrants in fixed order; every rescue gate applies unchanged at each
    orientation. Named-intersection anchors make a wrong-quadrant cluster
    implausible (street names break the grid's 90-deg symmetry), and the
    disjoint-pair rule still holds.
    """
    m: list[list[float]] | None = None
    anchors: list[Candidate] = []
    for extra_deg in (90.0, 180.0, 270.0):
        m, anchors = translation_fit(
            cands,
            pinned_linear(page_scale, pin_rot + extra_deg),
            require_disjoint=False,
            aliases=aliases,
        )
        if m is not None:
            logger.info("p%s: rescue cluster found at +%g deg quadrant", page, extra_deg)
            break
    return m, anchors


def _fit_on_own_grid(
    cands: list[Candidate],
    page: str,
    page_scale: float,
    pin_rot: float,
    own_rot: float,
    aliases: Mapping[str, str] | None,
) -> tuple[list[list[float]] | None, list[Candidate]]:
    """Retry the pinned fit at the sheet's OWN estimated rotation.

    Runs only when the VOLUME's pin found nothing, so it evicts no cluster that
    pin found. It DOES pre-empt the opt-in quadrant retry below, deliberately: a
    cluster at the axis the sheet's own labels point to beats a blind sweep.
    ``own_rot`` is an axis, so both ends are tried. Both bars sit above
    ``MIN_AGREE`` — an estimated pin at two ends is more exposure than a declared
    one — and a shared-street cluster takes one more, because the disjoint-pair
    rule already distrusts that configuration.
    """
    best_m: list[list[float]] | None = None
    best: tuple[bool, int] = (False, 0)
    best_anchors: list[Candidate] = []
    for end in (own_rot, own_rot + 180.0):
        m, anchors = translation_fit(
            cands, pinned_linear(page_scale, end), require_disjoint=False, aliases=aliases
        )
        if m is None:
            continue
        disjoint = has_disjoint_pair([c.streets for c in anchors], aliases)
        floor = OWN_GRID_MIN_AGREE if disjoint else OWN_GRID_SHARED_STREET_MIN_AGREE
        if len(anchors) < floor:
            logger.info(
                "p%s: own-grid cluster of %d at %.2f deg is below the %s bar of %d — refused",
                page,
                len(anchors),
                end,
                "disjoint" if disjoint else "shared-street",
                floor,
            )
            continue
        # The two ends are contradictory hypotheses 180 deg apart, so ranking
        # them by size alone lets a shared-street cluster displace a disjoint
        # one — the eviction ``translation_fit``'s rail guard forbids INSIDE one
        # fit, happening between two of them. Disjointness outranks size; a tie
        # keeps the end nearer the volume.
        rank = (disjoint, len(anchors))
        if rank <= best:
            continue
        best_m, best_anchors, best = m, anchors, rank
        logger.info(
            "p%s: rescue cluster of %d found on the sheet's own grid at %.2f deg "
            "(volume pin %.2f deg, %s)",
            page,
            len(anchors),
            end,
            pin_rot,
            "disjoint" if disjoint else "shared-street",
        )
    return best_m, best_anchors


def _own_grid_pin(ctx: _RescueContext, ann: dict[str, Any], pin_rot: float) -> float | None:
    """The sheet's own rotation when it differs from the volume's, else None.

    Gated on the difference because an estimate that agrees with the volume adds
    no hypothesis — only the upside-down end, which is a wrong pin.
    """
    if ctx.own_grid is None:
        return None
    own = ctx.own_grid.estimate(ann)
    if own is None:
        return None
    dev = abs(((own - pin_rot + 45.0) % 90.0) - 45.0)
    turn = abs(((own - pin_rot + 90.0) % 180.0) - 90.0)
    return own if dev > OWN_GRID_DEV_GATE or turn > OWN_GRID_TURN_GATE else None


def _note_index_window_offset(
    r: dict[str, Any],
    page: str,
    m: list[list[float]],
    full_size: tuple[float, float],
    window: Any | None,
) -> None:
    """Log-only rescue plausibility vs the street-index prior window.

    Never gates: a suspicious offset is logged and recorded on the result
    (``index_window_offset_m``) for audit, and the rescue proceeds.
    """
    if window is None:
        return
    cx, cy = full_size[0] / 2, full_size[1] / 2
    px = m[0][0] + m[0][1] * cx + m[0][2] * cy
    py = m[1][0] + m[1][1] * cx + m[1][2] * cy
    offset = math.hypot(px - window.center_3857[0], py - window.center_3857[1])
    if offset > window.radius_m:
        logger.warning(
            "p%s: rescue translation lands %.0f m from its street-index "
            "window (radius %.0f m) — suspicious, recorded but not gated",
            page,
            offset,
            window.radius_m,
        )
        r["index_window_offset_m"] = round(offset, 1)


class RescueOutcome(NamedTuple):
    """A rescue attempt that produced a recordable placement."""

    disjoint: bool  # anchors span two disjoint streets -> commit directly


@dataclass(frozen=True)
class _RescueContext:
    """The volume-wide inputs every page's rescue attempt shares."""

    index: CenterlineIndex
    vol: VolumeConfig
    aliases: Mapping[str, str]
    pin_scale: float
    pin_rot: float
    rail_index: Any | None
    index_windows: Mapping[str, Any] | None
    own_grid: OwnGridEstimator | None


def _rescue_page(
    ctx: _RescueContext,
    page: str,
    r: dict[str, Any],
    rp: Path,
    ann_path: Path,
    info: dict[str, Any],
) -> RescueOutcome | None:
    """One REJECTED page's translation-only rescue at the pinned constants.

    Mutates ``r`` in place. Returns ``None`` when the page stays flagged:
    either no agreeing cluster (nothing written) or a cluster whose recorded
    GCPs were refused (that refusal IS persisted here). On success the caller
    decides the status from the outcome and writes the record.
    """
    index, vol, aliases = ctx.index, ctx.vol, ctx.aliases
    sheet = sheet_input_from(page, json.loads(ann_path.read_text()), info)
    ann, ann_scale = sheet.annotation, sheet.scale
    # rescue pins the LINEAR part and solves only for translation, so a page
    # printed at another scale must be pinned at ITS scale — pinning a 4x
    # sheet at the volume's would fit a translation to a linear part that is
    # 4x wrong and cluster on nothing (volume.constraints_for_page)
    page_scale = ctx.pin_scale * vol.page_scale_multiples.get(page, 1.0)
    linear = pinned_linear(page_scale, ctx.pin_rot)
    cands: list[Candidate] = candidate_gcps(ann, index, ann_scale, aliases)
    if ctx.rail_index is not None:
        from ..rail import rail_crossing_candidates

        cands.extend(rail_crossing_candidates(ann, ctx.rail_index, index, aliases, ann_scale))
    # aliases must reach the fit: the rail eviction guard enforces the
    # disjoint clause there, and it has to agree with the post-fit
    # disjointness decision below or a doomed cluster can still win
    m, anchors = translation_fit(cands, linear, require_disjoint=False, aliases=aliases)
    own_rot = None if m is not None else _own_grid_pin(ctx, ann, ctx.pin_rot)
    if own_rot is not None:
        m, anchors = _fit_on_own_grid(cands, page, page_scale, ctx.pin_rot, own_rot, aliases)
    # ...and only if THAT is what placed it: the quadrant retry below can still
    # fire after the own-grid arm comes up empty
    on_own_grid = m is not None and own_rot is not None
    if m is None and vol.quadrant_rescue:
        m, anchors = _translation_fit_any_quadrant(cands, page, page_scale, ctx.pin_rot, aliases)
    if m is None:
        return None
    disjoint = has_disjoint_pair([c.streets for c in anchors], aliases)
    full_size = sheet.full_size
    _note_index_window_offset(r, page, m, full_size, (ctx.index_windows or {}).get(page))
    gcps = with_synthetic_corners(anchors, m, full_size)
    # Assert on the RECORDED correspondences, not on ``m``: no result record
    # carries a model, so what every consumer warps is the unconstrained refit
    # of these points. This cannot fire today — which is the point. It is the
    # tripwire for the day an anchor producer changes and the record stops
    # reproducing the model that passed the gate.
    recorded = fit_affine_checked(
        [(c.pixel[0], c.pixel[1], *TO_3857.transform(*c.world4326)) for c in gcps]
    )
    if recorded is None or model_determinant(recorded) >= 0:
        why = "singular" if recorded is None else f"{model_determinant(recorded):+.3e}"
        logger.error(
            "p%s: rescue GCPs do not reproduce an upright placement (det=%s) — page left flagged",
            page,
            why,
        )
        # Persist the refusal on the record. The page keeps its REJECTED
        # status, so without this `status` and `report` could not tell
        # "no cluster agreed" from "a cluster agreed and its record was
        # refused" — and the index-window diagnostic above would be lost
        # for precisely the page that needs it.
        r["rescue_record_refused"] = why
        write_result(rp, r)
        return None
    r["n_inliers"] = len(anchors)
    r["rescue_anchors"] = [list(c.streets) for c in anchors]
    r["gcps_geojson"] = gcps_geojson_from(gcps)
    if on_own_grid:
        # audit only, and read off the RECORDED correspondences so it describes
        # the placement a consumer refits rather than the model that passed
        r["rescue_pin_rotation_deg"] = round(model_rotation_deg(recorded), 4)
    return RescueOutcome(disjoint=disjoint)


def stage_rescue(
    paths: VolumePaths,
    index: CenterlineIndex,
    vol: VolumeConfig,
    rail_index: Any | None = None,
    index_windows: Mapping[str, Any] | None = None,
    bounds: tuple[float, float, float, float] | None = None,
) -> tuple[list[str], list[str]]:
    """Translation-only rescue of REJECTED pages at pinned volume constants.

    A cluster WITH a disjoint pair commits as ``OK (rescued)``; a cluster whose
    anchors all share one street records its PROVISIONAL placement revoked, for
    corroboration to vouch for or refuse; no agreeing cluster stays flagged.
    A page whose own drawn grid differs from the volume's gets a second, stricter
    attempt there when the volume's pin found nothing. Returns
    ``(rescued, provisional_revoked)``.
    """
    # constants priority: config pin, else what stage_match derived+persisted
    pins = resolve_constants(paths, vol)
    if pins is None:
        raise PipelineError(
            f"{vol.identifier}: rescue needs pinned scale/rotation "
            f"(none in config and no usable {paths.constants} — run match first)"
        )
    pin_scale, pin_rot = pins
    manifest = json.loads(paths.manifest.read_text())
    ctx = _RescueContext(
        index=index,
        vol=vol,
        aliases=index.aliases,
        pin_scale=pin_scale,
        pin_rot=pin_rot,
        rail_index=rail_index,
        index_windows=index_windows,
        own_grid=OwnGridEstimator(index, bounds),
    )
    rescued: list[str] = []
    provisional: list[str] = []
    for page, r, rp in iter_results(paths):
        status = str(r.get("status", ""))
        if not status.startswith(REJECTED_PREFIX) or status.startswith(REVOKED_PREFIX):
            continue
        ann_path = paths.annotations / f"p{page}.json"
        info = manifest.get(f"p{page}")
        if not ann_path.exists() or info is None:
            continue
        outcome = _rescue_page(ctx, page, r, rp, ann_path, info)
        if outcome is None:
            continue
        if outcome.disjoint:
            r["status"] = STATUS_RESCUED
            rescued.append(page)
        else:
            r["status"] = STATUS_RESCUE_REVOKED
            r["layer"] = None
            provisional.append(page)
        write_result(rp, r)
    logger.info(
        "%s: rescued %d pages, %d provisional (await corroboration)",
        vol.identifier,
        len(rescued),
        len(provisional),
    )
    return rescued, provisional


def stage_revoke_shared_street_rescues(
    paths: VolumePaths, aliases: dict[str, str] | None = None
) -> list[str]:
    """Revoke recorded rescues whose anchors all share one street.

    This supports resuming records created before the disjoint-pair rule.
    """
    revoked: list[str] = []
    for page, r, rp in iter_results(paths):
        if r.get("status") != STATUS_RESCUED or not r.get("rescue_anchors"):
            continue
        anchors = [tuple(a) for a in r["rescue_anchors"]]
        if not has_disjoint_pair(anchors, aliases):
            r["status"] = STATUS_RESCUE_REVOKED
            r["layer"] = None
            write_result(rp, r)
            revoked.append(page)
    return revoked
