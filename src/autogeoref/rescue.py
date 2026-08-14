"""Rescue pass for flagged sheets: translation-only fit at the volume's
pinned scale and rotation.

Why this is sound: the committed sheets establish the volume's scan scale and
print orientation precisely. Pinning both leaves only a 2-parameter
translation — so 2+ independent street-pair intersections that agree on the
same translation are sufficient evidence, where the full affine fit rightly
demanded 4+ well-spread inliers.

Acceptance gate per sheet, contract-tested against every recorded revocation
decision in the fixtures: at least ``MIN_AGREE`` candidates whose implied
translations agree within ``TOL_M`` metres, drawn from at least 2 distinct
street pairs and 2 distinct pixel points, and THE DISJOINT-PAIR RULE — at least
two agreeing anchors sharing NO street.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .affine import TO_3857, TO_4326
from .matching import Candidate
from .names import Aliases, normalize

TOL_M = 12.0
MIN_AGREE = 2

SYNTHETIC_STREETS = ("synthetic", "rescue-model-corner")

#: THE synthetic-GCP grammar. A model-derived corner is identified by this
#: marker inside its ``note`` property. There is NO boolean ``synthetic`` key
#: on the feature: a reader that checks for one silently sees zero synthetics
#: on every sheet. Every consumer goes through :func:`is_synthetic_gcp`.
SYNTHETIC_NOTE_MARKER = "synthetic"


#: The narrower marker: a corner THIS module generated from the pinned volume
#: constants, as opposed to one the review UI generated from a human placement.
#: Both are synthetic; only this one means "the linear part is the volume's".
RESCUE_CORNER_MARKER = SYNTHETIC_STREETS[1]


def is_synthetic_gcp(feature: Mapping[str, Any]) -> bool:
    """True when a ``gcps_geojson`` feature is a model-derived corner, of EITHER kind.

    Synthetic corners lie exactly on the placement model, so they carry no evidence: seam ties,
    corroboration vouching and the vouch pool all skip them, and a fit residual measured through
    them is measuring the model against itself. That is equally true of a reviewer's placement
    corners, so this test is deliberately the broad one — use :func:`is_rescue_model_corner`
    when the question is specifically whose orientation the placement carries.
    """
    note = (feature.get("properties") or {}).get("note") or ""
    return SYNTHETIC_NOTE_MARKER in note


def is_rescue_model_corner(feature: Mapping[str, Any]) -> bool:
    """True when a corner was generated from the VOLUME's pinned constants.

    Distinguishes :func:`with_synthetic_corners`' output from the review UI's ``synthetic:
    reviewer placement corner``. The distinction is not cosmetic: a corner of this kind is a
    statement that rotation and scale came from ``pinned_linear``, which is what
    :func:`autogeoref.placement_records.pinned_orientation` reports.
    """
    note = (feature.get("properties") or {}).get("note") or ""
    return RESCUE_CORNER_MARKER in note


def pinned_linear(scale_m_per_px: float, rotation_deg: float) -> list[list[float]]:
    """Fixed 2x2 linear part: ``world = T + A @ pixel``.

    ``scale_m_per_px`` is EPSG:3857 meters per full-res pixel (the
    ``VolumeConfig.scale_m_per_px`` frame contract), applied with no
    cos(lat) correction. Note the y flip: pixel y grows downward, 3857 y
    grows upward.
    """
    rot = math.radians(rotation_deg)
    s = scale_m_per_px * math.sin(rot)
    c = scale_m_per_px * math.cos(rot)
    # rotation matrix [[c,-s],[s,c]] with its second column negated (y flip)
    return [[c, s], [s, -c]]


def _apply_linear(a: list[list[float]], px: float, py: float) -> tuple[float, float]:
    return (a[0][0] * px + a[0][1] * py, a[1][0] * px + a[1][1] * py)


def translation_fit(
    cands: list[Candidate],
    linear: list[list[float]],
    tol_m: float = TOL_M,
    min_agree: int = MIN_AGREE,
    require_disjoint: bool = True,
    aliases: Aliases | None = None,
) -> tuple[list[list[float]] | None, list[Candidate]]:
    """Largest cluster of candidate-implied translations, gated as documented.

    Returns ``(M, anchors)`` with ``M`` a 2x3 affine (translation + pinned linear part), or
    ``(None, [])``. With ``require_disjoint=False`` the disjoint-pair rule is NOT enforced;
    callers use this to obtain the PROVISIONAL placement of a shared-street cluster, which is
    recorded as revoked and offered to neighbour corroboration. Such a placement must never be
    committed directly, and a rail-bearing cluster without a disjoint pair enters that lifecycle
    only when it displaces nothing.
    """
    pts: list[tuple[tuple[float, float], Candidate]] = []
    for c in cands:
        x, y = TO_3857.transform(*c.world4326)
        ax, ay = _apply_linear(linear, *c.pixel)
        pts.append(((x - ax, y - ay), c))

    def largest(allow_shared_street_rail: bool) -> list[tuple[tuple[float, float], Candidate]]:
        best: list[tuple[tuple[float, float], Candidate]] = []
        for t, _ in pts:
            cluster = [q for q in pts if math.hypot(q[0][0] - t[0], q[0][1] - t[1]) <= tol_m]
            if len(cluster) <= len(best):
                continue
            if any(name.startswith("RR ") for q in cluster for name in q[1].streets):
                pairs = {tuple(sorted(q[1].streets)) for q in cluster}
                pixels = {(round(q[1].pixel[0]), round(q[1].pixel[1])) for q in cluster}
                if len(cluster) < min_agree or len(pairs) < 2 or len(pixels) < 2:
                    # One rail/street image crossing can yield several modern
                    # geometry alternatives. They are not independent evidence
                    # and may not evict an otherwise usable street cluster.
                    continue
                if not allow_shared_street_rail and not has_disjoint_pair(
                    [q[1].streets for q in cluster], aliases
                ):
                    # A rail cluster whose anchors all share one street can
                    # never be accepted directly, so it may not displace a
                    # cluster that passes the fit gates. It is retried below
                    # only when nothing else wins.
                    continue
            best = cluster
        return best

    def gated(best: list[tuple[tuple[float, float], Candidate]]) -> bool:
        if len(best) < min_agree:
            return False
        pairs = {tuple(sorted(q[1].streets)) for q in best}
        pixels = {(round(q[1].pixel[0]), round(q[1].pixel[1])) for q in best}
        if len(pairs) < 2 or len(pixels) < 2:
            return False
        return not require_disjoint or has_disjoint_pair([q[1].streets for q in best], aliases)

    best = largest(allow_shared_street_rail=False)
    if not gated(best) and not require_disjoint:
        # nothing displaced: a shared-street rail cluster may take the
        # provisional (revoked, corroboration-eligible) route after all
        best = largest(allow_shared_street_rail=True)
    if not gated(best):
        return None, []
    tx = sum(q[0][0] for q in best) / len(best)
    ty = sum(q[0][1] for q in best) / len(best)
    m = [[tx, linear[0][0], linear[0][1]], [ty, linear[1][0], linear[1][1]]]
    return m, [q[1] for q in best]


def has_disjoint_pair(
    anchor_streets: list[tuple[str, str]],
    aliases: Aliases | None = None,
) -> bool:
    """True if at least two anchors share NO street (NORMALIZED comparison).

    Labels are normalized (func:`autogeoref.names.normalize`, with the
    volume's alias table when given) before the set comparison: the same
    street under two raw spellings ("WASHBURN AV." vs "Washburn Av.") is one
    street, so case/punctuation variance can no longer fake disjointness.
    Normalization only merges equivalent labels; it cannot manufacture a
    disjoint pair from spelling differences.
    """
    sets = [frozenset(normalize(name, aliases) for name in s) for s in anchor_streets]
    return any(not (sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets)))


def with_synthetic_corners(
    anchors: list[Candidate],
    m: list[list[float]],
    full_size: tuple[float, float],
) -> list[Candidate]:
    """Serialize the rescue model as synthetic corner GCPs. ALWAYS.

    GDAL's poly1 warp needs >= 3 NON-COLLINEAR GCPs, and a rescue's real anchors can be too few
    or collinear. The synthetic corners lie exactly on ``m`` — no new information — so the
    warper can reproduce the transform. **They are unconditional, and that is the point.** A
    result record carries no model field, so every consumer re-derives the placement with an
    UNCONSTRAINED ``fit_affine`` over these correspondences, and the pinned linear part is
    nowhere else on disk. This makes the record reproduce the placement; it does not make the
    placement more accurate.
    """
    gcp_cands = list(anchors)
    w, h = full_size
    linear = [[m[0][1], m[0][2]], [m[1][1], m[1][2]]]
    for cx, cy in ((w * 0.1, h * 0.1), (w * 0.9, h * 0.1), (w * 0.1, h * 0.9)):
        ax, ay = _apply_linear(linear, cx, cy)
        lng, lat = TO_4326.transform(m[0][0] + ax, m[1][0] + ay)
        gcp_cands.append(Candidate(pixel=(cx, cy), world4326=(lng, lat), streets=SYNTHETIC_STREETS))
    return gcp_cands
