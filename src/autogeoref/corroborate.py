"""Second-chance reinstatement of revoked rescues via NEIGHBOR corroboration.

The disjoint-pair revocation pulls every rescued sheet whose anchors all rode one street
— that gate cannot tell a correct placement from a parallel-street mistake. Committed
NEIGHBOUR sheets can: a street corner is one physical point, so when a revoked sheet's
GCP refers to the same centerline node as a committed neighbour's, comparing the two
sheets' warped positions of that node measures agreement directly.

The parallel-street failure CANNOT pass: a sheet matched to the wrong street holds world
nodes on it while its true neighbours hold the right corners — no shared nodes, no
corroboration. Zero-tie and mismatched-tie sheets stay revoked, honestly.

Gate to reinstate, contract-tested against every recorded decision: ``MIN_NODES`` shared
nodes agreeing within ``TOL_M``, against sheets that survived the strict gates.
"""

from __future__ import annotations

import math

from .affine import apply_affine
from .seam import SheetFit, node_key

TOL_M = 8.0
MIN_NODES = 2
#: Second-node slack for the VERIFIED-ACCEPT channel shape only (see
#: :func:`is_corroborated_near`): a second node within ``NEAR_FACTOR x TOL_M``
#: corroborates when a first node holds at full tolerance. Derived from the
#: The near band tolerates ordinary registration noise without relaxing the
#: standalone reinstatement gate.
NEAR_FACTOR = 2.0

#: (node_key, committed_page, distance_m)
Hit = tuple[tuple[float, float], str, float]


def committed_nodes(
    sheets: dict[str, SheetFit],
) -> dict[tuple[float, float], list[tuple[str, tuple[float, float]]]]:
    """Rounded 3857 node -> [(page, warped position of that node)].

    The PURE builder over caller-selected fits: contract tests and replay tooling pick their own
    sheet sets and decision frames. Production voucher-pool policy (status exclusions, pre-seam
    node keying) lives in :func:`autogeoref.vouchers.committed_vouch_nodes` — do not add policy
    here.
    """
    nodes: dict[tuple[float, float], list[tuple[str, tuple[float, float]]]] = {}
    for page, s in sheets.items():
        for px, py, x, y, synthetic in s.gcps:
            if synthetic:
                continue
            nodes.setdefault(node_key(x, y), []).append((page, apply_affine(s.coef, px, py)))
    return nodes


def corroborations(
    revoked: SheetFit,
    nodes: dict[tuple[float, float], list[tuple[str, tuple[float, float]]]],
) -> list[Hit]:
    """All shared-node observations between a revoked sheet and committed ones."""
    hits: list[Hit] = []
    for px, py, x, y, synthetic in revoked.gcps:
        if synthetic:
            continue
        key = node_key(x, y)
        for page, (nx, ny) in nodes.get(key, []):
            wx, wy = apply_affine(revoked.coef, px, py)
            hits.append((key, page, math.hypot(wx - nx, wy - ny)))
    return hits


def is_corroborated(
    hits: list[Hit],
    tol_m: float = TOL_M,
    min_nodes: int = MIN_NODES,
) -> bool:
    """Reinstate iff >= ``min_nodes`` DISTINCT nodes agree within ``tol_m``."""
    good_nodes = {h[0] for h in hits if h[2] <= tol_m}
    return len(good_nodes) >= min_nodes


def is_corroborated_near(
    hits: list[Hit],
    tol_m: float = TOL_M,
    min_nodes: int = MIN_NODES,
    near_factor: float = NEAR_FACTOR,
) -> bool:
    """The verified-accept CHANNEL shape: one strong node plus near agreement.

    True iff at least ``min_nodes`` distinct nodes exist whose best observations are, sorted
    ascending, [<= tol_m, <= near_factor x tol_m, ...]. A strict superset of
    :func:`is_corroborated`, and NEVER the standalone reinstatement gate: this shape is only a
    channel vote inside the >=2-channels contract, and a single agreeing node never votes.
    Note the scaling — only the FIRST node is held to full tolerance — so raising ``min_nodes``
    adds NEAR nodes, not strong ones. Re-derive the shape before using another count.
    """
    best: dict[tuple[float, float], float] = {}
    for key, _page, d in hits:
        if key not in best or d < best[key]:
            best[key] = d
    dists = sorted(best.values())
    if len(dists) < min_nodes:
        return False
    return dists[0] <= tol_m and dists[min_nodes - 1] <= near_factor * tol_m
