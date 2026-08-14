"""Committed placement voucher nodes shared by corroboration consumers."""

from __future__ import annotations

from .affine import apply_affine
from .paths import VolumePaths, iter_results
from .placement_records import preseam_record
from .seam import node_key, sheet_fit_from_result
from .volume import STATUS_CORROBORATED, STATUS_VERIFIED_PREFIX, is_committed

VouchNodes = dict[tuple[float, float], list[tuple[str, tuple[float, float]]]]


def committed_vouch_nodes(paths: VolumePaths) -> VouchNodes:
    """Build voucher nodes keyed in the pre-seam frame from committed records."""
    nodes: VouchNodes = {}
    for page, record, _result_path in iter_results(paths):
        status = str(record.get("status", ""))
        if (
            not is_committed(record)
            or status == STATUS_CORROBORATED
            or status.startswith(STATUS_VERIFIED_PREFIX)
        ):
            continue
        fit = sheet_fit_from_result(page, record)
        if fit is None:
            continue
        key_fit = sheet_fit_from_result(page, preseam_record(record)) or fit
        for (px, py, _x, _y, synthetic), (_, _, key_x, key_y, _) in zip(
            fit.gcps, key_fit.gcps, strict=True
        ):
            if synthetic:
                continue
            nodes.setdefault(node_key(key_x, key_y), []).append(
                (page, apply_affine(fit.coef, px, py))
            )
    return nodes
