"""The corroboration stage: reinstate a revoked rescue its neighbours vouch for."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..corroborate import corroborations, is_corroborated
from ..paths import iter_results, write_result
from ..seam import SheetFit, sheet_fit_from_result
from ..volume import REVOKED_PREFIX, STATUS_CORROBORATED
from ..vouchers import committed_vouch_nodes

if TYPE_CHECKING:
    from pathlib import Path

    from ..paths import VolumePaths

logger = logging.getLogger(__name__)


def stage_corroborate(paths: VolumePaths) -> list[str]:
    """Reinstate revoked rescues vouched by committed neighbors (>=2 nodes <=8 m)."""
    revoked: dict[str, tuple[SheetFit, Path, dict[str, Any]]] = {}
    for page, r, rp in iter_results(paths):
        st = str(r.get("status", ""))
        if not st.startswith(REVOKED_PREFIX):
            continue
        fit = sheet_fit_from_result(page, r)
        if fit is None:
            continue
        revoked[page] = (fit, rp, r)
    nodes = committed_vouch_nodes(paths)
    reinstated: list[str] = []
    for page, (fit, rp, r) in sorted(revoked.items()):
        if is_corroborated(corroborations(fit, nodes)):
            r["status"] = STATUS_CORROBORATED
            write_result(rp, r)
            reinstated.append(page)
    logger.info("corroborated %d of %d revoked pages", len(reinstated), len(revoked))
    return reinstated
