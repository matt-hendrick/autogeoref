"""The seam stage: one joint translation solve over the committed sheets.

Always solves in the pre-seam frame and applies the difference from what is
already recorded, so a re-run on unchanged inputs applies nothing.
"""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, Any, NamedTuple

from ..paths import atomic_write_text, iter_results, write_result
from ..placement_records import preseam_record
from ..seam import (
    MIN_SHIFT_M,
    SheetFit,
    build_ties,
    rms,
    sheet_fit_from_result,
    shift_gcps_geojson,
    solve,
)
from ..volume import (
    STATUS_CORROBORATED,
    STATUS_VERIFIED_PREFIX,
    is_committed,
    is_reviewer_verified,
)

if TYPE_CHECKING:
    from collections.abc import Collection

    from ..paths import VolumePaths

logger = logging.getLogger(__name__)


class _SeamInputs(NamedTuple):
    """What the classification pass feeds the seam solve."""

    sheets: dict[str, SheetFit]  # pre-seam fits entering the tie set
    applied: dict[str, tuple[float, float]]  # already-applied shift per sheet
    withheld: list[str]  # committed overview pages kept out of the solve
    reverted: list[str]  # pages whose stale applied shift was removed


def _collect_seam_inputs(
    paths: VolumePaths,
    overview_pages: Collection[str],
) -> _SeamInputs:
    """Classify every result record for :func:`stage_seam`.

    Read-only over the results except for one write: reverting a withheld
    overview page's previously applied shift.
    """
    sheets: dict[str, SheetFit] = {}
    applied: dict[str, tuple[float, float]] = {}
    withheld: list[str] = []
    reverted: list[str] = []
    for page, r, _rp in iter_results(paths):
        # a corroborated sheet keeps the exact placement its neighbours
        # vouched for; seam-shifting it afterwards would un-validate that
        # evidence. The same holds for verifier- and reviewer-accepted sheets:
        # the verdict scored THIS placement. These skips come FIRST so the
        # overview withdrawal below can never move a pinned placement.
        status = str(r.get("status", ""))
        if (
            status == STATUS_CORROBORATED
            or is_reviewer_verified(status)
            or status.startswith(STATUS_VERIFIED_PREFIX)
        ):
            continue
        if page in overview_pages:
            # withheld from the solve entirely (see the docstring). A shift an
            # earlier solve applied is reverted, not kept: the withdrawal means
            # these sheets neither receive nor exert seam corrections — and
            # that holds for an uncommitted (revoked / over-gate) record too,
            # whose leftover shift would otherwise survive and pin a later
            # reinstatement at the shifted placement.
            if is_committed(r):
                withheld.append(page)
            sa = r.get("seam_adjusted") or {}
            dx, dy = float(sa.get("dx_m", 0.0)), float(sa.get("dy_m", 0.0))
            if (dx, dy) != (0.0, 0.0):
                rp = paths.results / f"p{page}.json"
                record_on_disk = json.loads(rp.read_text())
                shift_gcps_geojson(record_on_disk.get("gcps_geojson") or {}, -dx, -dy)
                record_on_disk.pop("seam_adjusted", None)
                write_result(rp, record_on_disk)
                reverted.append(page)
            continue
        if not is_committed(r):
            continue
        sa = r.get("seam_adjusted") or {}
        applied[page] = (float(sa.get("dx_m", 0.0)), float(sa.get("dy_m", 0.0)))
        # the tie set is built on the PRE-SEAM geometry, never the current one:
        # this solve MOVES sheets, so reversing the recorded shift is what makes
        # the stage's input independent of its own output and the re-run a no-op
        fit = sheet_fit_from_result(page, preseam_record(r))
        if fit is not None:
            sheets[page] = fit
    if reverted:
        logger.info(
            "seam: reverted applied shifts on %d withheld overview page(s): %s",
            len(reverted),
            ", ".join(sorted(reverted)),
        )
    return _SeamInputs(sheets, applied, withheld, reverted)


def stage_seam(
    paths: VolumePaths,
    *,
    overview_pages: Collection[str] = (),
) -> dict[str, Any]:
    """Joint translation solve; applies TOTAL deltas >= MIN_SHIFT_M.

    Idempotent across re-runs: the solve always happens in the PRE-seam frame,
    producing per-sheet TOTAL deltas, and each record moves by the difference
    between its new total and what was applied, so a repeat run on unchanged
    inputs applies zero. ``overview_pages`` are withheld from the tie set — a
    district-scale sheet shares nodes with many detail sheets and its ties dominate
    the squared mismatch — and a withheld page's applied shift is reverted. Nothing
    here consults human pins; the scoring pass grades the solve afterwards.
    """
    inputs = _collect_seam_inputs(paths, overview_pages)
    sheets, applied = inputs.sheets, inputs.applied
    ties = build_ties(sheets)
    record: dict[str, Any] = {"ties": len(ties)}
    if inputs.withheld:
        record["overview_withheld"] = sorted(inputs.withheld)
    deltas: dict[str, tuple[float, float]] = {}
    if not ties:
        record["gate"] = "N/A"
    else:
        deltas, before, after = solve(sheets, ties)
        record.update(
            rms_before_m=round(rms(before), 3),
            rms_after_m=round(rms(after), 3),
            deltas={p: [round(dx, 3), round(dy, 3)] for p, (dx, dy) in deltas.items()},
        )
        # "N/A" because nothing in a run refuses a solve any more: the human-pin
        # check that used to sit here is a diagnostic the scoring pass writes to
        # the score sidecar, AFTER the shift is on disk. The key stays so a
        # reader of the recorded volumes still finds their verdict where it was.
        record["gate"] = "N/A"
        record["applied"] = True
    atomic_write_text(paths.seam_deltas, json.dumps(record, indent=2))

    for page, (dx, dy) in deltas.items():
        # apply EXACTLY what gets stored (rounded): the next run reverses the
        # stored value to reconstruct the pre-seam frame, so applying the raw
        # float would perturb node keys at rounding boundaries and destabilize
        # the tie set across re-runs
        if math.hypot(dx, dy) >= MIN_SHIFT_M:
            total = (round(dx, 3), round(dy, 3))
        else:
            total = (0.0, 0.0)
        already = applied.get(page, (0.0, 0.0))
        ddx, ddy = total[0] - already[0], total[1] - already[1]
        if math.hypot(ddx, ddy) < 0.01:
            continue
        rp = paths.results / f"p{page}.json"
        r = json.loads(rp.read_text())
        shift_gcps_geojson(r.get("gcps_geojson") or {}, ddx, ddy)
        if total == (0.0, 0.0):
            r.pop("seam_adjusted", None)
        else:
            r["seam_adjusted"] = {"dx_m": total[0], "dy_m": total[1]}
        write_result(rp, r)
    return record
