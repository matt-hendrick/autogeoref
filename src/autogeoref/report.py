"""Per-volume scoreboard: the human-facing contract artifact.

A volume is "done" when its report exists and its numbers are in the project
log. The report answers: how many sheets are provably placed (strict /
rescued / corroborated), how many are honestly flagged, at what residuals,
with what rotation spread, and what the run cost.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .placement_records import pinned_orientation
from .scoring import GT_COMMIT_RMSE_M
from .volume import (
    REVOKED_PREFIX,
    STATUS_CORROBORATED,
    STATUS_RESCUED,
    STATUS_VERIFIED_PREFIX,
    is_committed,
    is_reviewer_verified,
)


@dataclass(frozen=True)
class VolumeReport:
    volume: str
    n_sheets: int
    strict_accepted: int
    rescued: int
    corroborated: int
    #: accepted via the >=2-independent-verifiers path — its own counter so
    #: the anti-red-flag check stays checkable: acceptance may rise ONLY here
    #: while strict/rescued/corroborated stay put
    verified: int
    #: placed or confirmed by a human REVIEWER in the review UI. NOT part of
    #: ``accepted_total`` and excluded from every residual/RMSE statistic:
    #: human placements must never inflate the auto-acceptance rate.
    reviewer_verified: int
    revoked: int
    flagged: int
    median_auto_residual_m: float | None
    p90_auto_residual_m: float | None
    #: Median grid-RMSE vs the human GCPs — over the accepts that CARRY a score.
    #: Always read it next to the two counters below: it is a statement about
    #: ``accepts_scored_vs_human`` sheets and about no others. A volume with no
    #: ground truth scores nothing at all, and the median is then ``None`` rather
    #: than flattering.
    median_rmse_vs_human_m: float | None
    #: Accepts measured against human GCPs...
    accepts_scored_vs_human: int
    #: ...and accepts no score describes, so NOTHING has checked where they
    #: landed — either the volume has no pins for that page, or nobody has run
    #: the scoring pass over the placements now on disk. They are invisible to
    #: the median above. Printing the number is the point: a headline median
    #: over 4 of 30 sheets must not read like a headline median over 30.
    accepts_unscored_vs_human: int
    #: Accepts scored BEYOND the 15 m commit gate. They keep their OK status and
    #: they SERVE: no run consults a human score, so this is a finding for a
    #: later pass to act on, not a withdrawal that has already happened.
    accepts_over_commit_gate: int
    #: AUTO-accepts that clear ``volume.is_committed``. Reviewer-verified sheets
    #: are excluded — as they are from every other counter here, so that human
    #: placements never inflate an auto-acceptance number — even though they are
    #: committed and DO get served. Served sheets = this + ``reviewer_verified``.
    committed: int
    #: AUTO-accepts whose rotation and scale were PINNED rather than fitted
    #: (:func:`placement_records.pinned_orientation`) — the rescue path.
    #: Correct by design, but not evidence-fitted: nothing checked their
    #: orientation, and the residual statistics above EXCLUDE them, because a
    #: residual through synthetic corners is the model measured against itself.
    pinned_orientation: int
    rotation_min_deg: float | None
    rotation_max_deg: float | None
    seam_adjusted_sheets: int
    seam_gate: str | None
    annotation_cost_usd: float | None = None
    notes: list[str] = field(default_factory=list)
    #: Committed sheets among the volume's DECLARED overview pages
    #: (``VolumeConfig.overview_pages``). Its own row because the class is
    #: withheld from the seam solve and from the detail mosaic, so nothing
    #: serves it — a reader comparing committed counts across volumes needs to
    #: see how many are district-scale paint that never reaches the map.
    overview_committed: int = 0

    @property
    def accepted_total(self) -> int:
        """AUTO-accepted sheets only; reviewer-verified placements count separately."""
        return self.strict_accepted + self.rescued + self.corroborated + self.verified


def build_report(
    volume: str,
    results: dict[str, dict[str, Any]],
    seam_record: dict[str, Any] | None = None,
    annotation_cost_usd: float | None = None,
    notes: list[str] | None = None,
    overview_pages: Collection[str] = (),
    scores: Mapping[str, float] | None = None,
) -> VolumeReport:
    """Aggregate per-sheet result records into the volume scoreboard.

    ``scores`` is the scoring pass's sidecar (``{page: rmse_vs_human_m}``); a
    volume nobody has scored has none, and its ground-truth counters then say
    so rather than reading zero as agreement.
    """
    gt = scores or {}
    strict = rescued = corroborated = verified = reviewer = revoked = flagged = 0
    residuals: list[float] = []
    rmses: list[float] = []
    rotations: list[float] = []
    seam_adjusted = unscored = over_gate = committed = pinned = 0
    overview_committed = 0
    for page, r in results.items():
        if page in overview_pages and is_committed(r):
            overview_committed += 1
    for page, r in results.items():
        status = str(r.get("status", ""))
        if status == STATUS_CORROBORATED:
            corroborated += 1
        elif status == STATUS_RESCUED:
            rescued += 1
        elif status.startswith(STATUS_VERIFIED_PREFIX):
            verified += 1
        elif is_reviewer_verified(status):
            # human placement: counted apart, and skipped below so it never
            # feeds the residual/RMSE statistics the gates are judged by
            reviewer += 1
            continue
        elif status.startswith("OK"):
            strict += 1
        elif status.startswith(REVOKED_PREFIX):
            revoked += 1
            flagged += 1
        else:
            flagged += 1
        if status.startswith("OK"):
            is_pinned = pinned_orientation(r)
            pinned += is_pinned
            # a pinned-orientation sheet's fit passes through control points
            # the placement model generated, so a residual over them measures
            # the model against itself. Such a record carries no
            # ``auto_residuals_m`` today, so this exclusion is a no-op — it is
            # here so that stays true by rule rather than by accident.
            if not is_pinned:
                residuals.extend(r.get("auto_residuals_m") or [])
            # every accept is either scored against human GCPs or is not; a
            # median that quietly averages the first group while the second is
            # never mentioned is how a demotion looks like an improvement
            rmse = gt.get(page)
            if rmse is None:
                unscored += 1
            else:
                rmses.append(float(rmse))
                if float(rmse) > GT_COMMIT_RMSE_M:
                    over_gate += 1
            if is_committed(r):
                committed += 1
            if r.get("rotation_deg") is not None:
                rotations.append(float(r["rotation_deg"]))
            if r.get("seam_adjusted"):
                seam_adjusted += 1

    def med(v: list[float]) -> float | None:
        return round(float(statistics.median(v)), 2) if v else None

    def p90(v: list[float]) -> float | None:
        if not v:
            return None
        s = sorted(v)
        # nearest-rank percentile: ceil(0.9 n)-th value (1-based)
        return round(s[max(0, math.ceil(0.9 * len(s)) - 1)], 2)

    return VolumeReport(
        volume=volume,
        n_sheets=len(results),
        strict_accepted=strict,
        rescued=rescued,
        corroborated=corroborated,
        verified=verified,
        reviewer_verified=reviewer,
        revoked=revoked,
        flagged=flagged,
        median_auto_residual_m=med(residuals),
        p90_auto_residual_m=p90(residuals),
        median_rmse_vs_human_m=med(rmses),
        accepts_scored_vs_human=len(rmses),
        accepts_unscored_vs_human=unscored,
        accepts_over_commit_gate=over_gate,
        committed=committed,
        pinned_orientation=pinned,
        rotation_min_deg=min(rotations) if rotations else None,
        rotation_max_deg=max(rotations) if rotations else None,
        seam_adjusted_sheets=seam_adjusted,
        seam_gate=(seam_record or {}).get("gate"),
        annotation_cost_usd=annotation_cost_usd,
        notes=list(notes or []),
        overview_committed=overview_committed,
    )


def load_results_dir(results_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for f in sorted(results_dir.glob("p*.json")):
        r = json.loads(f.read_text())
        out[str(r.get("page", f.stem.removeprefix("p")))] = r
    return out


def report_json(report: VolumeReport) -> str:
    d = asdict(report)
    d["accepted_total"] = report.accepted_total
    return json.dumps(d, indent=2)


def report_markdown(report: VolumeReport) -> str:
    r = report
    lines = [
        f"# Volume report: {r.volume}",
        "",
        "| sheets | accepted | strict | rescued | corroborated | verified | flagged (revoked) |",
        "|---|---|---|---|---|---|---|",
        f"| {r.n_sheets} | **{r.accepted_total}** | {r.strict_accepted} | {r.rescued} "
        f"| {r.corroborated} | {r.verified} | {r.flagged} ({r.revoked}) |",
        "",
        f"- median auto residual: {r.median_auto_residual_m} m (p90 {r.p90_auto_residual_m} m) "
        f"— over EVIDENCE-FITTED accepts only",
    ]
    if r.pinned_orientation:
        # the residual line above would otherwise read as a quality score over
        # every accept. It is not one for these sheets and never was: their
        # control points were generated by the placement model, so the fit
        # cannot contradict itself. Print the count next to the median so the
        # population each number describes is visible without reading the code.
        lines.append(
            f"- pinned-orientation accepts (rotation and scale pinned, not fitted "
            f"— no residual is a quality score for these): "
            f"{r.pinned_orientation} of {r.accepted_total}"
        )
    if r.reviewer_verified:
        # deliberately outside the accepted table: reviewer placements are not
        # auto-acceptances and must not blend into the funnel numbers
        lines.append(f"- reviewer-verified (review UI, not auto-accepted): {r.reviewer_verified}")
    # never print the median without the size of the population it describes:
    # a demotion that moves a bad sheet out of the scored set improves the
    # median without improving a single placement, and the counts are what make
    # that visible instead of flattering
    if r.median_rmse_vs_human_m is not None:
        lines.append(
            f"- median grid-RMSE vs human ground truth: {r.median_rmse_vs_human_m} m "
            f"(over {r.accepts_scored_vs_human} of {r.accepted_total} accepts)"
        )
    if r.accepts_unscored_vs_human:
        lines.append(
            f"- accepts with NO ground truth (nothing has checked where they landed): "
            f"{r.accepts_unscored_vs_human}"
        )
    if r.accepts_over_commit_gate:
        lines.append(
            f"- accepts beyond the {GT_COMMIT_RMSE_M:g} m commit gate (they still vouch, "
            f"seam and serve — no run reads a human score): {r.accepts_over_commit_gate}"
        )
    # auto-accepts only, like every counter here — reviewer sheets are served too
    lines.append(f"- committed auto-accepts (the funnel accepted them; they serve): {r.committed}")
    if r.overview_committed:
        lines.append(
            f"- declared overview sheets committed (district-scale paint — baked as a "
            f"separate artifact nothing serves, withheld from the seam solve): "
            f"{r.overview_committed}"
        )
    if r.rotation_min_deg is not None:
        lines.append(
            f"- rotation across accepted sheets: {r.rotation_min_deg} .. {r.rotation_max_deg} deg"
        )
    lines.append(
        f"- seam-adjusted sheets: {r.seam_adjusted_sheets}"
        + (f" (gate: {r.seam_gate})" if r.seam_gate else "")
    )
    if r.annotation_cost_usd is not None:
        lines.append(f"- annotation cost: ${r.annotation_cost_usd:.2f}")
    lines.extend(f"- NOTE: {n}" for n in r.notes)
    lines.append("")
    return "\n".join(lines)
