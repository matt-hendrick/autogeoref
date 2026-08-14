"""The one model-call spend estimator, shared by every command that prints one.

Three estimators used to bound this spend and none bounded the same thing, so
the operator approved against whichever command they happened to type. This
states one set of assumptions and publishes one PLANNING bound — not a billing
guarantee:

- The floor is one clean primary read per unread page, net of cached reads.
- A transient failure is retried once, and a retry is a SECOND BILLABLE CALL,
  so the ceiling multiplies by :data:`ATTEMPTS` and ``escalation_attempts``.
- Escalation re-reads the gated pool once per ladder tier. That pool is an
  ESTIMATE, ``ceil(sheets * gated_fraction)``; the ACTUAL pool is decided at
  run time by drawn-street evidence and can exceed it.
- The bounds bootstrap can read before this prints; counted as cached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Reads per page, worst case — one attempt plus one retry of a transient
#: failure. A retry is a second billable call, so the ceiling multiplies by it.
ATTEMPTS = 2

#: Reads per gated page per ladder tier, worst case — escalation always bills
#: up to ``escalate.MAX_ATTEMPTS`` (= 2) per tier, independent of the annotate
#: plan's ``attempts``.
ESCALATION_ATTEMPTS = 2

#: Default fraction of sheets expected to reach the evidence-gated pool.
#: Cities may declare their own value; it affects spend estimates only.
DEFAULT_GATED_FRACTION = 0.41


@dataclass(frozen=True)
class SpendEstimate:
    """One volume's model-call range. See the module docstring for the bound.

    Plain fields, not properties: the console board serializes this with
    ``dataclasses.asdict`` and the queue UI reads ``low``/``ceiling`` from the
    JSON. Build instances through :func:`estimate_spend`.
    """

    sheets: int
    #: pages already paid for (cached; they replay free)
    cached: int
    #: the floor: every unread page read once, cleanly, no escalation
    low: int
    #: estimated evidence-gated pool (0 when the run cannot escalate)
    gated: int
    escalation_tiers: int
    attempts: int
    #: the planning ceiling: every read retried, every ESTIMATED gated page
    #: climbing every escalation tier, every tier's read retried too. Not a
    #: billing guarantee — the run-time escalation pool can exceed the
    #: estimated one (module docstring).
    ceiling: int

    def __str__(self) -> str:
        return f"{self.low}" if self.low == self.ceiling else f"{self.low}-{self.ceiling}"

    def render(self) -> str:
        """The range with its caveats said out loud, for command output."""
        if self.low == self.ceiling:
            return f"{self} model calls"
        caveats = []
        if self.attempts > 1:
            caveats.append("a transient failure is retried, and a retry is a second call")
        if self.gated and self.escalation_tiers:
            caveats.append(
                f"the ceiling escalates ~{self.gated} gated sheet(s) "
                f"through {self.escalation_tiers} tier(s)"
            )
        return f"{self} model calls ({'; '.join(caveats)})" if caveats else f"{self} model calls"


def estimate_spend(
    *,
    sheets: int,
    cached: int = 0,
    unread: int | None = None,
    escalation_tiers: int = 0,
    gated_fraction: float = DEFAULT_GATED_FRACTION,
    attempts: int = ATTEMPTS,
    escalation_attempts: int = ESCALATION_ATTEMPTS,
) -> SpendEstimate:
    """The planning bound on a place run's model-call spend.

    ``unread`` may be passed exactly when the caller has a real page plan;
    otherwise it is inferred as ``max(0, sheets - cached)``, floored at zero
    rather than going negative when annotations outnumber addressable sheets.
    ``escalation_attempts`` is separate from ``attempts`` because escalation
    always bills up to ``escalate.MAX_ATTEMPTS`` per tier.
    """
    if unread is None:
        unread = max(0, sheets - cached)
    gated = math.ceil(sheets * gated_fraction) if escalation_tiers > 0 else 0
    return SpendEstimate(
        sheets=sheets,
        cached=min(cached, sheets),
        low=unread,
        gated=gated,
        escalation_tiers=escalation_tiers,
        attempts=attempts,
        ceiling=attempts * unread + escalation_attempts * gated * escalation_tiers,
    )
