"""Street-name match coverage: the run-time alias-gap tripwire.

A sheet drawn before a street was renamed, or before its district was annexed
and renumbered, carries names the modern centerline reference does not hold.
Those reads normalize to keys absent from the index, so the matcher builds no
candidates and RANSAC never has an anchor: the volume fails wholesale, and its
funnel looks the same as a volume that is simply hard. The remedy is a volume
alias table; the problem is noticing one is needed.

``the match stage`` counts how many street reads resolve to a centerline
and persists the counts; ``stage_report`` pairs them with the zero-candidate
page share and emits an ADVISORY note. The bars are advisory precisely because
they misclassify in both directions — they exist to point a reader at the
corpus instrument, which shares this module counting rather than restating it.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeGuard

from .names import normalize
from .paths import VolumePaths, atomic_write_text

if TYPE_CHECKING:
    from .centerlines import CenterlineIndex
    from .volume import SheetInput

logger = logging.getLogger(__name__)

#: Advisory bar on the share of street reads that resolve to a centerline.
#: On the reference corpus healthy volumes sit well above it and every
#: confirmed alias-gap volume below
#: for the reference corpus.
LOW_MATCH_RATE = 0.80

#: Advisory bar on the share of pages whose match produced no candidate at
#: all — the half of the signal that survives without the sidecar, and the
#: one that separated the diagnosed volume from every healthy one in the
#: same sweep.
HIGH_ZERO_CANDIDATE_SHARE = 0.10

#: Each half of the note has its own floor, because each is a share of a
#: different thing and a tripwire that cries on every special sheet is one the
#: reader learns to skip. Pages, for the zero-candidate half: a two-segment
#: index sheet reading 0 of 2 is not a funnel measurement.
MIN_PAGES_FOR_NOTE = 10

#: Reads, for the match half — NOT pages. A volume can be two segments and
#: still carry a real vocabulary sample (the corpus has one at 139 reads
#: whose unmatched leaders are genuine retired names), so gating this half
#: on page count would suppress a true positive using a proxy for a question
#: it does not measure. Below this many reads the rate swings further on a
#: handful of names than the bar it is compared against.
MIN_READS_FOR_NOTE = 100

#: Where the reader is sent for what these bars do and do not mean; the note is
#: a prompt to measure, not a diagnosis.
SWEEP_RECORD = "docs/ADDING-A-CITY.md"


@dataclass(frozen=True)
class NameMatch:
    """How much of a volume's street vocabulary the centerline index holds.

    ``unmatched`` is carried for the corpus instrument, which reports each
    volume's unmatched leaders; it is deliberately NOT persisted — the
    sidecar is a tripwire input, and the names belong in the instrument's
    output where someone is already reading them to write an alias table.
    """

    reads: int
    matched: int
    unmatched: Counter[str] = field(default_factory=Counter)

    @property
    def match_rate(self) -> float | None:
        """Matched share, rounded as persisted; None with no reads at all."""
        return round(self.matched / self.reads, 4) if self.reads else None

    def document(self) -> dict[str, Any]:
        """The persisted shape — and the instrument's row for this volume."""
        return {"reads": self.reads, "matched": self.matched, "match_rate": self.match_rate}


def count_name_matches(sheets: Sequence[SheetInput], index: CenterlineIndex) -> NameMatch:
    """Count annotated street reads and how many resolve to a centerline.

    THE one implementation of the measurement, shared by the match stage and
    the corpus instrument so a run-time number and a sweep-table number are
    the same number. An alias table already applied to the index counts as a
    match, because it is one: measuring raw names instead would flag exactly
    the volumes whose gap has already been closed.
    """
    reads = matched = 0
    unmatched: Counter[str] = Counter()
    for sheet in sheets:
        for street in sheet.annotation.get("streets", []):
            name = street.get("name")
            if not name:
                continue
            reads += 1
            if normalize(str(name), index.aliases) in index.by_name:
                matched += 1
            else:
                unmatched[str(name)] += 1
    return NameMatch(reads=reads, matched=matched, unmatched=unmatched)


def write_name_match(
    paths: VolumePaths, volume: str, sheets: Sequence[SheetInput], index: CenterlineIndex
) -> None:
    """Persist ``name-match.json`` for the volume report to read back.

    Deliberately a sidecar and not a field on the result records: the metric
    is a property of the volume's vocabulary, not of any one placement, and
    the results schema is a contract several consumers read.
    """
    counts = count_name_matches(sheets, index)
    document = {"volume": volume, **counts.document()}
    atomic_write_text(paths.name_match, json.dumps(document, indent=2))
    rate = counts.match_rate
    if rate is not None and rate < LOW_MATCH_RATE:
        logger.warning(
            "%s: only %.0f%% of %d street reads match the centerline index — "
            "suspect a historic street-name alias gap (see %s)",
            volume,
            100 * rate,
            counts.reads,
            SWEEP_RECORD,
        )


def load_name_match(paths: VolumePaths) -> dict[str, Any] | None:
    """The persisted counts, or None when absent, unreadable or malformed.

    Absent is the normal state for a volume matched before this sidecar existed,
    and it stays SILENT: a report warning about its own missing metric would
    fire on every previously matched volume and teach the reader to skip the
    note that matters. Every field the note formats is validated here rather
    than at use, so a hand-edited sidecar can only silence an advisory line —
    never fail the report stage.
    """
    path = paths.name_match
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text())
        if not isinstance(doc, dict):
            raise ValueError("not a JSON object")
        reads, matched, rate = doc.get("reads"), doc.get("matched"), doc.get("match_rate")
        if not _is_count(reads) or not _is_count(matched):
            raise ValueError("reads/matched are not counts")
        if rate is not None and not (_is_number(rate) and 0.0 <= rate <= 1.0):
            raise ValueError("match_rate is not a fraction")
    except (OSError, ValueError) as exc:
        logger.warning("%s: unusable name-match sidecar (%s)", path, exc)
        return None
    return doc


def _is_number(value: Any) -> TypeGuard[float]:
    """True for a real int/float. ``bool`` is an int in Python and is not one."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_count(value: Any) -> TypeGuard[float]:
    return _is_number(value) and value >= 0


def alias_gap_note(
    stats: Mapping[str, Any] | None, results: Mapping[str, Mapping[str, Any]]
) -> str | None:
    """One advisory report line when a volume looks alias-starved, else None.

    Each half fires on its own, needs its own sample to be worth reading
    (:data:`MIN_READS_FOR_NOTE`, :data:`MIN_PAGES_FOR_NOTE`), and validates
    its own inputs — the note is public, so it stays total rather than
    trusting a caller to have come through :func:`load_name_match`.
    """
    parts: list[str] = []
    rate = (stats or {}).get("match_rate")
    reads = (stats or {}).get("reads")
    if (
        _is_number(rate)
        and rate < LOW_MATCH_RATE
        and _is_count(reads)
        and reads >= MIN_READS_FOR_NOTE
    ):
        parts.append(
            f"{rate:.0%} of {reads} street reads match the centerline index "
            f"(below {LOW_MATCH_RATE:.0%})"
        )
    zero = sum(1 for r in results.values() if r.get("n_candidates") == 0)
    if len(results) >= MIN_PAGES_FOR_NOTE and zero / len(results) >= HIGH_ZERO_CANDIDATE_SHARE:
        parts.append(
            f"{zero} of {len(results)} pages produced no match candidates "
            f"({zero / len(results):.0%}, at or above {HIGH_ZERO_CANDIDATE_SHARE:.0%})"
        )
    if not parts:
        return None
    return (
        "suspect a historic street-name alias gap — "
        + "; ".join(parts)
        + ". Measure it with scripts/audit_alias_coverage.py, then write a "
        "volume alias table if it confirms. ADVISORY only, nothing is gated: these bars "
        f"misclassify funnels in both directions ({SWEEP_RECORD})"
    )


__all__ = [
    "HIGH_ZERO_CANDIDATE_SHARE",
    "LOW_MATCH_RATE",
    "MIN_PAGES_FOR_NOTE",
    "MIN_READS_FOR_NOTE",
    "SWEEP_RECORD",
    "NameMatch",
    "alias_gap_note",
    "count_name_matches",
    "load_name_match",
    "write_name_match",
]
