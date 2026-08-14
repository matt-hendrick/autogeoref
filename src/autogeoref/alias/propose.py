"""Propose volume alias entries from a pinned rename source, and tier them.

The chain: parse the city's rename source (:mod:`autogeoref.alias.source`) ->
intersect with the volume's UNMATCHED read families -> keep only values that
are keys of the volume's own bounded centerline index -> corroborate with a
locality measured from sheet evidence -> corroborate close calls with printed
address numerals.

Then a tier — ``clean`` (safe to write unread), ``held`` (reachable but short
of the bar, an agent queue rather than a discard), ``refused``,
``no-candidate``, ``already-aliased``. The tier is the whole safety story and
`docs/INTERNALS.md` defines each one. None of the bars below is an acceptance
threshold: no gate, funnel, or placement decision reads this module.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from shapely.geometry import Point
from shapely.ops import nearest_points

from ..address_grid import NS, AddressGrid
from ..name_match import NameMatch, count_name_matches
from ..names import normalize
from .source import (
    DIRS,
    RenameSource,
    address_ranges,
    alias_key_for,
    clean,
    snap_value,
)

if TYPE_CHECKING:
    from ..centerlines import CenterlineIndex
    from ..volume import SheetInput

logger = logging.getLogger(__name__)

#: Locality accept bar, metres. Correct recorded entries measured 34-384 m
#: from their sheets' locality; wrong rivals 589 m and up.
ACCEPT_LOCALITY_M = 400.0

#: How much further a rival must be for the locality to have DECIDED between
#: two candidates: twice as far, or 300 m further.
MARGIN_RATIO = 2.0
MARGIN_M = 300.0

#: Sheets a locality must be measured on before it decides anything. One
#: sheet's intersection cloud is a single sample and the record says so
#: explicitly; two is the floor the recorded forward test used.
MIN_SHEETS = 2

#: A key map reads the whole volume, so its "locality" is the volume. Sheets
#: with more matched co-reads than this are excluded from locality entirely.
KEYMAP_READ_CAP = 35

#: Flat-earth metres per degree at the reference corpus's latitude. These
#: distances are compared against a 400 m bar, so a projection is not worth its
#: dependency — the record's limits section says the same.
M_PER_DEG_LAT = 111_100.0
M_PER_DEG_LON = 82_900.0

#: Printed numerals needed before the numeral check says anything, and the
#: share of them that must land inside the winning candidate's documented
#: range. One numeral is an OCR read, not a corroboration.
MIN_NUMERALS = 2
NUMERAL_INSIDE_SHARE = 0.6

#: Blocks of slack when comparing a printed numeral to a documented range. Both
#: sides are house numbers quoted in the same grid, so one block is generous;
#: it exists because a documented range is the RENAMED segment and a sheet can
#: print a number just past its end.
NUMERAL_PAD_BLOCKS = 1.0

#: Blocks of slack in the source-against-geography agreement check, which asks
#: only "is this quoted range anywhere near the street it claims to describe".
#: A mile, because the grid model is LINEAR and a real grid is not
#: (:mod:`autogeoref.address_grid` says why). Tightening this would withhold
#: valid auto-writes; it is not what stops a wrong one. That is the numerals
#: landing inside the winner's quoted range and outside the runner-up's.
AGREEMENT_PAD_BLOCKS = 8.0

#: Tier names, in the order a report lists them.
CLEAN = "clean"
HELD = "held"
#: A family every candidate of which is unreachable — no measurable locality, or
#: every candidate beyond the accept bar. This is a REFUSAL, not a call for more
#: corroboration: a correctly documented rename is still unwritable when the
#: modern value has no in-bounds geometry anywhere near the reads, which is the
#: whole out-of-city edge grid. Separating it from HELD keeps the agent queue
#: honest — the held tier is work someone can finish, this tier is not.
REFUSED = "refused"
NO_CANDIDATE = "no-candidate"
#: The family is already covered by a landed entry and these reads are variants
#: of it. Not a proposal at all; kept as an outcome so the counts add up.
ALREADY_ALIASED = "already-aliased"


@dataclass(frozen=True)
class SheetReads:
    """One sheet's reads, in the three shapes the evidence checks need."""

    page: str
    #: Normalized keys of every street read (alias-free).
    keys: frozenset[str]
    #: Cleaned raw read strings — where direction-qualified and full-string
    #: keys live, since those never survive normalization.
    raws: frozenset[str]
    #: Printed address numerals the annotation attributed to a read, keyed by
    #: the read's cleaned raw string.
    numerals: Mapping[str, tuple[int, ...]]

    def has_key(self, key: str) -> bool:
        """Does this sheet carry a read an alias table would key as ``key``?

        EXACT match on either shape. A prefix match here would be a false-permit
        path with teeth: ``PARK RIDGE AV`` would answer for ``PARK``, so three
        sheets that never mention Park could satisfy the two-sheet floor and
        their intersection clouds would set the locality. Both shapes
        :func:`autogeoref.alias.source.alias_key_for` can produce — the
        normalized key and a full court/place string — match exactly.
        """
        return key in self.keys or key in self.raws


def sheet_reads(sheets: Sequence[SheetInput]) -> list[SheetReads]:
    """Per-sheet read keys, raw strings, and attributed address numerals."""
    out: list[SheetReads] = []
    for sheet in sheets:
        keys: set[str] = set()
        raws: set[str] = set()
        for street in sheet.annotation.get("streets") or ():
            name = street.get("name")
            if not name:
                continue
            key = normalize(str(name))
            if key:
                keys.add(key)
            raws.add(clean(str(name)))
        numerals: dict[str, list[int]] = defaultdict(list)
        for numeral in sheet.annotation.get("address_numerals") or ():
            street_name, value = numeral.get("street"), numeral.get("value")
            if not street_name or isinstance(value, bool) or not isinstance(value, int):
                continue
            numerals[clean(str(street_name))].append(value)
        out.append(
            SheetReads(
                page=sheet.page,
                keys=frozenset(keys),
                raws=frozenset(raws),
                numerals={k: tuple(v) for k, v in numerals.items()},
            )
        )
    return out


#: Long direction spellings folded onto their letter, so a family reading both
#: ``N. PARK`` and ``NORTH PARK`` is not mistaken for a per-side rename.
_DIRECTION_LETTER = {"NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W"}


def _leading_direction(raw: str) -> set[str]:
    """The printed leading direction of a raw read, or the empty set.

    A set so callers can union across a read family and see divergence without
    a sentinel; an undirected read contributes nothing rather than a distinct
    "no direction" value, which would make every mixed family look diverging.
    """
    tokens = clean(raw).split()
    if not tokens or tokens[0] not in DIRS:
        return set()
    return {_DIRECTION_LETTER.get(tokens[0], tokens[0])}


def _distance_m(point: tuple[float, float], geometry: Any) -> float:
    a, b = nearest_points(Point(point), geometry)
    dx = (a.x - b.x) * M_PER_DEG_LON
    dy = (a.y - b.y) * M_PER_DEG_LAT
    return float((dx * dx + dy * dy) ** 0.5)


def locality_distance(
    index: CenterlineIndex, reads: Sequence[SheetReads], key: str, candidate: str
) -> tuple[int, float | None]:
    """``(sheets used, median minimum distance in metres)`` for one candidate.

    Each sheet carrying the read is localized by the pairwise intersection
    points of its DIRECTLY matched co-reads; the candidate street's distance to
    that cloud is that sheet's measurement, and the median over sheets is the
    answer. Sheets whose only matched co-reads are parallel give no
    intersection (or a spurious far one) and are skipped rather than counted as
    evidence against.
    """
    geometry = index.merged(candidate)
    if geometry is None:
        return 0, None
    distances: list[float] = []
    for sheet in reads:
        if not sheet.has_key(key):
            continue
        matched = sorted(k for k in sheet.keys if k in index.by_name and k != candidate)
        if len(matched) < 2 or len(matched) > KEYMAP_READ_CAP:
            continue
        points: list[tuple[float, float]] = []
        for i, a in enumerate(matched):
            for b in matched[i + 1 :]:
                points.extend(index.intersections(a, b))
        if points:
            distances.append(min(_distance_m(p, geometry) for p in points))
    if not distances:
        return 0, None
    distances.sort()
    return len(distances), distances[len(distances) // 2]


@dataclass(frozen=True)
class Candidate:
    """One in-bounds modern street the source offers for a read family."""

    value: str
    sheets: int
    locality_m: float | None
    source_line: str

    @property
    def distance(self) -> float:
        """Sortable locality: an unmeasured candidate sorts last."""
        return self.locality_m if self.locality_m is not None else float("inf")


@dataclass(frozen=True)
class DocumentedExtent:
    """The extents a source line quotes ALONG one street, in signed numbers.

    Kept as SEPARATE ranges, never merged into a hull: the source writes
    disjoint segments on one line, and merging them would manufacture a range
    covering the gap — a numeral there would then "corroborate" a rename
    nothing documents. ``sign`` is which side of the numbering origin the
    STREET lies on, taken from its own geometry: a printed house number carries
    no direction, so nothing else can say whether ``350`` means 350 North or
    South. A street crossing the origin gets no numeral check at all.
    """

    ranges: tuple[tuple[float, float], ...]
    axis: str
    sign: int

    def covers(self, numeral: int, pad: float) -> bool:
        """Is a printed house number inside ANY documented extent, padded?"""
        value = self.sign * float(numeral)
        return any(lo - pad <= value <= hi + pad for lo, hi in self.ranges)

    @property
    def label(self) -> str:
        letter = (
            ("N" if self.sign > 0 else "S") if self.axis == NS else ("E" if self.sign > 0 else "W")
        )
        parts = []
        for lo, hi in self.ranges:
            low, high = sorted((abs(lo), abs(hi)))
            parts.append(f"{low:.0f}-{high:.0f}{letter}")
        return "+".join(parts)


@dataclass(frozen=True)
class NumeralCheck:
    """Whether printed numerals independently support the winning candidate.

    ``supported`` is the only field the tiering reads; the rest is what the
    report and the provenance comment quote so a reader can re-derive it.
    """

    supported: bool
    reason: str
    numerals: tuple[int, ...] = ()
    winner_range: str | None = None
    runner_range: str | None = None
    inside_winner: int = 0
    inside_runner: int = 0


def documented_along(
    grid: AddressGrid, index: CenterlineIndex, candidate: Candidate, block: float
) -> DocumentedExtent | None:
    """The documented along-street extents for a candidate, if geography agrees.

    A source line quotes a cross position and, sometimes, extents along the
    street; only the ALONG extents compare with printed house numbers, and
    which one a quoted range is depends on the street's geometry — hence the
    grid. Four filters, each a way a quoted number can look like an extent
    without being one: a clear numbering axis (not a diagonal); wholly on ONE
    side of the numbering origin; each extent at least a block wide; and each
    on the street's own side, meeting its own geometric extent.
    """
    geometry = index.merged(candidate.value)
    if geometry is None:
        return None
    span = grid.span(geometry.bounds)
    if span is None:  # no clear numbering axis (diagonal or stub geometry)
        return None
    if span.along_lo >= 0 and span.along_hi >= 0:
        sign = 1
    elif span.along_lo <= 0 and span.along_hi <= 0:
        sign = -1
    else:
        return None  # the street crosses the origin: a numeral cannot be placed
    pad = AGREEMENT_PAD_BLOCKS * block
    kept = tuple(
        (lo, hi)
        for lo, hi in address_ranges(candidate.source_line).spans_on(span.axis)
        if hi - lo >= block and (sign > 0) == (hi > 0) and span.along_overlaps(lo, hi, pad=pad)
    )
    if not kept:
        return None
    return DocumentedExtent(ranges=kept, axis=span.axis, sign=sign)


def numeral_check(
    grid: AddressGrid | None,
    index: CenterlineIndex,
    reads: Sequence[SheetReads],
    key: str,
    winner: Candidate,
    runner: Candidate | None,
    *,
    block: float,
    era_modern: bool,
) -> NumeralCheck:
    """Do the sheet's printed numerals pick the winner over the runner-up?

    Every way this can decline is a HOLD, never a pass: no grid configured for
    the city, an address era that predates the city's renumbering (the source's
    ranges are post-renumbering numbers, so comparing them with pre-renumbering
    sheet numerals is arithmetic across two different grids), no documented
    range for either candidate, or too few numerals to be more than an OCR
    read.
    """
    if grid is None:
        return NumeralCheck(False, "no address grid configured for this city")
    if not era_modern:
        return NumeralCheck(False, "volume address era is not modern; source ranges do not apply")
    if runner is None:
        return NumeralCheck(False, "no runner-up to separate")
    winner_range = documented_along(grid, index, winner, block)
    if winner_range is None:
        return NumeralCheck(False, f"no documented along-street range for {winner.value}")
    runner_range = documented_along(grid, index, runner, block)
    if runner_range is None:
        # Without a range for the rival there is nothing to fall OUTSIDE of.
        return NumeralCheck(
            False,
            f"no documented along-street range for the runner-up {runner.value}",
            winner_range=winner_range.label,
        )
    if (runner_range.axis, runner_range.sign) != (winner_range.axis, winner_range.sign):
        # The two candidates number on different lines — the other axis, or the
        # other side of the origin — so one printed numeral does not mean the
        # same thing for both, and "inside one, outside the other" establishes
        # nothing. What separates a north-south street from an east-west one is
        # the read's own printed orientation: a different channel, and not this
        # one's to claim.
        return NumeralCheck(
            False,
            f"{runner.value} numbers on a different grid line ({runner_range.label}"
            f" vs {winner_range.label}); numerals cannot separate them",
            winner_range=winner_range.label,
            runner_range=runner_range.label,
        )
    values: list[int] = []
    for sheet in reads:
        if not sheet.has_key(key):
            continue
        for raw, numerals in sheet.numerals.items():
            if alias_key_for(raw) == key:
                values.extend(numerals)
    pad = NUMERAL_PAD_BLOCKS * block
    inside_winner = sum(1 for v in values if winner_range.covers(v, pad))
    inside_runner = sum(1 for v in values if runner_range.covers(v, pad))
    check = NumeralCheck(
        supported=False,
        reason="",
        numerals=tuple(sorted(values)),
        winner_range=winner_range.label,
        runner_range=runner_range.label,
        inside_winner=inside_winner,
        inside_runner=inside_runner,
    )
    if len(values) < MIN_NUMERALS:
        return dataclasses.replace(
            check, reason=f"{len(values)} attributed numerals (need {MIN_NUMERALS})"
        )
    if inside_runner:
        return dataclasses.replace(
            check, reason=f"{inside_runner} numeral(s) also fit {runner.value}"
        )
    if inside_winner < MIN_NUMERALS or inside_winner < NUMERAL_INSIDE_SHARE * len(values):
        return dataclasses.replace(
            check, reason=f"only {inside_winner} of {len(values)} numerals fit {winner.value}"
        )
    return dataclasses.replace(
        check,
        supported=True,
        reason=(
            f"{inside_winner} of {len(values)} numerals inside {winner.value}"
            f" {winner_range.label}, none inside {runner.value} {runner_range.label}"
        ),
    )


@dataclass(frozen=True)
class Proposal:
    """One read family's proposal, tiered."""

    key: str
    reads: int
    tier: str
    reason: str
    value: str | None = None
    raw_examples: tuple[str, ...] = ()
    candidates: tuple[Candidate, ...] = ()
    numerals: NumeralCheck | None = None

    @property
    def winner(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None


@dataclass(frozen=True)
class VolumeProposal:
    """Everything the proposer concluded about one volume."""

    volume: str
    counts: NameMatch
    proposals: tuple[Proposal, ...]

    def tier(self, name: str) -> tuple[Proposal, ...]:
        return tuple(p for p in self.proposals if p.tier == name)

    @property
    def clean(self) -> tuple[Proposal, ...]:
        return self.tier(CLEAN)

    @property
    def held(self) -> tuple[Proposal, ...]:
        return self.tier(HELD)

    @property
    def refused(self) -> tuple[Proposal, ...]:
        return self.tier(REFUSED)

    @property
    def no_candidate(self) -> tuple[Proposal, ...]:
        return self.tier(NO_CANDIDATE)

    @property
    def already_aliased(self) -> tuple[Proposal, ...]:
        return self.tier(ALREADY_ALIASED)

    @property
    def table(self) -> dict[str, str]:
        """The clean tier as an alias table."""
        return {p.key: p.value for p in self.clean if p.value}


def propose_volume(
    volume: str,
    source: RenameSource,
    index: CenterlineIndex,
    sheets: Sequence[SheetInput],
    *,
    grid: AddressGrid | None = None,
    block: float = 100.0,
    era_modern: bool = False,
    existing: Mapping[str, str] | None = None,
    entry_guard: Callable[[str, str], Sequence[str]] | None = None,
    counts: NameMatch | None = None,
) -> VolumeProposal:
    """Tiered alias proposals for one volume.

    ``index`` must be the volume's OWN bounded index — the value-in-bounds
    filter is exactly "is this key in ``index.by_name``", and those verdicts
    are bounds-provenance-sensitive. ``existing`` is the already-landed table:
    its keys are never re-proposed, so an owner-signed entry cannot be restated
    or contradicted. ``entry_guard`` is the structural veto, one entry at a
    time (:func:`autogeoref.alias.validate.validate_table`), so one bad key
    cannot cost a volume its sound entries; whole-table validation still runs.
    """
    existing = existing or {}
    # The caller usually measured this already (it is the tripwire number it
    # decided to propose on); recounting is correct but wasteful.
    counts = counts if counts is not None else count_name_matches(sheets, index)
    reads = sheet_reads(sheets)
    keys = set(index.by_name)

    families: Counter[str] = Counter()
    raw_examples: dict[str, list[str]] = defaultdict(list)
    directions: dict[str, set[str]] = defaultdict(set)
    for raw, n in counts.unmatched.items():
        key = alias_key_for(raw)
        if key:
            families[key] += n
            raw_examples[key].append(raw)
            directions[key] |= _leading_direction(raw)

    tier_ctx = _TierContext(
        grid=grid,
        index=index,
        reads=reads,
        block=block,
        era_modern=era_modern,
        entry_guard=entry_guard,
    )
    proposals: list[Proposal] = []
    for key, reads_n in families.most_common():
        examples = tuple(sorted(raw_examples[key])[:3])
        if key in existing:
            proposals.append(
                Proposal(
                    key=key,
                    reads=reads_n,
                    tier=ALREADY_ALIASED,
                    reason=f"already aliased to {existing[key]!r}; these reads are variants",
                    raw_examples=examples,
                )
            )
            continue
        lines: dict[str, str] = {}
        for name, line in source.candidates(key):
            value = snap_value(normalize(name), keys)
            # The empty key is a real bucket in a bounded index (unnamed
            # segments land in it), so a source phrase that normalizes away
            # would otherwise "snap" to it and score as a candidate street.
            if value and value != key:
                lines.setdefault(value, line)
        if not lines:
            proposals.append(
                Proposal(
                    key=key,
                    reads=reads_n,
                    tier=NO_CANDIDATE,
                    reason="the source documents no rename for this read",
                    raw_examples=examples,
                )
            )
            continue
        scored: list[Candidate] = []
        for value, line in lines.items():
            n_sheets, median = locality_distance(index, reads, key, value)
            scored.append(
                Candidate(value=value, sheets=n_sheets, locality_m=median, source_line=line)
            )
        candidates = tuple(sorted(scored, key=lambda c: (c.distance, c.value)))
        proposals.append(
            _tier(
                tier_ctx,
                key=key,
                reads_n=reads_n,
                examples=examples,
                candidates=candidates,
                read_directions=frozenset(directions[key]),
            )
        )
    return VolumeProposal(volume=volume, counts=counts, proposals=tuple(proposals))


@dataclass(frozen=True)
class _TierContext:
    """The volume-wide inputs every read family's tiering shares."""

    grid: AddressGrid | None
    index: CenterlineIndex
    reads: Sequence[SheetReads]
    block: float
    era_modern: bool
    entry_guard: Callable[[str, str], Sequence[str]] | None


def _tier(
    ctx: _TierContext,
    *,
    key: str,
    reads_n: int,
    examples: tuple[str, ...],
    candidates: tuple[Candidate, ...],
    read_directions: frozenset[str] = frozenset(),
) -> Proposal:
    """Apply the clean-tier rules to one family's scored candidates."""
    winner = candidates[0]
    runner = candidates[1] if len(candidates) > 1 else None
    base = Proposal(
        key=key,
        reads=reads_n,
        tier=HELD,
        reason="",
        value=winner.value,
        raw_examples=examples,
        candidates=candidates,
    )
    # Unreachable candidates are a REFUSAL, not a request for corroboration: no
    # amount of further checking makes a rename writable when its modern value
    # has no in-bounds geometry near the reads. That is the out-of-city edge
    # grid, and the distinction keeps the held tier a queue someone can finish.
    if winner.locality_m is None:
        return dataclasses.replace(
            base,
            tier=REFUSED,
            reason="no candidate has a measurable locality on any sheet carrying the read",
        )
    if winner.locality_m > ACCEPT_LOCALITY_M:
        return dataclasses.replace(
            base,
            tier=REFUSED,
            reason=(
                f"nearest candidate {winner.value} is {winner.locality_m:.0f} m away"
                f" (bar {ACCEPT_LOCALITY_M:.0f} m)"
            ),
        )
    if winner.sheets < MIN_SHEETS:
        return dataclasses.replace(
            base, reason=f"locality measured on {winner.sheets} sheet(s), need {MIN_SHEETS}"
        )
    if len(read_directions) > 1:
        # The family's reads carry more than one printed direction, and this key
        # is the direction-STRIPPED form: one entry would map both sides to one
        # street. A numbered avenue can diverge, each half renamed differently,
        # and only the qualified keys can express that — so the safe automatic
        # answer is to hold the family.
        return dataclasses.replace(
            base,
            reason=(
                f"reads carry diverging printed directions ({', '.join(sorted(read_directions))});"
                f" a direction-stripped key cannot express a per-side rename"
            ),
        )
    structural = list(ctx.entry_guard(key, winner.value)) if ctx.entry_guard else []
    if structural:
        return dataclasses.replace(
            base,
            reason=(
                f"{winner.value} is the evidence's answer, but the sweep cannot"
                f" write this key on its own here: {'; '.join(structural)}"
            ),
        )
    if runner is None:
        return dataclasses.replace(
            base,
            tier=CLEAN,
            reason=(
                f"unique in-bounds candidate; locality {winner.locality_m:.0f} m"
                f" over {winner.sheets} sheets"
            ),
        )
    rival = runner.distance
    decided = rival >= MARGIN_RATIO * winner.locality_m or rival - winner.locality_m >= MARGIN_M
    if not decided:
        return dataclasses.replace(
            base,
            reason=(
                f"{winner.value} {winner.locality_m:.0f} m vs {runner.value} {rival:.0f} m"
                f" does not clear the {MARGIN_RATIO:g}x / {MARGIN_M:.0f} m margin"
            ),
        )
    check = numeral_check(
        ctx.grid,
        ctx.index,
        ctx.reads,
        key,
        winner,
        runner,
        block=ctx.block,
        era_modern=ctx.era_modern,
    )
    if not check.supported:
        return dataclasses.replace(
            base,
            numerals=check,
            reason=(
                f"locality decided {winner.value} {winner.locality_m:.0f} m over"
                f" {runner.value} {rival:.0f} m, but numerals did not corroborate:"
                f" {check.reason}"
            ),
        )
    return dataclasses.replace(
        base,
        tier=CLEAN,
        numerals=check,
        reason=(
            f"locality {winner.locality_m:.0f} m over {winner.sheets} sheets beats"
            f" {runner.value} {rival:.0f} m; {check.reason}"
        ),
    )


__all__ = [
    "ACCEPT_LOCALITY_M",
    "AGREEMENT_PAD_BLOCKS",
    "ALREADY_ALIASED",
    "CLEAN",
    "HELD",
    "KEYMAP_READ_CAP",
    "MARGIN_M",
    "MARGIN_RATIO",
    "MIN_NUMERALS",
    "MIN_SHEETS",
    "NO_CANDIDATE",
    "NUMERAL_INSIDE_SHARE",
    "NUMERAL_PAD_BLOCKS",
    "REFUSED",
    "Candidate",
    "DocumentedExtent",
    "NumeralCheck",
    "Proposal",
    "SheetReads",
    "VolumeProposal",
    "documented_along",
    "locality_distance",
    "numeral_check",
    "propose_volume",
    "sheet_reads",
]
