"""Chain stitching, validation, and block-range compression.

The transcription driver supplies dual-read rows; this module attributes,
validates, and compresses them without silently correcting uncertain data.
"""

from __future__ import annotations

import difflib
import itertools
import re
from dataclasses import dataclass, field
from typing import Any

#: Maximum new-number gap when continuing a chain.
MAX_CONT_DELTA_NEW = 600
#: Old-number continuation slack: |dOld| <= max(OLD_SLACK_RATIO*dNew, OLD_SLACK_MIN).
OLD_SLACK_RATIO = 3.0
OLD_SLACK_MIN = 80
#: A header attributes a run only when it sits within this many pixels
#: above the run's first pair (~5 row pitches at scale 4).
HEADER_REACH_PX = 150

#: Row statuses usable as data: ``agreed`` (both primary reads identical),
#: ``tiebreak`` (a disagreement settled 2-of-3 by the independent CHM
#: imaging), and ``concat`` (both engines read the same digit string; one
#: glued the typewritten pair, the other's segmentation resolves it).
ACCEPTED_STATUSES = frozenset({"agreed", "tiebreak", "concat"})

_CONTINUED = re.compile(r"cont", re.IGNORECASE)
_DIRWORD = {
    "NO": "N",
    "SO": "S",
    "EA": "E",
    "WE": "W",
    "NORTH": "N",
    "SOUTH": "S",
    "EAST": "E",
    "WEST": "W",
    "N": "N",
    "S": "S",
    "E": "E",
    "W": "W",
}


@dataclass
class PairRow:
    """One table row carrying numerals, with its dual-read verdict."""

    page: int
    strip: int
    y: float
    status: str  # agreed | disagree | only_a | only_b
    new: int | None
    old: int | None
    new_suffix: str = ""
    old_suffix: str = ""
    a_text: str = ""
    b_text: str = ""
    flags: list[str] = field(default_factory=list)


@dataclass
class Chain:
    """A street-attributed odd or even number chain."""

    chain_id: int
    street_raw: str
    street: str
    parity: str
    page: int
    pairs: list[PairRow] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def agreed(self) -> list[PairRow]:
        return [
            p
            for p in self.pairs
            if p.status in ACCEPTED_STATUSES and not any(f.startswith("uncertain") for f in p.flags)
        ]

    def last_agreed(self) -> PairRow | None:
        for p in reversed(self.pairs):
            if p.status in ACCEPTED_STATUSES:
                return p
        return None


def clean_street(raw: str) -> str:
    """Convert a printed street header to matcher form.

    Only mechanical cleanup happens here; ``names.normalize`` does the real
    suffix/direction folding at lookup time.
    """
    t = re.sub(r"[^A-Za-z0-9' ]+", " ", raw)
    t = re.sub(r"\bcontinued\b", " ", t, flags=re.IGNORECASE)
    words = [w for w in t.upper().split() if w]
    if words and words[0] in _DIRWORD:
        words[0] = _DIRWORD[words[0]]
    return " ".join(words)


@dataclass
class _Event:
    y: float
    kind: str  # header | label | pair | reset-safe context kinds
    text: str = ""
    parity: str = ""
    cont: bool = False
    row: PairRow | None = None


_SUFFIX_WORD = re.compile(
    r"\b(ST|STREET|AV|AVE|AVENUE|PL|PLACE|CT|COURT|BLVD|BOULEVARD|RD|ROAD|DR|DRIVE"
    r"|TER|TERRACE|PK|PARK|SQ|LN|WAY|CONTINUED)\b",
    re.IGNORECASE,
)


def _effective(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Pick the read to represent a non-pair row.

    A label verdict beats a header verdict: the engine garbles the rotated
    'Odd Nos.' column labels into header-looking noise ('"WON PPO'), and a
    noise header would hijack street attribution, while the text layer
    usually still reads the label words. For real headers, prefer the read
    whose text carries a street-suffix word.
    """
    a = row.get("a") or {}
    b = row.get("b") or {}
    for src in (a, b):
        if src.get("kind") == "label":
            return "label", src
    headers = [s for s in (a, b) if s.get("kind") == "header"]
    if headers:

        def _quality(src: dict[str, Any]) -> float:
            text = src.get("text", "")
            alpha = sum(c.isalpha() or c in " ." for c in text)
            ratio = alpha / len(text) if text else 0.0
            return (2.0 if _SUFFIX_WORD.search(text) else 0.0) + ratio

        return "header", max(headers, key=_quality)
    for src in (a, b):
        if src.get("kind"):
            return str(src["kind"]), src
    return "junk", {}


_FURNITURE = re.compile(r"new\s+and\s+old|house\s+numbers", re.IGNORECASE)


_ANNOTATION = re.compile(r"\bcor\b|\brear\b|\bnr\b", re.IGNORECASE)


def _is_furniture(text: str) -> bool:
    """Running heads, page numbers, corner annotations, and number-soup
    misclusters are not street headers.

    Inline annotations between pair rows are not headers. A real section
    header never starts with a digit.
    """
    if _FURNITURE.search(text):
        return True
    stripped = text.strip()
    if stripped[:1].isdigit() and not re.match(r"\d{1,3}(st|nd|rd|th)\b", stripped, re.IGNORECASE):
        return True
    if _ANNOTATION.search(stripped):
        return True
    digits = sum(c.isdigit() for c in text)
    alpha = sum(c.isalpha() for c in text)
    return digits >= alpha or alpha < 3


def _events_for_strip(page_no: int, strip_no: int, strip: dict[str, Any]) -> list[_Event]:
    events: list[_Event] = []
    for row in strip["rows"]:
        status = row["st"]
        if status in ("agreed", "tiebreak", "concat", "disagree", "only_a", "only_b"):
            a = row.get("a") or {}
            b = row.get("b") or {}
            src = a if status in ("agreed", "only_a") else (a or b)
            if status == "only_b":
                src = b
            if status in ("tiebreak", "concat"):
                src = row["v"]
            pr = PairRow(
                page=page_no,
                strip=strip_no,
                y=row["y"],
                status=status,
                new=src.get("new"),
                old=src.get("old"),
                new_suffix=src.get("new_suffix", ""),
                old_suffix=src.get("old_suffix", ""),
                a_text=a.get("text", ""),
                b_text=b.get("text", ""),
            )
            events.append(_Event(y=row["y"], kind="pair", row=pr))
        else:
            kind, src = _effective(row)
            if kind == "header" and _is_furniture(src.get("text", "")):
                continue
            if kind in ("header", "label"):
                events.append(
                    _Event(
                        y=row["y"],
                        kind=kind,
                        text=src.get("text", ""),
                        parity=src.get("parity", ""),
                        cont=bool(src.get("cont")),
                    )
                )
    events.sort(key=lambda e: e.y)
    merged: list[_Event] = []
    for e in events:
        if (
            e.kind == "header"
            and merged
            and merged[-1].kind == "header"
            and e.y - merged[-1].y < 55
        ):
            merged[-1] = _Event(y=merged[-1].y, kind="header", text=f"{merged[-1].text} {e.text}")
        else:
            merged.append(e)
    return merged


def _name_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, clean_street(a), clean_street(b)).ratio()


def _merge_adjacent_headers(per_strip: list[list[_Event]]) -> None:
    """A street header is printed once, centered over its odd+even strip
    pair, so each strip catches only its share of the words ('Henry' /
    'Court'). Rejoin neighbors at the same y into the full name on both."""
    for i in range(len(per_strip) - 1):
        for ev in per_strip[i]:
            if ev.kind != "header":
                continue
            for other in per_strip[i + 1]:
                if other.kind == "header" and abs(other.y - ev.y) < 14:
                    combined = f"{ev.text} {other.text}".strip()
                    ev.text = combined
                    other.text = combined
                    break


class Stitcher:
    """Walks pages/strips in reading order and grows chains."""

    def __init__(self) -> None:
        self.chains: list[Chain] = []
        self._page_open: list[Chain] = []  # chains active on the current page
        self._prev_page_open: list[Chain] = []
        self._next_id = 1

    def _new_chain(self, page: int, street_raw: str, flags: list[str]) -> Chain:
        ch = Chain(
            chain_id=self._next_id,
            street_raw=street_raw,
            street=clean_street(street_raw),
            parity="",
            page=page,
            flags=flags,
        )
        self._next_id += 1
        self.chains.append(ch)
        self._page_open.append(ch)
        return ch

    def _old_direction(self, ch: Chain) -> int:
        ag = [p for p in ch.agreed() if p.old is not None]
        if len(ag) < 2:
            return 0
        deltas = [b.old - a.old for a, b in itertools.pairwise(ag) if a.old and b.old]
        if not deltas:
            return 0
        pos = sum(1 for d in deltas if d > 0)
        neg = sum(1 for d in deltas if d < 0)
        return 1 if pos >= neg else -1

    def _continuity_attach(self, first: PairRow, parity: str) -> Chain | None:
        assert first.new is not None
        best: Chain | None = None
        best_delta = MAX_CONT_DELTA_NEW + 1
        second_delta = best_delta
        for ch in self._page_open:
            if ch.parity != parity:
                continue
            # spike-tolerant tail: a single agreed-but-wrong digit at a strip
            # bottom (both engines misreading the same worn numeral) must not
            # break the snake, so continuity may hold against either of the
            # last two agreed pairs
            tail = [p for p in ch.agreed() if p.new is not None][-2:]
            last = None
            for cand in reversed(tail):
                if cand.new is not None and 0 < first.new - cand.new <= MAX_CONT_DELTA_NEW:
                    last = cand
                    break
            if last is None or last.new is None:
                continue
            delta_new = first.new - last.new
            if first.old is not None and last.old is not None:
                delta_old = first.old - last.old
                direction = self._old_direction(ch)
                if direction and delta_old and (delta_old > 0) != (direction > 0):
                    continue
                if abs(delta_old) > max(OLD_SLACK_RATIO * delta_new, OLD_SLACK_MIN):
                    continue
            if first.old_suffix != (last.old_suffix or ""):
                continue
            if delta_new < best_delta:
                best, second_delta, best_delta = ch, best_delta, delta_new
            elif delta_new < second_delta:
                second_delta = delta_new
        if best is not None and second_delta <= 2 * best_delta:
            best.flags.append("ambiguous_attach")
        return best

    def _link_continued(self, street_raw: str, first: PairRow, parity: str) -> Chain | None:
        for ch in self._prev_page_open:
            if ch.parity != parity or _name_similarity(ch.street_raw, street_raw) < 0.55:
                continue
            last = ch.last_agreed()
            if last is None or last.new is None or first.new is None:
                continue
            if 0 < first.new - last.new <= MAX_CONT_DELTA_NEW:
                if ch not in self._page_open:
                    self._page_open.append(ch)
                return ch
        return None

    def _flush_run(
        self,
        page: int,
        run: list[PairRow],
        header: _Event | None,
        label: _Event | None,
        active: Chain | None,
    ) -> Chain | None:
        firsts = [p for p in run if p.status in ACCEPTED_STATUSES and p.new is not None]
        if not firsts:
            if active is not None:
                active.pairs.extend(run)
            return active
        first = firsts[0]
        parity = "odd" if first.new % 2 else "even"  # type: ignore[operator]

        chain: Chain | None = None
        header_usable = (
            header is not None
            and first.y - header.y <= HEADER_REACH_PX
            and _SUFFIX_WORD.search(header.text) is not None
        )
        if header is not None and header_usable:
            if _CONTINUED.search(header.text):
                chain = self._link_continued(header.text, first, parity)
                if chain is None:
                    chain = self._new_chain(page, header.text, ["continued_unlinked"])
            else:
                chain = self._new_chain(page, header.text, [])
        elif active is not None and active.parity in ("", parity):
            last = active.last_agreed()
            if last is not None and last.new is not None and first.new is not None:
                delta = first.new - last.new
                if 0 < delta <= MAX_CONT_DELTA_NEW:
                    chain = active
        if chain is None:
            chain = self._continuity_attach(first, parity)
        if chain is None:
            if header is not None and not header_usable:
                # a garbled header: keep its text for later repair, but the
                # attribution is unproven
                chain = self._new_chain(page, header.text, ["header_unparsed"])
            else:
                flags = ["orphan_run"] if label is None or not label.cont else ["cont_unlinked"]
                chain = self._new_chain(page, "", flags)
        if not chain.parity:
            chain.parity = parity
        elif chain.parity != parity:
            chain.flags.append("parity_mixed")
        chain.pairs.extend(run)
        return chain

    def feed_page(self, page_no: int, page: dict[str, Any]) -> None:
        self._prev_page_open = self._page_open
        self._page_open = []
        per_strip = [
            _events_for_strip(page_no, strip_no, strip)
            for strip_no, strip in enumerate(page["strips"])
        ]
        _merge_adjacent_headers(per_strip)
        for events in per_strip:
            header: _Event | None = None
            label: _Event | None = None
            active: Chain | None = None
            run: list[PairRow] = []
            for ev in events:
                if ev.kind == "pair":
                    assert ev.row is not None
                    run.append(ev.row)
                elif ev.kind in ("header", "label"):
                    if run:
                        active = self._flush_split(page_no, run, header, label, active)
                        run = []
                    if ev.kind == "header":
                        header, label = ev, None
                        active = None
                    else:
                        label = ev
                        if not ev.cont and header is None:
                            active = None
            if run:
                self._flush_split(page_no, run, header, label, active)

    def _flush_split(
        self,
        page_no: int,
        run: list[PairRow],
        header: _Event | None,
        label: _Event | None,
        active: Chain | None,
    ) -> Chain | None:
        for k, sub in enumerate(_split_sustained(run)):
            if k == 0:
                active = self._flush_run(page_no, sub, header, label, active)
            else:
                # a sustained new-number drop mid-run: a new section whose
                # header the reads missed
                active = self._flush_run(page_no, sub, None, None, None)
        return active


def _split_sustained(run: list[PairRow]) -> list[list[PairRow]]:
    """Split a run where the new numbers drop AND stay dropped.

    A single agreed-but-wrong digit (both engines misreading the same worn
    numeral, e.g. printed 3622 read '2622' between 3620 and 3628) must stay
    in its run — validation flags it as a spike. Only a drop confirmed by
    the following agreed row is a genuine new-section boundary.
    """
    idx = [i for i, p in enumerate(run) if p.status in ACCEPTED_STATUSES and p.new is not None]
    splits: list[int] = []
    for k in range(1, len(idx)):
        prev_new = run[idx[k - 1]].new
        cur_new = run[idx[k]].new
        assert prev_new is not None and cur_new is not None
        if cur_new < prev_new:
            nxt = run[idx[k + 1]].new if k + 1 < len(idx) else None
            if nxt is not None and nxt < prev_new:
                splits.append(idx[k])
    out: list[list[PairRow]] = []
    start = 0
    for s in splits:
        out.append(run[start:s])
        start = s
    out.append(run[start:])
    return [r for r in out if r]


def validate_chain(chain: Chain) -> None:
    """Flag defects verbatim: new-number non-monotonic rows, old-number
    direction breaks, implausible old jumps. Flags feed the queue and are
    excluded from compression; the pairs themselves are never altered."""
    ag = chain.agreed()
    if len(ag) < 2:
        return
    deltas = [(b.old or 0) - (a.old or 0) for a, b in itertools.pairwise(ag) if a.old and b.old]
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    direction = 1 if pos >= neg else -1
    # single-row spikes first: when row i breaks new-monotonicity but its
    # neighbors chain cleanly around it, row i is the defect (both engines
    # misreading the same worn digit), not its successor
    for i in range(1, len(ag) - 1):
        p_prev, p, p_next = ag[i - 1], ag[i], ag[i + 1]
        if p.new is None or p_prev.new is None or p_next.new is None:
            continue
        if (p.new <= p_prev.new or p.new >= p_next.new) and p_prev.new < p_next.new:
            p.flags.append("uncertain:new_spike")
        elif p.old is not None and p_prev.old is not None and p_next.old is not None:
            d1 = (p.old - p_prev.old) * direction
            d2 = (p_next.old - p.old) * direction
            span = (p_next.old - p_prev.old) * direction
            if (d1 < 0 or d2 < 0) and span >= 0:
                p.flags.append("uncertain:old_spike")
    clean = [p for p in ag if not p.flags]
    prev: PairRow | None = None
    for p in clean:
        if prev is not None and p.new is not None and prev.new is not None:
            delta_new = p.new - prev.new
            if delta_new <= 0:
                p.flags.append("uncertain:new_nonmonotonic")
                continue
            if p.old is not None and prev.old is not None:
                delta_old = p.old - prev.old
                if delta_old and (delta_old > 0) != (direction > 0):
                    p.flags.append("uncertain:old_direction")
                    continue
                if abs(delta_old) > max(OLD_SLACK_RATIO * delta_new, OLD_SLACK_MIN):
                    p.flags.append("uncertain:old_jump")
                    continue
        prev = p


def tiebreak_strip(rows: list[dict[str, Any]], chm_pairs: list[tuple[int, int, str, str]]) -> int:
    """Settle a strip's non-agreed rows 2-of-3 against the CHM imaging.

    ``chm_pairs`` is the third read's pair sequence for the same strip (new, old, new_suffix,
    old_suffix). A queued row is anchored by its nearest agreed neighbors above and below; the
    CHM rows strictly between those anchor values are matched positionally to the queued rows
    between the same anchors, and a queued row is upgraded to ``tiebreak`` only when the CHM
    value equals one of its two original reads exactly. Rows without two anchors, or with a
    third distinct value, stay queued. Mutates ``rows`` in place; returns the number of
    upgrades.
    """

    def _pair_of(src: dict[str, Any] | None) -> tuple[int, int, str, str] | None:
        if src and src.get("kind") == "pair" and src.get("new") and src.get("old"):
            return (
                int(src["new"]),
                int(src["old"]),
                str(src.get("new_suffix", "")),
                str(src.get("old_suffix", "")),
            )
        return None

    agreed_idx = [i for i, r in enumerate(rows) if r["st"] == "agreed"]
    upgrades = 0
    for k in range(len(agreed_idx) - 1):
        lo, hi = agreed_idx[k], agreed_idx[k + 1]
        queued = [i for i in range(lo + 1, hi) if rows[i]["st"] in ("disagree", "only_a", "only_b")]
        if not queued:
            continue
        lo_pair = _pair_of(rows[lo].get("a"))
        hi_pair = _pair_of(rows[hi].get("a"))
        if lo_pair is None or hi_pair is None:
            continue
        try:
            c_lo = chm_pairs.index(lo_pair)
            c_hi = chm_pairs.index(hi_pair)
        except ValueError:
            continue
        between = list(range(c_lo + 1, c_hi))
        if len(between) != len(queued):
            continue
        for qi, ci in zip(queued, between, strict=True):
            verdict = chm_pairs[ci]
            row = rows[qi]
            for src in (row.get("a"), row.get("b")):
                if _pair_of(src) == verdict:
                    row["st"] = "tiebreak"
                    row["v"] = {
                        "kind": "pair",
                        "new": verdict[0],
                        "old": verdict[1],
                        "new_suffix": verdict[2],
                        "old_suffix": verdict[3],
                    }
                    upgrades += 1
                    break
    return upgrades


def repair_names_by_alpha_bracket(
    chains: list[Chain], vocabulary: set[str], cutoff: float = 0.6
) -> dict[str, str]:
    """Second name-repair pass exploiting the book's alphabetical layout.

    A garbled header still starts with its street's first letters almost
    always (the damage lands mid-word: 'E. Raven pooMs Pk.'), and the book's
    alphabetical sections mean the true name shares that prefix. Within the
    two-character prefix family a lower fuzzy cutoff is safe — the family
    does the disambiguation the global 0.78 cutoff had to. Applies only
    clear unique winners; everything else keeps its verbatim text and flag.
    """
    from autogeoref.names import normalize

    vocab_sorted = sorted(vocabulary)
    repairs: dict[str, str] = {}
    for ch in chains:
        if "name_unmatched" not in ch.flags or not ch.street:
            continue
        core = normalize(ch.street)
        # normalize keeps PARK/PK-class suffixes; strip them here so a
        # trailing suffix word can't dilute the similarity ratio
        core = " ".join(w for w in core.split() if not _SUFFIX_WORD.fullmatch(w))
        if len(core) < 4:
            continue
        bracket = [v for v in vocab_sorted if v[:2] == core[:2]]
        if not bracket:
            continue
        # compare space-stripped: OCR garble inserts spurious spaces, and a
        # short prefix name (RAVEN) otherwise scores deceptively close to
        # the true longer one (RAVENSWOOD)
        core_cmp = core.replace(" ", "")
        # a candidate that is a strict prefix of the core with a 3+-char
        # tail left over cannot be the street: suffix words are already
        # stripped, so a real 'Raven Street' header would have produced
        # exactly 'RAVEN' — a long tail is the damaged rest of a longer name
        candidates = [
            v
            for v in bracket
            if not (
                core_cmp.startswith(v.replace(" ", ""))
                and len(core_cmp) - len(v.replace(" ", "")) >= 3
            )
        ]
        scored = sorted(
            (
                (difflib.SequenceMatcher(None, core_cmp, v.replace(" ", "")).ratio(), v)
                for v in candidates
            ),
            reverse=True,
        )
        if not scored:
            continue
        best_ratio, best = scored[0]
        second_ratio = scored[1][0] if len(scored) > 1 else 0.0
        if best_ratio >= cutoff and best_ratio - second_ratio > 0.05:
            words = ch.street.split()
            direction = words[0] if words and words[0] in ("N", "S", "E", "W") else ""
            repaired = " ".join(w for w in (direction, best) if w)
            repairs[ch.street] = repaired
            ch.flags.remove("name_unmatched")
            ch.flags.append("name_repaired_bracket")
            ch.street = repaired
    return repairs


def adopt_orphans_by_sibling(chains: list[Chain]) -> int:
    """Name page-spanning orphans from their interleaved opposite-parity twin.

    A street's odd and even columns travel together through the page snake
    (layout fact, verified p76/p12/p122), so when a long unnamed chain's
    strips interleave with exactly ONE named chain of the opposite parity on
    the same page and their new-number ranges substantially overlap, the
    orphan is that street's other side. This rescues streets whose head
    column is unreadable in both reads (tiny S-suffixed print). Uniqueness
    is required — two candidate twins means no adoption.
    """
    adopted = 0
    by_page: dict[int, list[Chain]] = {}
    for ch in chains:
        by_page.setdefault(ch.page, []).append(ch)
    for ch in chains:
        if ch.street or len(ch.agreed()) < 30:
            continue
        ag = ch.agreed()
        strips = {p.strip for p in ag}
        lo = min(p.new for p in ag if p.new is not None)
        hi = max(p.new for p in ag if p.new is not None)
        candidates = []
        for other in by_page.get(ch.page, []):
            if not other.street or other.parity == ch.parity:
                continue
            other_ag = other.agreed()
            if len(other_ag) < 30:
                continue
            other_strips = {p.strip for p in other_ag}
            if not all(s - 1 in other_strips or s + 1 in other_strips for s in strips):
                continue
            o_lo = min(p.new for p in other_ag if p.new is not None)
            o_hi = max(p.new for p in other_ag if p.new is not None)
            overlap = min(hi, o_hi) - max(lo, o_lo)
            if overlap > 0.3 * min(hi - lo, o_hi - o_lo):
                candidates.append(other)
        if len(candidates) == 1:
            ch.street = candidates[0].street
            ch.street_raw = f"(sibling of) {candidates[0].street_raw}"
            ch.flags.append("name_from_sibling")
            if "orphan_run" in ch.flags:
                ch.flags.remove("orphan_run")
            adopted += 1
    return adopted


def select_shipped(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split compressed entries into the shippable table and a conflict list.

    Shippable: a non-empty street attribution. Conflict rule (abstain-safe):
    when two entries for the same normalized street and parity have
    overlapping old ranges but disagree on the implied conversion by more
    than 4 house numbers, BOTH are dropped to the conflict list — a wrong
    conversion is worse than none (the channel abstains on a miss).
    """
    from autogeoref.names import normalize

    named = [e for e in entries if e.get("street")]
    rest = [e for e in entries if not e.get("street")]

    def _predict(e: dict[str, Any], old: int) -> int:
        old_dir = -1 if e["old_range"][1] < e["old_range"][0] else 1
        new_dir = -1 if e["new_range"][1] < e["new_range"][0] else 1
        return int(e["new_range"][0] + old_dir * new_dir * (old - e["old_range"][0]))

    conflicts: list[dict[str, Any]] = list(rest)
    dropped: set[int] = set()
    by_key: dict[tuple[str, str, str], list[int]] = {}
    for i, e in enumerate(named):
        key = (normalize(str(e["street"])), str(e.get("side", "")), str(e.get("old_suffix", "")))
        by_key.setdefault(key, []).append(i)
    for idxs in by_key.values():
        for a_pos, i in enumerate(idxs):
            for j in idxs[a_pos + 1 :]:
                ea, eb = named[i], named[j]
                lo = max(min(ea["old_range"]), min(eb["old_range"]))
                hi = min(max(ea["old_range"]), max(eb["old_range"]))
                if lo > hi:
                    continue
                mid = (lo + hi) // 2
                if abs(_predict(ea, mid) - _predict(eb, mid)) > 4:
                    dropped.add(i)
                    dropped.add(j)
    shipped = [e for i, e in enumerate(named) if i not in dropped]
    conflicts.extend(named[i] for i in sorted(dropped))
    return shipped, conflicts


def repair_street_names(chains: list[Chain], vocabulary: set[str]) -> dict[str, str]:
    """Repair OCR-damaged street names against a known-streets vocabulary.

    ``vocabulary`` holds normalized centerline street cores. Only clear,
    unique fuzzy matches are repaired; other text remains verbatim and flagged.
    """
    from autogeoref.names import normalize

    vocab_list = sorted(vocabulary)
    repairs: dict[str, str] = {}
    for ch in chains:
        if not ch.street:
            continue
        core = normalize(ch.street)
        if not core:
            continue
        if core in vocabulary:
            continue
        matches = difflib.get_close_matches(core, vocab_list, n=2, cutoff=0.78)
        if len(matches) == 1 or (
            len(matches) == 2
            and difflib.SequenceMatcher(None, core, matches[0]).ratio()
            - difflib.SequenceMatcher(None, core, matches[1]).ratio()
            > 0.05
        ):
            words = ch.street.split()
            direction = words[0] if words and words[0] in ("N", "S", "E", "W") else ""
            # the matched vocabulary core is what volume aliases target (and
            # already carries PL/CT for numbered twins) — re-attaching the raw
            # suffix would break normalize-key equality ('RAVENSWOOD PK' !=
            # 'RAVENSWOOD')
            repaired = " ".join(w for w in (direction, matches[0]) if w)
            repairs[ch.street] = repaired
            ch.flags.append("name_repaired")
            ch.street = repaired
        else:
            ch.flags.append("name_unmatched")
    return repairs


def compress_chain(chain: Chain) -> list[dict[str, Any]]:
    """Block-level ranges: first/last usable pair per new-hundred (~30:1)."""
    out: list[dict[str, Any]] = []
    usable = [p for p in chain.agreed() if p.new is not None and p.old is not None]
    by_block: dict[tuple[int, str], list[PairRow]] = {}
    for p in usable:
        by_block.setdefault((p.new // 100, p.old_suffix), []).append(p)  # type: ignore[operator]
    for (block, suffix), rows in sorted(by_block.items()):
        first, last = rows[0], rows[-1]
        n_tiebroken = sum(1 for r in rows if r.status == "tiebreak")
        # measured worst deviation of interior pairs from the linear model
        # convert() applies — 0-2 on the per-house 1909 book, up to ~10 on
        # the entrance-level 1911 register (new 411 AND 413 <- old 291)
        old_dir = -1 if last.old < first.old else 1  # type: ignore[operator]
        max_dev = 0
        for r in rows:
            assert r.new is not None and r.old is not None and first.new is not None
            predicted = first.new + old_dir * (r.old - first.old)  # type: ignore[operator]
            max_dev = max(max_dev, abs(predicted - r.new))
        entry: dict[str, Any] = {
            "street": chain.street or "",
            "street_raw": chain.street_raw,
            "side": f"{chain.parity}(new)",
            "old_range": [first.old, last.old],
            "new_range": [first.new, last.new],
            "provenance": "dual_read_chain",
            "pdf_page": first.page,
            "chain_id": chain.chain_id,
            "n_pairs": len(rows),
        }
        if n_tiebroken:
            entry["n_tiebroken"] = n_tiebroken
        if max_dev:
            entry["max_dev"] = max_dev
        if suffix:
            entry["old_suffix"] = suffix
        if chain.flags:
            entry["chain_flags"] = sorted(set(chain.flags))
        out.append(entry)
        _ = block
    return out
