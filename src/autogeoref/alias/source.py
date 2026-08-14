"""Pinned rename sources: parse a documented street-renaming list into pairs.

A sheet drawn before a street was renamed carries a name the modern centerline
reference does not hold, and :mod:`autogeoref.name_match` is the tripwire that
says so. The remedy is a volume alias table, and the asset behind one is the
city's *documented rename list*, from which a deterministic parser recovers
nearly everything a manual pass would.

This module owns the source registry — a city CONFIGURES which text and parser
it has, and a city with none is still scanned and reported, never proposed for
— plus the parsers, the key conventions a table must respect, and the address
ranges the source quotes beside each entry, which make a numeral check possible
at all. `docs/INTERNALS.md` states the policy. Zero model calls, zero network;
the source text itself is not committed.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from ..address_grid import LETTER_AXIS, signed
from ..names import normalize

#: Suffix spellings collapsed to one canonical token before matching. The
#: source writes them long, sheets abbreviate, and both must meet.
SUFFIX_CANON = {
    "AVENUE": "AV",
    "AVE": "AV",
    "AV": "AV",
    "STREET": "ST",
    "ST": "ST",
    "BOULEVARD": "BLVD",
    "BOUL": "BLVD",
    "BLVD": "BLVD",
    "BL": "BLVD",
    "PLACE": "PL",
    "PL": "PL",
    "COURT": "CT",
    "CT": "CT",
    "ROAD": "RD",
    "RD": "RD",
    "DRIVE": "DR",
    "DR": "DR",
    "TERRACE": "TER",
    "TERR": "TER",
    "TER": "TER",
    "PARKWAY": "PKWY",
    "PKWY": "PKWY",
    "LANE": "LN",
    "LN": "LN",
    "SQUARE": "SQ",
    "SQ": "SQ",
    "WAY": "WAY",
}
SUFFIXES = frozenset(SUFFIX_CANON.values())
DIRS = frozenset({"N", "S", "E", "W", "NORTH", "SOUTH", "EAST", "WEST"})

#: Range tokens and prose connectors break a street-name token run. Without
#: them the parser glues Martin's explanatory sentences onto street names and
#: the ambiguity tier fills with junk candidates.
STOP_WORDS = frozenset(
    {
        "TO",
        "AND",
        "AT",
        "FROM",
        "BETWEEN",
        "VACATED",
        "IN",
        "THE",
        "OR",
        "NEAR",
        "NAMED",
        "AFTER",
        "WAS",
        "WHO",
        "A",
        "AN",
        "OF",
        "HE",
        "SHE",
        "IT",
        "FIRST",
        "ALSO",
        "NOW",
        "SEE",
        "ON",
        "BY",
        "WITH",
        "RUNNING",
        "LOCATED",
        "SIDE",
    }
)
COURT_WORDS = frozenset({"PL", "PLACE", "CT", "COURT"})

#: Ordinals the source spells out, and the OCR shapes they arrive in.
SPELLED = {
    "FORTIETH": "40TH",
    "FORTY FIRST": "41ST",
    "FORTY SECOND": "42ND",
    "FORTY THIRD": "43RD",
    "FORTY FOURTH": "44TH",
    "FORTY FIFTH": "45TH",
    "FORTY SIXTH": "46TH",
    "FORTY SEVENTH": "47TH",
    "FORTY EIGHTH": "48TH",
    "FORTY NINTH": "49TH",
    "FIFTIETH": "50TH",
    "FIFTY FIRST": "51ST",
    "FIFTY SECOND": "52ND",
    "FIFTY THIRD": "53RD",
    "FIFTH": "5TH",
    "FOURTH": "4TH",
    "THIRD": "3RD",
    "SECOND": "2ND",
    "FIRST": "1ST",
    "TWELFTH": "12TH",
}

_RANGE_TOK = re.compile(r"^\d+(N|S|E|W|NE|NW|SE|SW)?$")
_D_ORDINAL = re.compile(r"^(\d+)D$")  # period ordinals: 52D -> 52ND
_SPLIT_ORDINAL = re.compile(r"^(\d+) (ST|ND|RD|TH)$")  # OCR split: '42 ND'
_BARE_NUM = re.compile(r"^(\d+)$")
#: ``fr. 2200 to 2799N`` — an extent along the street. The ``fr.`` is optional
#: because the source writes it both ways (``1200 to 1250 N.``), and dropping
#: the requirement is most of this pattern's coverage. Prose can therefore
#: produce a spurious match ("in 1909 to 1911 N. Clark"), which is why every
#: consumer additionally requires the extent to be at least a block wide and to
#: agree with the candidate street's own geometry.
_SPAN_RANGE = re.compile(r"\b(?:FR|FROM)?\.?\s*(\d{1,5})\s+TO\s+(\d{1,5})\s*([NSEW])\b")
#: ``1420W`` — a single position, usually the cross-axis one.
_POINT_RANGE = re.compile(r"\b(\d{1,5})\s*([NSEW])\b")


def clean(text: str) -> str:
    """Uppercase, punctuation-stripped form the parser and keys both work in."""
    t = text.upper().replace("’", "").replace("'", "")  # noqa: RUF001 — curly quote intended
    t = re.sub(r"[^A-Z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def canon_tokens(phrase: str) -> list[str]:
    """Canonical tokens, with the source's solid ``McHenry`` spelling joined."""
    toks = [SUFFIX_CANON.get(t, t) for t in clean(phrase).split() if t]
    out: list[str] = []
    i = 0
    while i < len(toks):
        if toks[i] == "MC" and i + 1 < len(toks):
            out.append("MC" + toks[i + 1])
            i += 2
        else:
            out.append(toks[i])
            i += 1
    return out


def canon_key(key: str) -> str:
    """Canonical base form of an alias key for source lookup.

    Spelled-out and period ordinals become digits (``FIFTY SECOND`` /
    ``52D`` -> ``52ND``), the leading direction is dropped (it is a key
    detail, not part of the name the source lists), and suffixes canonicalize.
    """
    k = clean(key)
    for spelled, digit in SPELLED.items():
        if k.startswith(spelled):
            k = digit + k[len(spelled) :]
            break
    toks = k.split()
    while toks and toks[0] in DIRS:
        toks = toks[1:]
    if toks:
        m = _SPLIT_ORDINAL.match(" ".join(toks[:2]))
        if m:
            toks = [m.group(1) + m.group(2), *toks[2:]]
        m = _D_ORDINAL.match(toks[0])
        if m:
            n = m.group(1)
            toks[0] = n + ("ND" if n[-1] == "2" else "RD" if n[-1] == "3" else "TH")
        m = _BARE_NUM.match(toks[0])
        if m:
            n = m.group(1)
            if n[-2:] in {"11", "12", "13"}:
                toks[0] = n + "TH"
            else:
                toks[0] = n + {"1": "ST", "2": "ND", "3": "RD"}.get(n[-1], "TH")
    return " ".join(SUFFIX_CANON.get(t, t) for t in toks)


def edit_distance_one(x: str, y: str) -> bool:
    """One substitution or one deletion apart."""
    if abs(len(x) - len(y)) > 1:
        return False
    if len(x) == len(y):
        return sum(c != d for c, d in zip(x, y, strict=True)) <= 1
    if len(x) > len(y):
        x, y = y, x
    return any(x == y[:i] + y[i + 1 :] for i in range(len(y)))


def tokens_match(a: list[str], b: list[str]) -> bool:
    """Token lists match, tolerating one edit on longer same-prefix tokens.

    Source spellings drift against sheet reads and against the modern
    reference alike (Eldredge/Eldridge, Marcy/Marcey, Reese/Rees). The
    two-character prefix and four-character floor are what keep the tolerance
    from merging genuinely different short names.
    """
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if x == y:
            continue
        if min(len(x), len(y)) >= 4 and x[:2] == y[:2] and edit_distance_one(x, y):
            continue
        return False
    return True


def snap_value(val: str, keys: Iterable[str]) -> str | None:
    """Snap a parsed value onto a volume index key, or None.

    One edit absorbs source typos (``Delware``). INDEX MEMBERSHIP is what
    makes the fuzz safe: nothing outside the volume's own bounded street list
    can ever be produced.
    """
    key_set = set(keys)
    if val in key_set:
        return val
    if len(val) >= 6:
        hits = [
            k for k in key_set if len(k) >= 6 and k[:2] == val[:2] and edit_distance_one(k, val)
        ]
        if len(hits) == 1:
            return hits[0]
    return None


def alias_key_for(raw: str) -> str:
    """The alias key the recorded conventions would write for a raw read.

    Court-like reads take the full cleaned string — the pre-guard full-string
    lookup — because ``normalize`` refuses to alias a court or place through its
    stripped stem. Everything else takes the normalized key.
    """
    toks = clean(raw).split()
    if toks and toks[-1] in COURT_WORDS:
        return " ".join(SUFFIX_CANON.get(t, t) if t in COURT_WORDS else t for t in toks)
    return normalize(raw)


class RenamePair(NamedTuple):
    """One documented rename, with the source line it was read from."""

    old: str
    new: str
    line: str


class SourceRanges(NamedTuple):
    """Address extents quoted on a source line, as signed grid numbers.

    ``spans`` are ``(lo, hi, axis)`` extents (``fr. 2200 to 2799N``);
    ``points`` are ``(number, axis)`` single positions (``1420W``). Which one
    is the street's ALONG range and which its cross position depends on the
    street's geometry, so this type keeps both and decides nothing.
    """

    spans: tuple[tuple[float, float, str], ...]
    points: tuple[tuple[float, str], ...]

    def spans_on(self, axis: str) -> tuple[tuple[float, float], ...]:
        return tuple((lo, hi) for lo, hi, a in self.spans if a == axis)

    def points_on(self, axis: str) -> tuple[float, ...]:
        return tuple(n for n, a in self.points if a == axis)


def address_ranges(line: str) -> SourceRanges:
    """Signed grid extents quoted on a source line.

    A span consumes its own numbers so the trailing ``2799N`` of
    ``fr. 2200 to 2799N`` is not also read as a standalone position.
    """
    text = clean(line)
    spans: list[tuple[float, float, str]] = []
    for m in _SPAN_RANGE.finditer(text):
        letter = m.group(3)
        lo, hi = signed(letter, float(m.group(1))), signed(letter, float(m.group(2)))
        spans.append((min(lo, hi), max(lo, hi), LETTER_AXIS[letter]))
    remainder = _SPAN_RANGE.sub(" ", text)
    points = [
        (signed(m.group(2), float(m.group(1))), LETTER_AXIS[m.group(2)])
        for m in _POINT_RANGE.finditer(remainder)
    ]
    return SourceRanges(tuple(spans), tuple(points))


def _name_runs(text: str) -> list[list[str]]:
    """Suffix-terminated street-name token runs in a clause.

    A short trailing suffix-less run is kept so a bare new name
    (``Shakespeare``) survives; runs that are nothing but suffixes and
    directions are dropped.
    """
    runs: list[list[str]] = []
    cur: list[str] = []
    for t in canon_tokens(text):
        if _RANGE_TOK.match(t) or t in STOP_WORDS:
            cur = []
            continue
        cur.append(t)
        if t in SUFFIXES:
            runs.append(cur)
            cur = []
    if cur and len(cur) <= 3:
        runs.append(cur)
    out = []
    for run in runs:
        name = [t for t in run if t not in DIRS]
        if name and not all(t in SUFFIXES for t in name):
            out.append(name)
    return out


def _joined_lines(text: str) -> list[str]:
    """Source lines with continuations merged.

    A line not ending in ``.`` flows into the next unless that next line opens
    a new former-name entry — the shape the PDF extraction leaves behind.
    """
    raw = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out: list[str] = []
    for ln in raw:
        if out and not out[-1].endswith(".") and not ln.lstrip().startswith("-"):
            out[-1] += " " + ln.strip()
        else:
            out.append(ln.strip())
    return out


def parse_martin1948(text: str) -> list[RenamePair]:
    """``(old, new, line)`` pairs from both entry shapes in Martin's list.

    Former-name lines (``-Old ..., New ...``) give old -> new directly.
    Current-street lines (``Fulton St., 300N ... Frink St., Pleasant St.``)
    give the same pairs reversed: each listed former name -> the current
    street opening the entry. Using both directions is what lifts recovery to
    98% of a manual pass.
    """
    entries: list[RenamePair] = []
    for line in _joined_lines(text):
        if line.startswith("-"):
            body = line[1:]
            toks = canon_tokens(body.split(",", 1)[0] if "," in body else body)
            old: list[str] = []
            for t in toks:
                if _RANGE_TOK.match(t) or t in STOP_WORDS:
                    break
                old.append(t)
            if not old:
                continue
            rest = body.split(",", 1)[1] if "," in body else body
            # sentence-split so trailing prose never glues onto a name run
            entries.extend(
                RenamePair(" ".join(old), " ".join(name), line)
                for clause in re.split(r"\.\s+", rest)
                for name in _name_runs(clause)
            )
        else:
            toks = canon_tokens(line.split(",", 1)[0])
            if not toks or toks[-1] not in SUFFIXES or "," not in line:
                continue
            current = [t for t in toks if t not in DIRS]
            if all(t in SUFFIXES for t in current):
                # An OCR-clipped entry opening with nothing but a suffix
                # ("St., Anna Pl., Bingo St., ...") would pair every former name
                # it lists to the current street "ST" — which normalizes to the
                # EMPTY key, a real bucket in a bounded index (unnamed segments
                # land there). Every listed name would then score against it.
                continue
            entries.extend(
                RenamePair(" ".join(name), " ".join(current), line)
                for clause in re.split(r"\.\s+", line.split(",", 1)[1])
                for name in _name_runs(clause)
                if name != current
            )
    return entries


#: Parsers a city may name. Adding a city means adding a parser here and
#: pointing its config at a source text — no other code changes.
PARSERS: dict[str, Callable[[str], list[RenamePair]]] = {"martin1948": parse_martin1948}


class RenameSourceError(RuntimeError):
    """The configured rename source is unusable (missing text, unknown parser)."""


@dataclass(frozen=True)
class RenameSource:
    """A city's pinned rename list: which parser, which text, how to cite it."""

    parser: str
    text_path: Path
    citation: str
    #: The cached document the text is extracted from, quoted in the error a
    #: missing extraction raises. Never read here — extraction is offline.
    pdf_path: Path | None = None
    _pairs: list[RenamePair] = field(default_factory=list, repr=False, compare=False)

    def pairs(self) -> list[RenamePair]:
        """Every parsed rename pair, parsed once per instance."""
        if self._pairs:
            return self._pairs
        if self.parser not in PARSERS:
            raise RenameSourceError(
                f"unknown rename_source_parser {self.parser!r} (known: {sorted(PARSERS)})"
            )
        if not self.text_path.is_file():
            hint = f" Extract it offline from {self.pdf_path}." if self.pdf_path else ""
            raise RenameSourceError(
                f"rename source text missing: {self.text_path}.{hint} "
                "See the alias-sweep section of docs/OPERATIONS.md."
            )
        self._pairs.extend(PARSERS[self.parser](self.text_path.read_text()))
        return self._pairs

    def candidates(self, key: str) -> list[tuple[str, str]]:
        """``(new phrase, source line)`` for source old-names matching a key.

        The court/twin rules live here because they are conventions of the
        SOURCE-to-key match, not of any one volume: a key carrying a
        court/place suffix must find that suffix on the source line, numbered
        courts demand it exactly (the twin rule — ``42ND CT`` and ``42ND`` are
        different parallel streets), and a suffix-less key must NOT accept a
        court/place source line, which documents the twin.
        """
        want = canon_key(key).split()
        want_names = [t for t in want if t not in SUFFIXES]
        want_sfx = [t for t in want if t in SUFFIXES]
        out = []
        for old, new, raw in self.pairs():
            toks = old.split()
            names = [t for t in toks if t not in SUFFIXES and t not in DIRS]
            sfx = [t for t in toks if t in SUFFIXES]
            if not tokens_match(names, want_names):
                continue
            numbered = bool(want_names and re.match(r"^\d", want_names[0]))
            if want_sfx:
                if numbered:
                    if want_sfx[-1] not in sfx:
                        continue
                elif not (set(want_sfx) & set(sfx) or COURT_WORDS & set(sfx)):
                    continue
            elif sfx and sfx[-1] in {"PL", "CT"}:
                continue
            out.append((new, raw))
        return out


__all__ = [
    "COURT_WORDS",
    "DIRS",
    "PARSERS",
    "SPELLED",
    "STOP_WORDS",
    "SUFFIXES",
    "SUFFIX_CANON",
    "RenamePair",
    "RenameSource",
    "RenameSourceError",
    "SourceRanges",
    "address_ranges",
    "alias_key_for",
    "canon_key",
    "canon_tokens",
    "clean",
    "edit_distance_one",
    "parse_martin1948",
    "snap_value",
    "tokens_match",
]
