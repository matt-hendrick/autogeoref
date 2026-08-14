"""Street-name normalization and historical alias application.

The contract-tested ordering is: uppercase, with apostrophes DELETED outright (spacing them
would break suffix stripping); punctuation to space and parentheticals dropped; a number
rejoined to its own ordinal tail, ahead of the loop that would eat it; an exact full-string
alias lookup, before the court/place guard so renamed PL streets alias correctly; postfix
direction tokens stripped, but only when they trail a suffix token and a non-direction name
token remains; trailing suffix tokens stripped in a loop, keeping numbered PLACE/COURT twins
distinct; a second alias lookup BEFORE leading-direction stripping, skipped for court-like
names; leading direction tokens stripped; and a final alias lookup, again court-skipped.

Neither strip loop may consume the LAST token, so a name of only suffix or direction words
("TERRACE DRIVE", "SOUTH STREET") keys itself and not the empty string; a name that merely
ENDS in a direction keeps it. Aliases are volume-scoped because names vary by edition.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

SUFFIXES = frozenset(
    {"AVE", "AV", "ST", "BLVD", "BL", "PL", "CT", "DR", "RD", "TER", "PKWY", "SQ", "WAY", "LN"}
)
LONG_SUFFIXES = frozenset(
    {
        "BOUL",
        "BOULEVARD",
        "STREET",
        "AVENUE",
        "PLACE",
        "COURT",
        "ROAD",
        "DRIVE",
        "TERRACE",
        "PARKWAY",
    }
)
DIRECTIONS = frozenset({"N", "S", "E", "W", "NORTH", "SOUTH", "EAST", "WEST"})
# postfix quadrants may also arrive as one token ("MILES AV SE"); compounds stay
# out of DIRECTIONS so leading-strip and the ordinal-twin filter are unchanged
POSTFIX_DIRECTIONS = DIRECTIONS | frozenset(
    {"NE", "NW", "SE", "SW", "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST"}
)
COURT_LIKE = frozenset({"CT", "COURT", "PL", "PLACE"})

_NUMERIC_ORDINAL = re.compile(r"^\d+(ST|ND|RD|TH)$")
#: Seen after punctuation is spaced, so every separator reads as a break. The period
#: stays in the class for callers that pass a raw name.
_SPLIT_ORDINAL = re.compile(r"(?<![A-Z0-9])(\d+)[ .]+(ST|ND|RD|TH)(?![A-Z0-9])")
_PARENTHETICAL = re.compile(r"\([^)]*\)")
_NON_ALNUM = re.compile(r"[^A-Z0-9 ]")
_WS = re.compile(r"\s+")

Aliases = Mapping[str, str]


def load_aliases(path: Path | None) -> dict[str, str]:
    """Load a volume-scoped alias table; keys starting with ``_`` are comments."""
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def ordinal_suffix(number: int) -> str:
    """The tail an ordinal takes: 73 -> ``RD``, 11 -> ``TH``, 41 -> ``ST``."""
    if 10 <= number % 100 <= 20:
        return "TH"
    return {1: "ST", 2: "ND", 3: "RD"}.get(number % 10, "TH")


def join_split_ordinal(raw: str) -> str:
    """``'W. 73 RD ST.'`` -> ``'W. 73RD ST.'``, when the tail is that number's own.

    A reader that splits the ordinal off a numbered street leaves a tail the
    suffix loop eats, because ``RD`` and ``ST`` are also Road and Street, and a
    bare number keys nothing. ``'12 ST'`` is left alone: 12 takes ``TH``, so
    that ``ST`` really is a suffix. Expects an UPPERCASED name and is a no-op
    otherwise. Kept separate from :func:`normalize` so an instrument can patch
    it to the identity and run both arms in one process.
    """
    return _SPLIT_ORDINAL.sub(
        lambda m: (
            f"{m.group(1)}{m.group(2)}"
            if ordinal_suffix(int(m.group(1))) == m.group(2)
            else m.group(0)
        ),
        raw,
    )


def normalize(name: str, aliases: Aliases | None = None) -> str:
    """``'W. GARFIELD BOUL.'`` -> ``'GARFIELD'``; ``'31ST PL.'`` -> ``'31ST PL'``.

    See the module docstring for the (contract-tested) ordering.
    """
    aliases = aliases or {}
    # delete apostrophes outright (BL'VD -> BLVD); spacing them breaks suffix stripping
    n = name.upper().replace("'", "").replace("’", "")  # noqa: RUF001 — curly quote intended
    n = n.replace(".", " ").replace(",", " ")
    n = _PARENTHETICAL.sub(" ", n)  # drop parenthetical alternate names
    n = _NON_ALNUM.sub(" ", n)
    n = _WS.sub(" ", n).strip()
    n = join_split_ordinal(n)  # before the strip loop can eat the ordinal tail
    # exact full-string alias (suffix included, e.g. "MACALISTER PL": "LEXINGTON")
    # runs before the court/place guard: some PL streets were renamed outright
    if n in aliases:
        return aliases[n]
    toks = n.split(" ")
    # postfix directionals ("MILES AV S E"): strip trailing direction tokens
    # only when a suffix token precedes them and a NON-DIRECTION name token
    # remains, so names that merely end in a direction word ("SOUTH WEST",
    # "AVENUE E") stay intact and direction-word names ("S AVENUE E", the
    # lettered avenues) are not collapsed to nothing by the later leading strip
    i = len(toks)
    while i > 1 and toks[i - 1] in POSTFIX_DIRECTIONS:
        i -= 1
    if (
        i < len(toks)
        and i > 1
        and (toks[i - 1] in SUFFIXES or toks[i - 1] in LONG_SUFFIXES)
        and any(t not in DIRECTIONS for t in toks[: i - 1])
    ):
        toks = toks[:i]
    court_like = False  # aliases target the avenue/boulevard, not same-named courts/places
    # len > 1 keeps the last token whatever it is: a name made only of suffix words
    # ("TERRACE DRIVE") would otherwise strip to nothing. The empty key is not inert
    # on either side — the centerline index collects every such street under it, so
    # the streets merge into one chimeric geometry and a lookup against it returns
    # intersections belonging to none of them
    while len(toks) > 1 and (toks[-1] in SUFFIXES or toks[-1] in LONG_SUFFIXES):
        if toks[-1] in COURT_LIKE:
            court_like = True
            # numbered place/court twins stay distinct: '31ST PL' != '31ST'
            rest = [t for t in toks[:-1] if t not in DIRECTIONS]
            if len(rest) == 1 and _NUMERIC_ORDINAL.match(rest[0]):
                suffix = "PL" if toks[-1] in {"PL", "PLACE"} else "CT"
                return f"{rest[0]} {suffix}"
        toks = toks[:-1]
    pre = " ".join(toks)
    if pre in aliases and not court_like:
        return aliases[pre]
    # ...and the same guard here, which the module docstring already promises but
    # only the postfix branch enforced: "SOUTH STREET" must not reduce to nothing
    while len(toks) > 1 and toks[0] in DIRECTIONS:
        toks = toks[1:]
    out = " ".join(toks)
    if court_like:
        return out
    return aliases.get(out, out)
