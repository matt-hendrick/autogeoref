"""Candidate normalization rules for bare numerals inside street names.

The library the two numeral instruments share: the rules themselves, the
``normalize``-shaped variant they build, and the patch that swaps that variant
into every module which bound ``normalize`` at import time.

Nothing here is shipped behaviour. A rule is a rewrite of the RAW name applied
ahead of :func:`autogeoref.names.normalize`, so the shipped function still owns
aliasing, suffix stripping, the direction rules and the numbered PLACE/COURT
twin guard. A variant that post-processed the finished key could not tell
``'73 PL'`` from ``'73'`` — normalize has collapsed both by then.

Two rules, and they are separate defects:

``split``
    ``'W. 73 RD ST.'`` -> ``'W. 73RD ST.'``. The reader split the ordinal off a
    numbered street and the tail is also a suffix abbreviation, so the strip
    loop eats it and leaves a bare number. Joins ONLY when the tail is the
    ordinal that number actually takes, so ``'12 ST'`` stays a Street.

``interp``
    ``'BALMORAL 45 AV.'`` -> ``'BALMORAL AV.'``. A house number printed beside
    the label was read into the name. Drops a bare numeral only when a real
    non-direction word precedes it and an alphabetic token follows, so a
    trailing numbered alley (``'PLANKED ALLEY 6'``) and a split ordinal are
    both left alone.

``interp-suffix``
    the same rule with the right flank narrowed to a suffix token the strip
    loop knows, which is what keeps ``'PRIVATE 4 ALLEY'`` off the generic
    ``PRIVATE ALLEY`` key.

``defer`` is a modifier, not a rule: skip the rewrite when a volume alias
already claims the whole cleaned name. It changes WHERE the rule runs relative
to normalize's first alias lookup.
"""

from __future__ import annotations

import contextlib
import pkgutil
import re
from collections.abc import Iterator, Mapping
from importlib import import_module
from typing import Any

import autogeoref
from autogeoref import names

RULES = ("split", "interp", "interp-suffix")

#: Both rules see the raw name after apostrophe deletion only, so a decimal
#: point or a period between the number and its tail still reads as a break.
_SPLIT = re.compile(r"(?<![A-Z0-9])(\d+)[ .]+(ST|ND|RD|TH)(?![A-Z0-9])")
_ORDINAL_TAILS = frozenset({"ST", "ND", "RD", "TH"})
_WORD = re.compile(r"^[A-Z]+$")
#: Directions are excluded as the left flank: ``'W. 73 RD ST.'`` is a split
#: ordinal, not a house number, and the two rules must not both fire on it.
_LEFT_FLANK_MIN = 3

_PARENTHETICAL = re.compile(r"\([^)]*\)")
_NON_ALNUM = re.compile(r"[^A-Z0-9 ]")
_WS = re.compile(r"\s+")


def ordinal_of(number: str) -> str:
    """``'73'`` -> ``'73RD'``, by the English rule the centerline names use."""
    n = int(number)
    tail = "TH" if 10 <= n % 100 <= 20 else {1: "ST", 2: "ND", 3: "RD"}.get(n % 10, "TH")
    return f"{number}{tail}"


def join_split_ordinal(raw: str) -> str:
    """Join a number to a following ordinal tail, when that tail is its own.

    ``'W. 73 RD ST.'`` -> ``'W. 73RD ST.'``; ``'12 ST'`` is untouched because
    12 takes ``TH``, so that ``ST`` is a Street suffix the strip loop owns.
    """
    return _SPLIT.sub(
        lambda m: (
            f"{m.group(1)}{m.group(2)}"
            if ordinal_of(m.group(1)).endswith(m.group(2))
            else m.group(0)
        ),
        raw,
    )


def clean_tokens(raw: str) -> list[str]:
    """The token list normalize would build, up to its first alias lookup."""
    n = raw.replace(".", " ").replace(",", " ")
    n = _PARENTHETICAL.sub(" ", n)
    n = _NON_ALNUM.sub(" ", n)
    return _WS.sub(" ", n).strip().split(" ")


def drop_interpolated_numeral(raw: str, suffix_right_flank: bool = False) -> str:
    """Drop a bare numeral a real word precedes and an alphabetic token follows.

    ``'BALMORAL 45 AV.'`` -> ``'BALMORAL AV'``. With ``suffix_right_flank`` the
    follower must be a suffix token the strip loop knows. Returns ``raw``
    unchanged when no numeral qualifies, and never returns an empty string.
    """
    toks = clean_tokens(raw)
    out: list[str] = []
    changed = False
    for i, tok in enumerate(toks):
        prev = toks[i - 1] if i else ""
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        right_ok = bool(_WORD.match(nxt)) and nxt not in _ORDINAL_TAILS
        if suffix_right_flank:
            right_ok = nxt in names.SUFFIXES or nxt in names.LONG_SUFFIXES
        drop = (
            tok.isdigit()
            and len(prev) >= _LEFT_FLANK_MIN
            and bool(_WORD.match(prev))
            and prev not in names.DIRECTIONS
            and right_ok
        )
        if drop:
            changed = True
            continue
        out.append(tok)
    if not changed:
        return raw
    joined = " ".join(out)
    return joined or raw


def _cleaned(raw: str) -> str:
    """The exact string normalize tests against its first alias lookup."""
    return " ".join(clean_tokens(raw))


def make_variant(rules: frozenset[str], defer_to_alias: bool = False) -> Any:
    """A ``normalize``-shaped callable applying ``rules`` ahead of the shipped one.

    With ``defer_to_alias`` the rewrite is skipped whenever the cleaned full
    string is already an alias key, which is where normalize's first lookup
    would have caught it.
    """
    unknown = rules - set(RULES)
    if unknown:
        raise ValueError(f"unknown rule(s): {', '.join(sorted(unknown))}")

    def variant(name: str, aliases: Mapping[str, str] | None = None) -> str:
        # apostrophes first, exactly as the shipped normalize does them:
        # otherwise BOULEV'D reads as BOULEV plus a stray D
        raw = str(name).upper().replace("'", "").replace("’", "")  # noqa: RUF001
        if defer_to_alias and aliases and _cleaned(raw) in aliases:
            return names.normalize(raw, aliases)
        if "split" in rules:
            raw = join_split_ordinal(raw)
        if "interp" in rules or "interp-suffix" in rules:
            raw = drop_interpolated_numeral(raw, "interp" not in rules)
        return names.normalize(raw, aliases)

    return variant


def patch_targets() -> tuple[Any, ...]:
    """Every package module that bound ``normalize`` at import time.

    Discovered rather than listed: a module added later would otherwise keep
    the shipped rule while the rest of the arm ran the variant, which reads as
    a measurement and is not one. ``names`` itself is excluded — the variant
    calls the shipped function, so patching its home is infinite recursion.
    """
    found = []
    for info in pkgutil.walk_packages(autogeoref.__path__, f"{autogeoref.__name__}."):
        if info.name == names.__name__:
            continue
        try:
            module = import_module(info.name)
        except Exception:
            continue
        if getattr(module, "normalize", None) is names.normalize:
            found.append(module)
    return tuple(found)


@contextlib.contextmanager
def patched(variant: Any, targets: tuple[Any, ...]) -> Iterator[None]:
    """Swap ``normalize`` in every module that bound it, for the block."""
    saved = [(m, m.normalize) for m in targets]
    for module, _ in saved:
        module.normalize = variant
    try:
        yield
    finally:
        for module, original in saved:
            module.normalize = original
