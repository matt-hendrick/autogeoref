"""Structural validation of a volume alias table against its own index.

An alias table is read by ``normalize`` at match time and by ``centerline_key``
when the index is built, and both directions can go wrong quietly. The rules a
table must satisfy — values are exact index keys, no key shadows an in-bounds
street, no chains, no centerline joins another street's bucket — are stated in
`docs/INTERNALS.md` and implemented once here, because two consumers need them:
the automated sweep, which must ABORT a volume rather than write a failing
table, and the corpus regression instrument.

Every rule asks what an entry DOES, so one that provably does nothing is exempt
from all of them; the caller decides whether dead weight is worth reporting.

Zero model calls, zero network.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..centerlines import CenterlineIndex, centerline_key
from ..names import _NUMERIC_ORDINAL, normalize


def alias_files(aliases_dir: Path) -> list[Path]:
    """Every alias table in a city's alias directory, sorted.

    Discovery, not a hardcoded list: a sweep that writes a new volume's table
    must not also have to be edited to validate it.
    """
    return sorted(aliases_dir.glob("aliases-*.json"))


def volume_of(path: Path) -> str:
    """``aliases-<volume>.json`` -> ``<volume>``."""
    return path.stem.removeprefix("aliases-")


def is_inert(key: str, aliases: Mapping[str, str]) -> bool:
    """True when an entry cannot change any outcome, on either side.

    Decided by removing the entry and asking the normalizer, because the answer
    turns on the WHOLE table: a lookup that misses carries on through two more,
    and any of the strings it tries next may be a key as well.
    """
    rest = {k: v for k, v in aliases.items() if k != key}
    return normalize(key, rest) == aliases[key]


def twin_hold(key: str, index: CenterlineIndex) -> list[str]:
    """Why the auto-writer must not write this key by itself, if it must not.

    A numbered street and the place or court beside it are two streets, and a
    documented rename of the first says nothing about the second. Nothing
    breaks if the key is written — ``centerline_key`` keeps the twin's own key,
    and a read of the twin cannot reach a bare numbered key — so this is a
    change somebody should decide, not a rule the table violates.
    """
    twins = [f"{key} {suffix}" for suffix in ("PL", "CT") if f"{key} {suffix}" in index.by_name]
    if not twins or not _NUMERIC_ORDINAL.match(key):
        return []
    return [
        f"bare numbered key, and this volume also carries {', '.join(twins)}:"
        f" a rename of the street is not a rename of its place or court twin"
    ]


def redirects(aliases: Mapping[str, str], index: CenterlineIndex) -> list[str]:
    """Entries that take a spelling away from a street that is still here.

    NOT failures, and the distinction is the whole point: this is the shape a
    one-directional rename is supposed to have, and it is also the shape a
    mistake has. Nothing structural tells them apart — the entry is right
    exactly when the old qualified name meant a different street from the
    surviving bare one, which is a question about the city. So it is reported
    for a reader and never gates anything.
    """
    return [
        f"{key!r} -> {value!r}: takes {stripped!r} reads off a street still in bounds"
        for key, value in sorted(aliases.items())
        if not is_inert(key, aliases)
        and (stripped := normalize(key)) != key
        and stripped in index.by_name
        and value != stripped
    ]


def _origins(
    aliases: Mapping[str, str],
    index: CenterlineIndex,
    name_property: str,
    type_property: str,
) -> dict[str, set[str]]:
    """Each key the table produces -> the alias-free keys whose segments land in it."""
    origins: dict[str, set[str]] = {}
    for bare_key, segments in index.by_name.items():
        for segment in segments:
            aliased = centerline_key(segment["props"], aliases, name_property, type_property)
            origins.setdefault(str(aliased), set()).add(bare_key)
    return origins


def validate_table(
    aliases: Mapping[str, str],
    index: CenterlineIndex,
    name_property: str,
    type_property: str,
) -> list[str]:
    """Every rule violation in one table, as reader-facing lines.

    ``index`` must be the volume's **alias-free** bounded index: it is the
    ground the aliases have to land on, and building it with the table applied
    would hide exactly the shadowing and merging this checks for.
    """
    origins = _origins(aliases, index, name_property, type_property)
    failures: list[str] = []
    for key, value in aliases.items():
        if is_inert(key, aliases):
            continue
        if value not in index.by_name and value not in origins:
            failures.append(f"{key!r} -> {value!r}: VALUE NOT AN INDEX KEY")
        if key in index.by_name:
            failures.append(f"{key!r}: KEY SHADOWS AN IN-BOUNDS STREET (twin)")
        if aliases.get(value, value) != value:
            failures.append(f"{key!r} -> {value!r}: VALUE IS ALSO A KEY (chain)")
    failures.extend(_merges(origins))
    return failures


def _merges(origins: dict[str, set[str]]) -> list[str]:
    """The re-keyings that pool two different streets into one bucket.

    The harm is a chimeric geometry: a lookup against the merged key returns
    intersections belonging to neither street. That happens exactly when one
    key is fed by two, so a bucket fed by one is left alone however it got
    there — a whole key renamed, or a shared bucket split back apart, pools
    nothing. What a rename does leave behind is a vacated key that reads can
    still produce, and no rule here sees the read side.
    """
    failures: list[str] = []
    for aliased, sources in sorted(origins.items()):
        if len(sources) == 1:
            continue
        failures.extend(
            f"centerline {bare_key!r} RE-KEYED to {aliased!r}, "
            f"joining {sorted(sources - {bare_key})}"
            for bare_key in sorted(source for source in sources if source != aliased)
        )
    return failures


__all__ = [
    "alias_files",
    "is_inert",
    "redirects",
    "twin_hold",
    "validate_table",
    "volume_of",
]
