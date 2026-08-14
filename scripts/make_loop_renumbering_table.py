"""Merge two renumbering books into one table, collisions abstaining.

City-specific data production; the pipeline consumes only the generic
``renumbering_table`` JSON this writes.

WHY a third table exists: a city can renumber one district on its own date, so
volumes covering it need that register — but their bounds reach into ground
only the city-wide book covers, and neither book alone serves them. Merging
naively is worse than either: on the old numbers both books claim they disagree
essentially always, by hundreds of metres, and a wrong conversion makes a
numeral *votable* in the addresses channel, where enough of them dilute a
yes-majority into a REFUTE or fabricate support.

So this takes the union and SUBTRACTS the collision spans (:func:`collision_cuts`).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autogeoref.names import load_aliases, normalize

ROOT = Path(__file__).resolve().parents[1]
T1909 = ROOT / "configs" / "chicago" / "renumbering-chicago-1909.json"
T1911 = ROOT / "configs" / "chicago" / "renumbering-chicago-1911-loop.json"
OUT = ROOT / "configs" / "chicago" / "renumbering-chicago-loop-merged.json"
ALIASES = ROOT / "configs" / "chicago" / "aliases"

Span = tuple[int, int]


def key(entry: dict[str, Any], aliases: dict[str, str]) -> str:
    """The street key the pipeline actually matches on (directions stripped)."""
    return normalize(str(entry["street"]), aliases)


def convertible(entry: dict[str, Any]) -> bool:
    """Entries ``RenumberingTable.from_json`` will actually load.

    ``old_suffix`` entries (a separate old scheme sharing the street) are
    skipped there, so they can neither convert nor collide -- they ride along
    verbatim.
    """
    return not entry.get("old_suffix")


def old_span(entry: dict[str, Any]) -> Span:
    lo, hi = int(entry["old_range"][0]), int(entry["old_range"][1])
    return (min(lo, hi), max(lo, hi))


def new_at(entry: dict[str, Any], old: int) -> int:
    """``RenumberingTable.convert``'s arithmetic, verbatim."""
    o0, o1 = int(entry["old_range"][0]), int(entry["old_range"][1])
    n0, n1 = int(entry["new_range"][0]), int(entry["new_range"][1])
    old_dir = -1 if o1 < o0 else 1
    new_dir = -1 if n1 < n0 else 1
    return n0 + old_dir * new_dir * (old - o0)


def collision_cuts(
    a: list[dict[str, Any]], b: list[dict[str, Any]], alias_maps: list[dict[str, str]]
) -> dict[int, list[Span]]:
    """Per ENTRY (by ``id``), the old-number spans the other book also claims.

    Keyed by entry identity, not by street name, and unioned over every alias
    map the consuming volumes actually use. Both are load-bearing: the pipeline
    normalizes with ONE alias file per volume, never a city-wide table, and
    aliasing is not symmetric between the books, so a rename can move one
    book's entry onto a different key and the pair silently stops overlapping.
    Cutting against the ENTRY means a pair colliding under ANY volume's map is
    cut for all of them — adding a map can only cut MORE, the safe direction.
    """
    cuts: dict[int, list[Span]] = defaultdict(list)
    for aliases in alias_maps:
        by_key_b: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in b:
            by_key_b[key(e, aliases)].append(e)
        for ea in a:
            alo, ahi = old_span(ea)
            for eb in by_key_b.get(key(ea, aliases), ()):
                blo, bhi = old_span(eb)
                lo, hi = max(alo, blo), min(ahi, bhi)
                if lo <= hi:
                    cuts[id(ea)].append((lo, hi))
                    cuts[id(eb)].append((lo, hi))
    return dict(cuts)


def subtract(span: Span, cuts: list[Span]) -> list[Span]:
    """``span`` minus every cut -- the runs of old numbers that survive."""
    runs = [span]
    for clo, chi in cuts:
        nxt: list[Span] = []
        for lo, hi in runs:
            if chi < lo or clo > hi:  # disjoint
                nxt.append((lo, hi))
                continue
            if lo < clo:
                nxt.append((lo, clo - 1))
            if chi < hi:
                nxt.append((chi + 1, hi))
        runs = nxt
    return runs


def split_entry(entry: dict[str, Any], cuts: list[Span]) -> list[dict[str, Any]]:
    """Re-emit an entry as the sub-entries the collision spans leave behind.

    Ranges are re-paired from the surviving endpoints (``new_at`` reproduces the
    original arithmetic), so each sub-entry converts exactly as the source book
    does -- only over fewer old numbers.
    """
    out: list[dict[str, Any]] = []
    for lo, hi in subtract(old_span(entry), cuts):
        sub = dict(entry)
        sub["old_range"] = [lo, hi]
        sub["new_range"] = [new_at(entry, lo), new_at(entry, hi)]
        if (lo, hi) != old_span(entry):
            flags = list(sub.get("chain_flags") or [])
            flags.append("split_at_book_collision")
            sub["chain_flags"] = flags
        out.append(sub)
    return out


#: The volumes this table is FOR. Their alias files are the only maps the
#: pipeline will normalize with when it looks a numeral up (one per volume),
#: so they are the maps collisions must be detected under.
CONSUMING_VOLUMES = ("sanborn01790_017", "sanborn01790_018")


@dataclass(frozen=True)
class Merge:
    """The merged table, plus the counts :func:`main` reports."""

    entries: list[dict[str, Any]]
    convertible_1909: int
    convertible_1911: int
    collided_entries: int
    collided_streets: list[str]
    #: (street, old number) pairs both books claim -- these convert through neither
    abstained_old_numbers: int
    split: int
    dropped: int


def build_merged_table(t1909: Path, t1911: Path, alias_maps: list[dict[str, str]]) -> Merge:
    """Merge the two books, subtracting every old number they both claim.

    Pure: no file is written. ``tests/test_renumbering_table.py`` calls this to
    assert the committed ``configs/chicago/renumbering-chicago-loop-merged.json`` is still
    what the two source tables produce, so editing a source book cannot leave the
    generated table silently stale.
    """
    raw09 = json.loads(t1909.read_text())
    raw11 = json.loads(t1911.read_text())
    for e in raw09:
        e["source_table"] = "1909-outside-loop"
    for e in raw11:
        e["source_table"] = "1911-loop"

    c09 = [e for e in raw09 if convertible(e)]
    c11 = [e for e in raw11 if convertible(e)]
    cuts_by_entry = collision_cuts(c09, c11, alias_maps)

    merged: list[dict[str, Any]] = []
    dropped = 0
    split = 0
    collided_streets: set[str] = set()
    # district book first: on a street where only it has a rule, its rule
    # applies. Order is not load-bearing any more, since collisions are cut out
    # of BOTH sides, but it reads better in the file.
    for entries in (c11, c09):
        for e in entries:
            cuts = cuts_by_entry.get(id(e), [])
            if not cuts:
                merged.append(e)
                continue
            collided_streets.add(str(e["street"]))
            subs = split_entry(e, cuts)
            if not subs:
                dropped += 1
            else:
                split += 1
            merged.extend(subs)
    # non-convertible entries ride along verbatim: from_json skips them, so they
    # neither convert nor collide
    merged.extend(e for e in raw11 + raw09 if not convertible(e))

    # each collision span is recorded against BOTH entries, so count the
    # (street, old number) pairs once -- via the 1911 side, since every cut is
    # an intersection and so appears on both
    abstained = {
        (str(e["street"]), old)
        for e in c11
        for lo, hi in cuts_by_entry.get(id(e), [])
        for old in range(lo, hi + 1)
    }
    return Merge(
        entries=merged,
        convertible_1909=len(c09),
        convertible_1911=len(c11),
        collided_entries=len(cuts_by_entry),
        collided_streets=sorted(collided_streets),
        abstained_old_numbers=len(abstained),
        split=split,
        dropped=dropped,
    )


def main() -> None:
    alias_maps = [load_aliases(ALIASES / f"aliases-{vid}.json") for vid in CONSUMING_VOLUMES]
    m = build_merged_table(T1909, T1911, alias_maps)
    OUT.write_text(json.dumps(m.entries, indent=1) + "\n")
    print(f"1909 convertible entries: {m.convertible_1909}")
    print(f"1911 convertible entries: {m.convertible_1911}")
    print(f"alias maps used: {', '.join(CONSUMING_VOLUMES)}")
    print(f"entries carrying a collision: {m.collided_entries} on {m.collided_streets}")
    print(f"old numbers abstained (claimed by both books): {m.abstained_old_numbers}")
    print(f"entries split at a collision: {m.split}; entries dropped whole: {m.dropped}")
    print(f"merged entries written: {len(m.entries)} -> {OUT}")


if __name__ == "__main__":
    main()
