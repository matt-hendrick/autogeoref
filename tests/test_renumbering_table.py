"""Canaries for the shipped renumbering table."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType

from autogeoref.addresses import RenumberingEntry, RenumberingTable
from autogeoref.names import load_aliases, normalize

ROOT = Path(__file__).parent.parent
TABLE_PATH = ROOT / "configs" / "chicago" / "renumbering-chicago-1909.json"


def _load_script(name: str) -> ModuleType:
    """Import a `scripts/` module by path — `scripts/` is not an installed package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # registered before exec: @dataclass resolves its class's module out of
    # sys.modules, and raises on a module that is not there yet
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _table() -> RenumberingTable:
    return RenumberingTable.from_json(TABLE_PATH)


def test_full_table_loads() -> None:
    # A substantial drop indicates lost table coverage.
    table = _table()
    assert len(table.entries) >= 6500


def test_canary_hermitage() -> None:
    assert _table().convert("N. Hermitage Av.", 2464) == 4303


def test_canary_paulina() -> None:
    assert _table().convert("North Paulina St.", 2466) == 4303


def test_canary_ashland() -> None:
    assert _table().convert("N ASHLAND AV", 2466) == 4309


def test_canary_east_ravenswood_own_numbering() -> None:
    assert _table().convert("RAVENSWOOD", 1108) == 4303


def test_convert_abstains_when_the_table_contradicts_itself() -> None:
    """Contradictory conversions must abstain."""
    table = _table()
    assert table.covering_answers("ALEXANDER", 26) == [936, 228]
    assert table.convert("ALEXANDER", 26) is None


def test_convert_still_answers_when_the_entries_agree() -> None:
    """Compatible overlapping entries may still convert."""
    agreeing = RenumberingTable(
        entries=(
            RenumberingEntry(street="MAIN", old_range=(100, 200), new_range=(1000, 1100)),
            RenumberingEntry(street="MAIN", old_range=(100, 200), new_range=(1020, 1120)),
        )
    )
    assert agreeing.convert("MAIN", 150) == 1050  # first match; the other says 1070

    contradicting = RenumberingTable(
        entries=(
            RenumberingEntry(street="MAIN", old_range=(100, 200), new_range=(1000, 1100)),
            RenumberingEntry(street="MAIN", old_range=(100, 200), new_range=(4000, 4100)),
        )
    )
    assert contradicting.convert("MAIN", 150) is None


LOOP_TABLE_PATH = ROOT / "configs" / "chicago" / "renumbering-chicago-1911-loop.json"


def _loop_table() -> RenumberingTable:
    return RenumberingTable.from_json(LOOP_TABLE_PATH)


def test_loop_table_loads() -> None:
    assert len(_loop_table().entries) >= 140


def test_loop_canaries_clark_within_entrance_tolerance() -> None:
    # Entrance-level source rows require interpolation tolerance.
    table = _loop_table()
    for old, want in [(291, 411), (299, 431), (262, 314), (272, 328)]:
        got = table.convert("S CLARK ST", old)
        assert got is not None, f"no conversion for S CLARK {old}"
        assert abs(got - want) <= 12, f"S CLARK {old} -> {got}, expected ~{want}"


def test_loop_table_has_no_outside_loop_streets() -> None:
    # The district table must not include citywide entries.
    table = _loop_table()
    for e in table.entries:
        assert "HERMITAGE" not in e.street and "PAULINA" not in e.street


MERGED_TABLE_PATH = ROOT / "configs" / "chicago" / "renumbering-chicago-loop-merged.json"


def _merged_table() -> RenumberingTable:
    return RenumberingTable.from_json(MERGED_TABLE_PATH)


def test_merged_table_keeps_both_books() -> None:
    # The merged table retains conversions from both sources.
    table = _merged_table()
    assert table.convert("N. HERMITAGE AV.", 2464) == 4303  # 1909 book
    got = table.convert("S CLARK ST", 291)  # 1911 register (entrance-level +-)
    assert got is not None and abs(got - 411) <= 12


#: Consuming volumes define the alias key spaces to validate.
LOOP_VOLUMES = ("sanborn01790_017", "sanborn01790_018")
ALIASES_DIR = ROOT / "configs" / "chicago" / "aliases"


def _volume_aliases(vid: str) -> dict[str, str]:
    return load_aliases(ALIASES_DIR / f"aliases-{vid}.json")


def _by_key(table: RenumberingTable, aliases: dict[str, str]) -> dict[str, RenumberingTable]:
    """One single-street table per key, entry ORDER preserved.

    ``convert`` is first-match over entries and only ever matches its own
    street, so restricting it to one street's entries is exactly the same
    lookup — just without rescanning 7,000 irrelevant entries per numeral (the
    difference between a 7-minute test and a 3-second one).
    """
    groups: dict[str, list[RenumberingEntry]] = defaultdict(list)
    for e in table.entries:
        groups[normalize(e.street, aliases)].append(e)
    return {k: RenumberingTable(entries=tuple(v)) for k, v in groups.items()}


def test_merged_table_abstains_on_every_number_both_books_claim() -> None:
    """All conversions claimed by both source books must abstain."""
    merged, t09, t11 = _merged_table(), _table(), _loop_table()
    for vid in LOOP_VOLUMES:
        aliases = _volume_aliases(vid)
        g09, g11, gm = (_by_key(t, aliases) for t in (t09, t11, merged))
        checked = 0
        for key in set(g09) & set(g11):  # only shared streets can collide
            entries = [*g09[key].entries, *g11[key].entries]
            lo = min(min(e.old_range) for e in entries)
            hi = max(max(e.old_range) for e in entries)
            for old in range(lo, hi + 1):
                a = g09[key].convert(key, old, aliases)
                b = g11[key].convert(key, old, aliases)
                if a is None or b is None:
                    continue  # at most one book claims it: not a collision
                got = gm[key].convert(key, old, aliases) if key in gm else None
                assert got is None, (
                    f"{vid}: {key} {old} -> {got}, but BOTH books claim it "
                    f"(1909={a}, 1911={b}) — must abstain"
                )
                checked += 1
        assert checked > 300, f"{vid}: only {checked} collisions found — detection broke"


def test_merged_table_is_regenerable_from_its_two_sources() -> None:
    """The committed merged table matches regeneration from its sources."""
    script = _load_script("make_loop_renumbering_table")
    alias_maps = [_volume_aliases(vid) for vid in LOOP_VOLUMES]
    assert tuple(script.CONSUMING_VOLUMES) == LOOP_VOLUMES  # same key spaces
    regenerated = script.build_merged_table(TABLE_PATH, LOOP_TABLE_PATH, alias_maps).entries
    committed = json.loads(MERGED_TABLE_PATH.read_text())
    # A mismatch indicates a stale committed table.
    assert regenerated == committed


def test_merged_table_jackson_canary() -> None:
    # This collision is visible only through volume-specific aliases.
    aliases = _volume_aliases("sanborn01790_017")
    for old in range(37, 45):
        assert _merged_table().convert("JACKSON PL", old, aliases) is None


def test_merged_table_conversions_are_never_invented() -> None:
    """Every merged entry reproduces its source book across its WHOLE range.

    The collision subtraction re-pairs the surviving range endpoints, so an
    off-by-one would silently shift a block's conversions. Endpoints AND
    interior are checked (an earlier version tested only ``min(old_range)``,
    which a shifted upper endpoint would have sailed through), on the split
    entries specifically — they are the ones the arithmetic touched.
    """
    merged, t09, t11 = _merged_table(), _table(), _loop_table()
    split_ranges = {
        (str(e["street"]), int(e["old_range"][0]), int(e["old_range"][1]))
        for e in json.loads(MERGED_TABLE_PATH.read_text())
        if "split_at_book_collision" in (e.get("chain_flags") or [])
    }
    assert split_ranges, "no split entries — the collision subtraction did not run"
    for street, o0, o1 in split_ranges:
        lo, hi = min(o0, o1), max(o0, o1)
        mid = (lo + hi) // 2
        for old in {lo, mid, hi}:
            got = merged.convert(street, old)
            assert got is not None
            assert got in {t09.convert(street, old), t11.convert(street, old)}, (
                f"{street} {old} -> {got} matches neither source book"
            )


def test_canary_north_wood_absent_above_3286() -> None:
    # This street must not cover the unrelated high-number range.
    table = _table()
    for e in table.entries:
        if e.street.replace("N ", "").replace("S ", "").strip() == "WOOD":
            assert max(e.new_range) < 3300
