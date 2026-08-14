"""Tests for the fixture-manifest checksum plumbing (fixture_sums)."""

from __future__ import annotations

from pathlib import Path

from autogeoref.fixture_sums import sha256_of, update_sums


def test_update_sums_touches_only_the_lines_it_wrote(tmp_path: Path) -> None:
    """A refresh must be readable as a diff: four changed lines, nothing moved."""
    root = tmp_path
    (root / "fixtures").mkdir()
    written = root / "fixtures" / "aliases-x.json"
    written.write_text('{"A": "B"}')
    manifest = root / "FIXTURE-SHA256SUMS"
    manifest.write_text(
        f"{'0' * 64}  fixtures/a.json\n"
        f"{'1' * 64}  fixtures/z.json\n"
        f"{'2' * 64}  other-tree/thing.json\n"
    )

    changed = update_sums(manifest, root, [written])
    assert changed == ["fixtures/aliases-x.json"]
    lines = manifest.read_text().splitlines()
    assert lines == [
        f"{'0' * 64}  fixtures/a.json",
        f"{sha256_of(written)}  fixtures/aliases-x.json",
        f"{'1' * 64}  fixtures/z.json",
        f"{'2' * 64}  other-tree/thing.json",
    ]

    # idempotent: an unchanged file changes no line
    assert update_sums(manifest, root, [written]) == []

    # a rewritten file updates its digest in place, moving nothing
    written.write_text('{"A": "C"}')
    assert update_sums(manifest, root, [written]) == ["fixtures/aliases-x.json"]
    assert [line.split(None, 1)[1] for line in manifest.read_text().splitlines()] == [
        "fixtures/a.json",
        "fixtures/aliases-x.json",
        "fixtures/z.json",
        "other-tree/thing.json",
    ]


def test_a_new_entry_joins_its_own_group_not_the_global_sort(tmp_path: Path) -> None:
    """The insert follows the file's grouping, not a whole-file sort.

    ``a-tree/`` sorts before ``fixtures/`` but sits after it in the manifest, so
    a globally sorted insert would put its new entry at the very top, away from
    its sibling. Nothing else in this module distinguishes the two rules.
    """
    root = tmp_path
    (root / "a-tree").mkdir()
    written = root / "a-tree" / "aaa.json"
    written.write_text("{}")
    manifest = root / "FIXTURE-SHA256SUMS"
    manifest.write_text(
        f"{'0' * 64}  fixtures/a.json\n{'1' * 64}  fixtures/z.json\n{'2' * 64}  a-tree/thing.json\n"
    )

    assert update_sums(manifest, root, [written]) == ["a-tree/aaa.json"]
    assert [line.split(None, 1)[1] for line in manifest.read_text().splitlines()] == [
        "fixtures/a.json",
        "fixtures/z.json",
        "a-tree/aaa.json",
        "a-tree/thing.json",
    ]


def test_update_sums_is_a_no_op_without_a_manifest(tmp_path: Path) -> None:
    written = tmp_path / "f.json"
    written.write_text("{}")
    assert update_sums(tmp_path / "absent", tmp_path, [written]) == []


def test_update_sums_warns_rather_than_raising_on_a_path_outside_the_root(
    tmp_path: Path,
) -> None:
    """It runs AFTER the caller's files are written; a bookkeeping miss must
    not raise."""
    root = tmp_path / "repo"
    root.mkdir()
    manifest = root / "FIXTURE-SHA256SUMS"
    manifest.write_text(f"{'0' * 64}  fixtures/a.json\n")
    outside = tmp_path / "elsewhere.json"
    outside.write_text("{}")
    assert update_sums(manifest, root, [outside]) == []
    assert manifest.read_text() == f"{'0' * 64}  fixtures/a.json\n"
