"""Targeted maintenance of ``FIXTURE-SHA256SUMS``, the fixture integrity manifest.

The manifest's grouping order is owned by ``scripts/make_fixture_manifest.py``
(the whole-tree regenerator); this module refreshes or inserts only the lines
for files a command just wrote, preserving that order.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

from .paths import atomic_write_text

logger = logging.getLogger(__name__)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_rel(line: str) -> str:
    parts = line.split(None, 1)
    return parts[1] if len(parts) == 2 else ""


def _insert_at(lines: list[str], rel: str) -> int:
    """Where a new manifest line belongs, keeping the file's existing grouping.

    An entry goes at its sorted position *within the run of lines sharing its
    top-level directory*, and at the end when there is no such run. Sorting the
    whole file instead would churn lines this command never touched, and would
    move any group that sits later than it sorts.
    """
    tree = rel.split("/", 1)[0]
    block = [i for i, line in enumerate(lines) if _manifest_rel(line).split("/", 1)[0] == tree]
    if not block:
        return len(lines)
    for i in block:
        if _manifest_rel(lines[i]) > rel:
            return i
    return block[-1] + 1


def update_sums(manifest: Path, root: Path, written: Sequence[Path]) -> list[str]:
    """Refresh only the manifest lines for the files the caller wrote.

    Deliberately not a whole-tree regeneration (``scripts/make_fixture_manifest.py``
    is that, and running it is a human's decision): a refresh that re-hashed
    every fixture would quietly absorb unrelated on-disk drift into its own
    commit, and re-ordering would bury four real changes in thousands of moved
    lines. Returns the manifest paths it changed or added.
    """
    if not manifest.is_file():
        return []
    lines = [line for line in manifest.read_text().splitlines() if line.strip()]
    changed: list[str] = []
    for path in written:
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            # A configured output directory outside the repo has no manifest
            # line to refresh. This runs AFTER the caller's files are written,
            # so it warns and carries on rather than raising over bookkeeping.
            logger.warning("%s: outside %s, no %s line to refresh", path, root, manifest.name)
            continue
        entry = f"{sha256_of(path)}  {rel}"
        existing = next((i for i, line in enumerate(lines) if _manifest_rel(line) == rel), None)
        if existing is None:
            lines.insert(_insert_at(lines, rel), entry)
            changed.append(rel)
        elif lines[existing] != entry:
            lines[existing] = entry
            changed.append(rel)
    if changed:
        atomic_write_text(manifest, "\n".join(lines) + "\n")
    return changed
