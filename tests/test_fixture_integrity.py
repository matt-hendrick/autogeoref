"""The frozen fixtures are the spec — verify they are still the same bytes.

`fixtures/` is gitignored (large, machine-local), so nothing in version
control would notice corruption or an accidental write to the tree the
golden tests treat as ground truth. The
tracked `FIXTURE-SHA256SUMS` manifest closes that hole: every listed file
must exist and hash-match, or this test FAILS (a missing fixture tree still
skips, loudly, via conftest).

Regenerate the manifest ONLY for legitimate fixture changes:
`python scripts/make_fixture_manifest.py` (see FIXTURES.md).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "FIXTURE-SHA256SUMS"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.mark.golden
def test_fixture_tree_matches_manifest() -> None:
    assert MANIFEST.is_file(), (
        "FIXTURE-SHA256SUMS is missing — regenerate with "
        "scripts/make_fixture_manifest.py and commit it"
    )
    entries = [line.split(None, 1) for line in MANIFEST.read_text().splitlines() if line.strip()]
    if not any((ROOT / rel).is_file() for _, rel in entries if rel.startswith("fixtures/")):
        pytest.skip("FIXTURE TREE NOT PRESENT — see FIXTURES.md for provenance")
    missing: list[str] = []
    mismatched: list[str] = []
    n = 0
    for digest, rel in entries:
        p = ROOT / rel
        n += 1
        if not p.is_file():
            missing.append(rel)
        elif _sha256(p) != digest:
            mismatched.append(rel)
    assert n > 0, "manifest is empty"
    assert not missing, f"{len(missing)} manifest fixtures missing, e.g. {missing[:5]}"
    assert not mismatched, (
        f"{len(mismatched)} fixtures changed since the manifest was written "
        f"(the fixtures are read-only spec!), e.g. {mismatched[:5]} — if the "
        f"change is legitimate, regenerate via scripts/make_fixture_manifest.py"
    )


def test_manifest_pins_nothing_git_already_tracks() -> None:
    """A tracked entry would turn the skip above into a failure.

    The guard reads "no listed file on disk" as "no fixture tree here". A
    tracked file is on disk in every clone, so pinning one makes a bare clone
    claim the tree is present and then fail on thousands of missing entries.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--", "fixtures"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git cannot list files here")
    tracked = {p for p in out.split("\0") if p}
    pinned = {line.split(None, 1)[1] for line in MANIFEST.read_text().splitlines() if line.strip()}
    overlap = sorted(tracked & pinned)
    assert not overlap, (
        f"{len(overlap)} tracked file(s) are pinned in the manifest, which breaks the "
        f"tree-absent skip: {overlap[:5]} — scripts/make_fixture_manifest.py skips these"
    )
