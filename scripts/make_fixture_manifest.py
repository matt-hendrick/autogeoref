"""Regenerate FIXTURE-SHA256SUMS — the integrity manifest for the frozen spec.

The fixture tree is gitignored (too large / machine-local), yet it IS the
spec the golden tests validate against. This manifest is the tracked
stand-in: `tests/test_fixture_integrity.py`
fails (not skips) when a listed file is missing or its hash changed, so
silent fixture corruption or an accidental write can no longer hide behind
a green suite.

Run this ONLY when the fixture tree legitimately changes (e.g. a new frozen
volume is copied in per FIXTURES.md), then commit the updated manifest:

    .venv/bin/python scripts/make_fixture_manifest.py
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "FIXTURE-SHA256SUMS"


def tracked_paths() -> set[str]:
    """Repo-relative paths git already tracks under `fixtures/`.

    They must NOT be pinned. `tests/test_fixture_integrity.py` decides the
    tree is absent by asking whether ANY listed `fixtures/` file is on disk,
    so one tracked entry — present in every clone — makes that guard say the
    tree is here and the test fail on thousands of missing files instead of
    skipping. Git already pins these; the manifest exists for what it cannot.
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
        print("cannot ask git what is tracked — run this from a checkout", file=sys.stderr)
        raise SystemExit(1) from None
    return {p for p in out.split("\0") if p}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    fixtures = ROOT / "fixtures"
    if not fixtures.is_dir():
        print("no fixtures/ tree here — refusing to write an empty manifest", file=sys.stderr)
        return 1
    files = sorted(p for p in fixtures.rglob("*") if p.is_file())
    tracked = tracked_paths()
    lines = []
    skipped = 0
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        if rel in tracked:
            skipped += 1
            continue
        lines.append(f"{sha256_of(p)}  {rel}")
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"wrote {MANIFEST.name}: {len(lines)} files ({skipped} tracked, left to git)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
