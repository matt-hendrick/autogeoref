"""Comments must not cite a roadmap item number."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CITATION = re.compile(r"ROADMAP" + r" \[\d+\]")
SCANNED_SUFFIXES = {".py", ".html", ".toml"}


def test_roadmap_citations_resolve() -> None:
    citations: list[str] = []
    for tree in ("src", "tests"):
        for path in sorted((ROOT / tree).rglob("*")):
            if path.suffix not in SCANNED_SUFFIXES or not path.is_file():
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                if CITATION.search(line):
                    rel = path.relative_to(ROOT)
                    citations.append(f"{rel}:{lineno}: {line.strip()}")
    assert not citations, (
        "roadmap citations are forbidden; state the reason in the comment instead:\n"
        + "\n".join(citations)
    )
