"""Every doc/script citation in the tree must point at a file that exists.

Companion to ``test_roadmap_citations.py``: that test bans citing the live
roadmap; this one keeps citations of planning docs and scripts from
dangling after a merge, rename, or delete. The policy it enforces lives in
the planning-docs README: a doc may be merged, renamed, or deleted only if
every inbound citation is updated in the same commit.

A line may quote a deliberately broken path (for example a record describing a
deleted script) by carrying an ``<!-- no-cite -->`` marker on the same line.
A fenced code block may be exempted wholesale by putting ``no-cite`` on its
opening fence (for verbatim quotes of retired file content). The escapes are
per-line and per-block, never per-file, so a real break elsewhere in the same
file still fails.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Trees that carry citations worth checking. Data trees (fixtures, exports)
# and generated artifacts are excluded.
SCANNED_TREES = ("src", "tests", "scripts", "configs", "viewer", "docs", "planning_docs")
SCANNED_SUFFIXES = {".py", ".md", ".html", ".toml", ".txt", ".json"}
NO_CITE = "<!-- no-cite -->"

CITATION = re.compile(
    r"(?<![\w/-])("
    r"(?:planning_docs/)?(?:open|record|aspirational)/[A-Za-z0-9._-]+\.md"
    r"|planning_docs/[A-Za-z0-9._-]+\.md"
    r"|scripts/(?:experiments/(?:archive/)?)?[A-Za-z0-9._-]+\.(?:py|json)"
    r")"
)

# Inside planning_docs, the dominant citation style is a bare backticked filename. Those must
# resolve too, or a rename silently strands them; the convention for a *historical* mention of a
# doc that no longer exists (a provenance note, a fold banner) is to drop the `.md` suffix — a
# name without `.md` is a label, not a link. Only names in the planning-doc convention (CAPS-
# WITH-HYPHENS) are enforced, so generated artifacts (`report.md`) and untracked scratch files
# (`todo.md`) stay out of scope.
BARE_CITATION = re.compile(r"`([A-Z][A-Z0-9]*(?:[-.][A-Za-z0-9]+)+\.md)`")
BARE_SEARCH_DIRS = (
    "planning_docs/record",
    "planning_docs/open",
    "planning_docs/aspirational",
    "planning_docs",
    "docs",
    "",
)


def _normalize(target: str) -> str | None:
    # A sentence-final period can be absorbed into the filename.
    target = target.rstrip(".")
    # Glob patterns and prefixes quoted as prose are not citations.
    stem = target.rsplit("/", 1)[-1]
    if "*" in target or "{" in target or stem.split(".")[0].endswith("_"):
        return None
    if target.startswith(("open/", "record/", "aspirational/")):
        target = "planning_docs/" + target
    return target


def _bare_resolves(name: str) -> bool:
    return any((ROOT / d / name).exists() for d in BARE_SEARCH_DIRS)


def _iter_files() -> list[Path]:
    files: list[Path] = list(ROOT.glob("*.md"))
    for tree in SCANNED_TREES:
        root = ROOT / tree
        if not root.is_dir():
            continue
        files.extend(
            p
            for p in sorted(root.rglob("*"))
            if p.is_file()
            and p.suffix in SCANNED_SUFFIXES
            and "__pycache__" not in p.parts
            and ".venv" not in p.parts
        )
    return files


def test_doc_citations_resolve() -> None:
    dangling: list[str] = []
    for path in _iter_files():
        rel = path.relative_to(ROOT)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        in_exempt_fence = False
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("```"):
                if in_exempt_fence:
                    in_exempt_fence = False
                elif "no-cite" in stripped:
                    in_exempt_fence = True
                continue
            if in_exempt_fence or NO_CITE in line:
                continue
            for match in CITATION.finditer(line):
                target = _normalize(match.group(1))
                if target is None:
                    continue
                if not (ROOT / target).exists():
                    dangling.append(f"{rel}:{lineno}: {target}")
            if rel.parts[0] == "planning_docs":
                for bare in BARE_CITATION.finditer(line):
                    name = bare.group(1)
                    if not _bare_resolves(name):
                        dangling.append(f"{rel}:{lineno}: {name} (bare)")
    assert not dangling, (
        f"{len(dangling)} doc/script citation(s) point at files that do not exist; "
        "update the citing line or mark a deliberate quote with "
        f"{NO_CITE!r}:\n" + "\n".join(dangling)
    )
