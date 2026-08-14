"""The container build context must equal the tracked tree.

`.dockerignore` is load-bearing rather than hygiene. A host cache that rides
into the image can make a cold-start check falsely green — a mypy cache built
on the host is the worst case, since the whole point of the image is to run
where the host has not been. A tracked file dropped from the context is the
mirror failure: the image is then not a cold clone at all.

These hold the parts of that comparison a test can reach: what the two ignore
files agree on, whether an exclusion swallows tracked content, and whether a
pattern that has to match at depth says so. The exact file-by-file diff needs a
daemon and lives in the cold-clone CI job.

Skips where git cannot list files, as in a source tarball or inside the image.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DOCKERIGNORE = REPOSITORY_ROOT / ".dockerignore"
GITIGNORE = REPOSITORY_ROOT / ".gitignore"

#: In the context on purpose, and in no `.gitignore`: git's own directory, and
#: the maintainer's scratch list.
CONTEXT_ONLY = frozenset({".git", "todo.md"})

#: Artifact classes a tool can create below the top level. A `.dockerignore`
#: pattern is ANCHORED AT THE CONTEXT ROOT where the same `.gitignore` line
#: matches at any depth, so each of these needs `**/`. Most data directories are
#: deliberately absent: a blanket `**/` on one could swallow a tracked path that
#: shares its name. `cache` is here because a city keeps its own at
#: `configs/<city>/cache/`, and the swallowing test fails the day one is tracked.
NESTS = (
    "cache",
    "__pycache__",
    "*.pyc",
    "*.egg-info",
    "node_modules",
    ".pytest_cache",
    ".hypothesis",
    ".mypy_cache",
    ".ruff_cache",
    ".grimp_cache",
    ".import_linter_cache",
    "dist",
    "htmlcov",
)


def in_a_git_checkout() -> bool:
    """False in a source tarball or the container's copy, where git lists nothing."""
    return (REPOSITORY_ROOT / ".git").exists()


def _lines(path: Path) -> list[str]:
    return [
        stripped
        for line in path.read_text(encoding="utf-8").splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    ]


def _normalize(line: str) -> str:
    """Strip the spellings that carry no meaning for comparing the two files.

    A trailing ``/`` or ``/**`` says "this directory" to both tools; a leading
    ``**/`` says "at any depth", which `.gitignore` means by default and
    `.dockerignore` has to be told.
    """
    return line.removeprefix("!").removeprefix("**/").removesuffix("/**").rstrip("/")


def _patterns(path: Path) -> tuple[list[str], list[str]]:
    """Return (excludes, negations) from an ignore file, normalized."""
    excludes: list[str] = []
    negations: list[str] = []
    for line in _lines(path):
        (negations if line.startswith("!") else excludes).append(_normalize(line))
    return excludes, negations


def _tracked_under(pattern: str, *, anywhere: bool) -> list[str]:
    """Tracked paths a `.dockerignore` exclusion would drop from the context.

    ``anywhere`` mirrors the pattern's ``**/`` prefix: without it an exclusion
    is anchored at the context root, with it every depth matches.
    """
    specs = [pattern, f"{pattern}/**"]
    if anywhere:
        specs.append(f":(glob)**/{pattern}/**")
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--", *specs],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if listing.returncode != 0:
        return []
    return [name for name in listing.stdout.split("\0") if name]


def test_everything_git_ignores_the_build_context_ignores_too() -> None:
    """A tool that invents a new cache directory gets one `.gitignore` line and
    is then one forgotten line away from riding into the image."""
    missing = sorted(set(_patterns(GITIGNORE)[0]) - set(_patterns(DOCKERIGNORE)[0]))
    assert not missing, (
        "these are ignored by git but would be sent to the docker daemon as build "
        f"context; add them to .dockerignore: {missing}"
    )


def test_the_build_context_ignores_nothing_beyond_that_unaccounted_for() -> None:
    """The reverse direction: a `.dockerignore` line git knows nothing about is
    either deliberate or a tracked file class about to go missing from the image."""
    extra = sorted(set(_patterns(DOCKERIGNORE)[0]) - set(_patterns(GITIGNORE)[0]) - CONTEXT_ONLY)
    assert not extra, (
        "these .dockerignore lines are not in .gitignore, so they exclude something "
        f"git tracks or knows nothing about: {extra}"
    )


@pytest.mark.skipif(not in_a_git_checkout(), reason="no git repository to list files from")
def test_no_exclusion_swallows_a_tracked_file() -> None:
    """An image missing tracked files is not a cold clone.

    A blanket directory rule is fine as long as every tracked file under it is
    negated back in — which is how the one tracked file under `fixtures/`
    travels. A new one would get no negation and no warning.
    """
    excludes, negations = _patterns(DOCKERIGNORE)
    anywhere = {_normalize(line) for line in _lines(DOCKERIGNORE) if line.startswith("**/")}
    dropped: list[str] = []
    for pattern in excludes:
        kept = _tracked_under(pattern, anywhere=pattern in anywhere)
        dropped.extend(
            path
            for path in kept
            if not any(path == n or path.startswith(f"{n}/") for n in negations)
        )
    assert not dropped, (
        "these tracked files fall under a .dockerignore exclusion with no '!' negation, "
        f"so the image would not carry them: {sorted(set(dropped))}"
    )


def test_an_artifact_class_that_nests_is_excluded_at_every_depth() -> None:
    """The bare spelling reads correct and silently excludes only the top copy."""
    anywhere = {_normalize(line) for line in _lines(DOCKERIGNORE) if line.startswith("**/")}
    unanchored = [name for name in NESTS if name not in anywhere]
    assert not unanchored, (
        "a tool can create these below the top level, and a root-anchored .dockerignore "
        f"pattern misses every copy there; write them as '**/<name>': {unanchored}"
    )


@pytest.mark.skipif(not in_a_git_checkout(), reason="no git repository to list files from")
def test_the_guard_reads_both_files_and_reaches_git() -> None:
    """A guard that parses nothing passes forever."""
    assert len(_patterns(DOCKERIGNORE)[0]) > 10
    assert len(_patterns(GITIGNORE)[0]) > 10
    assert _tracked_under("fixtures", anywhere=False) == []
