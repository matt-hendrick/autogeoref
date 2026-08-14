"""Bound the size and prose content of docstrings and comment blocks.

Ruff, darglint, and pydoclint all govern docstring *format*; nothing upstream
bounds length or inspects prose. ``RULES`` lists what is checked and ``_tier``
scopes it per tree. The gating trees are at zero, so any violation fails.

    python scripts/lint/check_docstrings.py                 # gate
    python scripts/lint/check_docstrings.py --report        # list every site

Suppress one site with ``# noqa: DS002`` on the ``def``/``class`` line, or on a
comment line above a module docstring. ``pyproject.toml`` declares ``DS`` in
``lint.external`` so ruff accepts the code.
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

#: The checkout root — this file sits two directories below it.
ROOT = Path(__file__).resolve().parents[2]

MODULE_DOC_MAX = 15
DEF_DOC_MAX = 10
COMMENT_BLOCK_MAX = 6

RULES = {
    "DS001": f"module docstring over {MODULE_DOC_MAX} lines",
    "DS002": f"def/class docstring over {DEF_DOC_MAX} lines",
    "DS003": f"comment block over {COMMENT_BLOCK_MAX} lines",
    "DS101": "planning-doc citation",
    "DS102": "ISO date",
    "DS103": "roadmap item number",
    "DS104": "volume id",
    "DS105": "corpus lore",
    "DS106": "superseded-alternative narrative",
}

#: DS106 is a keyword heuristic and false-positives on ordinary prose, so it is
#: reported and never gated.
ADVISORY = {"DS106"}

_PLANNING_DOC = re.compile(r"planning_docs?/")
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_ROADMAP_ITEM = re.compile(r"(?<![\w\]\)])\[\d{1,3}\]")
_VOLUME_ID = re.compile(r"\b(?:sanborn\d+)?_0\d{2}\b")
_LORE = re.compile(r"\b(?:Chicago|Cicero|Cleveland|CBD|the Loop)\b")
_SUPERSEDED = re.compile(
    r"\b(?:superseded|refuted|historically|used to|[Cc]hosen over|no longer)\b"
)

_PROSE_RULES = (
    ("DS101", _PLANNING_DOC),
    ("DS102", _ISO_DATE),
    ("DS103", _ROADMAP_ITEM),
    ("DS104", _VOLUME_ID),
    ("DS105", _LORE),
    ("DS106", _SUPERSEDED),
)

_NOQA = re.compile(r"#\s*noqa:\s*(?P<codes>[A-Z]+\d+(?:\s*,\s*[A-Z]+\d+)*)", re.IGNORECASE)


@dataclass(frozen=True)
class Violation:
    """One rule tripped at one place. ``detail`` is the offending text or size."""

    path: str
    line: int
    rule: str
    detail: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.rule} {RULES[self.rule]} — {self.detail}"


def _tier(rel: str) -> tuple[frozenset[str], bool]:
    """Return the rules that apply to a file and whether its result gates.

    Test docstrings may name the corpus — the volume, and the city it belongs
    to, are the subject under test.
    """
    if rel.startswith("scripts/experiments/"):
        return frozenset(RULES), False
    if rel.startswith("tests/"):
        return frozenset(RULES) - {"DS104", "DS105"}, True
    return frozenset(RULES), True


def _noqa_codes(lines: list[str], start: int, end: int) -> set[str]:
    """Codes suppressed by any ``# noqa:`` in the inclusive 1-based line range."""
    codes: set[str] = set()
    for lineno in range(max(start, 1), min(end, len(lines)) + 1):
        match = _NOQA.search(lines[lineno - 1])
        if match is not None:
            codes |= {code.strip().upper() for code in match["codes"].split(",")}
    return codes


def _prose_violations(path: str, text: str, lineno: int) -> Iterator[Violation]:
    for offset, line in enumerate(text.splitlines()):
        for rule, pattern in _PROSE_RULES:
            found = pattern.search(line)
            if found is not None:
                yield Violation(path, lineno + offset, rule, found.group(0).strip())


def _docstring_nodes(tree: ast.Module) -> Iterator[tuple[str, ast.Constant, int, int]]:
    """Yield ``(rule, docstring node, noqa first line, noqa last line)`` per docstring.

    The noqa range runs from the owning ``def``/``class`` line — or from the top
    of the file, for a module — up to the line before the docstring opens.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        body = node.body
        if not body or not isinstance(body[0], ast.Expr):
            continue
        value = body[0].value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        rule = "DS001" if isinstance(node, ast.Module) else "DS002"
        start = 1 if isinstance(node, ast.Module) else node.lineno
        yield rule, value, start, max(value.lineno - 1, start)


def _comment_blocks(source: str) -> Iterator[tuple[int, list[str]]]:
    """Yield ``(first line, comment texts)`` for each run of whole-line comments."""
    lines = source.splitlines()
    block: list[str] = []
    start = 0
    previous = 0
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        lineno = token.start[0]
        if not lines[lineno - 1].lstrip().startswith("#"):
            continue
        if block and lineno == previous + 1:
            block.append(token.string)
        else:
            if block:
                yield start, block
            block = [token.string]
            start = lineno
        previous = lineno
    if block:
        yield start, block


def check_source(source: str, rel: str) -> list[Violation]:
    """Return every violation in one file's source. ``rel`` selects the scope tier."""
    rules, _ = _tier(rel)
    lines = source.splitlines()
    found: list[Violation] = []

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [Violation(rel, exc.lineno or 1, "DS001", f"unparseable: {exc.msg}")]

    for rule, node, noqa_start, noqa_end in _docstring_nodes(tree):
        span = (node.end_lineno or node.lineno) - node.lineno + 1
        cap = MODULE_DOC_MAX if rule == "DS001" else DEF_DOC_MAX
        suppressed = _noqa_codes(lines, noqa_start, noqa_end)
        if span > cap and rule not in suppressed:
            found.append(Violation(rel, node.lineno, rule, f"{span} lines"))
        text = node.value if isinstance(node.value, str) else ""
        found.extend(
            v for v in _prose_violations(rel, text, node.lineno) if v.rule not in suppressed
        )

    for start, block in _comment_blocks(source):
        suppressed = _noqa_codes(lines, start, start)
        if len(block) > COMMENT_BLOCK_MAX and "DS003" not in suppressed:
            found.append(Violation(rel, start, "DS003", f"{len(block)} lines"))
        for offset, comment in enumerate(block):
            found.extend(
                v
                for v in _prose_violations(rel, comment, start + offset)
                if v.rule not in suppressed
            )

    return [v for v in found if v.rule in rules]


def _scan(paths: Iterable[Path]) -> list[Violation]:
    found: list[Violation] = []
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        found.extend(check_source(path.read_text(encoding="utf-8"), rel))
    return sorted(found, key=lambda v: (v.path, v.line, v.rule))


def _python_files(trees: Iterable[str], include_experiments: bool) -> list[Path]:
    files: list[Path] = []
    for tree in trees:
        root = ROOT / tree
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if not include_experiments and rel.startswith("scripts/experiments/"):
                continue
            files.append(path)
    return files


def scan_repo(include_experiments: bool = False) -> list[Violation]:
    """Every violation in the checked trees, sorted by path and line."""
    return _scan(_python_files(("src", "scripts", "tests"), include_experiments))


def gating(violations: Iterable[Violation]) -> list[Violation]:
    """The violations that fail the build: gating trees, advisory rules dropped."""
    return [v for v in violations if _tier(v.path)[1] and v.rule not in ADVISORY]


def _report(violations: list[Violation]) -> None:
    for violation in violations:
        if violation.rule not in ADVISORY:
            print(violation.render())
    advisory = [v for v in violations if v.rule in ADVISORY]
    if advisory:
        print(f"\nadvisory ({len(advisory)}), never a failure:")
        for violation in advisory:
            print(f"  {violation.render()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true", help="list every site and exit 0")
    parser.add_argument(
        "--include-experiments",
        action="store_true",
        help="include scripts/experiments/, which never gates",
    )
    args = parser.parse_args(argv)

    violations = scan_repo(args.include_experiments or args.report)
    if args.report:
        _report(violations)
        print(f"\n{len(violations)} sites in {len({v.path for v in violations})} files")
        return 0

    failures = gating(violations)
    if failures:
        print("docstring budget exceeded:", file=sys.stderr)
        for violation in failures:
            print(f"  {violation.render()}", file=sys.stderr)
        print(
            "\nshorten the docstring or move the evidence to a dated record; "
            "`# noqa: DSNNN` is the deliberate, reviewable exception",
            file=sys.stderr,
        )
        return 1

    print("docstrings ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
