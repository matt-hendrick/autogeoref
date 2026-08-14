"""Find package symbols no live consumer imports.

The module gate (`.importlinter`) asks whether a MODULE is reachable. This asks
the same of each module-level `def`/`class`: a function whose only importer is
its own test is reachable at module level and dead at symbol level, and a
package `__init__` is exempt from the module contract entirely.

Import edges are RESOLVED, never name-matched. A word-frequency pass reports
dead symbols as live: a locally defined `def` of the same name in a harness, or
a keyword argument sharing the name, both read as hits. Nothing is imported or
executed; this is an AST pass.

    python scripts/lint/check_symbol_reachability.py            # gate
    python scripts/lint/check_symbol_reachability.py --report   # list every finding
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

PACKAGE = "autogeoref"

#: The ratchet, not a target: lower it freely, never raise it. `--report` names
#: every finding, and `git log` says which tree the number was taken on.
FROZEN_FINDINGS = 5

#: Consumers whose import does NOT keep a symbol alive, most specific first.
#: A test-only symbol is the class the module gate structurally cannot see;
#: letting one hold a symbol alive would pin the package's surface forever.
DEAD_TIERS = (("tests", "tests"),)

#: Trees scanned for live consumers. The package itself is one.
LIVE_TREES = ("src", "scripts")


@dataclass(frozen=True)
class Finding:
    """One module-level symbol with no live importer."""

    module: str
    symbol: str
    line: int
    reason: str

    def render(self) -> str:
        return f"{self.module}.{self.symbol}:{self.line}: {self.reason}"


def _module_name(path: Path, package_root: Path) -> str:
    """Dotted name of a file inside the package."""
    parts = list(path.relative_to(package_root.parent).parts)
    parts[-1] = parts[-1].removesuffix(".py")
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _anchor(module: str, is_package: bool, level: int) -> str:
    """Base a relative import resolves against.

    A package anchors at itself, a module at its parent — getting this backwards
    reports a hundred plausible findings.
    """
    parts = module.split(".")
    if not is_package:
        parts.pop()
    return ".".join(parts[: len(parts) - (level - 1)]) if level > 1 else ".".join(parts)


def _resolved(node: ast.ImportFrom, module: str | None, is_package: bool) -> str | None:
    """Absolute module an ``ImportFrom`` names, or None when it cannot be anchored."""
    if not node.level:
        return node.module
    if module is None:
        return None
    base = _anchor(module, is_package, node.level)
    return f"{base}.{node.module}" if node.module else base


def _definitions(body: Iterable[ast.stmt]) -> Iterator[tuple[str, int]]:
    """Module-level ``def``/``class``, skipping the PEP 562 module hooks.

    ``__getattr__`` and ``__dir__`` are called by the interpreter, never
    imported, so a reachability pass reports them as dead. A ``def`` under a
    module-level ``if``/``try`` still binds a module attribute, so it counts.
    """
    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in body:
        if isinstance(node, definitions) and not _is_dunder(node.name):
            yield node.name, node.lineno
        elif isinstance(node, (ast.If, ast.Try)):
            yield from _definitions(node.body)
            yield from _definitions(node.orelse)
            if isinstance(node, ast.Try):
                yield from _definitions(node.finalbody)
                for handler in node.handlers:
                    yield from _definitions(handler.body)


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _dunder_all(tree: ast.Module) -> set[str]:
    """Names listed in a module-level ``__all__`` literal."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
            return {
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
    return set()


def _local_references(tree: ast.Module, defined: set[str]) -> set[str]:
    """Defined names this module mentions anywhere other than their own header.

    A decorator, a default argument, an ``argparse`` ``set_defaults(func=...)``,
    or a call all count — the symbol is in use where it lives. An attribute
    access does NOT: ``other.helper()`` names a method, and counting it would
    be the name matching this whole check exists to avoid.
    """
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in defined:
            used.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            used |= {d.id for d in node.decorator_list if isinstance(d, ast.Name)} & defined
    return used


def _dotted(node: ast.Attribute) -> list[str] | None:
    """Flatten ``a.b.c`` to ``["a", "b", "c"]``; None when the head is not a name."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return list(reversed(parts))


def _imported_symbols(
    tree: ast.Module, module: str | None, is_package: bool, modules: set[str]
) -> set[tuple[str, str]]:
    """``(module, symbol)`` pairs this file binds from the package.

    Covers ``from pkg.mod import name``, ``import pkg.mod as alias`` followed by
    ``alias.name``, and ``from pkg import mod`` followed by ``mod.name``.
    """
    used: set[tuple[str, str]] = set()
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PACKAGE or alias.name.startswith(f"{PACKAGE}."):
                    aliases[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
        elif isinstance(node, ast.ImportFrom):
            target = _resolved(node, module, is_package)
            if target is None or not (target == PACKAGE or target.startswith(f"{PACKAGE}.")):
                continue
            for alias in node.names:
                if alias.name == "*":
                    # a star import binds names this pass cannot see; ruff's
                    # F403 refuses one anywhere `make lint` reaches, so the
                    # edge cannot exist rather than being quietly dropped
                    raise ValueError(f"star import of {target}: this check cannot resolve one")
                child = f"{target}.{alias.name}"
                if child in modules:
                    aliases[alias.asname or alias.name] = child
                else:
                    used.add((target, alias.name))
    if aliases:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            parts = _dotted(node)
            if parts is None or parts[0] not in aliases:
                continue
            parts = aliases[parts[0]].split(".") + parts[1:]
            for cut in range(len(parts) - 1, 0, -1):
                candidate = ".".join(parts[:cut])
                if candidate in modules:
                    used.add((candidate, parts[cut]))
                    break
    return used


def _tier(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    for name, prefix in DEAD_TIERS:
        if relative.startswith(f"{prefix}/"):
            return name
    return "live"


def _consumers(root: Path) -> Iterator[Path]:
    for tree in (*LIVE_TREES, "tests"):
        directory = root / tree
        if directory.is_dir():
            yield from sorted(directory.rglob("*.py"))


def scan(root: Path) -> list[Finding]:
    """Every module-level symbol in the package with no live importer."""
    package_root = root / "src" / PACKAGE
    sources = {
        _module_name(path, package_root): (path, path.name == "__init__.py")
        for path in sorted(package_root.rglob("*.py"))
    }
    modules = set(sources)

    trees = {
        module: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module, (path, _) in sources.items()
    }

    findings: list[Finding] = []
    unused: dict[tuple[str, str], int] = {}
    for module, tree in trees.items():
        defined = dict(_definitions(tree.body))
        if not defined:
            continue
        alive = _local_references(tree, set(defined))
        if sources[module][1]:
            # only in a package __init__: ruff's F401 and mypy's implicit-reexport
            # ban make __all__ the thing holding a re-export up. Elsewhere, with
            # no star import in the tree, it certifies itself and nothing else.
            alive |= _dunder_all(tree)
        for symbol, line in defined.items():
            if symbol not in alive:
                unused[(module, symbol)] = line

    if not unused:
        return findings

    reached: dict[tuple[str, str], set[str]] = {}
    for path in _consumers(root):
        owner = _module_name(path, package_root) if path.is_relative_to(package_root) else None
        is_package = path.name == "__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        tier = _tier(path, root)
        for pair in _imported_symbols(tree, owner, is_package, modules):
            if pair in unused:
                reached.setdefault(pair, set()).add(tier)

    for (module, symbol), line in sorted(unused.items()):
        tiers = reached.get((module, symbol), set())
        if "live" in tiers:
            continue
        # every tier that reaches it, not the first: naming one reader of a
        # symbol two tiers import reads as safe to delete when it is not
        names = [name for name, _ in DEAD_TIERS if name in tiers]
        reason = (
            f"reached only by {' and '.join(sorted(names))}"
            if names
            else "no importer outside its own module"
        )
        findings.append(Finding(module, symbol, line, reason))
    return findings


def _report(findings: Iterable[Finding]) -> None:
    for finding in findings:
        print(f"  {finding.render()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true", help="list every finding and exit 0")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)

    findings = scan(args.root)
    if args.report:
        _report(findings)
        print(f"\n{len(findings)} findings, ceiling {FROZEN_FINDINGS}")
        return 0

    if len(findings) > FROZEN_FINDINGS:
        print(
            f"unreachable package symbols: {len(findings)}, over the frozen {FROZEN_FINDINGS}:",
            file=sys.stderr,
        )
        for finding in findings:
            print(f"  {finding.render()}", file=sys.stderr)
        print(
            "\nwire it into the product, delete it with its test, or lower the "
            "ceiling if you removed one",
            file=sys.stderr,
        )
        return 1

    print(f"symbol reachability ok ({len(findings)}/{FROZEN_FINDINGS})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
