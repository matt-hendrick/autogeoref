"""Every live script's ``autogeoref`` imports resolve.

Nothing else executes a script, so an API change used to leave one broken until
somebody ran it. This parses each file and resolves the symbols it imports; it
never imports or runs a script module, which would spend time, budget, or both.

Signature drift is invisible here — an import check cannot see an added
parameter or a changed type. That is what the strict mypy pass covers.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import re
import tomllib
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIRECTORY = REPOSITORY_ROOT / "scripts"
EXPERIMENTS_DIRECTORY = SCRIPTS_DIRECTORY / "experiments"
PYPROJECT = REPOSITORY_ROOT / "pyproject.toml"
PACKAGE = "autogeoref"


@cache
def _mypy_settings() -> dict[str, Any]:
    settings: dict[str, Any] = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["mypy"]
    return settings


def _imported_symbols(tree: ast.AST) -> list[tuple[int, str, str | None]]:
    """Return ``(line, module, symbol)`` for each package import; symbol None for a plain import."""
    found: list[tuple[int, str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(
                (node.lineno, alias.name, None)
                for alias in node.names
                if alias.name == PACKAGE or alias.name.startswith(f"{PACKAGE}.")
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or not (module == PACKAGE or module.startswith(f"{PACKAGE}.")):
                continue
            found.extend(
                (node.lineno, module, alias.name) for alias in node.names if alias.name != "*"
            )
    return found


def _load(module: str, seen: dict[str, ModuleType | None]) -> ModuleType | None:
    """Import a package module, or None when it does not exist."""
    if module not in seen:
        try:
            seen[module] = importlib.import_module(module)
        except ImportError as error:
            # A missing optional dependency is an environment gap, not a stale
            # import; only a missing package module is this test's business.
            if error.name is not None and not error.name.startswith(PACKAGE):
                raise pytest.skip.Exception(f"{module} needs {error.name}") from error
            seen[module] = None
    return seen[module]


def _relative(path: Path) -> str:
    return str(path.relative_to(REPOSITORY_ROOT)) if path.is_relative_to(REPOSITORY_ROOT) else ""


def _unresolvable(path: Path, seen: dict[str, ModuleType | None]) -> list[str]:
    """Return one message per import of this file that no longer resolves."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative = _relative(path) or path.name
    broken: list[str] = []
    for line, module, symbol in _imported_symbols(tree):
        loaded = _load(module, seen)
        if loaded is None:
            broken.append(f"{relative}:{line}: no module {module}")
            continue
        if symbol is None or hasattr(loaded, symbol):
            continue
        # A package can also export a submodule nobody has imported yet.
        if not hasattr(loaded, "__path__") or _load(f"{module}.{symbol}", seen) is None:
            broken.append(f"{relative}:{line}: {module} has no {symbol}")
    return broken


@cache
def _broken_imports() -> tuple[str, ...]:
    """Scan every shipped script once; the module import cache is the expensive part."""
    seen: dict[str, ModuleType | None] = {}
    return tuple(message for path in _live_paths() for message in _unresolvable(path, seen))


def _live_paths() -> list[Path]:
    """Every script this repository ships.

    Filtered by the mypy exclusion rather than a second list of its own, so the
    scan and the type pass cannot come to cover different sets. Both walk the
    disk, which in a working checkout still holds untracked harnesses.
    """
    excluded = re.compile(_mypy_settings()["exclude"])
    return [
        path
        for path in sorted(SCRIPTS_DIRECTORY.rglob("*.py"))
        if not excluded.match(path.relative_to(REPOSITORY_ROOT).as_posix())
    ]


def test_every_live_script_import_resolves() -> None:
    """The gate: no maintained tool or live instrument names a symbol that moved."""
    assert list(_broken_imports()) == []


def test_the_scan_covers_both_tiers() -> None:
    """A scan that reads nothing passes forever."""
    live = _live_paths()
    assert SCRIPTS_DIRECTORY / "annotate_volume.py" in live
    assert any(path.parent == EXPERIMENTS_DIRECTORY for path in live)


def test_the_scan_catches_a_moved_symbol_and_a_moved_module(tmp_path: Path) -> None:
    """The two shapes the gate exists for, and one live import that must stay quiet."""
    source = tmp_path / "stale.py"
    source.write_text(
        "from autogeoref.scoring import GT_COMMIT_RMSE_M, gt_commit_rmse_m\n"
        "import autogeoref.nowhere\n"
        "from autogeoref.queue import run\n",
        encoding="utf-8",
    )
    broken = _unresolvable(source, {})

    assert broken == [
        "stale.py:1: autogeoref.scoring has no gt_commit_rmse_m",
        "stale.py:2: no module autogeoref.nowhere",
    ]


def test_the_mypy_pass_is_the_only_check_on_a_bare_sibling_import() -> None:
    """A `scripts/` library is imported by bare name, and the scan above cannot see it.

    `_imported_symbols` follows `autogeoref` imports only, so a library that
    moves or is deleted breaks nothing here. mypy is what catches it, and only
    while its search path finds siblings and a missing import is an error
    rather than ignored.
    """
    settings = _mypy_settings()
    assert set(settings["mypy_path"]) == {"scripts", "scripts/experiments"}
    assert "ignore_missing_imports" not in settings
    assert not settings.get("disable_error_code")


def test_the_mypy_pass_covers_the_script_tiers() -> None:
    """One strict pass, or a script an API change left behind lands green."""
    settings = _mypy_settings()
    assert settings["strict"] is True
    assert "scripts" in settings["files"]


def test_the_mypy_exclusion_matches_the_untracked_harnesses_exactly() -> None:
    """mypy walks the disk, so the exclusion has to mirror what git ignores.

    A checkout that still holds the untracked harnesses would otherwise be
    graded on files it does not ship, and one that does not hold them would
    silently stop checking the three that it does.
    """
    pattern = re.compile(_mypy_settings()["exclude"])
    tier = f"{SCRIPTS_DIRECTORY.name}/{EXPERIMENTS_DIRECTORY.name}"
    for kept in ("numeral_rules", "qualification", "water"):
        assert not pattern.match(f"{tier}/{kept}.py")
    # composed, not written whole: a literal path here reads as a citation
    for gone in (f"{tier}/archive/probe.py", f"{tier}/g1_probe.py"):
        assert pattern.match(gone)
    assert not pattern.match(f"{SCRIPTS_DIRECTORY.name}/annotate_volume.py")


#: Names that reach a model backend. A script calling one of these spends, and
#: `scripts/README.md` promises the marker grep is the whole list before you run.
SPEND_SURFACE = (
    "annotate_volume",
    "_default_annotator",
    "stage_escalate",
    "run_cli",
    "ClaudeCLIBackend",
    "CodexCLIBackend",
    "backend_for_model",
)
MARKER = "SPENDS BUDGET:"


def test_every_script_that_reaches_a_model_backend_carries_the_spend_marker() -> None:
    """The marker is a safety claim, so it cannot rest on someone remembering.

    A reader greps for it before running anything; a spender without one reads
    as free. Add the marker rather than trimming this list.
    """
    unmarked = []
    for path in sorted(SCRIPTS_DIRECTORY.rglob("*.py")):
        source = path.read_text(encoding="utf-8", errors="replace")
        # CALLS, not mentions: the probes and replays name these in prose while
        # spending nothing, and a grep over raw text reports them as spenders.
        called = {
            node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for node in ast.walk(ast.parse(source, filename=str(path)))
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
        }
        if called & set(SPEND_SURFACE) and MARKER not in source:
            unmarked.append(str(path.relative_to(REPOSITORY_ROOT)))
    assert not unmarked, f"reaches a model backend with no {MARKER!r} marker: {unmarked}"
