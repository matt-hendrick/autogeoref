"""The symbol-reachability gate: what it flags, what it must never flag, its ceiling.

Every case builds a small package tree on disk rather than a fake graph, because
the defects this gate has already had were in resolution — anchoring a relative
import at the wrong package, and matching a name instead of an edge.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parent.parent

#: Fixture paths are composed, not written whole: `tests/test_doc_citations.py`
#: reads a literal script path in any tracked file as a citation and requires
#: the file to exist, and none of these do.
SCRIPTS = "scripts"
EXPERIMENTS = f"{SCRIPTS}/experiments"


def _load_script(name: str) -> ModuleType:
    """Import a `scripts/` module by path — `scripts/` is not an installed package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "lint" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_script("check_symbol_reachability")


def build(root: Path, files: dict[str, str]) -> Path:
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    return root


def findings(root: Path) -> set[tuple[str, str, str]]:
    return {(f.module, f.symbol, f.reason) for f in checker.scan(root)}


def names(root: Path) -> set[str]:
    return {f"{f.module}.{f.symbol}" for f in checker.scan(root)}


# ----------------------------------------------------------------------
# The classes it exists to separate
# ----------------------------------------------------------------------


def test_a_live_importer_keeps_a_symbol(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/thing.py": "def used() -> None: ...\n",
            "src/autogeoref/cli.py": "from .thing import used\n",
        },
    )
    assert names(root) == set()


def test_a_test_only_importer_is_dead_and_says_so(tmp_path: Path) -> None:
    """The class the module gate structurally cannot see."""
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/thing.py": "def seam() -> None: ...\n",
            "tests/test_thing.py": "from autogeoref.thing import seam\n",
        },
    )
    assert findings(root) == {("autogeoref.thing", "seam", "reached only by tests")}


def test_a_live_script_does_hold_a_symbol_alive(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/thing.py": "def instrument() -> None: ...\n",
            f"{EXPERIMENTS}/measure.py": "from autogeoref.thing import instrument\n",
        },
    )
    assert names(root) == set()


def test_a_symbol_nobody_imports_is_distinguished_from_a_test_only_one(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/thing.py": "def orphan() -> None: ...\n",
        },
    )
    assert findings(root) == {("autogeoref.thing", "orphan", "no importer outside its own module")}


# ----------------------------------------------------------------------
# Resolve edges, never match names
# ----------------------------------------------------------------------


def test_a_same_named_local_def_in_a_script_does_not_count(tmp_path: Path) -> None:
    """The failure mode of a grep audit: a harness defining its own `run_volume`."""
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/volume.py": "def run_volume() -> None: ...\n",
            f"{SCRIPTS}/harness.py": """
                def run_volume(path: str) -> None: ...

                run_volume("x")
                """,
        },
    )
    assert names(root) == {"autogeoref.volume.run_volume"}


def test_a_same_named_keyword_argument_does_not_count(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/allmaps.py": "def viewer_url() -> None: ...\n",
            f"{SCRIPTS}/harness.py": "render(viewer_url='https://example.test')\n",
        },
    )
    assert names(root) == {"autogeoref.allmaps.viewer_url"}


# ----------------------------------------------------------------------
# Relative-import anchoring
# ----------------------------------------------------------------------


def test_the_anchor_rule_itself() -> None:
    """Backwards here reports a hundred plausible findings, so pin it directly.

    A tree-level case can pass by luck: whichever anchor the scan happens to
    use may be right for the one file the fixture holds.
    """
    assert checker._anchor("autogeoref.cli", is_package=False, level=1) == "autogeoref"
    assert checker._anchor("autogeoref.mask", is_package=True, level=1) == "autogeoref.mask"
    assert checker._anchor("autogeoref.mask.move", is_package=False, level=2) == "autogeoref"
    assert checker._anchor("autogeoref.mask", is_package=True, level=2) == "autogeoref"


def test_a_relative_import_anchors_a_module_at_its_parent(tmp_path: Path) -> None:
    """`from .x import y` inside `autogeoref.cli` means `autogeoref.x`, not `autogeoref.cli.x`."""
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/thing.py": "def used() -> None: ...\n",
            "src/autogeoref/cli.py": "from .thing import used\n",
        },
    )
    assert names(root) == set()


def test_a_relative_import_anchors_a_package_at_itself(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/mask/__init__.py": "from .geometry import heal\n",
            "src/autogeoref/mask/geometry.py": "def heal() -> None: ...\n",
        },
    )
    assert names(root) == set()


def test_a_two_level_relative_import_resolves(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/thing.py": "def used() -> None: ...\n",
            "src/autogeoref/mask/__init__.py": "",
            "src/autogeoref/mask/move.py": "from ..thing import used\n",
        },
    )
    assert names(root) == set()


# ----------------------------------------------------------------------
# The false-positive classes that were live defects in the prototype
# ----------------------------------------------------------------------


def test_module_level_dunders_are_never_reported(tmp_path: Path) -> None:
    """PEP 562 hooks are called by the interpreter and imported by nobody."""
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/lazy.py": """
                def __getattr__(name: str) -> object: ...

                def __dir__() -> list[str]: ...
                """,
        },
    )
    assert names(root) == set()


def test_dunder_all_in_a_package_init_counts_as_a_use(tmp_path: Path) -> None:
    """Ruff's F401 and mypy's implicit-reexport ban make the list load-bearing there."""
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/facade/__init__.py": """
                def helper() -> None: ...

                __all__ = ["helper"]
                """,
        },
    )
    assert names(root) == set()


def test_dunder_all_in_an_ordinary_module_does_not_count(tmp_path: Path) -> None:
    """With no star import in the tree it certifies itself and holds up nothing."""
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/margins.py": """
                class MarginError(ValueError): ...

                __all__ = ["MarginError"]
                """,
        },
    )
    assert names(root) == {"autogeoref.margins.MarginError"}


def test_a_reference_in_the_symbols_own_module_counts(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/thing.py": """
                def helper() -> None: ...

                def caller() -> None:
                    helper()

                __all__ = ["caller"]
                """,
            "src/autogeoref/cli.py": "from .thing import caller\n",
        },
    )
    assert names(root) == set()


def test_a_method_call_does_not_revive_a_module_level_name(tmp_path: Path) -> None:
    """`other.helper()` names a method. Counting it is the name matching §3 rules out."""
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/thing.py": """
                def helper() -> None: ...


                class Other:
                    def helper(self) -> None: ...


                def caller(other: Other) -> None:
                    other.helper()
                """,
            "src/autogeoref/cli.py": "from .thing import caller\n",
        },
    )
    assert names(root) == {"autogeoref.thing.helper"}


def test_a_def_under_a_module_level_if_is_still_gated(tmp_path: Path) -> None:
    """It binds a module attribute like any other, so hiding one there must not exempt it."""
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/thing.py": """
                import sys

                if sys.version_info >= (3, 12):
                    def gated() -> None: ...
                else:
                    def gated() -> None: ...
                """,
        },
    )
    assert names(root) == {"autogeoref.thing.gated"}


def test_a_star_import_is_refused_rather_than_silently_dropped(tmp_path: Path) -> None:
    """It binds names this pass cannot resolve; ruff's F403 keeps one out of the tree."""
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/thing.py": "def used() -> None: ...\n",
            f"{SCRIPTS}/harness.py": "from autogeoref.thing import *\n",
        },
    )
    with pytest.raises(ValueError, match="star import"):
        checker.scan(root)


def test_ruff_still_refuses_the_star_import_this_gate_assumes_away() -> None:
    """Read the shipped config: the assumption is only safe while something enforces it."""
    config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lint = config.split("[tool.ruff.lint]", 1)[1].split("\n[", 1)[0]
    selected = lint.split("select = [", 1)[1].split("]", 1)[0]
    ignored = lint.split("ignore = [", 1)[1].split("]", 1)[0]
    assert '"F"' in selected or '"F403"' in selected
    assert "F403" not in ignored and '"F"' not in ignored


def test_a_docstring_mention_is_not_a_reference(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/thing.py": '''
                def helper() -> None:
                    """Prose naming :func:`helper` proves nothing."""
                ''',
        },
    )
    assert names(root) == {"autogeoref.thing.helper"}


# ----------------------------------------------------------------------
# Binding forms other than `from pkg.mod import name`
# ----------------------------------------------------------------------


def test_an_aliased_module_import_plus_attribute_access_counts(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/thing.py": "def used() -> None: ...\n",
            f"{SCRIPTS}/harness.py": """
                import autogeoref.thing as thing

                thing.used()
                """,
        },
    )
    assert names(root) == set()


def test_a_plain_dotted_module_import_plus_attribute_access_counts(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/mask/__init__.py": "",
            "src/autogeoref/mask/geometry.py": "def heal() -> None: ...\n",
            f"{SCRIPTS}/harness.py": """
                import autogeoref.mask.geometry

                autogeoref.mask.geometry.heal()
                """,
        },
    )
    assert names(root) == set()


def test_importing_a_submodule_by_name_then_using_it_counts(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/mask/__init__.py": "",
            "src/autogeoref/mask/geometry.py": "def heal() -> None: ...\n",
            f"{SCRIPTS}/harness.py": """
                from autogeoref.mask import geometry

                geometry.heal()
                """,
        },
    )
    assert names(root) == set()


def test_binding_a_submodule_alone_does_not_revive_its_symbols(tmp_path: Path) -> None:
    """`from pkg import mod` is a module edge; the module gate already owns that."""
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/mask/__init__.py": "",
            "src/autogeoref/mask/geometry.py": "def heal() -> None: ...\n",
            f"{SCRIPTS}/harness.py": "from autogeoref.mask import geometry\n",
        },
    )
    assert names(root) == {"autogeoref.mask.geometry.heal"}


# ----------------------------------------------------------------------
# The ceiling, read from the shipped configuration
# ----------------------------------------------------------------------


def test_the_shipped_tree_is_at_or_under_its_frozen_ceiling() -> None:
    """Reading the checker's own constant is the point: restating it here would
    stay green after someone deleted it."""
    assert len(checker.scan(ROOT)) <= checker.FROZEN_FINDINGS


def test_the_ceiling_is_a_ratchet_not_a_target() -> None:
    """A ceiling far above the real count silently permits new dead symbols."""
    assert checker.FROZEN_FINDINGS - len(checker.scan(ROOT)) <= 2


def test_the_gate_runs_in_make_lint_and_not_in_report_mode() -> None:
    """A checker nothing invokes is a report, and `--report` exits 0 on any tree.

    Asserting the file name alone stays green when someone silences the gate by
    appending `--report`, or comments the recipe line out.
    """
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    # `lint` delegates to prerequisite targets, so scan every recipe the tree
    # under it reaches rather than that one target's own lines.
    invocations = [
        line
        for line in makefile.splitlines()
        if line.startswith("\t") and "check_symbol_reachability.py" in line
    ]
    assert len(invocations) == 1, invocations
    assert "--report" not in invocations[0]


def test_the_gate_exits_nonzero_over_the_ceiling_and_zero_at_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = build(
        tmp_path,
        {
            "src/autogeoref/__init__.py": "",
            "src/autogeoref/thing.py": "def orphan() -> None: ...\n",
        },
    )
    monkeypatch.setattr(checker, "FROZEN_FINDINGS", 0)
    assert checker.main(["--root", str(root)]) == 1
    monkeypatch.setattr(checker, "FROZEN_FINDINGS", 1)
    assert checker.main(["--root", str(root)]) == 0
