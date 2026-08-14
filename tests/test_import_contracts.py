"""The two custom import contracts: what each flags, what it exempts, and its allowlist."""

from __future__ import annotations

import configparser
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import pytest
from grimp import ImportGraph

if TYPE_CHECKING:
    from importlinter import ContractCheck

ROOT = Path(__file__).parent.parent


def _load_script(name: str) -> ModuleType:
    """Import a `scripts/` module by path — `scripts/` is not an installed package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "lint" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contracts = _load_script("import_contracts")


def _graph(modules: list[str], imports: list[tuple[str, str]]) -> ImportGraph:
    graph = ImportGraph()
    for module in modules:
        graph.add_module(module)
    for importer, imported in imports:
        graph.add_import(importer=importer, imported=imported)
    return graph


def _check(graph: ImportGraph, allow: set[str] | None = None) -> ContractCheck:
    contract = contracts.ReachableFrom(
        name="test",
        session_options={"root_packages": ["pkg"]},
        contract_options={"root": "pkg.cli", "allow": sorted(allow or set())},
    )
    result: ContractCheck = contract.check(graph, verbose=False)
    return result


def test_a_module_no_import_chain_reaches_is_flagged() -> None:
    graph = _graph(["pkg", "pkg.cli", "pkg.used", "pkg.dead"], [("pkg.cli", "pkg.used")])

    check = _check(graph)

    assert not check.kept
    assert check.metadata["unlisted"] == ["pkg.dead"]


def test_reachability_is_transitive_not_direct() -> None:
    graph = _graph(
        ["pkg", "pkg.cli", "pkg.mid", "pkg.leaf"],
        [("pkg.cli", "pkg.mid"), ("pkg.mid", "pkg.leaf")],
    )

    assert _check(graph).kept


def test_an_allowed_module_is_the_written_exception() -> None:
    graph = _graph(["pkg", "pkg.cli", "pkg.dead"], [])

    assert not _check(graph).kept
    assert _check(graph, allow={"pkg.dead"}).kept


def test_package_init_modules_are_exempt_without_an_allow_entry() -> None:
    """A package that re-exports nothing is never imported, which is deliberate here."""
    graph = _graph(
        ["pkg", "pkg.cli", "pkg.sub", "pkg.sub.thing"],
        [("pkg.cli", "pkg.sub.thing")],
    )

    check = _check(graph)

    assert check.kept, check.metadata
    assert "pkg.sub" not in check.metadata["unlisted"]


def test_an_allow_entry_that_became_reachable_breaks_the_contract() -> None:
    graph = _graph(["pkg", "pkg.cli", "pkg.used"], [("pkg.cli", "pkg.used")])

    check = _check(graph, allow={"pkg.used"})

    assert not check.kept
    assert check.metadata["stale"] == ["pkg.used"]


def test_an_allow_entry_naming_no_module_breaks_the_contract() -> None:
    graph = _graph(["pkg", "pkg.cli"], [])

    check = _check(graph, allow={"pkg.typo"})

    assert not check.kept
    assert check.metadata["unknown"] == ["pkg.typo"]


def _shipped_config() -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(ROOT / ".importlinter", encoding="utf-8")
    return parser


def test_the_shipped_config_wires_the_contract_in() -> None:
    """Reading `.importlinter` is the point: deleting the section must fail something."""
    config = _shipped_config()

    assert (
        "reachable_from: import_contracts.ReachableFrom"
        in (config["importlinter"]["contract_types"])
    )
    assert config["importlinter:contract:reachable"]["type"] == "reachable_from"


def test_the_real_package_keeps_the_contract_as_configured() -> None:
    """The gate is only worth having if it passes, as configured, on the shipped tree."""
    import grimp

    section = _shipped_config()["importlinter:contract:reachable"]

    contract = contracts.ReachableFrom(
        name=section["name"],
        session_options={"root_packages": ["autogeoref"]},
        contract_options={
            "root": section["root"].strip(),
            "allow": section.get("allow", "").split(),
        },
    )
    check = contract.check(grimp.build_graph("autogeoref"), verbose=False)

    assert check.kept, check.metadata


def _fan_out(graph: ImportGraph, cap: int, allow: set[str] | None = None) -> ContractCheck:
    contract = contracts.MaxFanOut(
        name="test",
        session_options={"root_packages": ["pkg"]},
        contract_options={"max": str(cap), "allow": sorted(allow or set())},
    )
    result: ContractCheck = contract.check(graph, verbose=False)
    return result


def _star(importer: str, count: int) -> tuple[list[str], list[tuple[str, str]]]:
    leaves = [f"pkg.leaf{n}" for n in range(count)]
    return ["pkg", importer, *leaves], [(importer, leaf) for leaf in leaves]


def test_a_module_over_the_fan_out_cap_is_flagged_with_its_count() -> None:
    check = _fan_out(_graph(*_star("pkg.god", 4)), cap=3)

    assert not check.kept
    assert check.metadata["unlisted"] == [("pkg.god", 4)]


def test_the_fan_out_cap_is_inclusive() -> None:
    """At the cap is kept, one over is not: the ceiling names a permitted count."""
    at_cap = _graph(*_star("pkg.wide", 3))
    over = _graph(*_star("pkg.wide", 4))

    assert _fan_out(at_cap, cap=3).kept
    assert not _fan_out(over, cap=3).kept


def test_fan_out_counts_direct_imports_not_transitive_reach() -> None:
    """A deep chain is not a god module, and must not read as one."""
    graph = _graph(
        ["pkg", "pkg.a", "pkg.b", "pkg.c", "pkg.d"],
        [("pkg.a", "pkg.b"), ("pkg.b", "pkg.c"), ("pkg.c", "pkg.d")],
    )

    assert _fan_out(graph, cap=1).kept


def test_an_allowed_module_over_the_fan_out_cap_is_the_written_exception() -> None:
    modules, imports = _star("pkg.god", 4)

    assert not _fan_out(_graph(modules, imports), cap=3).kept
    assert _fan_out(_graph(modules, imports), cap=3, allow={"pkg.god"}).kept


def test_an_allow_entry_that_dropped_under_the_fan_out_cap_breaks_the_contract() -> None:
    """The entry is a plan to split; keeping it after the split hides the next one."""
    modules, imports = _star("pkg.slim", 2)

    check = _fan_out(_graph(modules, imports), cap=3, allow={"pkg.slim"})

    assert not check.kept
    assert check.metadata["stale"] == ["pkg.slim"]


def test_a_fan_out_allow_entry_naming_no_module_breaks_the_contract() -> None:
    check = _fan_out(_graph(["pkg", "pkg.cli"], []), cap=3, allow={"pkg.typo"})

    assert not check.kept
    assert check.metadata["unknown"] == ["pkg.typo"]


def test_the_shipped_config_wires_the_fan_out_contract_in() -> None:
    """Reading `.importlinter` is the point: deleting the section must fail something."""
    config = _shipped_config()

    assert "max_fan_out: import_contracts.MaxFanOut" in config["importlinter"]["contract_types"]
    assert config["importlinter:contract:fan-out"]["type"] == "max_fan_out"


def test_the_real_package_keeps_the_fan_out_contract_as_configured() -> None:
    """As configured, on the shipped tree — cap and allowlist both read from the file."""
    import grimp

    section = _shipped_config()["importlinter:contract:fan-out"]

    contract = contracts.MaxFanOut(
        name=section["name"],
        session_options={"root_packages": ["autogeoref"]},
        contract_options={
            "max": section["max"].strip(),
            "allow": section.get("allow", "").split(),
        },
    )
    check = contract.check(grimp.build_graph("autogeoref"), verbose=False)

    assert check.kept, check.metadata


def _names(section: str, key: str) -> set[str]:
    """Every module named in one contract field, `|` and `:` layer forms included."""
    raw = _shipped_config()[f"importlinter:contract:{section}"][key]
    return {name for name in raw.replace("|", " ").replace(":", " ").split() if name}


def _package_modules(package: str) -> set[str]:
    directory = ROOT / "src" / Path(*package.split("."))
    return {f"{package}.{path.stem}" for path in directory.glob("*.py") if path.stem != "__init__"}


#: package -> the contract field that has to name every module in it.
GOVERNED = {
    "autogeoref.cli": ("cli", "layers"),
    "autogeoref.stages": ("stages", "modules"),
    "autogeoref.bake": ("bake", "layers"),
    "autogeoref.runplan": ("runplan", "layers"),
    "autogeoref.config": ("config", "layers"),
    "autogeoref.mask": ("mask", "layers"),
}


@pytest.mark.parametrize("package", sorted(GOVERNED))
def test_the_contract_names_every_module_in_its_package(package: str) -> None:
    """A hand-written list is what a new sibling walks straight past.

    import-linter fails loudly on an entry naming nothing, so the risk runs one
    way only: a module that exists and is not listed is silently ungoverned.
    """
    section, key = GOVERNED[package]

    missing = _package_modules(package) - _names(section, key)

    assert not missing, f"{section} does not govern {sorted(missing)}"


def test_the_cli_sibling_list_is_the_cli_layers_bottom_row() -> None:
    """Two lists of the same six modules; if they drift, one of them stops binding."""
    bottom = _shipped_config()["importlinter:contract:cli"]["layers"].split("\n")[-1]

    assert {name.strip() for name in bottom.split("|")} == _names("cli-siblings", "modules")
