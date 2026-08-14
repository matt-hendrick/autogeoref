"""Custom import-linter contracts, named by dotted path from ``.importlinter``.

``ReachableFrom`` asserts every module is imported, directly or transitively,
by one entry point; a module that is not becomes dead weight no per-file linter
can see. ``MaxFanOut`` caps how many package modules one module may import,
which is the direct measure of a god module. Both take an ``allow`` set — a
written, reviewable claim, and both break when an entry goes stale.

``make lint`` puts this directory on ``PYTHONPATH`` so ``lint-imports`` can
import it. Package ``__init__`` modules are exempt from ``ReachableFrom``: a
package that re-exports nothing is never itself imported, which is deliberate.

Static analysis only, inherited from grimp: a function-local import is seen, an
``importlib.import_module`` on a computed name is not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from importlinter import Contract, ContractCheck, fields, output

if TYPE_CHECKING:
    from collections.abc import Iterable

    from grimp import ImportGraph


def _values(field: Any) -> Iterable[object]:
    """Read a declared field's parsed value, which its class attribute hides."""
    return cast("Iterable[object]", field or set())


class ReachableFrom(Contract):
    """Every non-package module is reachable from ``root``, or named in ``allow``.

    Breaks three ways: an unreachable module that is not allowed, an ``allow``
    entry that has become reachable (a stale claim), and an ``allow`` entry
    naming no module in the graph (a typo).
    """

    root = fields.ModuleField()
    allow = fields.SetField(subfield=fields.ModuleField(), required=False)

    def check(self, graph: ImportGraph, verbose: bool) -> ContractCheck:
        root = str(self.root)
        allowed = {str(m) for m in _values(self.allow)}

        output.verbose_print(verbose, f"Finding modules upstream of {root}...")
        reachable = {root} | graph.find_upstream_modules(root)

        unreachable = {
            module
            for module in graph.modules
            if module not in reachable and not graph.find_children(module)
        }

        unlisted = sorted(unreachable - allowed)
        stale = sorted(allowed & reachable)
        unknown = sorted(allowed - set(graph.modules))

        return ContractCheck(
            kept=not (unlisted or stale or unknown),
            metadata={
                "root": root,
                "unlisted": unlisted,
                "stale": stale,
                "unknown": unknown,
            },
        )

    def render_broken_contract(self, check: ContractCheck) -> None:
        root = check.metadata["root"]
        if check.metadata["unlisted"]:
            output.print_error(f"Not reachable from {root}, and not in the allowlist:")
            output.new_line()
            for module in check.metadata["unlisted"]:
                output.print_error(f"    {module}", bold=False)
            output.new_line()
            output.print_error(
                "    Wire it into the product, delete it, or add it to `allow` with a reason.",
                bold=False,
            )
            output.new_line()
        if check.metadata["stale"]:
            output.print_error(f"Allowed, but now reachable from {root} (drop the entry):")
            output.new_line()
            for module in check.metadata["stale"]:
                output.print_error(f"    {module}", bold=False)
            output.new_line()
        if check.metadata["unknown"]:
            output.print_error("Allowed, but no such module (typo?):")
            output.new_line()
            for module in check.metadata["unknown"]:
                output.print_error(f"    {module}", bold=False)
            output.new_line()


class MaxFanOut(Contract):
    """No module imports more than ``max`` other modules of the package.

    Breaks three ways: a module over the cap that is not allowed, an ``allow``
    entry now under it (a stale claim), and an ``allow`` entry naming no module
    in the graph (a typo). Direct imports only — a deep chain is not fan-out.
    """

    max = fields.IntegerField()
    allow = fields.SetField(subfield=fields.ModuleField(), required=False)

    def check(self, graph: ImportGraph, verbose: bool) -> ContractCheck:
        cap = cast(int, self.max)
        allowed = {str(m) for m in _values(self.allow)}

        output.verbose_print(verbose, f"Counting package imports per module, cap {cap}...")
        counts = {
            module: len(graph.find_modules_directly_imported_by(module)) for module in graph.modules
        }

        over = ((m, n) for m, n in counts.items() if n > cap and m not in allowed)
        unlisted = sorted(over, key=lambda pair: (-pair[1], pair[0]))
        stale = sorted(m for m in allowed & set(counts) if counts[m] <= cap)
        unknown = sorted(allowed - set(counts))

        return ContractCheck(
            kept=not (unlisted or stale or unknown),
            metadata={"cap": cap, "unlisted": unlisted, "stale": stale, "unknown": unknown},
        )

    def render_broken_contract(self, check: ContractCheck) -> None:
        cap = check.metadata["cap"]
        if check.metadata["unlisted"]:
            output.print_error(f"Imports more than {cap} package modules:")
            output.new_line()
            for module, count in check.metadata["unlisted"]:
                output.print_error(f"    {module} ({count})", bold=False)
            output.new_line()
            output.print_error(
                "    Split it, or add it to `allow` with a reason and a plan.",
                bold=False,
            )
            output.new_line()
        if check.metadata["stale"]:
            output.print_error(f"Allowed, but now within {cap} (drop the entry):")
            output.new_line()
            for module in check.metadata["stale"]:
                output.print_error(f"    {module}", bold=False)
            output.new_line()
        if check.metadata["unknown"]:
            output.print_error("Allowed, but no such module (typo?):")
            output.new_line()
            for module in check.metadata["unknown"]:
                output.print_error(f"    {module}", bold=False)
            output.new_line()
