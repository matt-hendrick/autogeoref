"""No module in the package may take part in an import cycle.

Complements ``.importlinter``, which pins the DIRECTION of a named spine, and
``test_import_boundaries.py``, which pins import COST. This one is total: it
walks every module pair, so it catches a cycle nobody thought to enumerate.
"""

from __future__ import annotations

import grimp


def test_package_has_no_import_cycles() -> None:
    graph = grimp.build_graph("autogeoref")
    modules = sorted(graph.modules)
    successors = {m: sorted(graph.find_modules_directly_imported_by(m)) for m in modules}

    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    counter = 0
    cycles: list[list[str]] = []

    for root in modules:
        if root in index:
            continue
        # iterative Tarjan: the graph is ~100 modules but the recursion depth
        # of a naive version tracks the longest chain, not the module count
        work: list[tuple[str, list[str]]] = [(root, list(successors[root]))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, pending = work[-1]
            descended = False
            while pending:
                nxt = pending.pop(0)
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, list(successors[nxt])))
                    descended = True
                    break
                if nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            if descended:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    popped = stack.pop()
                    on_stack.discard(popped)
                    component.append(popped)
                    if popped == node:
                        break
                if len(component) > 1:
                    cycles.append(sorted(component))

    assert not cycles, "import cycle(s) — break one edge, do not add a lazy import: " + "; ".join(
        " <-> ".join(c) for c in sorted(cycles)
    )
