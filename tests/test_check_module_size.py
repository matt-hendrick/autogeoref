"""The module-size checker: what it counts, its scope tiers, and the baseline ratchet."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).parent.parent


def _load_script(name: str) -> ModuleType:
    """Import a `scripts/` module by path — `scripts/` is not an installed package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "lint" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_script("check_module_size")

SOURCE = "src/autogeoref/thing.py"


def measure(
    source: str, rel: str = SOURCE, baseline: dict[str, dict[str, int]] | None = None
) -> Any:
    return checker.measure(source, rel, baseline or {})


def body(lines: int) -> str:
    """A module of exactly `lines` statements, all code and none of it public."""
    return "".join(f"_x{n} = {n}\n" for n in range(lines))


def test_blank_lines_are_not_code() -> None:
    assert measure("a = 1\n\n\n\nb = 2\n").lines == 2


def test_whole_line_comments_are_not_code() -> None:
    assert measure("# note\na = 1\n# more\n# and more\n").lines == 1


def test_a_trailing_comment_sits_on_a_line_of_code() -> None:
    """Excluding it would count the line as absent when it is not."""
    assert measure("a = 1  # why\n").lines == 1


def test_docstrings_are_not_code() -> None:
    source = '"""Module.\n\nProse.\n"""\n\n\ndef f() -> None:\n    """One.\n\n    Two.\n    """\n'
    assert measure(source).lines == 1


def test_a_string_that_is_not_a_docstring_is_code() -> None:
    """Only the first statement of a module, class, or def is a docstring."""
    assert measure('a = 1\n"""not a docstring"""\n').lines == 2


def test_the_line_cap_is_what_a_file_with_no_baseline_entry_gets() -> None:
    assert measure(body(checker.LINE_CAP)).violations == []
    assert measure(body(checker.LINE_CAP + 1)).violations == ["MS001"]


def test_a_baseline_entry_raises_the_allowance_to_its_own_number() -> None:
    over = body(checker.LINE_CAP + 50)
    recorded = {SOURCE: {"lines": checker.LINE_CAP + 50, "public": 0}}

    assert measure(over, baseline=recorded).violations == []
    assert measure(over + "_y = 1\n", baseline=recorded).violations == ["MS001"]


def test_a_baseline_entry_never_lowers_an_allowance_below_the_cap() -> None:
    """A file recorded for one rule must not be pinned at today's number for the other."""
    recorded = {SOURCE: {"lines": 900, "public": 2}}

    metrics = measure("".join(f"def f{n}() -> None: pass\n" for n in range(20)), baseline=recorded)

    assert metrics.public == 20
    assert metrics.violations == []


@pytest.mark.parametrize(
    "source",
    [
        "def f() -> None: pass\n",
        "class C: pass\n",
        "NAME = 1\n",
        "NAME: int = 1\n",
    ],
)
def test_public_top_level_symbols_are_counted(source: str) -> None:
    assert measure(source).public == 1


@pytest.mark.parametrize(
    "source",
    [
        "def _f() -> None: pass\n",
        "class _C: pass\n",
        "_NAME = 1\n",
        "def f() -> None:\n    def inner() -> None: pass\n",
    ],
)
def test_private_and_nested_symbols_are_not_public(source: str) -> None:
    assert measure(source).public <= 1
    assert measure(source).public == (1 if "def f() -> None:" in source else 0)


def test_the_public_cap_trips_ms003() -> None:
    wide = "".join(f"def f{n}() -> None: pass\n" for n in range(checker.PUBLIC_CAP + 1))

    assert "MS003" in measure(wide).violations


def test_a_test_module_is_not_measured_for_public_symbols() -> None:
    """Its public surface is its test functions, so the count means nothing."""
    rules, gates = checker._tier("tests/test_thing.py")

    assert gates
    assert "MS001" in rules
    assert "MS003" not in rules


def test_live_experiments_are_scanned_but_never_gate() -> None:
    probe = next((ROOT / "scripts" / "experiments").glob("*.py"))

    rules, gates = checker._tier(probe.relative_to(ROOT).as_posix())

    assert rules
    assert not gates


def test_the_scan_covers_the_gating_trees() -> None:
    """A scan that reads nothing passes forever."""
    scanned = {path.relative_to(ROOT).as_posix() for path in checker.python_files()}

    assert "src/autogeoref/paths.py" in scanned
    assert any(path.startswith("scripts/lint/check_") for path in scanned)
    assert any(path.startswith("tests/") for path in scanned)
    assert any(path.startswith("scripts/experiments/") for path in scanned)


def test_a_baseline_entry_for_a_deleted_file_is_stale() -> None:
    measured = checker.scan_repo({})

    assert checker.stale_entries(measured, {"src/autogeoref/gone.py": {"lines": 900, "public": 0}})


def test_a_baseline_entry_for_a_file_now_under_the_caps_is_stale() -> None:
    """The entry records a plan to shrink; keeping it after the shrink hides the next one."""
    small = next(m for m in checker.scan_repo({}) if m.lines < checker.LINE_CAP)

    stale = checker.stale_entries(checker.scan_repo({}), {small.path: {"lines": 900, "public": 0}})

    assert small.path in stale


def test_the_shipped_baseline_holds_no_stale_entries() -> None:
    baseline = checker.load_baseline()

    assert checker.stale_entries(checker.scan_repo(baseline), baseline) == []


def test_the_shipped_tree_passes_its_own_baseline() -> None:
    """The gate is only worth having if it passes, as configured, on the shipped tree."""
    baseline = checker.load_baseline()

    assert checker.gating(checker.scan_repo(baseline)) == []


def test_the_baseline_records_the_tree_it_was_measured_on() -> None:
    """A number with no provenance cannot be re-derived or trusted."""
    recorded = json.loads(checker.BASELINE.read_text(encoding="utf-8"))

    assert recorded["measured_on"] != "unknown"
    assert recorded["modules"]


def test_the_baseline_is_not_empty_of_the_files_it_exists_for() -> None:
    """Every entry names a real file, and every number in it is today's measurement.

    Pinning the numbers is what stops the baseline being hand-inflated into an
    allowance: raising one here fails, so the only way down is a smaller file.
    """
    baseline = checker.load_baseline()
    measured = {m.path: m for m in checker.scan_repo({})}
    fields = {field: rule for rule, (_, field, _cap) in checker.RULES.items()}

    assert baseline
    for path, recorded in baseline.items():
        assert path in measured, f"{path} no longer exists"
        assert recorded
        for field, value in recorded.items():
            assert value == measured[path].actual(fields[field]), f"{path}.{field}"


def test_a_baseline_entry_records_only_the_rules_its_tier_applies() -> None:
    """A test file has no MS003, so recording its public count would refuse a green change."""
    baseline = checker.load_baseline()

    for path, recorded in baseline.items():
        if path.startswith("tests/"):
            assert set(recorded) == {"lines"}, path


def _make_recipe(target: str) -> list[str]:
    """One target's command lines, plus every prerequisite target's, recursively.

    `lint` delegates its recipe to prerequisites, so reading the one target's
    own lines would report a wired gate as absent.
    """
    lines = (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{target}:"))
    body = []
    for line in lines[start + 1 :]:
        if not line.startswith("\t"):
            break
        body.append(line.strip())
    for prerequisite in lines[start].split(":", 1)[1].split():
        body.extend(_make_recipe(prerequisite))
    return body


def test_the_gate_is_wired_into_the_lint_target() -> None:
    """A checker nothing runs is a checker nothing enforces, and CI runs `make lint`."""
    assert any(
        line.endswith("python scripts/lint/check_module_size.py") for line in _make_recipe("lint")
    )


def test_the_report_target_is_not_what_gates() -> None:
    """`--report` exits 0 whatever it finds; wiring that in place of the gate is the mistake."""
    assert all("--report" not in line for line in _make_recipe("lint"))


#: Cap name -> the value it was adopted at. Lower freely, never raise.
ADOPTED = {"LINE_CAP": 400, "PUBLIC_CAP": 30}


@pytest.mark.parametrize("cap", sorted(ADOPTED))
def test_the_cap_is_at_or_below_its_adopted_value(cap: str) -> None:
    """The same ratchet the ruff ceilings get, for the two numbers this file owns."""
    assert getattr(checker, cap) <= ADOPTED[cap]


@pytest.mark.parametrize("rule", ["MS001", "MS003"])
def test_each_rule_is_still_declared(rule: str) -> None:
    """`_tier` derives from `RULES`, so deleting an entry silently retires the rule."""
    assert rule in checker.RULES


def test_ms002_is_not_reissued() -> None:
    """It was fan-out, and `.importlinter` owns that; the code must not mean something else."""
    assert "MS002" not in checker.RULES


def _tree(
    tmp_path: Path, files: dict[str, str], baseline: dict[str, dict[str, int]] | None = None
) -> None:
    """Point the checker at a throwaway repo, so `main` can be run end to end."""
    for rel, source in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    (tmp_path / "baseline.json").write_text(
        json.dumps({"measured_on": "test", "modules": baseline or {}}), encoding="utf-8"
    )


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(checker, "BASELINE", tmp_path / "baseline.json")
    monkeypatch.setattr(checker, "TREES", ("src", "tests"))
    return tmp_path


def test_the_gate_fails_and_names_the_file_that_is_too_long(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The behavioural check: a gate proven only by passing has never been seen to fail."""
    _tree(sandbox, {"src/autogeoref/big.py": body(checker.LINE_CAP + 1)})

    assert checker.main([]) == 1
    assert "MS001" in capsys.readouterr().err


def test_the_gate_fails_and_names_the_file_with_too_wide_a_surface(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wide = "".join(f"def f{n}() -> None: pass\n" for n in range(checker.PUBLIC_CAP + 1))
    _tree(sandbox, {"src/autogeoref/wide.py": wide})

    assert checker.main([]) == 1
    assert "MS003" in capsys.readouterr().err


def test_the_gate_passes_a_file_inside_both_caps(sandbox: Path) -> None:
    _tree(sandbox, {"src/autogeoref/small.py": body(10)})

    assert checker.main([]) == 0


def test_a_baselined_file_passes_until_it_grows(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    over = body(checker.LINE_CAP + 10)
    _tree(
        sandbox,
        {"src/autogeoref/big.py": over},
        baseline={"src/autogeoref/big.py": {"lines": checker.LINE_CAP + 10}},
    )
    assert checker.main([]) == 0

    (sandbox / "src/autogeoref/big.py").write_text(over + "_more = 1\n", encoding="utf-8")

    assert checker.main([]) == 1
    assert "big.py" in capsys.readouterr().err


def test_update_refuses_to_raise_an_existing_number(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _tree(
        sandbox,
        {"src/autogeoref/big.py": body(checker.LINE_CAP + 60)},
        baseline={"src/autogeoref/big.py": {"lines": checker.LINE_CAP + 10}},
    )

    assert checker.main(["--update"]) == 1
    assert "only goes down" in capsys.readouterr().err


def test_update_refuses_a_rename_that_carries_growth(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keyed on the path, the per-file check walks past a rename. The total does not."""
    _tree(
        sandbox,
        {"src/autogeoref/renamed.py": body(checker.LINE_CAP + 200)},
        baseline={"src/autogeoref/original.py": {"lines": checker.LINE_CAP + 10}},
    )

    assert checker.main(["--update"]) == 1
    assert "total baselined lines" in capsys.readouterr().err


def test_update_accepts_a_split_that_lowers_the_total(sandbox: Path) -> None:
    """The point of the whole gate: one big file becoming several smaller ones."""
    _tree(
        sandbox,
        {
            "src/autogeoref/part_a.py": body(300),
            "src/autogeoref/part_b.py": body(300),
        },
        baseline={"src/autogeoref/whole.py": {"lines": 900}},
    )

    assert checker.main(["--update"]) == 0
    assert checker.load_baseline() == {}


def test_a_barrel_is_measured_by_what_it_re_exports(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A package `__init__` re-exporting everything is the facade the convention forbids."""
    barrel = "".join(f"from .m{n} import name{n}\n" for n in range(checker.PUBLIC_CAP + 1))
    _tree(sandbox, {"src/autogeoref/thing/__init__.py": barrel})

    assert checker.main([]) == 1
    assert "MS003" in capsys.readouterr().err


def test_a_declared_all_is_the_public_surface(sandbox: Path) -> None:
    """`__all__` is the module's own statement of its API; believe it over the defs."""
    names = ", ".join(f'"n{n}"' for n in range(checker.PUBLIC_CAP + 1))

    metrics = measure(f"__all__ = [{names}]\n")

    assert metrics.public == checker.PUBLIC_CAP + 1


def test_an_ordinary_module_is_not_charged_for_its_imports() -> None:
    """Only a barrel re-exports; elsewhere an import is fan-out, which `.importlinter` owns."""
    source = "".join(f"from x import name{n}\n" for n in range(40))

    assert measure(source).public == 0


def test_an_unreadable_file_is_reported_not_raised(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One bad file must not take the whole gate down with a traceback."""
    _tree(sandbox, {"src/autogeoref/broken.py": "def (\n"})

    assert checker.main([]) == 1
    assert "unreadable" in capsys.readouterr().err


def test_a_byte_order_mark_is_not_a_syntax_error() -> None:
    """`utf-8` keeps the BOM as a character and `ast.parse` then rejects the file."""
    source = ROOT / "scripts" / "lint" / "check_module_size.py"
    encoded = ("﻿" + source.read_text(encoding="utf-8")).encode("utf-8")

    assert encoded.decode("utf-8-sig")[0] != "﻿"


def test_a_corrupt_baseline_is_a_named_failure(sandbox: Path) -> None:
    (sandbox / "baseline.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(SystemExit, match="unreadable"):
        checker.load_baseline()


def test_scripts_proper_is_governed_like_the_package() -> None:
    """An ungoverned tree is where code hides, and only experiments are exempt."""
    assert "scripts" in checker.TREES
    assert checker._tier("scripts/lint/check_module_size.py") == (frozenset(checker.RULES), True)
    assert checker._tier("scripts/experiments/water.py") == (frozenset(checker.RULES), False)
