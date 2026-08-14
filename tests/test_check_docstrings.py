"""The docstring budget checker: rules, scope tiers, escape hatch, and ratchet."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

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


checker = _load_script("check_docstrings")


def rules_hit(source: str, rel: str = "src/autogeoref/thing.py") -> list[str]:
    return [v.rule for v in checker.check_source(source, rel)]


def long_docstring(lines: int) -> str:
    """A `def` whose docstring spans exactly `lines` source lines."""
    body = "\n".join(f"line {n}" for n in range(lines - 3))
    return f'def f() -> None:\n    """Summary.\n\n{body}\n    """\n'


def test_short_docstring_passes() -> None:
    source = 'def f() -> None:\n    """Return nothing. Takes nothing."""\n'
    assert rules_hit(source) == []


def test_long_function_docstring_trips_ds002() -> None:
    assert "DS002" in rules_hit(long_docstring(20))


def test_function_docstring_at_the_cap_passes() -> None:
    assert "DS002" not in rules_hit(long_docstring(checker.DEF_DOC_MAX))


def test_long_module_docstring_trips_ds001() -> None:
    source = '"""Summary.\n' + "\n".join(f"line {n}" for n in range(20)) + '\n"""\n'
    assert "DS001" in rules_hit(source)


def test_module_docstring_at_the_cap_passes() -> None:
    body = "\n".join(f"line {n}" for n in range(checker.MODULE_DOC_MAX - 2))
    assert "DS001" not in rules_hit(f'"""Summary.\n{body}\n"""\n')


def test_long_comment_block_trips_ds003() -> None:
    source = "".join(f"# note {n}\n" for n in range(9)) + "x = 1\n"
    assert "DS003" in rules_hit(source)


def test_comment_block_at_the_cap_passes() -> None:
    source = "".join(f"# note {n}\n" for n in range(checker.COMMENT_BLOCK_MAX)) + "x = 1\n"
    assert "DS003" not in rules_hit(source)


def test_a_blank_line_ends_a_comment_block() -> None:
    half = "".join(f"# note {n}\n" for n in range(5))
    assert "DS003" not in rules_hit(f"{half}\n{half}x = 1\n")


def test_trailing_comments_do_not_form_a_block() -> None:
    source = "".join(f"x{n} = 1  # note {n}\n" for n in range(9))
    assert "DS003" not in rules_hit(source)


@pytest.mark.parametrize(
    ("prose", "rule"),
    [
        ("see planning_docs/ for evidence", "DS101"),
        ("measured 2026-07-31", "DS102"),
        ("closes [114]", "DS103"),
        ("only on sanborn01790_024", "DS104"),
        ("bare id _024", "DS104"),
        ("the grid north of the Loop", "DS105"),
        ("Cicero renumbered first", "DS105"),
    ],
)
def test_prose_rules_fire_in_a_docstring(prose: str, rule: str) -> None:
    assert rule in rules_hit(f'def f() -> None:\n    """Do a thing. {prose}."""\n')


@pytest.mark.parametrize(
    ("prose", "rule"),
    [
        ("see planning_docs/ for evidence", "DS101"),
        ("measured 2026-07-31", "DS102"),
        ("closes [114]", "DS103"),
    ],
)
def test_prose_rules_fire_in_a_comment(prose: str, rule: str) -> None:
    assert rule in rules_hit(f"# {prose}\nx = 1\n")


def test_indexing_is_not_a_roadmap_number() -> None:
    assert "DS103" not in rules_hit('def f() -> None:\n    """Return rows[0] and cols[12]."""\n')


def test_underscored_identifiers_are_not_volume_ids() -> None:
    assert "DS104" not in rules_hit('def f() -> None:\n    """Read page_012 from disk."""\n')


def test_tests_may_name_the_corpus_but_not_dates() -> None:
    source = (
        "def test_x() -> None:\n"
        '    """Replay sanborn01790_024 of Chicago as frozen 2026-07-31."""\n'
    )
    hit = rules_hit(source, "tests/test_x.py")
    assert "DS104" not in hit
    assert "DS105" not in hit
    assert "DS102" in hit


def test_ds106_is_advisory_only() -> None:
    source = 'def f() -> None:\n    """Do a thing; the ratio approach was refuted."""\n'
    assert rules_hit(source) == ["DS106"]


def test_noqa_on_the_def_line_suppresses_one_rule() -> None:
    source = long_docstring(20).replace("def f() -> None:", "def f() -> None:  # noqa: DS002")
    assert "DS002" not in rules_hit(source)


def test_noqa_is_rule_specific() -> None:
    source = 'def f() -> None:  # noqa: DS002\n    """Do it, per planning_docs/."""\n'
    assert rules_hit(source) == ["DS101"]


def test_noqa_above_a_module_docstring_suppresses_ds001() -> None:
    source = "# noqa: DS001\n" + '"""Summary.\n' + "\n".join("x" for _ in range(20)) + '\n"""\n'
    assert "DS001" not in rules_hit(source)


def test_experiments_are_scanned_but_never_gate() -> None:
    source = long_docstring(20)
    rel = "scripts/experiments/probe_thing.py"  # <!-- no-cite -->
    assert "DS002" in rules_hit(source, rel)
    assert checker.gating(checker.check_source(source, rel)) == []


def test_unparseable_source_is_reported_not_raised() -> None:
    assert rules_hit("def f(\n") == ["DS001"]


def test_the_gating_trees_are_at_zero() -> None:
    """The whole point of the cleanup: any violation now fails the build."""
    assert checker.gating(checker.scan_repo()) == []


def test_experiments_are_scanned_but_excluded_from_the_gate() -> None:
    scanned = {v.path for v in checker.scan_repo(include_experiments=True)}
    assert any(p.startswith("scripts/experiments/") for p in scanned)
    assert not any(
        v.path.startswith("scripts/experiments/")
        for v in checker.gating(checker.scan_repo(include_experiments=True))
    )


def test_advisory_rules_never_gate() -> None:
    source = 'def f() -> None:\n    """Do a thing; the ratio approach was refuted."""\n'
    assert checker.gating(checker.check_source(source, "src/a.py")) == []
