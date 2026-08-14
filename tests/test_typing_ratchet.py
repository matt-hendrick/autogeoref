"""The not-yet-strict list only ever shrinks, and every entry on it still binds.

The modules that cannot meet `pyproject.toml`'s strict pass are named in
per-module overrides — a reviewable claim, like `.importlinter`'s allowlists.
A list nobody checks becomes a dumping ground, so each claim is executed: mypy
is re-run with the weakenings stripped, and an entry naming a rule the module
no longer breaks fails here. Fixing a module means deleting its entry.

Counting entries is not enough. The codes are counted too — a longer entry is
as much a loosening as a new one — and the settings an override may carry are
an allowlist, because any OTHER weakening (`ignore_errors`, `follow_imports`)
turns a module off without touching either number.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parent.parent

#: The counts at adoption. Lower them as you fix; never raise either.
ADOPTED_MODULES = 32
ADOPTED_CODES = 53

#: The only settings a per-module override may carry. Anything else — most of
#: all `ignore_errors` — silences a module without appearing in either count.
PERMITTED_SETTINGS = frozenset(
    {"module", "disable_error_code", "ignore_missing_imports", "disallow_untyped_calls"}
)

#: The drift a caller cannot survive: a symbol that moved, a parameter added or
#: removed, a parameter whose type changed, and a return whose type changed
#: (which lands as `assignment` or `misc` at the call site). `tests/` may
#: disable these while it catches up; the package and the live scripts may not.
MUST_STAY_ENABLED = frozenset({"attr-defined", "call-arg", "arg-type", "assignment", "misc"})


def _mypy() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        settings: dict[str, Any] = tomllib.load(handle)["tool"]["mypy"]
    return settings


def _overrides() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = _mypy().get("overrides", [])
    return entries


def _ratchet() -> dict[str, set[str]]:
    """Module -> the error codes its override switches off."""
    out: dict[str, set[str]] = {}
    for entry in _overrides():
        for module in entry.get("module", []) if entry.get("disable_error_code") else []:
            out[str(module)] = set(entry["disable_error_code"])
    return out


def _untyped_callee_modules() -> set[str]:
    """Modules exempted because something they call ships no annotations."""
    return {
        str(module)
        for entry in _overrides()
        if entry.get("disallow_untyped_calls") is False
        for module in entry.get("module", [])
    }


def test_the_not_yet_strict_list_is_at_or_below_its_adopted_size() -> None:
    """The whole point of a ratchet: it turns one way."""
    assert len(_ratchet()) <= ADOPTED_MODULES


def test_the_adopted_size_is_not_stale() -> None:
    """A ceiling left above the real count silently buys room for a new entry."""
    assert len(_ratchet()) == ADOPTED_MODULES


def test_the_disabled_codes_are_at_or_below_their_adopted_count() -> None:
    """Appending a code to an entry loosens the gate without adding a module."""
    assert sum(len(codes) for codes in _ratchet().values()) <= ADOPTED_CODES


def test_the_adopted_code_count_is_not_stale() -> None:
    """Same reason as the module ceiling: headroom is a loosening nobody reviews."""
    assert sum(len(codes) for codes in _ratchet().values()) == ADOPTED_CODES


def test_no_override_carries_a_setting_outside_the_allowlist() -> None:
    """`ignore_errors = true` would pass every count above and check nothing."""
    stray = sorted(
        f"{entry['module']}: {key}"
        for entry in _overrides()
        for key in entry
        if key not in PERMITTED_SETTINGS
    )

    assert stray == [], f"not a ratchet knob; state the codes instead: {stray}"


def _weakened_modules() -> set[str]:
    """Every module a per-module override lets off a rule."""
    return set(_ratchet()) | _untyped_callee_modules()


def test_only_test_modules_are_exempt_from_a_rule() -> None:
    """`src/` and `scripts/` are strict outright; the list is a `tests/` fact.

    The two package modules on the untyped-callee block are named, not derived:
    they call pmtiles, and that is a third-party fact rather than a repair owed.
    """
    allowed = {"autogeoref.tiles", "autogeoref.viewer.bounds"}
    outside = sorted(
        m
        for m in _weakened_modules()
        if m not in allowed and not (ROOT / "tests" / f"{m}.py").is_file()
    )

    assert outside == [], f"the package and the live scripts hold no exemptions: {outside}"


def test_no_live_script_disables_the_codes_that_catch_a_broken_caller() -> None:
    """A `scripts/` exemption for these lets an API change land green over a caller.

    `tests/test_script_imports.py` states the same contract for the global
    settings; this is the per-module half, which is where a code can now hide.
    """
    scripts = {p.stem for p in (ROOT / "scripts").rglob("*.py")}
    offending = sorted(
        f"{module}: {sorted(codes & MUST_STAY_ENABLED)}"
        for module, codes in _ratchet().items()
        if module in scripts and codes & MUST_STAY_ENABLED
    )

    assert offending == [], f"signature drift would land unseen: {offending}"


def _module_name(relative: Path) -> str:
    """What mypy calls a file, given no `__init__.py` under `tests/` or `scripts/`."""
    if relative.parts[0] == "src":
        return ".".join(relative.with_suffix("").parts[1:])
    return relative.stem


def _run_mypy(config: Path, cache: Path) -> dict[str, set[str]]:
    """Every error code mypy reports, keyed by module."""
    result = subprocess.run(
        ["mypy", "--config-file", str(config), "--cache-dir", str(cache)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    found: dict[str, set[str]] = defaultdict(set)
    for line in result.stdout.splitlines():
        match = re.match(r"^(\S+\.py):\d+: error: .*\[([a-z-]+)\]$", line)
        if match:
            found[_module_name(Path(match.group(1)))].add(match.group(2))
    assert found, f"mypy reported nothing, so nothing was executed:\n{result.stdout[-2000:]}"
    return found


def _config(tmp_path: Path) -> Path:
    """pyproject's settings, minus every weakening whose necessity is being tested.

    The third-party `ignore_missing_imports` entries are the exception and are
    kept: dropping them would make shapely and pyproj `Any` and SWALLOW the very
    errors the list claims, reporting a live entry as fixed.
    """
    settings = _mypy()
    lines = [
        "[mypy]",
        f"python_version = {settings['python_version']}",
        "strict = True",
        f"warn_unreachable = {settings['warn_unreachable']}",
        f"mypy_path = {':'.join(settings['mypy_path'])}",
        f"files = {', '.join(settings['files'])}",
        f"exclude = {settings['exclude']}",
    ]
    for entry in _overrides():
        if not entry.get("ignore_missing_imports"):
            continue
        for module in entry["module"]:
            lines += [f"\n[mypy-{module}]", "ignore_missing_imports = True"]
    config = tmp_path / "strict.ini"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config


@pytest.fixture(scope="module")
def strict_codes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, set[str]]:
    """One mypy run with every weakened rule re-enabled, shared by the tests below."""
    if shutil.which("mypy") is None:
        pytest.skip("mypy is not on PATH")
    tmp = tmp_path_factory.mktemp("ratchet")
    return _run_mypy(_config(tmp), tmp / "cache")


def test_every_disabled_code_is_one_the_module_still_breaks(
    strict_codes: dict[str, set[str]],
) -> None:
    """The behavioural check. A claim that has come true is a failure, not a no-op."""
    stale = {
        module: sorted(codes - strict_codes.get(module, set()))
        for module, codes in _ratchet().items()
        if codes - strict_codes.get(module, set())
    }

    assert stale == {}, f"fixed — delete these from pyproject.toml: {stale}"


def test_the_untyped_callee_exemptions_are_still_needed(
    strict_codes: dict[str, set[str]],
) -> None:
    """A stub-shipping release retires one of these, and nothing else would notice."""
    unneeded = sorted(
        module
        for module in _untyped_callee_modules()
        if "no-untyped-call" not in strict_codes.get(module, set())
    )

    assert unneeded == [], f"no longer untyped — drop from pyproject.toml: {unneeded}"
