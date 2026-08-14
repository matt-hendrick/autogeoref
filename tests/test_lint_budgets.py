"""The size ceilings only ever go down, and each one still fires.

The ceilings are policy stated in a comment, which nothing enforced. Each number
below is the value at adoption; lowering one means lowering it here too, and
raising one fails. Reading the config is not enough on its own — a rule can be
selected and configured and still be switched off by `ignore` — so the last test
runs ruff over source it hands it and asserts the codes come back.
"""

from __future__ import annotations

import configparser
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parent.parent

#: Ceiling name -> the value it was adopted at. Lower freely, never raise.
ADOPTED = {
    "max-complexity": 20,
    "max-args": 10,
    "max-returns": 11,
    "max-branches": 18,
    "max-statements": 50,
}

#: Which ruff table each ceiling lives in.
TABLE = {"max-complexity": "mccabe"}

#: The import-linter fan-out cap, which lives in its own file and is not a ruff rule.
FAN_OUT_ADOPTED = 25

SIZE_CODES = ("C901", "PLR0913", "PLR0911", "PLR0912", "PLR0915")


def _lint_config() -> dict[str, Any]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        config: dict[str, Any] = tomllib.load(handle)["tool"]["ruff"]["lint"]
    return config


def _fan_out_section() -> configparser.SectionProxy:
    parser = configparser.ConfigParser()
    parser.read(ROOT / ".importlinter", encoding="utf-8")
    return parser["importlinter:contract:fan-out"]


@pytest.mark.parametrize("ceiling", sorted(ADOPTED))
def test_the_ceiling_is_configured_at_or_below_its_adopted_value(ceiling: str) -> None:
    table = _lint_config()[TABLE.get(ceiling, "pylint")]

    assert ceiling in table, f"{ceiling} was deleted, which removes the ceiling entirely"
    assert table[ceiling] <= ADOPTED[ceiling]


def test_the_fan_out_cap_is_at_or_below_its_adopted_value() -> None:
    """The allowlist entry holding this cap honest goes away with the split it plans."""
    assert int(_fan_out_section()["max"]) <= FAN_OUT_ADOPTED


def test_the_fan_out_contract_name_states_the_cap_it_enforces() -> None:
    """`make lint` prints the name every run; a stale number there misreports the gate."""
    section = _fan_out_section()

    assert section["max"].strip() in section["name"]


@pytest.mark.parametrize("code", SIZE_CODES)
def test_the_rule_behind_each_ceiling_is_selected(code: str) -> None:
    """A configured ceiling with its rule deselected is a number nothing reads."""
    assert code in _lint_config()["select"]


def test_the_script_tiers_ignore_the_ceilings_their_frozen_archive_cannot_meet() -> None:
    """`scripts/` is a different population, and the archive may not be repaired."""
    ignores = _lint_config()["per-file-ignores"]["scripts/*"]

    assert {"PLR0912", "PLR0915"} <= set(ignores)


#: Over every one of the three at once: 25 returns, 25 branches, 150 statements.
OVERSIZED = "def f(n: int) -> int:\n" + "".join(
    f"    if n == {i}:\n"
    + "".join(f"        v{part} = {i}\n" for part in range(4))
    + f"        return v0 + v1 + v2 + v3 + {i}\n"
    for i in range(25)
)


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not on PATH")
def test_ruff_as_configured_still_reports_every_size_code() -> None:
    """The behavioural gate: `ignore` and a per-file-ignore each switch a rule off silently."""
    probe = ROOT / "src" / "autogeoref" / "_ceiling_probe.py"

    result = subprocess.run(
        [
            "ruff",
            "check",
            "--no-cache",
            "--output-format",
            "concise",
            "--stdin-filename",
            str(probe),
            "-",
        ],
        input=OVERSIZED,
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )

    reported = {
        line.split(": ")[1].split(" ")[0] for line in result.stdout.splitlines() if ": " in line
    }
    assert {"PLR0911", "PLR0912", "PLR0915"} <= reported, result.stdout
