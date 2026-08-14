"""Build the wheel and prove the distribution contract pyproject already declares.

The test suite runs from an editable install, where a wheel that drops a
packaged UI directory or the console entry point would go unnoticed. This
builds the wheel, checks its metadata, entry point, and resource inclusion,
installs it into a fresh venv, then runs ``autogeoref --help`` and loads each
packaged resource through ``importlib.resources`` — the same mechanism the CLI
and dashboard use. Internal distribution contract only; nothing here implies a
published release.

    uv run --no-project python scripts/wheel_smoke.py

Requires ``uv`` on PATH (used for the build, the throwaway venv, and the
install). Exits nonzero with a one-line reason on the first broken check.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

#: UI files the installed package loads through importlib.resources; each must
#: exist and carry content. ``py.typed`` is checked for presence only — it is
#: an intentionally empty marker.
UI_RESOURCES = (
    "dashboard_ui/dashboard.css",
    "dashboard_ui/dashboard.html",
    "queue_ui/board.js",
    "queue_ui/index.html",
    "review_ui/affine.js",
    "review_ui/index.html",
)
REQUIRED_RESOURCES = ("py.typed", *UI_RESOURCES)

CONSOLE_SCRIPT = "autogeoref = autogeoref.cli.entry:main"

PYTHON_VERSION = "3.12"

# Runs inside the throwaway venv, where only the wheel and its runtime
# dependencies exist — a resource the wheel dropped cannot resolve here.
RESOURCE_SMOKE = f"""\
import sys
from importlib.resources import files

root = files("autogeoref")
bad = [rel for rel in {REQUIRED_RESOURCES!r} if not root.joinpath(rel).is_file()]
bad += [rel for rel in {UI_RESOURCES!r} if rel not in bad and not root.joinpath(rel).read_bytes()]
if bad:
    sys.exit("installed package resources missing or empty: " + ", ".join(bad))
"""


def _run(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"wheel smoke: `{' '.join(cmd)}` failed:\n{proc.stdout}{proc.stderr}")
    return proc.stdout


def _check_wheel_contents(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())

        missing = [rel for rel in REQUIRED_RESOURCES if f"autogeoref/{rel}" not in names]
        if missing:
            sys.exit(f"wheel smoke: wheel omits packaged resources: {', '.join(missing)}")

        infos = [n for n in names if re.fullmatch(r"autogeoref-[^/]+\.dist-info/METADATA", n)]
        if len(infos) != 1:
            sys.exit(f"wheel smoke: expected one dist-info METADATA, found {sorted(infos)}")
        metadata = zf.read(infos[0]).decode()
        if "Name: autogeoref" not in metadata.splitlines():
            sys.exit("wheel smoke: METADATA does not name the autogeoref distribution")

        entry_points = infos[0].replace("METADATA", "entry_points.txt")
        if entry_points not in names:
            sys.exit("wheel smoke: wheel ships no entry_points.txt (console script dropped?)")
        declared = zf.read(entry_points).decode()
        if "[console_scripts]" not in declared or CONSOLE_SCRIPT not in declared:
            sys.exit(f"wheel smoke: entry_points.txt lacks `{CONSOLE_SCRIPT}`:\n{declared}")


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    repo = Path(__file__).resolve().parents[1]

    with tempfile.TemporaryDirectory(prefix="wheel-smoke-") as tmpdir:
        tmp = Path(tmpdir)
        dist = tmp / "dist"
        _run(["uv", "build", "--wheel", "--out-dir", str(dist)], cwd=repo)
        wheels = sorted(dist.glob("*.whl"))
        if len(wheels) != 1:
            sys.exit(f"wheel smoke: expected exactly one wheel, found {wheels}")
        wheel = wheels[0]
        _check_wheel_contents(wheel)

        # Fresh venv, cwd outside the repo: nothing from the checkout can
        # shadow the installed package.
        venv = tmp / "venv"
        _run(["uv", "venv", "--python", PYTHON_VERSION, str(venv)], cwd=tmp)
        python = venv / "bin" / "python"
        _run(["uv", "pip", "install", "--python", str(python), str(wheel)], cwd=tmp)

        help_text = _run([str(venv / "bin" / "autogeoref"), "--help"], cwd=tmp)
        if not help_text.startswith("usage:"):
            sys.exit(f"wheel smoke: `autogeoref --help` printed no usage:\n{help_text}")
        _run([str(python), "-c", RESOURCE_SMOKE], cwd=tmp)

    print(f"wheel smoke: OK ({wheel.name}: console script + {len(REQUIRED_RESOURCES)} resources)")


if __name__ == "__main__":
    main()
