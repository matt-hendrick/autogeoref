"""Run one expression against a frontend script in node.

The hand-written frontend keeps its decisions in scripts a test can execute —
`viewer/lib.js`, `review_ui/affine.js`, `queue_ui/board.js`. A literal-source
assertion fails on a refactor and passes on a bug; this runs the code instead.
Needs `node` on PATH; without it the caller skips.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
EVAL_JS = Path(__file__).resolve().parent / "js" / "eval.js"
VIEWER_LIB = ROOT / "viewer" / "lib.js"
REVIEW_AFFINE = ROOT / "src" / "autogeoref" / "review_ui" / "affine.js"
QUEUE_BOARD = ROOT / "src" / "autogeoref" / "queue_ui" / "board.js"


def run_js(script: Path, expression: str) -> Any:
    """``expression`` evaluated in node with ``script``'s exports bound to ``L``."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH")
    done = subprocess.run(
        [node, str(EVAL_JS), str(script), expression],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed on {expression}:\n{done.stderr}")
    return json.loads(done.stdout)


def viewer(expression: str) -> Any:
    """One expression against ``viewer/lib.js``."""
    return run_js(VIEWER_LIB, expression)


def review(expression: str) -> Any:
    """One expression against ``review_ui/affine.js``."""
    return run_js(REVIEW_AFFINE, expression)


def queue(expression: str) -> Any:
    """One expression against ``queue_ui/board.js``."""
    return run_js(QUEUE_BOARD, expression)
