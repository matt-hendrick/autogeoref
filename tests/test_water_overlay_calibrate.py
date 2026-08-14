"""Contracts for the waterway calibration experiment's bind bar."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any


def _summary() -> Any:
    script = Path(__file__).parents[1] / "scripts" / "water_overlay_calibrate.py"
    return runpy.run_path(str(script))["summary"]


def test_calibration_summary_requires_both_offset_limits() -> None:
    summary = _summary()

    passing = [{"offset_m": 4.0, "ground_truth": True}, {"offset_m": 6.0, "ground_truth": False}]
    failing = [{"offset_m": 5.0, "ground_truth": False}, {"offset_m": 12.01, "ground_truth": False}]

    assert summary(passing)["bindable"]
    assert not summary(failing)["bindable"]
