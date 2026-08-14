"""Minimal file-target stage runner: idempotent, resume-safe, dual-marker.

Key guarantees:

1. **Idempotence / resume**: a stage is skipped when all its outputs exist
   and are at least as new as every input — re-running a crashed volume
   reproduces or skips, never duplicates.
2. **Dual markers**: every stage emits a machine-readable marker on BOTH
   success and failure — a monitor watching only for success cannot
   distinguish a crash from slow progress.

"""

from __future__ import annotations

import json
import logging
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .paths import atomic_write_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage:
    """One idempotent pipeline stage with declared file targets."""

    name: str
    run: Callable[[], object]  # return value ignored; lambdas may return freely
    inputs: Sequence[Path] = field(default_factory=tuple)
    outputs: Sequence[Path] = field(default_factory=tuple)
    #: Run despite fresh-looking outputs when the stage owns content comparison itself.
    always_run: bool = False
    #: Stage is skipped (with a marker) when this returns False.
    enabled: Callable[[], bool] = lambda: True


@dataclass(frozen=True)
class StageResult:
    name: str
    status: str  # "ok" | "skipped" | "fresh" | "failed" | "disabled"
    elapsed_s: float
    error: str | None = None


def _mtime(p: Path) -> float:
    return p.stat().st_mtime


def is_fresh(stage: Stage) -> bool:
    """True when all outputs exist and none is older than any input.

    Wall-clock mtimes, and a wall clock can step backwards, so an output can
    read as older than the input it was made from. That answer re-runs the
    stage, which is safe. The unsafe direction needs an input rewritten within
    one clock step of the output that consumed it; pass ``force`` for that.
    """
    if not stage.outputs:
        return False
    if not all(p.exists() for p in stage.outputs):
        return False
    existing_inputs = [p for p in stage.inputs if p.exists()]
    if not existing_inputs:
        return True
    newest_input = max(_mtime(p) for p in existing_inputs)
    oldest_output = min(_mtime(p) for p in stage.outputs)
    return oldest_output >= newest_input


class Runner:
    """Executes stages in order, writing a marker JSON per stage attempt."""

    def __init__(self, marker_dir: Path) -> None:
        self.marker_dir = marker_dir

    def _write_marker(self, result: StageResult, started: float) -> None:
        # created lazily so a --dry-run leaves no trace on disk
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        marker = {
            "stage": result.name,
            "status": result.status,
            "started": started,
            "finished": time.time(),
            "elapsed_s": round(result.elapsed_s, 3),
            "error": result.error,
        }
        path = self.marker_dir / f"{result.name}.marker.json"
        atomic_write_text(path, json.dumps(marker, indent=2))

    def _write_running_marker(self, name: str, started: float) -> None:
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.marker_dir / f"{name}.marker.json",
            json.dumps({"stage": name, "status": "running", "started": started}, indent=2),
        )

    def execute(
        self,
        stages: Sequence[Stage],
        dry_run: bool = False,
        force: bool = False,
        stop_on_failure: bool = True,
    ) -> list[StageResult]:
        results: list[StageResult] = []
        #: Outputs earlier stages would produce during this dry-run plan.
        planned: set[Path] = set()
        for stage in stages:
            started = time.time()
            if not stage.enabled():
                result = StageResult(stage.name, "disabled", 0.0)
                results.append(result)
                if not dry_run:
                    self._write_marker(result, started)
                continue
            if (
                not force
                and not stage.always_run
                and not any(path in planned for path in stage.inputs)
                and is_fresh(stage)
            ):
                result = StageResult(stage.name, "fresh", 0.0)
                results.append(result)
                logger.info("%s: outputs fresh, skipping", stage.name)
                if not dry_run:
                    self._write_marker(result, started)
                continue
            missing = [
                str(p) for p in stage.inputs if not p.exists() and not (dry_run and p in planned)
            ]
            if missing:
                result = StageResult(stage.name, "failed", 0.0, error=f"missing inputs: {missing}")
                results.append(result)
                if not dry_run:
                    self._write_marker(result, started)
                if stop_on_failure:
                    break
                continue
            if dry_run:
                planned.update(stage.outputs)
                results.append(StageResult(stage.name, "skipped", 0.0))
                logger.info("%s: would run (dry-run)", stage.name)
                continue
            logger.info("%s: running", stage.name)
            self._write_running_marker(stage.name, started)
            try:
                stage.run()
            except Exception as exc:  # noqa: BLE001 - a stage failure is a result, not a crash
                result = StageResult(
                    stage.name,
                    "failed",
                    time.time() - started,
                    error=f"{exc}\n{traceback.format_exc()}",
                )
                results.append(result)
                self._write_marker(result, started)
                logger.error("%s: FAILED: %s", stage.name, exc)
                if stop_on_failure:
                    break
                continue
            result = StageResult(stage.name, "ok", time.time() - started)
            results.append(result)
            self._write_marker(result, started)
            logger.info("%s: ok (%.1fs)", stage.name, result.elapsed_s)
        return results
