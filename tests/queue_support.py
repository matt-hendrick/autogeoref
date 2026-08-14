"""Builders shared by the queue test modules.

A queue entry needs a work tree to name, a drain needs a city and a publication
target, and almost every drain test needs to see what the drain spawned without
spawning it. These are the pieces that do that; the per-subject spies live with
the tests that use them.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from autogeoref.queue import command as qcommand
from autogeoref.queue import store as qstore
from autogeoref.viewer import publish as viewer_publish

if TYPE_CHECKING:
    import pytest


def _volume(work: Path, name: str, *, results: bool = False) -> Path:
    vol = work / name
    (vol / "sheets").mkdir(parents=True)
    (vol / "sheets" / "manifest.json").write_text(
        json.dumps({"p1": {"scale": 0.5}, "p2": {"scale": 0.5}, "_orientation_normalized": True})
    )
    if results:
        (vol / "results").mkdir()
        (vol / "results" / "p1.json").write_text(
            json.dumps({"page": "1", "status": "OK", "gcps_geojson": {"type": "FeatureCollection"}})
        )
        (vol / "results" / "p2.json").write_text(
            json.dumps({"page": "2", "status": "REJECTED (no valid RANSAC model)"})
        )
    return vol


CITY = Path("configs/chicago/chicago.toml")


def _ctx(work: Path, city: Path = CITY, **kwargs: object) -> qcommand.DrainContext:
    """A DrainContext for direct `_command` / `_run_leg` calls."""
    return qcommand.DrainContext(work=work, city=city, **kwargs)  # type: ignore[arg-type]


def _publication(work: Path) -> viewer_publish.PublicationConfig:
    return viewer_publish.PublicationConfig(
        work=work,
        city_toml=CITY,
        manifest=work / "viewer" / "chicago-ill" / "manifest.json",
        viewer_root=work / "viewer",
        tiles_root=work / "tiles",
    )


class _FakeProc:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def _spy(monkeypatch: pytest.MonkeyPatch, code: int = 0) -> list[list[str]]:
    """Record every command a drain spawns, in order."""
    spawned: list[list[str]] = []

    def _fake(cmd: list[str], **_k: object) -> _FakeProc:
        spawned.append(list(cmd))
        return _FakeProc(code)

    monkeypatch.setattr(subprocess, "run", _fake)
    return spawned


def _by(entries: list[qstore.QueueEntry]) -> dict[tuple[str, str], str]:
    return {(e.volume, e.track): e.status for e in entries}
