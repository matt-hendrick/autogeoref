"""Contracts for the replay/validate harness's read-selection and window.

Which cached reads the harness replays IS the measurement. Miss a spelling and
the primary-vs-second-read split it prints becomes vacuous; count a marker or a
pointer and the replay reports wins that no reading supports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autogeoref.volume import constraints_from_constants

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "replay_and_validate.py"


def _module() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("replay_and_validate", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_both_spellings_of_the_primary_read_are_found(tmp_path: Path) -> None:
    """The legacy bare read counts as a primary, or older volumes look uncached.

    The harness splits its findings into alias/index drift (the PRIMARY read
    places the page today) and genuine second-read wins. Missing the bare
    ``p<N>.json`` spelling would file every older volume's win under the wrong
    half of that split.
    """
    reads = _module()._cached_reads
    ann = tmp_path
    (ann / "p7.json").write_text("{}")
    (ann / "p7.annotation.some-model.json").write_text("{}")

    found = reads(ann, "7")
    assert [p.name for _producer, p in found] == ["p7.json", "p7.annotation.some-model.json"]
    assert {producer for producer, _p in found} == {"annotation"}


def test_pointers_and_failure_markers_are_not_reads(tmp_path: Path) -> None:
    """A pointer holds no reading, and a marker is a refusal this must not spend past."""
    reads = _module()._cached_reads
    ann = tmp_path
    (ann / "p7.annotation.active.json").write_text("{}")
    (ann / "p7.escalated.some-model.failed.json").write_text("{}")
    (ann / "p7.v2.some-model.json").write_text("{}")

    assert [(producer, p.name) for producer, p in reads(ann, "7")] == [
        ("v2", "p7.v2.some-model.json")
    ]


def test_every_producer_is_replayed_and_no_other_page_leaks_in(tmp_path: Path) -> None:
    """All three producers, this page only — a neighbour's cache is not evidence."""
    module = _module()
    ann = tmp_path
    for producer in module._PRODUCERS:
        (ann / f"p7.{producer}.some-model.json").write_text("{}")
    (ann / "p70.annotation.some-model.json").write_text("{}")

    assert [producer for producer, _p in module._cached_reads(ann, "7")] == list(module._PRODUCERS)


def test_the_persisted_volume_constants_outrank_the_config(tmp_path: Path) -> None:
    """A replay must reuse the window the finished run used, not re-derive one.

    The persisted constants are what the recorded placements were fitted under;
    replaying against the config's declared pair would judge the cached reads by
    a window the volume never ran with.
    """
    module = _module()
    volume = tmp_path / "vol"
    volume.mkdir()

    class _Config:
        scale_m_per_px = 2.0
        rotation_deg = 10.0

    config = _Config()
    assert module._constraints(volume, config) == constraints_from_constants(2.0, 10.0)

    (volume / "volume-constants.json").write_text(
        json.dumps({"scale_m_per_px": 0.5, "rotation_deg": -88.0})
    )
    assert module._constraints(volume, config) == constraints_from_constants(0.5, -88.0)


def test_a_volume_with_no_window_at_all_is_skipped_not_guessed(tmp_path: Path) -> None:
    """No persisted constants and no configured pair means no replay for this volume."""
    module = _module()
    volume = tmp_path / "vol"
    volume.mkdir()

    class _Config:
        scale_m_per_px = None
        rotation_deg = None

    assert module._constraints(volume, _Config()) is None


@pytest.mark.parametrize(
    ("direction", "bearing"), [("E", 0.0), ("N", 90.0), ("W", 180.0), ("S", -90.0)]
)
def test_the_margin_bearings_are_compass_directions(direction: str, bearing: float) -> None:
    """Margin adjacency compares against these, so a swapped pair inverts a verdict."""
    module = _module()
    assert module._BEARINGS[direction] == bearing
    assert 0 < module._BEARING_TOL_DEG < 90
