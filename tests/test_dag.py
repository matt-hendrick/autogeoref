"""Contracts for the stage runner: idempotence, resume, dual markers."""

import json
from pathlib import Path

import pytest

from autogeoref import paths
from autogeoref.dag import Runner, Stage, StageResult, is_fresh
from conftest import antedate


def _touch(p: Path, content: str = "x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def test_marker_written_on_success_and_failure(tmp_path: Path) -> None:
    ok_out = tmp_path / "out" / "a.txt"

    def boom() -> None:
        raise RuntimeError("kapow")

    stages = [
        Stage(name="good", run=lambda: _touch(ok_out), outputs=[ok_out]),
        Stage(name="bad", run=boom),
    ]
    runner = Runner(tmp_path / "markers")
    results = runner.execute(stages, stop_on_failure=False)
    assert [r.status for r in results] == ["ok", "failed"]
    good = json.loads((tmp_path / "markers" / "good.marker.json").read_text())
    bad = json.loads((tmp_path / "markers" / "bad.marker.json").read_text())
    assert good["status"] == "ok" and good["error"] is None
    assert bad["status"] == "failed" and "kapow" in bad["error"]


def test_failed_marker_replacement_preserves_previous_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = Runner(tmp_path / "markers")
    result = StageResult("stage", "ok", 0.0)
    runner._write_marker(result, started=0.0)
    marker = tmp_path / "markers" / "stage.marker.json"
    old_text = marker.read_text()

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replacement failed")

    monkeypatch.setattr(paths.Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        runner._write_marker(result, started=1.0)

    assert marker.read_text() == old_text
    assert json.loads(marker.read_text())["status"] == "ok"


def test_fresh_outputs_skip_and_force_reruns(tmp_path: Path) -> None:
    src = _touch(tmp_path / "in.txt")
    antedate(src)  # the input predates the output; do not ask the wall clock
    out = tmp_path / "out.txt"
    calls: list[int] = []

    def run() -> None:
        calls.append(1)
        _touch(out)

    stage = Stage(name="s", run=run, inputs=[src], outputs=[out])
    runner = Runner(tmp_path / "markers")
    assert runner.execute([stage])[0].status == "ok"
    assert runner.execute([stage])[0].status == "fresh"
    assert len(calls) == 1
    assert runner.execute([stage], force=True)[0].status == "ok"
    assert len(calls) == 2


def test_stale_output_reruns(tmp_path: Path) -> None:
    src = _touch(tmp_path / "in.txt")
    out = _touch(tmp_path / "out.txt")
    import os

    os.utime(out, (out.stat().st_atime, src.stat().st_mtime - 100))
    stage = Stage(name="s", run=lambda: _touch(out), inputs=[src], outputs=[out])
    assert not is_fresh(stage)
    # and the boundary the comparison is written around: two writes the kernel
    # clock could not tell apart are FRESH, or a coarse tick would re-run a stage
    antedate(src, out)
    assert is_fresh(stage)


def test_missing_input_fails_with_marker(tmp_path: Path) -> None:
    stage = Stage(
        name="s", run=lambda: None, inputs=[tmp_path / "nope.txt"], outputs=[tmp_path / "o"]
    )
    runner = Runner(tmp_path / "markers")
    results = runner.execute([stage])
    assert results[0].status == "failed"
    assert "missing inputs" in (results[0].error or "")
    marker = json.loads((tmp_path / "markers" / "s.marker.json").read_text())
    assert marker["status"] == "failed"


def test_stop_on_failure_halts_chain(tmp_path: Path) -> None:
    ran: list[str] = []

    def boom() -> None:
        raise RuntimeError("x")

    stages = [
        Stage(name="a", run=lambda: ran.append("a")),
        Stage(name="b", run=boom),
        Stage(name="c", run=lambda: ran.append("c")),
    ]
    results = Runner(tmp_path / "m").execute(stages)
    assert ran == ["a"]
    assert [r.name for r in results] == ["a", "b"]


def test_dry_run_runs_nothing_and_writes_no_markers(tmp_path: Path) -> None:
    ran: list[str] = []
    stages = [Stage(name="a", run=lambda: ran.append("a"))]
    results = Runner(tmp_path / "m").execute(stages, dry_run=True)
    assert ran == []
    assert results[0].status == "skipped"
    assert not list((tmp_path / "m").glob("*.marker.json"))


def test_disabled_stage(tmp_path: Path) -> None:
    stage = Stage(name="s", run=lambda: None, enabled=lambda: False)
    results = Runner(tmp_path / "m").execute([stage])
    assert results[0] == StageResult("s", "disabled", 0.0)


def test_tile_params_sidecar_drives_freshness(tmp_path: Path) -> None:
    """--max-zoom is a flag, not a file: the CLI persists the effective zoom
    range as a tile-stage input. Same range -> mtime untouched (fresh-skip
    survives); changed range -> content differs -> mtime bumps -> re-tile."""
    import os

    from autogeoref.bake.tiles import write_tile_params
    from autogeoref.paths import VolumePaths

    paths = VolumePaths(root=tmp_path / "vol")
    sidecar = write_tile_params(paths, max_zoom=None)
    assert json.loads(sidecar.read_text()) == {"min_zoom": 12, "max_zoom": 20}
    # pin mtimes far in the past: the coarse kernel mtime clock would
    # otherwise tie writes landing in the same tick
    os.utime(sidecar, (1000, 1000))
    pmtiles = _touch(tmp_path / "vol" / "v.pmtiles")
    os.utime(pmtiles, (2000, 2000))  # bake newer than the sidecar
    stage = Stage(name="tile", run=lambda: None, inputs=[sidecar], outputs=[pmtiles])
    assert is_fresh(stage)

    write_tile_params(paths, max_zoom=None)  # same range: content-compare, no write
    assert sidecar.stat().st_mtime == 1000
    assert is_fresh(stage)

    write_tile_params(paths, max_zoom=16)  # changed range: rewrite, now-mtime
    assert json.loads(sidecar.read_text())["max_zoom"] == 16
    assert sidecar.stat().st_mtime > 2000
    assert not is_fresh(stage)
