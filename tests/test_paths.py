"""Atomic persistence contracts for shared work-tree files."""

import json
import os
import stat
from pathlib import Path

import pytest

from autogeoref import paths


def test_write_result_replaces_complete_record(tmp_path: Path) -> None:
    result = tmp_path / "results" / "p1.json"
    result.parent.mkdir()
    result.write_text(json.dumps({"page": "1", "status": "old"}, indent=2))

    record = {"page": "1", "status": "OK"}
    paths.write_result(result, record)

    assert result.read_text() == json.dumps(record, indent=2)
    assert json.loads(result.read_text()) == record


def test_write_result_preserves_existing_file_mode(tmp_path: Path) -> None:
    result = tmp_path / "results" / "p1.json"
    result.parent.mkdir()
    result.write_text(json.dumps({"page": "1", "status": "old"}, indent=2))
    result.chmod(0o640)

    paths.write_result(result, {"page": "1", "status": "OK"})

    assert stat.S_IMODE(result.stat().st_mode) == 0o640


def test_write_result_replaces_write_only_record(tmp_path: Path) -> None:
    result = tmp_path / "results" / "p1.json"
    result.parent.mkdir()
    result.write_text(json.dumps({"page": "1", "status": "old"}, indent=2))
    result.chmod(0o200)

    paths.write_result(result, {"page": "1", "status": "OK"})

    assert stat.S_IMODE(result.stat().st_mode) == 0o200
    result.chmod(0o600)
    assert json.loads(result.read_text()) == {"page": "1", "status": "OK"}


def test_write_result_preserves_existing_owner_and_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = tmp_path / "results" / "p1.json"
    result.parent.mkdir()
    result.write_text(json.dumps({"page": "1", "status": "old"}, indent=2))
    target_stat = result.stat()
    calls: list[tuple[int, int, int]] = []

    def record_fchown(fd: int, uid: int, gid: int) -> None:
        calls.append((fd, uid, gid))

    monkeypatch.setattr(os, "fchown", record_fchown)
    paths.write_result(result, {"page": "1", "status": "OK"})

    assert calls and calls[0][1:] == (target_stat.st_uid, target_stat.st_gid)


def test_write_result_preserves_existing_extended_attributes(tmp_path: Path) -> None:
    result = tmp_path / "results" / "p1.json"
    result.parent.mkdir()
    result.write_text(json.dumps({"page": "1", "status": "old"}, indent=2))
    os.setxattr(result, "user.autogeoref-test", b"retained")

    paths.write_result(result, {"page": "1", "status": "OK"})

    assert os.getxattr(result, "user.autogeoref-test") == b"retained"


def test_write_result_does_not_bypass_read_only_target(tmp_path: Path) -> None:
    result = tmp_path / "results" / "p1.json"
    result.parent.mkdir()
    old_text = json.dumps({"page": "1", "status": "old"}, indent=2)
    result.write_text(old_text)
    result.chmod(0o444)

    with pytest.raises(PermissionError):
        paths.write_result(result, {"page": "1", "status": "new"})

    assert result.read_text() == old_text


def test_failed_result_replacement_preserves_old_complete_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = tmp_path / "results" / "p1.json"
    result.parent.mkdir()
    old_text = json.dumps({"page": "1", "status": "old"}, indent=2)
    result.write_text(old_text)

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replacement failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        paths.write_result(result, {"page": "1", "status": "new"})

    assert result.read_text() == old_text
    assert json.loads(result.read_text()) == {"page": "1", "status": "old"}
    assert not list(result.parent.glob(".p1.json.*.tmp"))


# ---------------------------------------------------------------------------
# volume_lock: one mutating owner per volume work tree
# ---------------------------------------------------------------------------


def _vp(tmp_path: Path, name: str = "vol_a") -> paths.VolumePaths:
    return paths.VolumePaths(root=tmp_path / name)


def test_volume_lock_refuses_a_second_owner_with_actionable_metadata(tmp_path: Path) -> None:
    vp = _vp(tmp_path)
    with paths.volume_lock(vp, operation="run"):
        held = json.loads(vp.lock.read_text())
        assert held["operation"] == "run"
        assert held["pid"] == os.getpid()
        with (
            pytest.raises(paths.VolumeBusyError) as exc_info,
            paths.volume_lock(vp, operation="review --apply"),
        ):
            raise AssertionError("a held volume must never be entered")
        message = str(exc_info.value)
        assert "vol_a is busy" in message
        assert "run" in message and str(os.getpid()) in message and str(vp.lock) in message
        assert exc_info.value.holder is not None and exc_info.value.holder["operation"] == "run"


def test_volume_lock_releases_on_exit_and_clears_holder_metadata(tmp_path: Path) -> None:
    vp = _vp(tmp_path)
    with paths.volume_lock(vp, operation="prep"):
        pass
    # the file is PERMANENT (the flock is the authority, never file existence),
    # but the released holder's identity is gone: nobody stale can be blamed
    assert vp.lock.exists()
    assert vp.lock.read_text() == ""
    with paths.volume_lock(vp, operation="run"):  # and it is immediately reacquirable
        pass


def test_volume_lock_releases_when_the_operation_raises(tmp_path: Path) -> None:
    vp = _vp(tmp_path)
    with pytest.raises(RuntimeError, match="stage failed"), paths.volume_lock(vp, operation="run"):
        raise RuntimeError("stage failed")
    with paths.volume_lock(vp, operation="run"):
        pass


def test_volume_lock_is_per_volume_so_independent_volumes_run_concurrently(
    tmp_path: Path,
) -> None:
    with (
        paths.volume_lock(_vp(tmp_path, "vol_a"), operation="run"),
        paths.volume_lock(_vp(tmp_path, "vol_b"), operation="run"),
    ):
        pass


def test_volume_lock_busy_error_survives_a_crashed_holders_stale_bytes(tmp_path: Path) -> None:
    """A holder killed mid-operation leaves metadata but NO flock: not busy."""
    vp = _vp(tmp_path)
    vp.root.mkdir(parents=True)
    vp.lock.write_text(json.dumps({"pid": 999999, "operation": "run", "started": 0}))
    with paths.volume_lock(vp, operation="run"):  # acquires: the kernel holds no lock
        held = json.loads(vp.lock.read_text())
        assert held["pid"] == os.getpid(), "stale bytes are replaced by the real holder"
