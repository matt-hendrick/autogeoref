"""Contracts for the alias-file corpus validator.

Two decisions carry this instrument. The recorded normalizer contract — the
reads a shipped table MUST catch and the reads it must NOT touch — is executed
here against the real tables, so a normalizer change that breaks a volume fails
in the suite rather than the next time an operator remembers to run the script.
And the excuse machinery: a checkout that cannot judge a table says so instead
of passing it, and an excuse that no longer fires is itself a failure.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from autogeoref import names
from autogeoref.names import load_aliases

if TYPE_CHECKING:
    import pytest

CENTERLINES = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"street_nam": name, "street_typ": "AVE"},
            "geometry": {"type": "LineString", "coordinates": [[0.1, 0.1], [0.1, 0.9]]},
        }
        for name in ("NEWNAME", "KEPT")
    ],
}


def _module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_alias_files.py"
    spec = importlib.util.spec_from_file_location("validate_alias_files", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_instrument_normalizes_through_the_shipped_normalizer() -> None:
    """A private copy would let a table pass here and fail at match time."""
    assert _module().normalize is names.normalize


def test_the_recorded_normalizer_contract_holds_for_every_shipped_table(
    aliases_dir: Path,
) -> None:
    """Every MUST-catch read reaches its key; every MUST-NOT-catch read is untouched.

    ``expected is None`` is the second half and the easier one to break: those
    reads are court guards, out-of-bounds twins and direct index matches that a
    widened table must not start absorbing.
    """
    module = _module()
    tables: dict[str, dict[str, str]] = {}
    failures: list[str] = []
    for volume, read, expected in module.CONTRACT:
        table = tables.setdefault(volume, load_aliases(aliases_dir / f"aliases-{volume}.json"))
        assert table, f"{volume}: no alias table on disk for a contract case"
        got = names.normalize(read, table)
        if expected is None:
            bare = names.normalize(read)
            if got != bare:
                failures.append(f"{volume} {read!r}: table changed {bare!r} -> {got!r}")
        elif got != expected:
            failures.append(f"{volume} {read!r}: expected {expected!r}, got {got!r}")
    assert not failures, "\n".join(failures)
    assert len(module.CONTRACT) > 50 and len(tables) > 5


def _city(tmp_path: Path, table: dict[str, str], *, bbox: bool) -> Path:
    aliases = tmp_path / "aliases"
    aliases.mkdir(parents=True)
    (aliases / "aliases-vol.json").write_text(json.dumps(table))
    (tmp_path / "streets.geojson").write_text(json.dumps(CENTERLINES))
    city = tmp_path / "city.toml"
    city.write_text(
        '[city]\nname = "X"\naliases_dir = "aliases"\ncenterlines = "streets.geojson"\n'
        "[volumes.vol]\n" + ("bounds_bbox = [0.0, 0.0, 1.0, 1.0]\n" if bbox else "")
    )
    return city


def _run(module: ModuleType, city: Path, work: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(
        "sys.argv",
        ["validate_alias_files.py", "--city", str(city), "--work", str(work)],
    )
    exit_code: int = module.main()
    return exit_code


def test_a_volume_with_no_bounds_is_reported_and_skipped_not_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fresh checkout cannot judge a table either way, and must not pretend to.

    Failing here would train operators to ignore the tool on a thin checkout;
    passing silently would let a real defect ride out under a green run.
    """
    module = _module()
    monkeypatch.setattr(module, "CONTRACT", [])
    monkeypatch.setattr(module, "PRE_EXISTING", {})
    city = _city(tmp_path, {"OLDNAME": "NEWNAME"}, bbox=False)

    assert _run(module, city, tmp_path / "work", monkeypatch) == 0
    out = capsys.readouterr().out
    assert "NO BOUNDS on this checkout — skipped" in out
    assert "1 volume(s) not checked" in out
    assert "OK: 0 failure(s)" in out


def test_a_clean_table_passes_and_a_shadowing_key_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With bounds on disk the table is judged against an alias-free index."""
    module = _module()
    monkeypatch.setattr(module, "CONTRACT", [])
    monkeypatch.setattr(module, "PRE_EXISTING", {})

    clean = _city(tmp_path / "clean", {"OLDNAME": "NEWNAME"}, bbox=True)
    assert _run(module, clean, tmp_path / "work", monkeypatch) == 0
    assert "OK: 0 failure(s)" in capsys.readouterr().out

    # KEPT is a surviving in-bounds street; keying it shadows the real one
    shadow = _city(tmp_path / "shadow", {"KEPT": "NEWNAME"}, bbox=True)
    assert _run(module, shadow, tmp_path / "work", monkeypatch) == 1
    assert "FAIL" in capsys.readouterr().out


def test_an_excuse_that_no_longer_fires_is_itself_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stale excuse would silently cover a DIFFERENT violation later."""
    module = _module()
    monkeypatch.setattr(module, "CONTRACT", [])
    monkeypatch.setattr(
        module, "PRE_EXISTING", {("vol", "'GONE' -> 'NEWNAME': VALUE NOT AN INDEX KEY"): "stale"}
    )
    city = _city(tmp_path, {"OLDNAME": "NEWNAME"}, bbox=True)

    assert _run(module, city, tmp_path / "work", monkeypatch) == 1
    assert "stale PRE_EXISTING entr" in capsys.readouterr().out


def test_a_skipped_volume_does_not_retire_its_excuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Not checked is not clean: an unjudged volume's excuses must survive the run."""
    module = _module()
    monkeypatch.setattr(module, "CONTRACT", [])
    monkeypatch.setattr(module, "PRE_EXISTING", {("vol", "some recorded violation"): "excused"})
    city = _city(tmp_path, {"OLDNAME": "NEWNAME"}, bbox=False)

    assert _run(module, city, tmp_path / "work", monkeypatch) == 0
    assert "stale" not in capsys.readouterr().out


def test_a_city_that_curates_no_aliases_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Having no table is a real state — a city whose reads already resolve, or
    one whose table the normalizer made redundant."""
    module = _module()
    monkeypatch.setattr(module, "CONTRACT", [])
    monkeypatch.setattr(module, "PRE_EXISTING", {})
    city = _city(tmp_path, {}, bbox=True)
    (tmp_path / "aliases" / "aliases-vol.json").unlink()

    assert _run(module, city, tmp_path / "work", monkeypatch) == 0
    assert "nothing to validate" in capsys.readouterr().out


def test_an_aliases_dir_that_is_not_there_is_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other half, and the reason the one above cannot just pass on emptiness.

    A typo in ``aliases_dir`` drops every table a city has, and the run stays
    green everywhere else — this is the one place positioned to notice.
    """
    module = _module()
    monkeypatch.setattr(module, "CONTRACT", [])
    monkeypatch.setattr(module, "PRE_EXISTING", {})
    city = _city(tmp_path, {}, bbox=True)
    city.write_text(city.read_text().replace('aliases_dir = "aliases"', 'aliases_dir = "aliasses"'))

    assert _run(module, city, tmp_path / "work", monkeypatch) == 1
    assert "does not exist" in capsys.readouterr().out


def test_an_empty_alias_file_is_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty table is a write that went wrong, not a volume with no renames."""
    module = _module()
    monkeypatch.setattr(module, "CONTRACT", [])
    monkeypatch.setattr(module, "PRE_EXISTING", {})
    city = _city(tmp_path, {}, bbox=True)

    assert _run(module, city, tmp_path / "work", monkeypatch) == 1
    assert "alias file has no entries" in capsys.readouterr().out
