"""A publish that fails leaves the previous publication exactly as it was.

Every refusal and every mid-flight failure is checked for the same three
things: the served archive still holds the old bytes, the manifest still reads
as it did, and no temporary or backup residue is left behind. A failure also
leaves the owed-publish marker standing, since a bake nobody served is still
owed, and it releases the publish lock rather than hanging every publisher.
"""

from __future__ import annotations

import fcntl
import json
import shutil
from pathlib import Path

import pytest

from autogeoref.viewer.publish import (
    PublicationError,
    publish_owed,
    publish_volume,
    record_publish_owed,
)
from viewer_support import (
    _archive,
    _fake_pmtiles_bounds,
    _publication_config,
    _refuse_directory_renames,
)


def test_a_failed_publish_keeps_the_debt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing landed, so the owed publish is still owed — clearing it would strand
    a complete bake with no record that it was never served."""
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    _archive(config, "vol_a")
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)
    monkeypatch.setattr(
        viewer_mod, "write_manifest", lambda *_a, **_k: (_ for _ in ()).throw(OSError("boom"))
    )
    record_publish_owed("vol_a", config.work, baked_at=1.0)

    with pytest.raises(PublicationError, match="could not publish"):
        publish_volume("vol_a", config)

    assert publish_owed("vol_a", config.work) is not None


def test_publish_refuses_a_wrong_volume_item_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sanborn pages restart at 1 every volume, so a wrong volume's item JSON
    would 'match' every page and aim the GCPs at the wrong imagery."""
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    _archive(config, "vol_a")
    item = json.loads(config.loc_item.read_text())
    item["item"] = {"id": "http://www.loc.gov/item/vol_b/"}
    config.loc_item.write_text(json.dumps(item))
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)

    with pytest.raises(PublicationError, match="item document of vol_b"):
        publish_volume("vol_a", config)

    assert not (config.city_tiles / "vol_a.pmtiles").exists()
    assert not config.manifest.exists()


def test_publish_that_cannot_export_lands_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No results tree -> the publish fails loudly BEFORE the archive or
    manifest land (no half-landed publication, no residue)."""
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    _archive(config, "vol_a")
    shutil.rmtree(config.work / "vol_a" / "sheets")
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)

    with pytest.raises(PublicationError, match="cannot export"):
        publish_volume("vol_a", config)

    assert not (config.city_tiles / "vol_a.pmtiles").exists()
    assert not config.manifest.exists()
    assert not config.exports_root.exists() or not list(config.exports_root.iterdir())


def test_publish_failure_restores_the_previous_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure at the manifest write rolls the already-landed export tree
    back to the previously published state along with the archive."""
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    _archive(config, "vol_a")
    old = config.exports_root / "vol_a" / "gcps"
    old.mkdir(parents=True)
    (old / "p9.json").write_text("previous export")
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)
    monkeypatch.setattr(
        viewer_mod, "write_manifest", lambda *_a, **_k: (_ for _ in ()).throw(OSError("boom"))
    )

    with pytest.raises(PublicationError, match="could not publish"):
        publish_volume("vol_a", config)

    assert (old / "p9.json").read_text() == "previous export"
    assert not (config.exports_root / "vol_a" / "allmaps.json").exists()
    assert not [p for p in config.exports_root.iterdir() if p.name.startswith(".")]


def test_rollback_restores_the_exports_where_renames_cross_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The restore leg has the same exposure as the swap: if it silently gave up
    on a filesystem that refuses directory renames, a failed publish would leave
    the previous export tree gone rather than put back."""
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    _archive(config, "vol_a")
    old = config.exports_root / "vol_a" / "gcps"
    old.mkdir(parents=True)
    (old / "p9.json").write_text("previous export")
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)
    monkeypatch.setattr(
        viewer_mod, "write_manifest", lambda *_a, **_k: (_ for _ in ()).throw(OSError("boom"))
    )
    _refuse_directory_renames(monkeypatch)

    with pytest.raises(PublicationError, match="could not publish"):
        publish_volume("vol_a", config)

    assert (old / "p9.json").read_text() == "previous export"
    assert not (config.exports_root / "vol_a" / "allmaps.json").exists()
    assert not [p for p in config.exports_root.iterdir() if p.name.startswith(".")]


@pytest.mark.parametrize("data", [None, b"", b"not pmtiles"])
def test_invalid_publish_source_never_replaces_existing_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: bytes | None
) -> None:
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    if data is not None:
        _archive(config, "vol_a", data)
    destination = config.city_tiles / "vol_a.pmtiles"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"PM old")
    config.manifest.parent.mkdir(parents=True)
    config.manifest.write_text('{"volumes": []}')
    before = config.manifest.read_text()
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)

    with pytest.raises(PublicationError):
        publish_volume("vol_a", config)

    assert destination.read_bytes() == b"PM old"
    assert config.manifest.read_text() == before


def test_manifest_failure_restores_the_previous_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    _archive(config, "vol_a")
    destination = config.city_tiles / "vol_a.pmtiles"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"PM old")
    config.manifest.parent.mkdir(parents=True)
    config.manifest.write_text('{"volumes": [{"id": "old"}]}')
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)
    monkeypatch.setattr(
        viewer_mod, "build_manifest", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("boom"))
    )

    with pytest.raises(PublicationError, match="could not publish"):
        publish_volume("vol_a", config)

    assert json.loads(config.manifest.read_text()) == {"volumes": [{"id": "old"}]}
    assert destination.read_bytes() == b"PM old"


def test_failed_publish_restores_archive_and_manifest_backups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest-write failure AFTER the archive landed restores both
    backed-up resources and leaves no residue."""
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    _archive(config, "vol_a")
    destination = config.city_tiles / "vol_a.pmtiles"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"PM old")
    config.manifest.parent.mkdir(parents=True)
    config.manifest.write_text('{"volumes": [{"id": "old"}]}')
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)
    monkeypatch.setattr(
        viewer_mod, "write_manifest", lambda *_a, **_k: (_ for _ in ()).throw(OSError("boom"))
    )

    with pytest.raises(PublicationError, match="could not publish"):
        publish_volume("vol_a", config)

    assert destination.read_bytes() == b"PM old"
    assert config.manifest.read_text() == '{"volumes": [{"id": "old"}]}'
    assert not list(config.city_tiles.glob(".*"))  # no tmp/backup residue


def test_publish_refuses_a_path_traversal_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)

    with pytest.raises(PublicationError, match="invalid volume identifier"):
        publish_volume("../outside", config)


def test_a_manifest_build_failure_rolls_back_publish_and_releases_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure while ASSEMBLING the manifest becomes PublicationError: the
    publish restores the prior archive+manifest state and releases its lock
    instead of hanging every publisher or leaking a half-built publication."""
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    _archive(config, "vol_a")
    destination = config.city_tiles / "vol_a.pmtiles"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"PM old")
    config.manifest.parent.mkdir(parents=True)
    config.manifest.write_text('{"volumes": [{"id": "old"}]}')
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)
    monkeypatch.setattr(
        viewer_mod,
        "build_manifest",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(PublicationError):
        publish_volume("vol_a", config)

    assert destination.read_bytes() == b"PM old"
    assert config.manifest.read_text() == '{"volumes": [{"id": "old"}]}'
    assert not list(config.city_tiles.glob(".*"))  # no tmp/backup residue
    # the lock is free again: a fresh exclusive non-blocking grab succeeds
    with (config.tiles_root / ".publish.lock").open("a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
