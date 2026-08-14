"""Publishing a volume: what lands, and what the queue and the exports are left holding.

A publish moves three resources together — the served archive, the manifest and
the researcher exports — and settles the owed-publish marker the queue writes
between bake and publish. Only the committed sheets are exported, an unchanged
volume re-exports byte for byte, and the bake's overview companion is left
alone. The CLI leg is here because it must publish without running any stage.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from autogeoref.allmaps import RIGHTS
from autogeoref.exports import README as EXPORTS_README
from autogeoref.viewer.publish import (
    _move_tree,
    publish_owed,
    publish_volume,
    record_publish_owed,
)
from viewer_support import (
    ROOT,
    _archive,
    _fake_pmtiles_bounds,
    _publication_config,
    _refuse_directory_renames,
    _result_record,
)


def _tree(root: Path) -> Path:
    """A directory with a nested file, so a shallow copy would be visible."""
    (root / "gcps").mkdir(parents=True)
    (root / "gcps" / "p1.json").write_text("payload")
    return root


def test_move_tree_copies_when_rename_crosses_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The post-condition is the rename's: source gone, destination holding the tree."""
    source = _tree(tmp_path / "src")
    _refuse_directory_renames(monkeypatch)

    _move_tree(source, tmp_path / "dst")

    assert not source.exists()
    assert (tmp_path / "dst" / "gcps" / "p1.json").read_text() == "payload"


def test_move_tree_does_not_swallow_other_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only EXDEV means 'try another way'. A permission error is a real failure
    and copying past it would land a tree the caller could not have renamed."""
    source = _tree(tmp_path / "src")

    def denied(self: Path, target: str | Path) -> Path:
        raise OSError(errno.EACCES, "denied", str(self))

    monkeypatch.setattr(Path, "rename", denied)

    with pytest.raises(OSError, match="denied"):
        _move_tree(source, tmp_path / "dst")

    assert (source / "gcps" / "p1.json").exists()
    assert not (tmp_path / "dst").exists()


def test_move_tree_leaves_nothing_behind_when_the_source_cannot_be_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-done move is the one outcome a rollback cannot reason about, so a
    failed removal takes the copy back out rather than leaving the tree twice."""
    source = _tree(tmp_path / "src")
    _refuse_directory_renames(monkeypatch)
    # only the source refuses to go: the cleanup of the copy must really run,
    # or this could not tell "cleaned up" from "never attempted"
    real_rmtree = shutil.rmtree

    def refuse_the_source(path: Any, *args: Any, **kwargs: Any) -> None:
        if Path(path) == source:
            raise OSError("cannot remove")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", refuse_the_source)

    with pytest.raises(OSError, match="cannot remove"):
        _move_tree(source, tmp_path / "dst")

    assert (source / "gcps" / "p1.json").exists()
    assert not (tmp_path / "dst").exists()


def test_publish_lands_where_the_export_tree_cannot_be_renamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The container case: an export tree that ships in the image cannot be
    renamed aside, and every tracked volume has one, so the publish must not
    depend on that rename."""
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    _archive(config, "vol_a")
    previous = config.exports_root / "vol_a" / "gcps"
    previous.mkdir(parents=True)
    (previous / "stale.json").write_text("previous export")
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)
    _refuse_directory_renames(monkeypatch)

    publish_volume("vol_a", config)

    assert (config.city_tiles / "vol_a.pmtiles").read_bytes() == b"PM archive"
    assert config.manifest.exists()
    assert (config.exports_root / "vol_a" / "allmaps.json").exists()
    assert not (config.exports_root / "vol_a" / "gcps" / "stale.json").exists()
    assert not [p for p in config.exports_root.iterdir() if p.name.startswith(".")]


def _record_publish(
    called: list[tuple[str, Path | None, Path | None]],
) -> Callable[..., Path | None]:
    """A `publish_volume` stand-in that logs its call and echoes `source` back."""

    def publish(volume: str, config: Any, *, source: Path | None = None) -> Path | None:
        called.append((volume, config.loc_catalog, source))
        return source

    return publish


def test_publish_lands_archive_and_keeps_prior_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A catalog-less publish lands the archive, stamps it `autogeoref`, and
    carries the previous manifest's title/year forward rather than reducing the
    entry to its id."""
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    source = _archive(config, "vol_a")
    # a sibling directory publish must NOT scan: only deploy/tiles/autogeoref
    # is this pipeline's serving root, and a foreign tree beside it is not ours
    foreign = config.tiles_root / "partner-archive" / "vol_b.pmtiles"
    foreign.parent.mkdir(parents=True)
    foreign.write_bytes(b"PM theirs")
    config.manifest.parent.mkdir(parents=True)
    config.manifest.write_text(
        json.dumps({"volumes": [{"id": "vol_a", "title": "Old title", "year": 1900}]})
    )
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)

    destination = publish_volume("vol_a", config)

    assert destination.read_bytes() == source.read_bytes()
    volumes = json.loads(config.manifest.read_text())["volumes"]
    assert [v["id"] for v in volumes] == ["vol_a"]  # the foreign tree is not served
    entry = volumes[0]
    assert entry["pmtiles"].endswith("autogeoref/vol_a.pmtiles")
    assert entry["title"] == "Old title" and entry["year"] == 1900


def test_publish_discharges_an_owed_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clearing lives with the publish, so a hand-run settles the debt the queue recorded.

    The marker exists so a drain killed between bake and publish can be recovered
    without re-baking (queue.publish.finish_owed_publishes). An operator who publishes by
    hand has done exactly that recovery, and must not leave the queue believing a
    publish is still owed.
    """
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    _archive(config, "vol_a")
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)
    record_publish_owed("vol_a", config.work, baked_at=1.0)
    assert publish_owed("vol_a", config.work) is not None

    publish_volume("vol_a", config)

    assert publish_owed("vol_a", config.work) is None


def test_publishing_an_explicit_source_leaves_the_work_trees_debt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The marker names the WORK-TREE archive. Publishing some other file — the
    legacy bake script's, a hand-built one — discharges a debt it never paid, and
    the work tree's complete bake would silently stop being owed to anyone."""
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    _archive(config, "vol_a")
    elsewhere = tmp_path / "elsewhere.pmtiles"
    elsewhere.write_bytes(b"PM other")
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)
    record_publish_owed("vol_a", config.work, baked_at=1.0)

    publish_volume("vol_a", config, source=elsewhere)

    assert publish_owed("vol_a", config.work) is not None


def test_publish_writes_the_researcher_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A publish rewrites exports/<volume>/: the committed sheets' records
    verbatim plus one Allmaps AnnotationPage in the full-res IIIF frame."""
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    _archive(config, "vol_a")
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)

    publish_volume("vol_a", config)

    export = config.exports_root / "vol_a"
    record = config.work / "vol_a" / "results" / "p1.json"
    assert (export / "gcps" / "p1.json").read_bytes() == record.read_bytes()
    page = json.loads((export / "allmaps.json").read_text())
    assert page["type"] == "AnnotationPage"
    assert len(page["items"]) == 1
    source = page["items"][0]["target"]["source"]
    assert (source["width"], source["height"]) == (100, 80)
    assert source["id"].startswith("https://tile.loc.gov/image-services/iiif/")
    assert not [p for p in config.exports_root.iterdir() if p.name.startswith(".")]

    # The licence, both copies. The tree's note is written HERE and nowhere
    # else, so nothing but a publish would put it beside the volumes.
    assert page["rights"] == RIGHTS
    assert page["items"][0]["rights"] == RIGHTS
    assert (config.exports_root / "README.md").read_text(encoding="utf-8") == EXPORTS_README


def test_publish_regenerates_the_exports_note_it_finds_edited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of generating it: a hand edit does not survive the next publish."""
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    _archive(config, "vol_a")
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)
    config.exports_root.mkdir(parents=True, exist_ok=True)
    (config.exports_root / "README.md").write_text("this tree is all rights reserved\n")

    publish_volume("vol_a", config)

    assert (config.exports_root / "README.md").read_text(encoding="utf-8") == EXPORTS_README


def test_publish_exports_only_committed_sheets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flagged sheet must not appear in the exports — the product promise
    (accepted or flagged) extends to the tracked data."""
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    _archive(config, "vol_a")
    root = config.work / "vol_a"
    (root / "results" / "p2.json").write_text(json.dumps(_result_record("2", status="FLAGGED")))
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)

    publish_volume("vol_a", config)

    export = config.exports_root / "vol_a"
    assert [p.name for p in sorted((export / "gcps").iterdir())] == ["p1.json"]
    assert len(json.loads((export / "allmaps.json").read_text())["items"]) == 1


def test_publish_replaces_stale_staging_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Residue from a crashed publish under the same staging name (pid reuse)
    is replaced wholesale — a stale record must never ride into the landed
    tree beside a fresh AnnotationPage that does not describe it."""
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    _archive(config, "vol_a")
    stale = config.exports_root / f".vol_a.tmp-{os.getpid()}" / "gcps"
    stale.mkdir(parents=True)
    (stale / "p9.json").write_text("crashed publish leftover")
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)

    publish_volume("vol_a", config)

    export = config.exports_root / "vol_a"
    assert [p.name for p in sorted((export / "gcps").iterdir())] == ["p1.json"]
    assert not [p for p in config.exports_root.iterdir() if p.name.startswith(".")]


def test_publish_reexport_is_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Republishing an unchanged volume reproduces every export byte, so a
    diff under exports/ only ever shows a real placement change."""
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    _archive(config, "vol_a")
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)

    publish_volume("vol_a", config)
    export = config.exports_root / "vol_a"
    before = {p: p.read_bytes() for p in export.rglob("*") if p.is_file()}
    publish_volume("vol_a", config)
    after = {p: p.read_bytes() for p in export.rglob("*") if p.is_file()}
    assert before == after


def test_publish_ignores_a_bakes_overview_companion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bake still writes ``<volume>-overview.pmtiles`` beside the archive and
    nothing serves it: publish neither carries it across nor disturbs whatever
    an earlier publish left in the served directory."""
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    source = _archive(config, "vol_a")
    source.with_name("vol_a-overview.pmtiles").write_bytes(b"PM new underlay")
    stale = config.city_tiles / "vol_a-overview.pmtiles"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"PM old underlay")
    config.manifest.parent.mkdir(parents=True)
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)

    publish_volume("vol_a", config)

    assert stale.read_bytes() == b"PM old underlay"
    entry = json.loads(config.manifest.read_text())["volumes"][0]
    assert entry["id"] == "vol_a" and "overview_pmtiles" not in entry
    # and the leftover never becomes a volume of its own
    assert [v["id"] for v in json.loads(config.manifest.read_text())["volumes"]] == ["vol_a"]


def test_concurrent_publishes_leave_both_archives_and_manifest_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autogeoref.viewer.publish as viewer_mod

    config = _publication_config(tmp_path)
    _archive(config, "vol_a")
    _archive(config, "vol_b")
    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def publish(volume: str) -> None:
        try:
            barrier.wait()
            publish_volume(volume, config)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=publish, args=(volume,)) for volume in ("vol_a", "vol_b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert {p.stem for p in config.city_tiles.glob("*.pmtiles")} == {"vol_a", "vol_b"}
    assert {v["id"] for v in json.loads(config.manifest.read_text())["volumes"]} == {
        "vol_a",
        "vol_b",
    }


def test_publish_cli_uses_the_publisher_without_spawning_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autogeoref.viewer.publish as viewer_mod
    from autogeoref.cli.entry import main

    source = tmp_path / "source.pmtiles"
    source.write_bytes(b"PM archive")
    city = _publication_config(tmp_path).city_toml
    catalog = tmp_path / "catalog.json"
    city.write_text(
        '[city]\nname = "Test City"\ncenterlines = "streets.geojson"\n'
        'aliases_dir = "aliases"\nloc_catalog = "catalog.json"\n'
    )
    called: list[tuple[str, Path | None, Path | None]] = []
    monkeypatch.setattr(viewer_mod, "publish_volume", _record_publish(called))
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: pytest.fail("publish must not run stages")
    )

    assert (
        main(
            [
                "publish",
                "vol_a",
                "--city",
                str(city),
                "--source",
                str(source),
            ]
        )
        == 0
    )
    assert called == [("vol_a", catalog, source)]


def test_publish_cli_closes_the_queue_row_it_recovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through the CLI: the documented hand-recovery must not leave the
    stranded row for the next drain to stamp "interrupted before completion"."""
    import autogeoref.queue.store as queue_store
    import autogeoref.viewer.publish as viewer_mod
    from autogeoref.cli.entry import main

    work = tmp_path / "work"
    config = _publication_config(tmp_path)
    _archive(config, "vol_a")
    (work / "vol_a").mkdir(parents=True, exist_ok=True)
    for name in ("sheets", "results"):
        (work / "vol_a" / name).mkdir(parents=True, exist_ok=True)
    (work / "vol_a" / "sheets" / "manifest.json").write_text(json.dumps({"p1": {}}))
    (work / "vol_a" / "results" / "p1.json").write_text(json.dumps(_result_record("1")))
    entry = queue_store.add(work, "vol_a", "serve")
    entry.status = "running"
    queue_store.persist(work, [entry])
    record_publish_owed("vol_a", work)
    city = config.city_toml
    city.write_text(
        '[city]\nname = "Test City"\ncenterlines = "streets.geojson"\naliases_dir = "aliases"\n'
    )
    monkeypatch.setattr(viewer_mod, "publish_volume", lambda *_a, **_k: Path("published.pmtiles"))
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: pytest.fail("publish must not run stages")
    )

    assert main(["publish", "vol_a", "--city", str(city), "--work", str(work)]) == 0

    assert queue_store.load_queue(work)[0].status == "done"


def test_legacy_bake_script_uses_the_shared_publish_helper() -> None:
    script = (ROOT / "scripts" / "run_bake_queue.sh").read_text()
    assert "autogeoref publish" in script
    assert ".manifest.lock" not in script
