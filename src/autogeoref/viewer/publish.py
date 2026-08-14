"""Archive publication: land tiles, exports, and the manifest transactionally."""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config.model import ConfigError
from ..paths import VolumePaths, atomic_write_text
from .bounds import pmtiles_bounds
from .config import ViewerConfig, load_viewer_config
from .coverage import SheetFootprints
from .layout import TILES_ROOT, VIEWER_ROOT, city_tiles, refresh_cities
from .manifest import (
    AreaSource,
    BoundsProbes,
    assert_serving_dirs_declared,
    build_manifest,
    write_manifest,
)
from .stories import stage_story_assets

if TYPE_CHECKING:
    from ..config.model import CityConfig

logger = logging.getLogger(__name__)


class PublicationError(RuntimeError):
    """A finished archive could not become a visible viewer layer."""


@dataclass(frozen=True)
class PublicationConfig:
    """The roots a publish operation owns, assembled once at the CLI boundary."""

    work: Path
    city_toml: Path
    #: this city's page manifest — ``viewer/<city-slug>/manifest.json``. It has
    #: no site-wide default because every input to it is one city's: a second
    #: city publishing to a shared path retitles the first city's page and
    #: re-derives its era chips. Resolve it with :func:`layout.city_manifest`.
    manifest: Path
    #: the viewer directory holding the page files and the city index beside
    #: them; the manifest normally sits one level under it
    viewer_root: Path = VIEWER_ROOT
    tiles_root: Path = TILES_ROOT
    #: the ``deploy/tiles/`` subdirectory this city's archives land in — its
    #: ``viewer.serving_dirs[0]``, resolved at the CLI boundary.
    serve_dir: str = "autogeoref"
    loc_catalog: Path | None = None
    #: researcher data exports (tracked): ``exports/<volume>/`` is rewritten by
    #: every publish so the committed data cannot drift from what is served
    exports_root: Path = Path("exports")
    #: local LOC item JSON for the export's IIIF service ids; ``None`` uses the
    #: cached LOC client (``loc_cache``)
    loc_item: Path | None = None
    loc_cache: Path = Path("cache/loc")

    @property
    def city_tiles(self) -> Path:
        """The archive directory this city's publishes land in."""
        return city_tiles(self.serve_dir, self.tiles_root)


@contextmanager
def publication_lock(config: PublicationConfig) -> Iterator[None]:
    """Serialize archive landing and manifest replacement across all serve workers."""
    lock = config.tiles_root / ".publish.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _manifest_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Read the metadata fields a catalog-less rebuild must retain."""
    if not path.exists():
        return {}
    try:
        volumes = json.loads(path.read_text(encoding="utf-8")).get("volumes") or []
    except (OSError, ValueError) as exc:
        raise PublicationError(f"cannot read existing manifest {path}: {exc}") from exc
    return {
        str(v["id"]): {k: v[k] for k in ("title", "year", "volume_number", "label") if k in v}
        for v in volumes
        if isinstance(v, Mapping) and v.get("id")
    }


def _assert_readable_archive(path: Path) -> None:
    """Prove ``path`` is a non-empty file with a readable PMTiles header."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            raise PublicationError(f"missing or empty PMTiles archive: {path}")
        pmtiles_bounds(path)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        raise PublicationError(f"unreadable PMTiles archive {path}: {exc}") from exc


def _verified_copy(source: Path, destination: Path) -> None:
    """Copy an archive, then prove the staged file has a readable PMTiles header."""
    try:
        shutil.copyfile(source, destination)
        if destination.stat().st_size == 0:
            raise PublicationError(f"copied archive is empty: {destination}")
        pmtiles_bounds(destination)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        raise PublicationError(f"unreadable PMTiles archive {source}: {exc}") from exc


class _Restorable:
    """One backed-up resource of the publish transaction.

    Two of the three resource groups — archive and manifest — share this shape:
    hard-link the currently visible file aside before replacing it, mark the new
    state ``landed``, and on failure either put the backup back or remove a file
    that had no predecessor. The exports group is rename-based with a
    ``.doomed`` slot, so it does not share this shape;
    :func:`_roll_back_exports` restores it.
    """

    def __init__(self, destination: Path, previous: Path) -> None:
        self.destination = destination
        self.previous = previous
        self.landed = False
        self.backed_up = False

    def backup(self) -> None:
        """Hard-link the visible file aside so a rollback can restore it."""
        os.link(self.destination, self.previous)
        self.backed_up = True

    def roll_back(self) -> None:
        """Restore the pre-publish state; a no-op unless ``landed`` was set."""
        if not self.landed:
            return
        if self.backed_up:
            self.previous.replace(self.destination)
        else:
            self.destination.unlink(missing_ok=True)


def record_publish_owed(
    volume: str, work: Path, *, baked_at: float | None = None, log: str | None = None
) -> Path:
    """Record that ``volume``'s work tree holds a complete, UNPUBLISHED bake.

    Written the moment a bake succeeds and cleared by :func:`publish_volume`, so
    the debt outlives the process that incurred it. Without it a drain killed
    between the two — the bake is long, the publish is seconds — leaves a
    ``running`` queue row reading only as "the bake was interrupted", steering
    recovery into a full re-bake the back half cannot skip.
    """
    paths = VolumePaths(root=work / volume)
    paths.root.mkdir(parents=True, exist_ok=True)
    payload = {"volume": volume, "baked_at": baked_at, "log": log}
    return atomic_write_text(paths.publish_owed, json.dumps(payload, indent=2))


def publish_owed(volume: str, work: Path) -> dict[str, Any] | None:
    """The recorded owed publish for ``volume``, or ``None`` when none is owed.

    Fails CLOSED: an unreadable or malformed marker reads as "nothing owed".
    The marker only ever ADDS a cheap recovery (publish what is already baked);
    treating a damaged one as a debt would instead assert a completed bake on
    no evidence, and the caller would publish an archive it never verified.
    """
    try:
        record = json.loads(VolumePaths(root=work / volume).publish_owed.read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def clear_publish_owed(volume: str, work: Path) -> None:
    """Discharge the debt. Idempotent — no marker means nothing was owed."""
    VolumePaths(root=work / volume).publish_owed.unlink(missing_ok=True)


@dataclass(frozen=True)
class _Scratch:
    """Where one publish lands, and the per-process scratch slots it rolls back through.

    Every name carries the pid, so two publishes of the same volume cannot
    collide on a temporary even though the lock already serializes them.
    """

    destination: Path
    temporary: Path
    previous_archive: Path
    previous_manifest: Path
    export_dir: Path
    export_tmp: Path
    export_prev: Path
    export_doomed: Path


def _scratch(volume: str, config: PublicationConfig) -> _Scratch:
    destination = config.city_tiles / f"{volume}.pmtiles"
    pid = os.getpid()
    return _Scratch(
        destination=destination,
        temporary=destination.with_name(f".{destination.name}.tmp-{pid}"),
        previous_archive=destination.with_name(f".{destination.name}.previous-{pid}"),
        previous_manifest=config.manifest.with_name(f".{config.manifest.name}.previous-{pid}"),
        export_dir=config.exports_root / volume,
        export_tmp=config.exports_root / f".{volume}.tmp-{pid}",
        export_prev=config.exports_root / f".{volume}.previous-{pid}",
        export_doomed=config.exports_root / f".{volume}.doomed-{pid}",
    )


def _move_tree(source: Path, destination: Path) -> None:
    """Move a directory to a destination that does not exist yet.

    ``rename`` is the fast path. It raises ``EXDEV`` when the source directory
    lives in a lower overlayfs layer, which is every export tree baked into a
    container image, so the fallback copies and then removes the source. The
    source is never left half-moved: if it will not go, the copy is taken back
    out. A copy that dies partway does leave one, for the caller's scratch
    cleanup to remove, and the fallback is not atomic the way the rename is.
    """
    try:
        source.rename(destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copytree(source, destination, symlinks=True)
        try:
            shutil.rmtree(source)
        except OSError:
            shutil.rmtree(destination, ignore_errors=True)
            raise


def _roll_back_exports(
    volume: str,
    config: PublicationConfig,
    paths: _Scratch,
    *,
    landed: bool,
    backed_up: bool,
) -> bool:
    """Put the previous export tree back; True when that failed and nothing may be cleaned.

    Moves, not rmtree: the landed tree is set aside as doomed (removed after
    the restore) so a failure here cannot delete the only copy of the previous
    export tree.
    """
    try:
        if landed:
            _move_tree(paths.export_dir, paths.export_doomed)
        if backed_up:
            _move_tree(paths.export_prev, paths.export_dir)
    except OSError as rollback_exc:
        logger.error(
            "export rollback for %s failed: %s; its .tmp/.previous/"
            ".doomed trees under %s are preserved for inspection",
            volume,
            rollback_exc,
            config.exports_root,
        )
        return True
    return False


def _refresh_index(config: PublicationConfig) -> None:
    """Rebuild the city index from the manifests the transaction actually left.

    Called after the rollback, not before it, so it describes the state this
    publish really ended in. Never fatal: the atlas is published either way and
    only the landing page needs the list.
    """
    try:
        refresh_cities(config.viewer_root)
    except OSError as exc:
        logger.warning("could not refresh the viewer city index: %s", exc)


def _publish_under_lock(
    volume: str,
    config: PublicationConfig,
    paths: _Scratch,
    *,
    city: CityConfig,
    viewer: ViewerConfig,
    source: Path,
    services: dict[str, str],
) -> None:
    """The transaction itself: archive, exports and manifest land or none of them do."""
    from ..exports import stage_export, write_exports_readme

    with publication_lock(config):
        config.city_tiles.mkdir(parents=True, exist_ok=True)
        archive = _Restorable(paths.destination, paths.previous_archive)
        manifest_file = _Restorable(config.manifest, paths.previous_manifest)
        exports_landed = False
        exports_backup = False
        exports_rollback_failed = False
        try:
            try:
                stage_export(
                    VolumePaths(root=config.work / volume),
                    page_services=services,
                    out_dir=paths.export_tmp,
                )
            except Exception as exc:
                raise PublicationError(f"cannot export {volume}: {exc}") from exc
            # Story images first, while nothing has landed yet: they are a city
            # fact, not this volume's, so they are outside the archive/exports/
            # manifest rollback set and must not be able to fail after it.
            stage_story_assets(viewer.stories, config.manifest.parent)
            _verified_copy(source, paths.temporary)
            if paths.destination.exists():
                archive.backup()
            paths.temporary.replace(paths.destination)
            archive.landed = True
            # exports land BEFORE the manifest is assembled: the story coverage
            # gate reads them, and a gate that read the previous publish's
            # sheets would judge this one on last week's placements
            if paths.export_dir.exists():
                _move_tree(paths.export_dir, paths.export_prev)
                exports_backup = True
            _move_tree(paths.export_tmp, paths.export_dir)
            exports_landed = True
            # The tree's licence note, beside the volumes rather than inside
            # one. Its content is constant, so a rollback deliberately leaves it
            # standing — there is no previous version to restore it to, and it
            # describes the tree rather than this volume. It is inside the try,
            # so it can still fail a publish; that is the right way round for a
            # tree the export leg could not finish writing.
            write_exports_readme(config.exports_root)
            manifest = build_manifest(
                city.name,
                viewer,
                out_path=config.manifest,
                pmtiles_dirs=(config.city_tiles,),
                loc_catalog=config.loc_catalog,
                metadata_fallback=_manifest_metadata(config.manifest),
                areas=AreaSource(city.community_areas_path),
                probes=BoundsProbes(pmtiles=pmtiles_bounds),
                footprints=SheetFootprints(config.exports_root),
            )
            if config.manifest.exists():
                manifest_file.backup()
            # marked before the write: a partial manifest must roll back too
            manifest_file.landed = True
            write_manifest(manifest, config.manifest)
        except Exception as exc:
            # The archive, exports, and manifest are one publication state.
            # Restore all before releasing the lock so a failed publish cannot
            # alter the visible layer or the tracked export tree.
            manifest_file.roll_back()
            if exports_landed or exports_backup:
                exports_rollback_failed = _roll_back_exports(
                    volume,
                    config,
                    paths,
                    landed=exports_landed,
                    backed_up=exports_backup,
                )
            archive.roll_back()
            if isinstance(exc, PublicationError):
                raise
            raise PublicationError(f"could not publish {volume}: {exc}") from exc
        finally:
            paths.temporary.unlink(missing_ok=True)
            paths.previous_archive.unlink(missing_ok=True)
            paths.previous_manifest.unlink(missing_ok=True)
            if not exports_rollback_failed:
                shutil.rmtree(paths.export_tmp, ignore_errors=True)
                shutil.rmtree(paths.export_prev, ignore_errors=True)
                shutil.rmtree(paths.export_doomed, ignore_errors=True)
            _refresh_index(config)


def publish_volume(
    volume: str,
    config: PublicationConfig,
    *,
    source: Path | None = None,
) -> Path:
    """Land one archive, its data exports, and the manifest under one lock.

    The source remains untouched on every failure. ``source`` only exists for the legacy bake
    script; queue and repair callers use the work-tree default. The researcher exports are
    staged as the transaction's first step and swapped in just before the manifest is BUILT
    (the story coverage gate reads them): a publish that cannot export fails loudly with
    nothing half-landed, and a failure after landing restores the previous export tree along
    with the archive and manifest. Publishing therefore needs the volume's work tree even with
    an explicit ``source``.
    """
    if not volume or Path(volume).name != volume or volume in {".", ".."}:
        raise PublicationError(f"invalid volume identifier: {volume!r}")
    work_archive = config.work / volume / f"{volume}.pmtiles"
    source = source or work_archive
    _assert_readable_archive(source)

    from ..config.load import load_city_config
    from ..exports import volume_page_services

    city = load_city_config(config.city_toml)
    viewer = load_viewer_config(config.city_toml)
    # Up front rather than at the manifest build: the same refusal fires there,
    # but by then the archive has landed and has to be rolled back out again.
    # Re-raised as this function's own error, which is what its callers catch.
    try:
        assert_serving_dirs_declared((config.city_tiles,), viewer.serving_dirs)
    except ConfigError as exc:
        raise PublicationError(str(exc)) from exc
    paths = _scratch(volume, config)
    try:
        # possibly a (cached, rate-limited) LOC fetch — resolve before the lock
        services = volume_page_services(
            volume, item_json=config.loc_item, cache_dir=config.loc_cache
        )
    except Exception as exc:
        raise PublicationError(f"cannot export {volume}: {exc}") from exc
    _publish_under_lock(
        volume, config, paths, city=city, viewer=viewer, source=source, services=services
    )
    # The debt is discharged, so clear it — here rather than in any one caller, so a
    # hand-run `autogeoref publish` settles it exactly as the queue does. AFTER the
    # lock: a crash between landing and this unlink leaves a marker claiming a publish
    # that already happened, and re-publishing is idempotent, while clearing first
    # would lose the debt on a rollback. Only when this WAS the work-tree archive: the
    # marker names that file, and an explicit --source publishes a different one.
    if source == work_archive:
        clear_publish_owed(volume, config.work)
    return paths.destination
