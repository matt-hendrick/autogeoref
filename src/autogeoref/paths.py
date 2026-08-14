"""Per-volume work-tree layout and shared results-directory access.

Pure path arithmetic plus the one place the ``results/p<N>.json`` iteration
pattern lives (func:`iter_results`) and the one place the
``rotation_applied`` frame guard lives (func:`small_sheet_entry`). Stage
modules import :class:`VolumePaths` from here (zero project dependencies, so
no import cycles) and type their ``paths`` parameters properly.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .slugs import page_from_slug

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VolumePaths:
    """Layout of one volume's work tree (``work_dir/<volume>/``)."""

    root: Path

    @property
    def sheets(self) -> Path:
        return self.root / "sheets"

    @property
    def annotations(self) -> Path:
        return self.root / "annotations"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def regions(self) -> Path:
        return self.root / "regions"

    @property
    def masks(self) -> Path:
        return self.root / "masks"

    @property
    def warped(self) -> Path:
        return self.root / "warped"

    @property
    def markers(self) -> Path:
        return self.root / "markers"

    @property
    def manifest(self) -> Path:
        return self.sheets / "manifest.json"

    @property
    def seam_deltas(self) -> Path:
        return self.root / "seam_deltas.json"

    @property
    def scores(self) -> Path:
        """Ground-truth scores for this volume's placements, keyed by page.

        A sidecar, never a field on a result record: the record says what the
        pipeline did and this says something about it, so nothing a run reads
        can be gated on a human score (:mod:`autogeoref.scoring`).
        """
        return self.root / "results-scores.json"

    @property
    def constants(self) -> Path:
        return self.root / "volume-constants.json"

    @property
    def name_match(self) -> Path:
        """Street-name match counts written by the match stage (name_match)."""
        return self.root / "name-match.json"

    @property
    def publish_owed(self) -> Path:
        """Set when a bake completes, cleared by the publish that lands it.

        The ONLY artifact that separates "this tree holds a complete archive
        nobody has published" from "the bake never finished". Neither the
        archive nor the served copy can say it: the archive's presence is
        silent about publication, and an EARLIER bake's published copy sits at
        the same destination path. :func:`viewer.record_publish_owed` owns the
        format; the queue reads it to describe and finish a dead drain's
        leftovers without re-baking (see :func:`queue.publish.finish_owed_publishes`).
        """
        return self.root / "publish-owed.json"

    @property
    def lock(self) -> Path:
        """THE volume-ownership lock file — one location for every entry point.

        Every mutating operation on this tree (a placement run, a ``--warp-only``
        bake, standalone prep, review apply) takes :func:`volume_lock` on this
        one path. The file is permanent — created on first use and never
        unlinked — because the exclusion authority is the kernel ``flock`` held
        on it, not the file's existence.
        """
        return self.root / "volume.lock"


#: Extensions a full-res sheet scan can arrive in. All consumers use these
#: helpers so prep, status, review, and baking agree on what is a sheet.
SHEET_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".jp2"})


def sheet_images(regions_dir: Path) -> list[Path]:
    """Every full-res sheet scan on disk, page-addressable or not."""
    if not regions_dir.is_dir():
        return []
    return [path for path in sorted(regions_dir.iterdir()) if path.suffix.lower() in SHEET_SUFFIXES]


def regions_by_page(regions_dir: Path) -> dict[str, Path]:
    """``{page: full-res image path}`` for every recognized sheet scan."""
    out: dict[str, Path] = {}
    for path in sheet_images(regions_dir):
        page = page_from_slug(path.stem)
        if page is not None:
            out[page] = path
    return out


def iter_results(
    paths: VolumePaths,
    *,
    sort_key: Callable[[Path], Any] | None = None,
) -> Iterator[tuple[str, dict[str, Any], Path]]:
    """Every recorded result: ``(page, record, path)``, path-sorted.

    The one owner of the ``results/p*.json`` read loop every stage repeats.
    Default ordering is the plain lexicographic path sort; pass ``sort_key``
    where page-numeric order matters. A record missing its ``page`` key falls
    back to the filename-derived page WITH a warning — pipeline-written records
    always carry it, so a firing warning means a damaged results directory.
    """
    files = paths.results.glob("p*.json")
    for rp in sorted(files) if sort_key is None else sorted(files, key=sort_key):
        r = json.loads(rp.read_text())
        page = r.get("page")
        if page is None:
            page = rp.stem.removeprefix("p")
            logger.warning("%s: record has no 'page' key; using filename page %s", rp, page)
        yield str(page), r, rp


@contextmanager
def atomic_output_path(path: Path, *, publish: bool = True) -> Iterator[Path]:
    """Yield a unique same-directory temporary path and optionally publish it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        target_fd = os.open(path, os.O_WRONLY)
    except FileNotFoundError:
        target_stat = None
    else:
        try:
            target_stat = os.fstat(target_fd)
        finally:
            os.close(target_fd)
    while True:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            continue
        break
    open_fd = fd
    try:
        if target_stat is not None:
            # Match an in-place write: retain mode bits and supported xattrs
            # (including POSIX ACLs) from the record being replaced.
            os.fchown(fd, target_stat.st_uid, target_stat.st_gid)
            shutil.copystat(path, temporary)
        os.close(fd)
        open_fd = -1
        yield temporary
        if publish:
            temporary_fd = os.open(temporary, os.O_WRONLY)
            try:
                os.fsync(temporary_fd)
            finally:
                os.close(temporary_fd)
            temporary.replace(path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        else:
            temporary.unlink()
    except BaseException:
        if open_fd != -1:
            os.close(open_fd)
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> Path:
    """Replace ``path`` with complete text written to a sibling temporary file."""
    with atomic_output_path(path) as temporary:
        temporary.write_text(text, encoding="utf-8")
    return path


def write_result(path: Path, record: Mapping[str, Any]) -> Path:
    """Atomically write one ``results/p<N>.json`` record in its stable format."""
    return atomic_write_text(path, json.dumps(record, indent=2))


def write_if_changed(path: Path, text: str) -> Path:
    """Atomically write ``text`` only when content differs (mtime-stable)."""
    if path.exists() and path.read_text() == text:
        return path
    return atomic_write_text(path, text)


class VolumeBusyError(RuntimeError):
    """Another process owns this volume's work tree right now.

    Carries the holder's recorded metadata (``holder``: pid / operation /
    started, best-effort) and the lock path, so every entry point can print
    the same actionable refusal.
    """

    def __init__(self, volume: str, lock_path: Path, holder: Mapping[str, Any] | None) -> None:
        self.volume = volume
        self.lock_path = lock_path
        self.holder = dict(holder) if holder else None
        if self.holder:
            operation = self.holder.get("operation") or "an operation"
            pid = self.holder.get("pid")
            started = self.holder.get("started")
            since = (
                time.strftime(" since %Y-%m-%d %H:%M:%S", time.localtime(started))
                if isinstance(started, (int, float))
                else ""
            )
            detail = f"{operation} (pid {pid}) has held {lock_path}{since}"
        else:
            detail = f"another process holds {lock_path}"
        super().__init__(
            f"{volume} is busy: {detail}. Two owners would duplicate model reads and "
            "interleave result/mask writes — wait for it to finish, or stop it."
        )


@contextmanager
def volume_lock(paths: VolumePaths, operation: str) -> Iterator[None]:
    """One mutating owner per volume work tree — nonblocking, kernel-held.

    THE per-volume exclusion every mutating entry point takes. Read-only commands and side-
    effect-free dry runs must NOT take it; acquire it before any model or GDAL work and hold it
    across the whole operation. The authority is a nonblocking ``flock`` on a permanent lock
    file, so a holder killed by any signal releases it automatically and there is no stale-file
    reclamation. The JSON metadata inside only makes the refusal actionable, and is cleared
    before release while the lock is still held.
    """
    lock = paths.lock
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                raw = json.loads(lock.read_text())
            except (OSError, ValueError):
                raw = None
            holder = raw if isinstance(raw, dict) else None
            raise VolumeBusyError(paths.root.name, lock, holder) from None
        # metadata AFTER acquiring: only ever describes the actual holder. A
        # crashed holder may leave its bytes behind — truncate first; the flock
        # is already gone with the process, so those bytes never refuse anyone.
        os.ftruncate(fd, 0)
        os.write(
            fd,
            json.dumps(
                {"pid": os.getpid(), "operation": operation, "started": time.time()}
            ).encode(),
        )
        try:
            yield
        finally:
            os.ftruncate(fd, 0)  # clear while still holding — never someone else's bytes
    finally:
        os.close(fd)  # releases the flock


def small_sheet_entry(
    paths: VolumePaths,
    manifest: Mapping[str, Any],
    page: str,
    *,
    stage: str,
    require_image: bool = True,
    skip_rotated: bool = True,
) -> tuple[dict[str, Any], Path] | None:
    """Manifest entry + small-image path for one page, with the frame guard.

    Bundles the checks every small-consuming stage must repeat: the manifest entry exists, the
    small image exists, and — the eternal trap — ``rotation_applied`` is absent, because
    evidence read off an upright small cannot be scored against the record's SOURCE-frame
    affine. ``skip_rotated=False`` is for stages that instead COMPOSE the recorded rotation,
    turning their evidence back into the source frame first. Pass it only with one of those in
    hand — it is the guard, not a nuisance.
    """
    info = manifest.get(f"p{page}")
    img = paths.sheets / f"p{page}_small.jpg"
    if info is None or (require_image and not img.exists()):
        return None
    if skip_rotated and info.get("rotation_applied"):
        logger.warning("p%s: rotation-normalized small; %s skipped", page, stage)
        return None
    return info, img


__all__ = [
    "SHEET_SUFFIXES",
    "VolumeBusyError",
    "VolumePaths",
    "atomic_output_path",
    "atomic_write_text",
    "iter_results",
    "regions_by_page",
    "sheet_images",
    "small_sheet_entry",
    "volume_lock",
    "write_if_changed",
    "write_result",
]
