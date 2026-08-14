"""The web board payload: freshness fingerprint, cache, and the board dict."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import drain_lock
from ..bounds import Bounds, BoundsError
from ..config.model import CityConfig, ConfigError
from ..loc import catalog_year
from ..paths import VolumePaths
from ..queue.progress import board as queue_board
from ..queue.store import TRACKS, queue_path
from ..run_inputs import resolve_bounds
from ..viewer.sources import AreaIndex
from .actions import drain_state
from .backlog import candidates
from .text import REVIEW_URL, VIEWER_URL

if TYPE_CHECKING:
    from ..status import VolumeStatus

logger = logging.getLogger(__name__)


def _manifest_bounds(path: Path | None) -> dict[str, Bounds]:
    """Published volume footprints, or an empty map when the manifest is unavailable."""
    if path is None:
        return {}
    try:
        entries = json.loads(path.read_text(encoding="utf-8")).get("volumes") or []
    except (OSError, ValueError, TypeError):
        return {}
    out: dict[str, Bounds] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        volume, bounds = entry.get("id"), entry.get("bounds")
        if (
            isinstance(volume, str)
            and isinstance(bounds, list | tuple)
            and len(bounds) == 4
            and all(isinstance(value, int | float) for value in bounds)
        ):
            out[volume] = tuple(bounds)
    return out


#: Parsed community-area polygons, one per path: static config, reparsed only
#: when the file's mtime moves (the board used to re-parse it on every poll).
_area_indexes: dict[Path, tuple[int, AreaIndex]] = {}


def _area_index(path: Path) -> AreaIndex:
    """The ``AreaIndex`` for ``path``, cached until the file changes."""
    stamp = path.stat().st_mtime_ns
    cached = _area_indexes.get(path)
    if cached is None or cached[0] != stamp:
        cached = (stamp, AreaIndex(path))
        _area_indexes[path] = cached
    return cached[1]


def _volume_context(
    rows: list[VolumeStatus],
    city: CityConfig | None,
    catalog: dict[str, dict[str, Any]] | None,
    viewer_manifest: Path | None,
) -> dict[str, dict[str, Any]]:
    """Optional city, year, and neighborhood context for the board.

    This is display metadata, never a pipeline input. A missing or malformed
    catalog, viewer manifest, city footprint, or area file simply omits the fact
    it would have supplied; the operator can still run the queue.
    """
    if city is None:
        return {}
    catalog = catalog or {}
    areas = None
    if city.community_areas_path is not None and city.community_areas_path.is_file():
        try:
            areas = _area_index(city.community_areas_path)
        except Exception as exc:  # noqa: BLE001 - optional display data, never the console
            logger.warning("community-area context unavailable: %s", exc)
    manifest_bounds = _manifest_bounds(viewer_manifest)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        bounds = manifest_bounds.get(row.volume)
        if bounds is None:
            with suppress(BoundsError, ConfigError, OSError, ValueError, TypeError):
                bounds = resolve_bounds(city, city.volume(row.volume), viewer_manifest)
        neighborhoods: list[str] = []
        if areas is not None and bounds is not None:
            try:
                neighborhoods = areas.names(bounds)
            except Exception as exc:  # noqa: BLE001 - malformed geometry is display-only
                logger.warning("%s: neighborhood context unavailable: %s", row.volume, exc)
        out[row.volume] = {
            "city": city.name,
            "year": catalog_year(catalog, row.volume),
            "neighborhoods": neighborhoods,
        }
    return out


def _stamp(path: Path) -> tuple[int, int] | None:
    """``(mtime_ns, size)`` of ``path``, or None when it does not exist."""
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _subdirs(root: Path) -> list[Path]:
    try:
        return sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return []


def board_fingerprint(
    *,
    work: Path,
    fixtures: Path,
    tiles: Path,
    ground_truth: Path,
    config_files: Sequence[Path] = (),
) -> tuple[Any, ...]:
    """A cheap freshness key over everything the board payload derives from.

    Stat calls only — no file is read or parsed. Every write path that feeds the
    board moves the key: results, markers, sidecars and manifests are written
    atomically so the rename bumps the parent directory, the queue file and
    drain locks carry their own mtimes, and a drain that DIES without touching a
    file still flips :func:`drain_lock.live_drain`. Fixture volumes are frozen by
    contract, so only the fixtures root is watched. ``config_files`` covers the
    flat inputs the caller re-reads per build.
    """
    parts: list[Any] = [_stamp(p) for p in config_files]
    parts.append(_stamp(queue_path(work)))
    for track in TRACKS:
        parts.append(_stamp(drain_lock.drain_lock_path(work, track)))
        parts.append(drain_lock.live_drain(work, track))
    parts.extend(_stamp(root) for root in (work, fixtures, tiles, ground_truth))
    parts.extend((provenance.name, _stamp(provenance)) for provenance in _subdirs(tiles))
    for volume_dir in _subdirs(work):
        paths = VolumePaths(volume_dir)
        parts.append(
            (
                volume_dir.name,
                _stamp(volume_dir),
                _stamp(paths.results),
                _stamp(paths.markers),
                _stamp(paths.annotations),
                _stamp(paths.regions),
                _stamp(paths.manifest),
            )
        )
    return tuple(parts)


class BoardCache:
    """Reuse the last board payload while the tree's fingerprint is unchanged.

    A freshness-bounded read cache, never a persisted verdict: the key is
    recomputed from the tree on EVERY request (func:`board_fingerprint`), so a
    write shows up on the next poll, and nothing survives the process. Only the
    long-lived ``queue --serve`` server wraps its builder in one; ``status`` and
    the one-shot ``queue`` views derive directly from the tree each time.
    """

    def __init__(
        self, build: Callable[[], dict[str, Any]], fingerprint: Callable[[], object]
    ) -> None:
        self._build = build
        self._fingerprint = fingerprint
        self._lock = threading.Lock()
        self._key: object | None = None
        self._payload: dict[str, Any] | None = None

    def __call__(self) -> dict[str, Any]:
        # Key before build: a write landing mid-build files the payload under
        # the OLDER key, so the next poll rebuilds instead of trusting a
        # snapshot that may have missed it.
        key = self._fingerprint()
        with self._lock:
            if self._payload is None or key != self._key:
                self._payload = self._build()
                self._key = key
            return self._payload


def board(
    *,
    work: Path,
    rows: list[VolumeStatus],
    city: CityConfig | None = None,
    catalog: dict[str, dict[str, Any]] | None = None,
    review_url: str = REVIEW_URL,
    viewer_url: str = VIEWER_URL,
    candidates_command: str = "autogeoref queue --candidates",
    era_command: str | None = None,
    can_act: bool = False,
    viewer_manifest: Path | None = None,
) -> dict[str, Any]:
    """The four-column board as plain data: runnable · running · needs you · served.

    A superset of :func:`queue.progress.board` — ``entries`` is that payload verbatim, so the
    terminal view and this one cannot drift. ``can_act`` is whether this server was given a city
    config, and therefore whether its buttons exist at all. ``candidates_command`` and
    ``era_command`` are echoed to the page rather than composed there: the HTML must not learn a
    city's name or config path one floor down. The blocked card appends its volume ids to
    ``era_command`` — ids the payload already carries, so nothing new leaks.
    """
    payload = queue_board(work)
    served = [
        {"volume": r.volume, "sheets": r.sheets, "accepted": r.accepted, "flagged": r.flagged}
        for r in rows
        if r.ours
    ]
    entries = payload["entries"]
    context = _volume_context(rows, city, catalog, viewer_manifest)
    return {
        "generated": payload["generated"],
        "entries": entries,
        "context": context,
        # The queues, in pipeline order. Sent rather than hardcoded in the page so the
        # page cannot disagree with the queue about what a track IS. It does not make
        # a new track render for free: the prose describing a queue lives with its
        # heading, so a track still needs a dot, a table and a target option in the
        # markup — but the page now reports a track it was given and cannot render,
        # instead of dropping its rows in silence.
        "tracks": list(TRACKS),
        "runnable": [asdict(c) for c in candidates(rows, work=work, city=city, catalog=catalog)],
        "served": served,
        # With no city config there is no address-era check, so EVERY candidate
        # renders as ready — including the ones `autogeoref run` refuses on its
        # first line. The page must say so out loud rather than quietly present a
        # backlog it could not vet: a console that is silently wrong about what is
        # runnable is worse than one that admits it does not know.
        "era_check": city is not None,
        "can_act": can_act,
        # Is a drain running, and is it one this page could stop? Read off the LOCK,
        # which is the queue's own answer to that question (`drain_lock.live_drain`) — the
        # page never infers "running" from a queue entry's status field, because a
        # drain killed at 3am leaves that field saying `running` forever.
        "drain": drain_state(work),
        # What the buttons have SPENT: reads are the budget, and this is the same
        # number `status.annotation_reads` bills — summed over the queue, so a drain
        # you started an hour ago tells you what it cost without opening a log.
        "spend": {
            "reads": sum(e["progress"]["reads"] for e in entries),
            "reads_running": sum(
                e["progress"]["reads"] for e in entries if e["status"] == "running"
            ),
        },
        "links": {
            "review": review_url,
            "viewer": viewer_url,
            "candidates": candidates_command,
            "era": era_command,
        },
    }
