"""The coverage gate: does a story stop actually look at placed sheets?

A caption pointing at blank paper is the failure this exists to catch. The
extents come from the researcher exports and are an INNER bound, so this
under-reports at the margins and says nothing about a hole inside a sheet.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..config.model import ConfigError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from .stories import StoriesConfig

logger = logging.getLogger(__name__)


class SheetFootprints:
    """Per-volume placed-sheet extents, read from the researcher exports.

    A sheet's extent is the axis-aligned box of its recorded control points.
    It is an INNER bound and it is approximate: the sheet reaches past its
    outermost junctions, and a sheet rotated off north extends past its own
    box's corners. So this over- and under-reports at the margins, and it says
    nothing about nodata inside a placed sheet. What it does see, and a volume
    envelope cannot, is a hole where sheets were never placed.
    """

    def __init__(self, exports_root: Path | None = None) -> None:
        self.exports_root = exports_root
        self._cache: dict[str, list[tuple[float, float, float, float]]] | None = None

    def _read_volume(self, gcps_dir: Path) -> list[tuple[float, float, float, float]]:
        boxes = []
        unreadable = 0
        for record_path in sorted(gcps_dir.glob("*.json")):
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
                points = [
                    feature["geometry"]["coordinates"]
                    for feature in record["gcps_geojson"]["features"]
                ]
                lngs = [float(p[0]) for p in points]
                lats = [float(p[1]) for p in points]
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                unreadable += 1
                continue
            if len(points) < 3:
                continue
            boxes.append((min(lngs), min(lats), max(lngs), max(lats)))
        if unreadable:
            # silence here would quietly demote this volume to an envelope check
            logger.warning("%s: %d unreadable export record(s); skipped", gcps_dir, unreadable)
        return boxes

    def by_volume(self) -> dict[str, list[tuple[float, float, float, float]]]:
        """Volume id -> its placed sheets' boxes. Empty when nothing is known."""
        if self._cache is None:
            self._cache = {}
            if self.exports_root is None:
                return self._cache
            if not self.exports_root.is_dir():
                # the gate silently weakens to an envelope check otherwise, and
                # a mistyped or wrong-cwd path is the likeliest way to get here
                logger.warning(
                    "%s is not a directory: story coverage falls back to volume envelopes",
                    self.exports_root,
                )
                return self._cache
            for volume_dir in sorted(self.exports_root.iterdir()):
                gcps = volume_dir / "gcps"
                if gcps.is_dir():
                    boxes = self._read_volume(gcps)
                    if boxes:
                        self._cache[volume_dir.name] = boxes
        return self._cache


#: What a manifest build knows about placed sheets when nobody says otherwise.
NO_FOOTPRINTS = SheetFootprints()


def _contains(box: Sequence[float], lng: float, lat: float) -> bool:
    return box[0] <= lng <= box[2] and box[1] <= lat <= box[3]


@dataclass(frozen=True)
class CoverageMatch:
    """One layer that covers a stop's camera, and how well it is known."""

    volume: str
    title: str
    era: str
    #: True when a placed sheet contains the point; False when only the
    #: volume's published envelope does, which cannot see a hole inside it.
    from_sheets: bool


def covering_layers(
    lng: float,
    lat: float,
    eras: Sequence[str],
    volumes: Sequence[Mapping[str, Any]],
    footprints: SheetFootprints,
) -> tuple[list[CoverageMatch], list[CoverageMatch]]:
    """``(covering, envelope_only)`` for one point, restricted to ``eras``.

    A volume counts as served on the same rule the deploy bundle uses: it has
    its own archive.
    """
    by_volume = footprints.by_volume()
    covering: list[CoverageMatch] = []
    envelope_only: list[CoverageMatch] = []
    wanted = set(eras)
    for volume in volumes:
        era = str(volume.get("era") or "")
        if wanted and era not in wanted:
            continue
        if "pmtiles" not in volume:
            continue  # listed, but nothing of it is on the map
        bounds = volume.get("bounds")
        if not bounds or not _contains(bounds, lng, lat):
            continue
        ident = str(volume.get("id"))
        match = CoverageMatch(
            volume=ident,
            title=str(volume.get("title") or ident),
            era=era,
            from_sheets=ident in by_volume,
        )
        if ident not in by_volume:
            # nothing placed by this pipeline to check against (a foreign or
            # not-yet-exported layer): the envelope is the best evidence there
            covering.append(match)
        elif any(_contains(box, lng, lat) for box in by_volume[ident]):
            covering.append(match)
        else:
            envelope_only.append(match)
    return covering, envelope_only


def assert_stops_are_covered(
    config: StoriesConfig,
    volumes: Sequence[Mapping[str, Any]],
    footprints: SheetFootprints = NO_FOOTPRINTS,
) -> list[str]:
    """Refuse a manifest whose story stops point at ground nothing covers.

    Returns one report line per stop naming the volume that will draw it, by
    TITLE: the archives hold specials whose envelopes look like ordinary
    districts, so "something is here" is not the same as "this is the volume
    the caption means".
    """
    lines: list[str] = []
    for story in config.stories:
        for stop in story.stops:
            lng, lat = stop.camera.center
            covering, envelope_only = covering_layers(lng, lat, stop.eras, volumes, footprints)
            if not covering:
                near = (
                    " Inside the published envelope of "
                    + ", ".join(f"{m.volume} ({m.title})" for m in envelope_only)
                    + ", but no placed sheet reaches it."
                    if envelope_only
                    else ""
                )
                raise ConfigError(
                    f"{config.path}: story {story.id!r} stop {stop.id!r} puts its camera at "
                    f"{lng}, {lat} in era(s) {', '.join(stop.eras) or 'any'}, which no served "
                    f"layer covers.{near}"
                )
            drawn = ", ".join(
                f"{m.volume} {m.title}" + ("" if m.from_sheets else " [no sheet record]")
                for m in covering
            )
            lines.append(f"{story.id}/{stop.id}: {drawn}")
    return lines
