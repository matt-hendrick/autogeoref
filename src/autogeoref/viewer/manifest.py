"""Viewer manifest assembly: served layers + catalog metadata + site config."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from ..config.model import ConfigError
from ..paths import atomic_write_text
from .bounds import pmtiles_bounds
from .config import ViewerConfig, site_dict
from .coverage import NO_FOOTPRINTS, SheetFootprints, assert_stops_are_covered
from .era import EraBucket, era_label
from .sources import (
    AreaIndex,
    _relpath,
    classify_pmtiles,
    loc_titles,
)

logger = logging.getLogger(__name__)


def assert_serving_dirs_declared(pmtiles_dirs: Sequence[Path], declared: Sequence[str]) -> None:
    """Refuse a serving directory this city has not vouched for.

    Every layer is published under the city's one credit line, so an undeclared
    directory would publish someone else's georeferencing as this project's.
    Matches the directory's BASE NAME, so it catches the wrong directory being
    passed, not a directory named to pass — treat it as a mistake-catcher.
    Resolve before naming: ``Path(".").name`` is the empty string, which matches
    nothing and would read as "no directory here".
    """
    known = set(declared)
    # name BOTH when they differ: a relative path reports as "." otherwise, and
    # the operator cannot see which name was actually checked
    foreign = sorted(
        f"{d} (name {name!r})" if str(d) != name else name
        for d in pmtiles_dirs
        if (name := d.name or d.resolve().name) not in known
    )
    if foreign:
        raise ConfigError(
            f"{', '.join(foreign)}: serving directory not declared in "
            f"viewer.serving_dirs ({', '.join(declared)}) — these layers would be "
            f"published under this city's credit line. Declare the directory's "
            f"name if the layers are yours to serve, or stop serving them."
        )


def no_layers_note(pmtiles_dirs: Sequence[Path]) -> str:
    """Why a build found nothing, worded once for both callers that refuse it.

    An empty manifest is a legal object — a city with nothing baked yet has one
    — but it is never a page worth publishing, so the two commands that write
    one for an operator say so instead.
    """
    named = ", ".join(str(d) for d in pmtiles_dirs) or "(no directory given)"
    return (
        f"no .pmtiles archives under {named} — nothing to publish. Bake an "
        "archive, or point --pmtiles at the directory holding one"
    )


def _collect_pmtiles(pmtiles_dirs: Sequence[Path]) -> dict[str, Path]:
    """Every served archive under the pmtiles roots, first-directory-wins."""
    volume_files: dict[str, Path] = {}
    for directory in pmtiles_dirs:
        for ident, path in classify_pmtiles(directory).items():
            volume_files.setdefault(ident, path)
    return volume_files


class BoundsProbes(NamedTuple):
    """How a manifest build reads an artifact's extent; injectable for tests."""

    pmtiles: Callable[[Path], list[float]] = pmtiles_bounds


class AreaSource(NamedTuple):
    """A community-area GeoJSON and the property its names live under."""

    path: Path | None = None
    name_property: str = "community"

    def load(self) -> AreaIndex | None:
        """The loaded index, or None when no readable file is declared."""
        if self.path is None or not self.path.exists():
            return None
        return AreaIndex(self.path, self.name_property)


#: the PMTiles header — an archive's own extent, and the only bounds source.
DEFAULT_PROBES = BoundsProbes()

#: No community-area overlay, which is what a city that declares none gets.
NO_AREAS = AreaSource()


def _volume_entries(
    identifiers: Sequence[str],
    *,
    meta: Mapping[str, Mapping[str, Any]],
    volume_files: Mapping[str, Path],
    areas: AreaIndex | None,
    declared_labels: Mapping[str, str],
    buckets: Sequence[EraBucket],
    probes: BoundsProbes,
    out_dir: Path,
) -> list[dict[str, Any]]:
    volumes: list[dict[str, Any]] = []
    for ident in identifiers:
        m = meta.get(ident) or {}
        bounds = probes.pmtiles(volume_files[ident])
        entry: dict[str, Any] = {
            "id": ident,
            "title": m.get("title", ident),
            "year": m.get("year"),
            "volume_number": m.get("volume_number"),
            "era": era_label(m.get("year"), buckets),
            "bounds": bounds,
            "areas": areas.names(bounds) if areas is not None else [],
        }
        # The district-list name, strongest claim first: a declared label, the
        # LOC-derived subject (unnumbered specials only), a retained label from
        # the previous manifest. Absent = the viewer shows community areas.
        vol_label = declared_labels.get(ident) or m.get("subject") or m.get("label")
        if vol_label:
            entry["label"] = vol_label
        entry["pmtiles"] = _relpath(volume_files[ident], out_dir)
        volumes.append(entry)
        logger.info("%s: %s [%s] -> %s", ident, entry["title"], entry["era"], entry["areas"])
    return volumes


def build_manifest(
    city_name: str,
    viewer: ViewerConfig,
    *,
    out_path: Path,
    pmtiles_dirs: Sequence[Path] = (),
    loc_catalog: Path | None = None,
    metadata_fallback: Mapping[str, Mapping[str, Any]] | None = None,
    areas: AreaSource = NO_AREAS,
    probes: BoundsProbes = DEFAULT_PROBES,
    footprints: SheetFootprints = NO_FOOTPRINTS,
) -> dict[str, Any]:
    """Assemble the viewer manifest (volumes, site). Paths in the manifest are
    relative to ``out_path``'s directory — no host, no scheme, so the output is
    serve-anywhere by construction.

    ``pmtiles_dirs`` is ordered and earlier directories win, so a volume served
    from the first directory shadows the same volume in a later one. A directory
    the city has not declared in ``serving_dirs`` stops the build.
    """
    out_dir = out_path.parent
    assert_serving_dirs_declared(pmtiles_dirs, viewer.serving_dirs)
    meta = loc_titles(loc_catalog, city_name) if loc_catalog else {}
    area_index = areas.load()
    volume_files = _collect_pmtiles(pmtiles_dirs)
    identifiers = sorted(volume_files)
    declared_labels = dict(viewer.volume_labels)
    for unmatched in sorted(set(declared_labels) - set(identifiers)):
        logger.warning("viewer.labels: %s names no served volume; label unused", unmatched)
    # A repair publish is often run without a local catalog: fall back to the
    # current viewer metadata rather than reducing every layer to its id.
    resolved = {
        ident: meta.get(ident) or (metadata_fallback or {}).get(ident) or {}
        for ident in identifiers
    }
    volumes = _volume_entries(
        identifiers,
        meta=resolved,
        volume_files=volume_files,
        areas=area_index,
        declared_labels=declared_labels,
        buckets=viewer.era_buckets,
        probes=probes,
        out_dir=out_dir,
    )

    manifest: dict[str, Any] = {"volumes": volumes}
    site = site_dict(city_name, viewer)
    if viewer.stories is not None:
        if not volumes:
            # Before the gate: with nothing served EVERY stop fails coverage,
            # and the gate's message names a stop that is in fact covered,
            # which reads as a guided-story bug and is not one.
            raise ConfigError(
                f"{no_layers_note(pmtiles_dirs)} — and every story stop would fail coverage."
            )
        # Before a stop can be published, the ground under its camera has to be
        # on the map. Say which volume will draw it, by title: an envelope that
        # looks like a district can belong to a special that draws something
        # else, and the person writing captions is not the one who would notice.
        for line in assert_stops_are_covered(viewer.stories, volumes, footprints):
            logger.info("story stop covered by %s", line)
    manifest["site"] = site
    return manifest


def write_manifest(manifest: Mapping[str, Any], out_path: Path) -> Path:
    # atomic: bake-queue landings regenerate the manifest while the viewer
    # server is range-reading it — a reader must never see a partial file
    return atomic_write_text(out_path, json.dumps(manifest, indent=1))
