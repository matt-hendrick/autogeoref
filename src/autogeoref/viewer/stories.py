"""Guided stories: an optional sidecar of ordered map stops, and its gates.

A story is a list of stops; a stop is a camera, an era selection, a swipe
fraction and some prose. Everything here is optional — a city that declares
no ``[viewer.stories]`` gets a manifest with no ``stories`` key and a viewer
with no story UI.

The schema is stated once in ``docs/ADDING-A-CITY.md``. This module refuses a
story rather than rendering a broken one; whether a stop actually looks at
placed sheets is ``coverage``'s question.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from ..config.model import ConfigError
from .era import EraBucket, era_label

#: Where local story assets are staged inside the viewer directory, and the
#: prefix their manifest URLs carry. One conventional name, so the deploy
#: bundle can copy it without knowing anything about the city.
ASSETS_DIR = "story-assets"

#: MapLibre's zoom range.
ZOOM_RANGE = (0.0, 24.0)


@dataclass(frozen=True)
class Camera:
    """Where a stop puts both panes."""

    center: tuple[float, float]  # lng, lat
    zoom: float
    bearing: float | None = None
    pitch: float | None = None


@dataclass(frozen=True)
class Overlay:
    """An outline drawn on the atlas pane — the claim the caption is making.

    ``geojson`` is inline; a ``file`` is read at config-load time and inlined,
    so the viewer needs no second fetch and the deploy bundle no extra copy.
    """

    geojson: dict[str, Any]
    style: dict[str, Any]


@dataclass(frozen=True)
class Media:
    """One image beside the caption. ``src`` is a local asset or an https URL."""

    src: str
    alt: str
    caption: str | None = None
    credit: str | None = None
    href: str | None = None


@dataclass(frozen=True)
class Source:
    """One citation line under a stop."""

    label: str
    href: str | None = None


@dataclass(frozen=True)
class Stop:
    """One position in a story: camera, era selection, and what to read."""

    id: str
    title: str
    body_html: str = ""
    camera: Camera = Camera(center=(0.0, 0.0), zoom=12.0)
    eras: tuple[str, ...] = ()
    swipe: float | None = None
    overlay: Overlay | None = None
    media: tuple[Media, ...] = ()
    sources: tuple[Source, ...] = ()


@dataclass(frozen=True)
class Story:
    """An ordered set of stops, read start to finish."""

    id: str
    title: str
    dek: str = ""
    stops: tuple[Stop, ...] = ()


@dataclass(frozen=True)
class StoriesConfig:
    """A parsed ``[viewer.stories]`` block: the sidecar and what it held."""

    path: Path
    stories: tuple[Story, ...]
    assets_dir: Path | None = None


def _table(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where}: expected an object, got {type(value).__name__}")
    return value


def _text(entry: Mapping[str, Any], key: str, where: str, *, required: bool = False) -> str:
    value = entry.get(key)
    if value is None:
        if required:
            raise ConfigError(f"{where}: {key} is required")
        return ""
    if not isinstance(value, str) or (required and not value.strip()):
        raise ConfigError(f"{where}: {key} must be a non-empty string, got {value!r}")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{where}: expected a number, got {value!r}")
    return float(value)


def _identifier(entry: Mapping[str, Any], where: str) -> str:
    """An id, which is also a permalink key. Restricted to what survives a URL
    fragment unescaped: an ``&`` or ``=`` in one would inject a second key into
    the shared hash, and the viewer would read it as somebody else's setting."""
    value = _text(entry, "id", where, required=True)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ConfigError(
            f"{where}: id {value!r} must be letters, digits, '-' and '_' only — it is a "
            f"permalink key"
        )
    return value


def _is_era_label(label: str, known: set[str], buckets: Sequence[EraBucket]) -> bool:
    """The rule ``default_eras`` uses: a declared bucket label, or a bare year
    outside every bucket (such a year labels its own chip)."""
    if label in known:
        return True
    return bool(re.fullmatch(r"\d{4}", label)) and era_label(int(label), buckets) == label


def _parse_camera(raw: Any, where: str) -> Camera:
    entry = _table(raw, where)
    center = entry.get("center")
    if not isinstance(center, Sequence) or isinstance(center, str) or len(center) != 2:
        raise ConfigError(f"{where}: camera.center must be [lng, lat]")
    lng, lat = (_number(center[0], where), _number(center[1], where))
    if not -180.0 <= lng <= 180.0 or not -90.0 <= lat <= 90.0:
        raise ConfigError(f"{where}: camera.center {[lng, lat]} is not a lng/lat pair")
    zoom = _number(entry.get("zoom", 15), where)
    if not ZOOM_RANGE[0] <= zoom <= ZOOM_RANGE[1]:
        raise ConfigError(f"{where}: camera.zoom {zoom} is outside {ZOOM_RANGE}")
    return Camera(
        center=(lng, lat),
        zoom=zoom,
        bearing=_number(entry["bearing"], where) if "bearing" in entry else None,
        pitch=_number(entry["pitch"], where) if "pitch" in entry else None,
    )


def _inside(root: Path, relative: str, where: str) -> Path:
    """``root/relative``, proven to stay inside ``root``.

    Resolved, not string-matched: ``a/../../secret`` starts with neither ``..``
    nor ``/`` and would otherwise read — and publish — a file outside the
    configured directory.
    """
    if PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).is_absolute():
        raise ConfigError(f"{where}: {relative!r} must be a path inside {root}, not absolute")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ConfigError(f"{where}: {relative!r} resolves outside {root}")
    return resolved


def _resolve_asset(src: str, where: str, assets_dir: Path | None) -> str:
    """A local asset path proven to sit inside ``assets_dir`` and prefixed for
    the manifest; an ``https://`` URL passed through untouched.

    Prefer local copies — the same reasoning that self-hosts the basemap.
    """
    if src.startswith("https://"):
        return src
    if "://" in src or src.split("/", 1)[0].endswith(":"):
        raise ConfigError(f"{where}: {src!r} must be an https:// URL or a local asset path")
    if assets_dir is None:
        raise ConfigError(f"{where}: {src!r} is a local asset but no viewer.stories.assets is set")
    if not _inside(assets_dir, src, where).is_file():
        raise ConfigError(f"{where}: {src!r} does not exist under {assets_dir}")
    return f"{ASSETS_DIR}/{src}"


def _link(value: Any, where: str) -> str | None:
    """An outbound link, or None. Only http(s) and mailto: are rendered — a
    ``javascript:`` href would run in the page as soon as a reader clicked."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.startswith(("https://", "http://", "mailto:")):
        raise ConfigError(f"{where}: href must be an http(s) or mailto: URL, got {value!r}")
    return value


def _parse_overlay(raw: Any, where: str, *, sidecar: Path, assets_dir: Path | None) -> Overlay:
    entry = _table(raw, where)
    inline, filename = entry.get("geojson"), entry.get("file")
    if (inline is None) == (filename is None):
        raise ConfigError(f"{where}: overlay needs exactly one of geojson or file")
    if filename is not None:
        if not isinstance(filename, str):
            raise ConfigError(f"{where}: overlay.file must be a string")
        # relative to the assets dir when there is one, else to the sidecar
        root = assets_dir or sidecar.parent
        path = _inside(root, filename, where)
        if not path.is_file():
            raise ConfigError(f"{where}: overlay.file {filename!r} does not exist under {root}")
        try:
            inline = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ConfigError(f"{where}: overlay.file {path} is not valid JSON: {exc}") from exc
    style = _table(entry.get("style") or {}, f"{where} overlay.style")
    # the style reaches setPaintProperty verbatim, so refuse a wrong type here
    # rather than in somebody's browser
    for key in ("width", "fill_opacity"):
        if key in style:
            _number(style[key], f"{where} overlay.style.{key}")
    if "color" in style and not isinstance(style["color"], str):
        raise ConfigError(f"{where} overlay.style.color must be a string, got {style['color']!r}")
    if "dash" in style and not (
        isinstance(style["dash"], list) and all(isinstance(n, int | float) for n in style["dash"])
    ):
        raise ConfigError(f"{where} overlay.style.dash must be a list of numbers")
    return Overlay(geojson=dict(_table(inline, where)), style=dict(style))


def _parse_stop(
    raw: Any, where: str, *, sidecar: Path, assets_dir: Path | None, buckets: Sequence[EraBucket]
) -> Stop:
    entry = _table(raw, where)
    stop_id = _identifier(entry, where)
    where = f"{where} (stop {stop_id})"
    if "camera" not in entry:
        raise ConfigError(f"{where}: camera is required")

    eras = entry.get("eras") or []
    if not isinstance(eras, list) or not all(isinstance(e, str) for e in eras):
        raise ConfigError(f"{where}: eras must be a list of era labels")
    known = {b.label for b in buckets}
    for label in eras:
        if not _is_era_label(label, known, buckets):
            raise ConfigError(
                f"{where}: era {label!r} is not an era bucket label "
                f"(expected one of {', '.join(sorted(known)) or 'none declared'})"
            )

    swipe = None
    if "swipe" in entry:
        swipe = _number(entry["swipe"], where)
        if not 0.0 <= swipe <= 1.0:
            raise ConfigError(f"{where}: swipe {swipe} is not a fraction between 0 and 1")

    media = []
    for item in entry.get("media") or []:
        m = _table(item, f"{where} media")
        media.append(
            Media(
                src=_resolve_asset(
                    _text(m, "src", f"{where} media", required=True), f"{where} media", assets_dir
                ),
                alt=_text(m, "alt", f"{where} media", required=True),
                caption=_text(m, "caption", f"{where} media") or None,
                credit=_text(m, "credit", f"{where} media") or None,
                href=_link(m.get("href"), f"{where} media"),
            )
        )
    sources = []
    for item in entry.get("sources") or []:
        source = _table(item, f"{where} sources")
        sources.append(
            Source(
                label=_text(source, "label", f"{where} sources", required=True),
                href=_link(source.get("href"), f"{where} sources"),
            )
        )
    return Stop(
        id=stop_id,
        title=_text(entry, "title", where, required=True),
        body_html=_text(entry, "body_html", where),
        camera=_parse_camera(entry["camera"], where),
        eras=tuple(eras),
        swipe=swipe,
        overlay=(
            _parse_overlay(entry["overlay"], where, sidecar=sidecar, assets_dir=assets_dir)
            if entry.get("overlay") is not None
            else None
        ),
        media=tuple(media),
        sources=tuple(sources),
    )


def load_stories(
    sidecar: Path, *, assets_dir: Path | None, buckets: Sequence[EraBucket]
) -> tuple[Story, ...]:
    """Parse and validate a story sidecar. Refuses rather than half-renders.

    ``buckets`` are the city's ``[[viewer.era]]`` declarations: a stop may only
    name an era the viewer will actually show a chip for.
    """
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"viewer.stories.file: cannot read {sidecar}: {exc}") from exc
    except ValueError as exc:
        raise ConfigError(f"{sidecar}: not valid JSON: {exc}") from exc

    entries = raw.get("stories") if isinstance(raw, Mapping) else raw
    if not isinstance(entries, list):
        raise ConfigError(f"{sidecar}: expected a list of stories, or an object with `stories`")

    stories: list[Story] = []
    seen_stories: set[str] = set()
    for index, item in enumerate(entries):
        where = f"{sidecar} story {index}"
        entry = _table(item, where)
        story_id = _identifier(entry, where)
        if story_id in seen_stories:
            raise ConfigError(f"{sidecar}: duplicate story id {story_id!r}")
        seen_stories.add(story_id)
        where = f"{sidecar} story {story_id!r}"
        stops: list[Stop] = []
        seen_stops: set[str] = set()
        for stop_raw in entry.get("stops") or []:
            stop = _parse_stop(
                stop_raw, where, sidecar=sidecar, assets_dir=assets_dir, buckets=buckets
            )
            if stop.id in seen_stops:
                raise ConfigError(f"{where}: duplicate stop id {stop.id!r}")
            seen_stops.add(stop.id)
            stops.append(stop)
        if not stops:
            raise ConfigError(f"{where}: a story needs at least one stop")
        stories.append(
            Story(
                id=story_id,
                title=_text(entry, "title", where, required=True),
                dek=_text(entry, "dek", where),
                stops=tuple(stops),
            )
        )
    return tuple(stories)


def stories_json(stories: Sequence[Story]) -> list[dict[str, Any]]:
    """The ``manifest.site.stories`` block the viewer reads."""
    out: list[dict[str, Any]] = []
    for story in stories:
        stops: list[dict[str, Any]] = []
        for stop in story.stops:
            camera: dict[str, Any] = {
                "center": list(stop.camera.center),
                "zoom": stop.camera.zoom,
            }
            if stop.camera.bearing is not None:
                camera["bearing"] = stop.camera.bearing
            if stop.camera.pitch is not None:
                camera["pitch"] = stop.camera.pitch
            entry: dict[str, Any] = {"id": stop.id, "title": stop.title, "camera": camera}
            if stop.body_html:
                entry["body_html"] = stop.body_html
            if stop.eras:
                entry["eras"] = list(stop.eras)
            if stop.swipe is not None:
                entry["swipe"] = stop.swipe
            if stop.overlay is not None:
                entry["overlay"] = {
                    "geojson": stop.overlay.geojson,
                    "style": stop.overlay.style,
                }
            if stop.media:
                entry["media"] = [
                    {
                        k: v
                        for k, v in (
                            ("src", m.src),
                            ("alt", m.alt),
                            ("caption", m.caption),
                            ("credit", m.credit),
                            ("href", m.href),
                        )
                        if v
                    }
                    for m in stop.media
                ]
            if stop.sources:
                entry["sources"] = [
                    {k: v for k, v in (("label", s.label), ("href", s.href)) if v}
                    for s in stop.sources
                ]
            stops.append(entry)
        story_entry: dict[str, Any] = {"id": story.id, "title": story.title, "stops": stops}
        if story.dek:
            story_entry["dek"] = story.dek
        out.append(story_entry)
    return out


def stage_story_assets(config: StoriesConfig | None, viewer_dir: Path) -> Path | None:
    """Copy the configured assets into ``<viewer_dir>/story-assets``.

    The deploy bundle copies that one directory, so nothing there needs to know
    where a city keeps its images. Copies aside and swaps by rename, so a
    failure mid-copy leaves the previous images serving and the visible gap is
    a rename rather than the length of a copy. Refuses when source and staging
    overlap in either direction: the swap removes the staged tree, and a source
    nested inside it would be deleted rather than copied.
    """
    if config is None or config.assets_dir is None:
        return None
    source = config.assets_dir.resolve()
    staged = viewer_dir / ASSETS_DIR
    settled = staged.resolve()
    if settled == source:
        return staged  # already the source of truth; there is nothing to copy
    if settled.is_relative_to(source) or source.is_relative_to(settled):
        raise ConfigError(
            f"viewer.stories.assets {source} overlaps the staging directory {settled} — "
            f"keep the source images outside the viewer's {ASSETS_DIR}/"
        )
    if staged.is_symlink() or (staged.exists() and not staged.is_dir()):
        raise ConfigError(f"{staged} is not a directory; move it aside before publishing")
    incoming = staged.with_name(f".{ASSETS_DIR}.tmp-{os.getpid()}")
    outgoing = staged.with_name(f".{ASSETS_DIR}.previous-{os.getpid()}")
    shutil.rmtree(incoming, ignore_errors=True)
    shutil.rmtree(outgoing, ignore_errors=True)
    try:
        shutil.copytree(source, incoming)
        replaced = staged.exists()
        if replaced:
            staged.rename(outgoing)
        try:
            incoming.rename(staged)
        except OSError:
            if replaced:
                outgoing.rename(staged)
            raise
    finally:
        shutil.rmtree(incoming, ignore_errors=True)
        shutil.rmtree(outgoing, ignore_errors=True)
    return staged
