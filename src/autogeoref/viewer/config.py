"""``[viewer]`` configuration: what the static viewer shows for a city.

This module itself imports neither shapely nor subprocess (those live in
:mod:`.sources` / :mod:`.bounds`) — but importing any ``viewer`` submodule
runs the eager facade ``__init__`` first, which pulls both in anyway. The
split keeps the dependency graph clean; making config imports actually
cheap would additionally need a lazy (``__getattr__``) facade.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.model import ConfigError
from .era import EraBucket, era_label
from .stories import StoriesConfig, load_stories, stories_json

#: ``deploy/tiles/`` subdirectory names this project vouches for. A manifest
#: build REFUSES any other serving directory instead of publishing it: a
#: directory nobody declared holds someone else's georeferencing, and the
#: site-wide credit would silently claim it. A city that genuinely serves an
#: imported archive widens this with ``serving_dirs``.
SERVING_DIRS = ("autogeoref",)


@dataclass(frozen=True)
class RegionBand:
    """One first-match-wins band over a volume-centroid coordinate."""

    label: str
    above: float | None = None
    below: float | None = None


@dataclass(frozen=True)
class RegionLabels:
    """Fallback district naming for volumes with no community-area names:
    ``combine`` with ``{lat}``/``{lng}`` filled from the first matching band on
    each axis, then ``collapse`` pairs applied verbatim."""

    lat_bands: tuple[RegionBand, ...] = ()
    lng_bands: tuple[RegionBand, ...] = ()
    combine: str = "{lat} {lng}"
    collapse: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class GeocoderConfig:
    """Address-search bias. ``suffix`` is appended to queries that don't
    already mention the city; ``bbox`` is west, south, east, north.

    Which geocoder answers is not configurable here: the page decides from the
    deployed token and the serving host.
    """

    suffix: str = ""
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class BasemapConfig:
    """The map under (and beside) the atlas.

    ``raster`` is a ``{z}/{x}/{y}`` tile template — the local-development
    shape. ``vector`` is the deployment shape: one self-hosted PMTiles archive
    plus a MapLibre style per pane (``style_atlas`` is the muted map under the
    historical sheets, ``style_now`` the modern one). Vector keeps every tile
    request inside our own hosting, which the traffic envelope requires:
    OSMF's tile policy forbids a public viewer on their raster tiles, and the
    failure mode is a referer block and a blank basemap."""

    type: str = "raster"
    tiles: str | None = None
    pmtiles: str | None = None
    style_atlas: str | None = None
    style_now: str | None = None
    attribution: str | None = None
    maxzoom: int | None = None


@dataclass(frozen=True)
class ViewerConfig:
    """Everything city-specific the static viewer shows. All fields are
    optional: the HTML has generic fallbacks, so a city with no ``[viewer]``
    block still renders a working (if plain) site."""

    title: str | None = None  # document title; "{era}" substituted live
    kicker: str | None = None
    heading: str | None = None  # h1 text; "{era}" substituted live
    heading_note: str | None = None
    dek: str | None = None
    loading_text: str | None = None
    now_label: str | None = None
    #: era chips selected on first load. Several may be on at once: eras that
    #: barely overlap compose into one continuous city. Parsed from
    #: ``default_eras`` (list) or the legacy ``default_era`` (single string).
    default_eras: tuple[str, ...] = ()
    home_point: tuple[float, float] | None = None  # lng, lat the initial view seeks
    footer_source_html: str | None = None
    #: the site-wide credit line: who georeferenced these layers. OPTIONAL —
    #: a city that sets nothing publishes no credit, which is a choice, not a
    #: fallback. HTML, so it can carry a link. Per-era ``credits`` on an
    #: ``[[viewer.era]]`` bucket overrides it for that era.
    optional_credits_html: str | None = None
    #: serving directories this city vouches for, the ``serving_dirs`` key.
    #: Defaults to :data:`SERVING_DIRS`; widening it is how a city declares
    #: that an imported archive is its to publish.
    serving_dirs: tuple[str, ...] = SERVING_DIRS
    #: declared display label per volume id, the ``[viewer.labels]`` table.
    #: Beats both the LOC-derived subject and the community-area names — the
    #: escape hatch for a name the catalog never recorded.
    volume_labels: tuple[tuple[str, str], ...] = ()
    geocoder: GeocoderConfig | None = None
    basemap: BasemapConfig | None = None
    era_buckets: tuple[EraBucket, ...] = ()
    region_labels: RegionLabels | None = None
    #: optional guided stories, the ``[viewer.stories]`` block. Absent means
    #: absent: the manifest carries no ``stories`` and the viewer builds no
    #: story UI at all, which is what keeps this out of every other city's way.
    stories: StoriesConfig | None = None


def _parse_bands(raw: Sequence[Mapping[str, Any]], where: str) -> tuple[RegionBand, ...]:
    bands = []
    for entry in raw:
        if "label" not in entry:
            raise ConfigError(f"{where}: every region band needs a label")
        bands.append(
            RegionBand(
                label=str(entry["label"]),
                above=float(entry["above"]) if "above" in entry else None,
                below=float(entry["below"]) if "below" in entry else None,
            )
        )
    return tuple(bands)


def _parse_basemap(raw: Mapping[str, Any], city_toml: Path) -> BasemapConfig | None:
    """``[viewer.basemap]`` -> a validated BasemapConfig (absent -> None).

    A vector basemap that is missing its archive or either pane style renders
    as a blank map with the atlas floating on it, and nothing in the browser
    says why. Refuse the config instead."""
    if not raw:
        return None
    kind = str(raw.get("type", "raster"))
    if kind not in ("raster", "vector"):
        raise ConfigError(
            f"{city_toml}: viewer.basemap.type must be raster or vector, got {kind!r}"
        )
    basemap = BasemapConfig(
        type=kind,
        tiles=raw.get("tiles"),
        pmtiles=raw.get("pmtiles"),
        style_atlas=raw.get("style_atlas"),
        style_now=raw.get("style_now"),
        attribution=raw.get("attribution"),
        maxzoom=int(raw["maxzoom"]) if "maxzoom" in raw else None,
    )
    if kind == "vector":
        missing = [
            key for key in ("pmtiles", "style_atlas", "style_now") if not getattr(basemap, key)
        ]
        if missing:
            raise ConfigError(f"{city_toml}: a vector viewer.basemap needs {', '.join(missing)}")
    elif not basemap.tiles:
        raise ConfigError(f"{city_toml}: a raster viewer.basemap needs tiles")
    return basemap


def _parse_stories(
    raw: Mapping[str, Any] | None, city_toml: Path, buckets: Sequence[EraBucket]
) -> StoriesConfig | None:
    """``[viewer.stories]`` -> a validated StoriesConfig (absent -> None).

    Prose lives in the sidecar rather than inline: caption text churns on a
    different cadence from thresholds and per-volume boxes. Both paths are
    relative to the city TOML's directory. An EMPTY block is a mistake, not an
    opt-out — a city that wants no stories writes no block at all.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{city_toml}: viewer.stories must be a table")
    filename = raw.get("file")
    if not isinstance(filename, str) or not filename.strip():
        raise ConfigError(f'{city_toml}: viewer.stories needs file = "<sidecar>.json"')
    sidecar = (city_toml.parent / filename).resolve()
    assets_dir = None
    if "assets" in raw:
        assets_dir = (city_toml.parent / str(raw["assets"])).resolve()
        if not assets_dir.is_dir():
            raise ConfigError(f"{city_toml}: viewer.stories.assets {assets_dir} is not a directory")
    return StoriesConfig(
        path=sidecar,
        stories=load_stories(sidecar, assets_dir=assets_dir, buckets=buckets),
        assets_dir=assets_dir,
    )


def _parse_serving_dirs(raw: Mapping[str, Any], city_toml: Path) -> tuple[str, ...]:
    """``serving_dirs`` -> the tripwire's allow-list (absent -> the default).

    An empty list would disable the refusal outright rather than mean "nothing
    serves", so it is rejected: a city that wants the default omits the key.
    """
    if "serving_dirs" not in raw:
        return SERVING_DIRS
    declared = raw["serving_dirs"]
    if (
        not isinstance(declared, list)
        or not declared
        or not all(isinstance(d, str) and d.strip() for d in declared)
    ):
        raise ConfigError(
            f"{city_toml}: viewer.serving_dirs must be a non-empty list of "
            f"deploy/tiles/ subdirectory names, got {declared!r}"
        )
    dirs = tuple(d.strip() for d in declared)
    # These are NAMES, matched against a directory's last component. A path
    # ("deploy/tiles/autogeoref") is the natural mistake and would declare
    # nothing at all — every build would then be refused with a message about
    # the directory, never about the typo here.
    bad = sorted(d for d in dirs if d != Path(d).name or d in {".", ".."})
    if bad:
        raise ConfigError(
            f"{city_toml}: viewer.serving_dirs takes bare directory NAMES, not "
            f"paths — {', '.join(repr(d) for d in bad)}. Use the last component "
            f"only (e.g. 'autogeoref', not 'deploy/tiles/autogeoref')."
        )
    return dirs


def _geocoder_config(raw: dict[str, Any] | None, city_toml: Path) -> GeocoderConfig | None:
    """The ``[viewer.geocoder]`` block: search bias, and nothing else.

    ``provider`` is refused rather than ignored. It used to be parsed and
    published while nothing read it, so setting it changed no behaviour and
    still contradicted the page in every manifest.
    """
    if raw is None:
        return None
    if "provider" in raw:
        raise ConfigError(
            f"{city_toml}: viewer.geocoder.provider is not configurable — the page "
            "uses Mapbox when a token is deployed and a dev-only fallback otherwise"
        )
    bbox = raw.get("bbox")
    if bbox is not None and len(bbox) != 4:
        raise ConfigError(f"{city_toml}: viewer.geocoder.bbox must have 4 numbers")
    return GeocoderConfig(
        suffix=raw.get("suffix", ""),
        bbox=tuple(bbox) if bbox is not None else None,
    )


def load_viewer_config(city_toml: Path) -> ViewerConfig:
    """Parse the ``[viewer]`` block of a city TOML (absent -> all defaults)."""
    raw = tomllib.loads(city_toml.read_text(encoding="utf-8")).get("viewer") or {}

    home = raw.get("home_point")
    if home is not None and len(home) != 2:
        raise ConfigError(f"{city_toml}: viewer.home_point must be [lng, lat]")

    geocoder = _geocoder_config(raw.get("geocoder"), city_toml)

    buckets = []
    for b in raw.get("era") or []:
        years = b.get("years")
        if not years or len(years) != 2 or "label" not in b:
            raise ConfigError(f"{city_toml}: each [[viewer.era]] needs years=[first,last], label")
        buckets.append(
            EraBucket(
                first_year=int(years[0]),
                last_year=int(years[1]),
                label=str(b["label"]),
                credits_html=b.get("credits"),
            )
        )

    regions = None
    if "region_labels" in raw:
        r = raw["region_labels"]
        regions = RegionLabels(
            lat_bands=_parse_bands(r.get("lat") or (), f"{city_toml}: viewer.region_labels"),
            lng_bands=_parse_bands(r.get("lng") or (), f"{city_toml}: viewer.region_labels"),
            combine=r.get("combine", "{lat} {lng}"),
            collapse=tuple((str(a), str(b)) for a, b in (r.get("collapse") or ())),
        )

    # `optional_credits` was `default_credits`. Refuse the old spelling rather
    # than ignore it: the key is optional, so an unrecognized one would parse
    # clean and publish a site with NO credit line — the failure nobody sees.
    if "default_credits" in raw:
        raise ConfigError(
            f"{city_toml}: viewer.default_credits is now viewer.optional_credits — "
            f"rename it. It was never a fallback for a missing value; it is the "
            f"site-wide credit, and omitting it publishes no credit at all."
        )
    # The viewer splits and joins this as a string. A number reaches that code
    # and throws BEFORE the footer is written, so the page boots with an empty
    # footer, sources panel and district list — refuse it here instead.
    credit_line = raw.get("optional_credits")
    if credit_line is not None and not isinstance(credit_line, str):
        raise ConfigError(
            f"{city_toml}: viewer.optional_credits must be a string (HTML), got {credit_line!r}"
        )

    serving_dirs = _parse_serving_dirs(raw, city_toml)

    # [viewer.labels]: volume id -> display label. A non-string (or blank)
    # value is a typo that would otherwise render literally in the district
    # list — reject it.
    raw_labels = raw.get("labels", {})
    if not isinstance(raw_labels, Mapping):
        raise ConfigError(f'{city_toml}: viewer.labels must be a table of volume = "label"')
    volume_labels: list[tuple[str, str]] = []
    for ident, text in raw_labels.items():
        if not isinstance(text, str) or not text.strip():
            raise ConfigError(
                f"{city_toml}: viewer.labels.{ident} must be a non-empty string, got {text!r}"
            )
        volume_labels.append((str(ident), text.strip()))

    # default_eras / default_era: one meaning, two spellings — reject both at
    # once rather than guess which the operator meant
    if "default_eras" in raw and "default_era" in raw:
        raise ConfigError(
            f"{city_toml}: viewer.default_era and viewer.default_eras are "
            "mutually exclusive — keep only default_eras"
        )
    default_eras_raw = raw.get("default_eras")
    if default_eras_raw is None:
        default_eras_raw = [raw["default_era"]] if "default_era" in raw else []
    if not isinstance(default_eras_raw, list) or not all(
        isinstance(e, str) for e in default_eras_raw
    ):
        raise ConfigError(f"{city_toml}: viewer.default_eras must be a list of era labels")
    if buckets:
        # A typo'd label would silently start the viewer on its fallback era.
        # A bare year is a real chip only when it falls outside every bucket
        # (era_label lets such years label themselves); a year a bucket covers
        # renders under the bucket's label, so as a default it is a typo.
        known = {b.label for b in buckets}
        for label in default_eras_raw:
            if label in known:
                continue
            if re.fullmatch(r"\d{4}", label) and era_label(int(label), buckets) == label:
                continue
            raise ConfigError(
                f"{city_toml}: viewer default era {label!r} is not an era "
                f"bucket label (expected one of {', '.join(sorted(known))})"
            )

    basemap = _parse_basemap(raw.get("basemap") or {}, city_toml)
    stories = _parse_stories(raw.get("stories"), city_toml, buckets)
    return ViewerConfig(
        title=raw.get("title"),
        kicker=raw.get("kicker"),
        heading=raw.get("heading"),
        heading_note=raw.get("heading_note"),
        dek=raw.get("dek"),
        loading_text=raw.get("loading_text"),
        now_label=raw.get("now_label"),
        default_eras=tuple(default_eras_raw),
        home_point=tuple(home) if home is not None else None,
        footer_source_html=raw.get("footer_source"),
        optional_credits_html=raw.get("optional_credits"),
        serving_dirs=serving_dirs,
        volume_labels=tuple(volume_labels),
        geocoder=geocoder,
        basemap=basemap,
        era_buckets=tuple(buckets),
        region_labels=regions,
        stories=stories,
    )


def _band_dict(bd: RegionBand) -> dict[str, Any]:
    d: dict[str, Any] = {"label": bd.label}
    if bd.above is not None:
        d["above"] = bd.above
    if bd.below is not None:
        d["below"] = bd.below
    return d


def site_dict(city_name: str, viewer: ViewerConfig) -> dict[str, Any]:
    """The ``manifest.site`` block: config facts the viewer HTML reads.
    Only set fields are emitted — the HTML supplies generic fallbacks."""
    site: dict[str, Any] = {"name": city_name}
    site.update(
        {
            key: value
            for key, value in (
                ("title", viewer.title),
                ("kicker", viewer.kicker),
                ("heading", viewer.heading),
                ("heading_note", viewer.heading_note),
                ("dek", viewer.dek),
                ("loading_text", viewer.loading_text),
                ("now_label", viewer.now_label),
                ("footer_source_html", viewer.footer_source_html),
                ("optional_credits_html", viewer.optional_credits_html),
            )
            if value is not None
        }
    )
    if viewer.default_eras:
        site["default_eras"] = list(viewer.default_eras)
    if viewer.home_point is not None:
        site["home_point"] = list(viewer.home_point)
    if viewer.geocoder is not None:
        g: dict[str, Any] = {}
        if viewer.geocoder.suffix:
            g["suffix"] = viewer.geocoder.suffix
        if viewer.geocoder.bbox is not None:
            g["bbox"] = list(viewer.geocoder.bbox)
        if g:  # an empty block would publish a geocoder that configures nothing
            site["geocoder"] = g
    if viewer.basemap is not None:
        b = viewer.basemap
        basemap: dict[str, Any] = {"type": b.type}
        basemap.update(
            {
                key: setting
                for key, setting in (
                    ("tiles", b.tiles),
                    ("pmtiles", b.pmtiles),
                    ("attribution", b.attribution),
                    ("maxzoom", b.maxzoom),
                )
                if setting is not None
            }
        )
        if b.type == "vector":
            basemap["styles"] = {"atlas": b.style_atlas, "now": b.style_now}
        site["basemap"] = basemap
    era_credits = {b.label: b.credits_html for b in viewer.era_buckets if b.credits_html}
    if era_credits:
        site["era_credits"] = era_credits
    if viewer.stories is not None:
        # inside `site`, so the deploy bundle carries it and the leak scan
        # already reads every caption
        site["stories"] = stories_json(viewer.stories.stories)
    if viewer.region_labels is not None:
        r = viewer.region_labels
        site["region_labels"] = {
            "lat": [_band_dict(b) for b in r.lat_bands],
            "lng": [_band_dict(b) for b in r.lng_bands],
            "combine": r.combine,
            "collapse": [list(pair) for pair in r.collapse],
        }
    return site
