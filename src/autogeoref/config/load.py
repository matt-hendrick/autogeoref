"""Read a city TOML into the dataclasses, refusing anything it cannot vouch for.

Every helper here is owned by `load_city_config` alone. Relative paths resolve
against the file, unknown and retired keys are refused rather than ignored, and
a model reference is checked before anything can spend on it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, overload

from ..address_grid import AddressGrid
from ..alias.source import PARSERS, RenameSource
from ..annotate.failures import AnnotateError
from ..annotate.providers import DEFAULT_MODEL as DEFAULT_ANNOTATION_MODEL
from ..budget import DEFAULT_GATED_FRACTION
from ..slugs import valid_review_page
from ..validation import finite_number, volume_id
from .escalation import resolve_escalation
from .fields import model_variant, require_str_list
from .model import (
    EVIDENCE_CHANNELS,
    VOLUME_KEYS,
    CityConfig,
    ConfigError,
    EscalationResolution,
    VolumeConfig,
    city_slug,
)

#: Bounds on a per-page scale multiple. A printed-scale step (200 ft/in against
#: a book's 50) is a small ratio > 1; anything outside this range is a typo or
#: someone reaching for a knob to make a stubborn sheet fit. The floor of 1.0
#: also bars this key's likeliest footgun — a metres-per-pixel value pasted in
#: where a MULTIPLE belongs. A page printed FINER than its book is possible and
#: simply absent here; lower the floor deliberately, with a sheet in hand.
MIN_PAGE_SCALE_MULTIPLE = 1.0
MAX_PAGE_SCALE_MULTIPLE = 10.0


def _page_scale_multiples(v: dict[str, Any], path: Path, vid: str) -> dict[str, float]:
    """Parse and validate `page_scale_multiples` (see VolumeConfig for the why).

    Rejects anything that is not a plain positive multiple: this key is a
    DECLARATION about a named printed page, not a tuning knob, and every
    validation here exists to keep it one.
    """
    raw = v.get("page_scale_multiples")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path}: volume {vid} page_scale_multiples must be a table of "
            f"page -> multiple (e.g. {{ cbd1 = 4.0 }}), got {raw!r}"
        )
    out: dict[str, float] = {}
    for page, mult in raw.items():
        if isinstance(mult, bool) or not isinstance(mult, (int, float)):
            raise ConfigError(
                f"{path}: volume {vid} page_scale_multiples[{page}] must be a number "
                f"(a MULTIPLE of the volume's measured scale, never meters-per-pixel), "
                f"got {mult!r}"
            )
        if not (MIN_PAGE_SCALE_MULTIPLE <= float(mult) <= MAX_PAGE_SCALE_MULTIPLE):
            raise ConfigError(
                f"{path}: volume {vid} page_scale_multiples[{page}] = {mult} is outside "
                f"[{MIN_PAGE_SCALE_MULTIPLE}, {MAX_PAGE_SCALE_MULTIPLE}]. This is a MULTIPLE "
                f"of the volume's measured scale (200 ft/in in a 50 ft/in book = 4.0), never "
                f"a meters-per-pixel value. If a sheet needs a value outside this range to "
                f"fit, the fit is the problem, not the gate."
            )
        out[str(page)] = float(mult)
    return out


#: Keys that together declare a city's pinned rename source. The parser and
#: the text are ONE unit: a parser with nothing to parse, or a text with no
#: declared grammar, is a half-configured source and reads as a config error
#: rather than as "this city has no source" (which is spelled by omitting all
#: of them).
_RENAME_SOURCE_KEYS = ("rename_source_parser", "rename_source_text", "rename_source_citation")


def _rename_source(city: dict[str, Any], path: Path, base: Path) -> RenameSource | None:
    """The city's pinned rename source, or None when it declares none."""
    present = [k for k in _RENAME_SOURCE_KEYS if k in city]
    if not present:
        return None
    missing = [k for k in _RENAME_SOURCE_KEYS if k not in city]
    if missing:
        raise ConfigError(
            f"{path}: a rename source needs all of {list(_RENAME_SOURCE_KEYS)} "
            f"together; missing {missing}. Omit all of them for a city with no "
            f"documented rename list — the alias sweep then reports it without proposing."
        )
    parser = city["rename_source_parser"]
    if parser not in PARSERS:
        raise ConfigError(
            f"{path}: unknown rename_source_parser {parser!r} (known: {sorted(PARSERS)})"
        )
    citation = city["rename_source_citation"]
    if not isinstance(citation, str) or not citation.strip():
        # The citation is copied verbatim into every alias file this source
        # writes; an empty one would ship provenance-free fixtures.
        raise ConfigError(
            f"{path}: rename_source_citation must be a non-empty string, got {citation!r}"
        )
    pdf = city.get("rename_source_pdf")
    return RenameSource(
        parser=parser,
        text_path=_respath(city["rename_source_text"], "city rename_source_text", path, base),
        citation=citation.strip(),
        pdf_path=_respath(pdf, "city rename_source_pdf", path, base) if pdf is not None else None,
    )


def _optional_respath(city: dict[str, Any], key: str, path: Path, base: Path) -> Path | None:
    """A ``[city]`` path key that need not be configured; None when it is absent."""
    return _respath(city[key], f"city {key}", path, base) if key in city else None


def _address_grid(city: dict[str, Any], path: Path) -> AddressGrid | None:
    """The city's house-number grid, or None when it declares none."""
    origin = city.get("address_grid_origin")
    units = city.get("address_grid_units_per_mile")
    if origin is None and units is None:
        return None
    if origin is None or units is None:
        raise ConfigError(
            f"{path}: address_grid_origin and address_grid_units_per_mile must be "
            f"configured together (omit both for a city with no declared grid)"
        )
    if (
        not isinstance(origin, list)
        or len(origin) != 2
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in origin)
    ):
        raise ConfigError(
            f"{path}: address_grid_origin must be [longitude, latitude], got {origin!r}"
        )
    lon, lat = float(origin[0]), float(origin[1])
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ConfigError(f"{path}: address_grid_origin is out of range, got {origin!r}")
    if not isinstance(units, (int, float)) or isinstance(units, bool) or units <= 0:
        raise ConfigError(
            f"{path}: address_grid_units_per_mile must be a positive number, got {units!r}"
        )
    return AddressGrid(origin_lon=lon, origin_lat=lat, units_per_mile=float(units))


#: Config keys that USED to do something and no longer do, with what to say.
#: A retired key must be REFUSED, never ignored: an operator who still writes
#: one believes a behaviour is on. Volume tables get this free from
#: `VOLUME_KEYS`, but the CITY table has no key allow-list, so this is all
#: that stands between a retired city-level key and silence.
_RETIRED_KEYS: dict[str, str] = {
    "consensus_fallback_model": (
        "named the third voice of the consensus-annotate stage, which was cut. The "
        "addresses channel buys no model reads at all now — it votes on the escalation ladder's "
        "tier caches and any sidecar already on disk — so there is no third voice to "
        "configure. Delete the key"
    ),
}


def _reject_retired_keys(path: Path, table: dict[str, Any], where: str) -> None:
    """Refuse a key that once configured behaviour this build no longer has."""
    for key, why in _RETIRED_KEYS.items():
        if key in table:
            raise ConfigError(f"{path}: {where} {key} is RETIRED — it {why}.")


def _respath(v: Any, where: str, path: Path, base: Path) -> Path:
    """A configured path, resolved against the config file's directory."""
    if not isinstance(v, str) or not v:
        raise ConfigError(f"{path}: {where} must be a non-empty path string, got {v!r}")
    p = Path(v)
    return p if p.is_absolute() else (base / p).resolve()


def _config_number(v: Any, where: str, path: Path, *, positive: bool = False) -> float:
    try:
        number = finite_number(v)
    except ValueError as exc:
        raise ConfigError(f"{path}: {where} must be a finite number, got {v!r}") from exc
    if positive and number <= 0:
        raise ConfigError(f"{path}: {where} must be positive, got {v!r}")
    return number


def _config_volume_id(v: Any, where: str, path: Path) -> str:
    try:
        return volume_id(v)
    except ValueError as exc:
        raise ConfigError(f"{path}: {where} {exc}") from exc


@overload
def _require_bool(
    table: dict[str, Any], key: str, where: str, path: Path, *, default: bool
) -> bool: ...


@overload
def _require_bool(table: dict[str, Any], key: str, where: str, path: Path) -> bool | None: ...


def _require_bool(
    table: dict[str, Any], key: str, where: str, path: Path, *, default: bool | None = None
) -> bool | None:
    """Reject a quoted boolean rather than coercing its string value."""
    val = table.get(key, default)
    if val is not None and not isinstance(val, bool):
        raise ConfigError(f"{path}: {where} {key} must be a TOML boolean, got {val!r}")
    return val


def _channels(table: dict[str, Any], where: str, path: Path) -> tuple[str, ...]:
    val = table.get("evidence_channels")
    if val is None:
        return ()
    require_str_list(
        val,
        f"{path}: {where} evidence_channels must be a list of channel names "
        f"(any of {list(EVIDENCE_CHANNELS)}; [] = explicitly off)",
        allow_empty_items=True,
    )
    unknown = [c for c in val if c not in EVIDENCE_CHANNELS]
    if unknown:
        # a typo must never read as "off": that is how a config slip becomes a
        # quietly degraded run — the volume finishes, reports a tidy funnel, and
        # nobody learns that its flagged pool never got a second opinion
        raise ConfigError(
            f"{path}: {where} unknown evidence_channels {unknown} "
            f"(known: {list(EVIDENCE_CHANNELS)})"
        )
    return tuple(dict.fromkeys(val))  # de-duplicated, order preserved


def _annotation_voice(table: dict[str, Any], where: str, default: str, path: Path) -> str:
    val = table.get("annotation_model")
    if val is None:
        return default
    if not isinstance(val, str) or not val:
        # No empty spelling here: this is the read every later stage
        # consumes — a volume with no annotation model has no annotations,
        # and there is nothing to match. Off is `--no-annotate`.
        raise ConfigError(f"{path}: {where} annotation_model must be a model name string")
    return val


def _centerlines(city: dict[str, Any], name: str, path: Path, base: Path) -> tuple[Path, bool]:
    """The street centerline file and whether it is the OSM cache rather than a configured one.

    OPTIONAL: absent (or empty — ``_respath("")`` would resolve to the config
    DIRECTORY) means the OSM default (module docstring); the run command
    materializes that cache file before anything reads it.
    """
    centerlines_raw = city.get("centerlines")
    if centerlines_raw is not None and not isinstance(centerlines_raw, str):
        raise ConfigError(
            f"{path}: city centerlines must be a path string when configured, "
            f"got {centerlines_raw!r}"
        )
    if not centerlines_raw:
        osm_cache_dir = city.get("osm_cache_dir", "cache")
        cache_dir = _respath(osm_cache_dir, "city osm_cache_dir", path, base)
        return cache_dir / f"osm-centerlines-{city_slug(name)}.geojson", True
    return _respath(centerlines_raw, "city centerlines", path, base), False


def _renumbering(city: dict[str, Any], path: Path) -> tuple[int | None, str | None]:
    """The address-renumbering year and its note, each None when undeclared."""
    renumbering_year = city.get("renumbering_year")
    if renumbering_year is not None and (
        not isinstance(renumbering_year, int)
        or isinstance(renumbering_year, bool)
        or not 1000 < renumbering_year < 3000
    ):
        raise ConfigError(
            f"{path}: renumbering_year must be a four-digit year, got {renumbering_year!r}"
        )
    renumbering_note = city.get("renumbering_note")
    if renumbering_note is not None and (
        not isinstance(renumbering_note, str) or not renumbering_note.strip()
    ):
        raise ConfigError(
            f"{path}: renumbering_note must be a non-empty string, got {renumbering_note!r}"
        )
    return renumbering_year, renumbering_note


def _alias_sweep_skip(city: dict[str, Any], path: Path) -> dict[str, str]:
    """Volume id -> why the alias sweep skips it; empty when none is declared."""
    alias_sweep_skip = city.get("alias_sweep_skip", {})
    if not isinstance(alias_sweep_skip, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in alias_sweep_skip.items()
    ):
        # A bare list would let an entry lose its reason, and a skip whose
        # reason nobody wrote down is indistinguishable from a bug.
        raise ConfigError(
            f"{path}: alias_sweep_skip must be a table of volume id -> reason "
            f"(non-empty strings), got {alias_sweep_skip!r}"
        )
    for vid in alias_sweep_skip:
        _config_volume_id(vid, "alias_sweep_skip key", path)
    return alias_sweep_skip


def load_city_config(path: Path) -> CityConfig:
    """Load and validate a city TOML; relative paths resolve against the file."""
    raw = tomllib.loads(path.read_text())
    base = path.parent

    try:
        city = raw["city"]
    except KeyError as exc:
        raise ConfigError(f"{path}: missing required key {exc}") from exc
    if not isinstance(city, dict):
        raise ConfigError(f"{path}: city must be a table")
    _reject_retired_keys(path, city, "city")
    try:
        name = city["name"]
        aliases_dir = _respath(city["aliases_dir"], "city aliases_dir", path, base)
    except KeyError as exc:
        raise ConfigError(f"{path}: missing required key {exc}") from exc
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{path}: city name must be a non-empty string, got {name!r}")
    centerlines, centerlines_from_osm = _centerlines(city, name, path, base)
    city_escalation = resolve_escalation(city, "city", path)
    city_channels = _channels(city, "city", path)
    city_annotation = _annotation_voice(city, "city", DEFAULT_ANNOTATION_MODEL, path)
    city_annotation_variant = model_variant(city, "annotation_variant", "city", path)
    rail_geojson = _optional_respath(city, "rail_geojson", path, base)
    rail_gazetteer = _optional_respath(city, "rail_gazetteer", path, base)
    if (rail_geojson is None) != (rail_gazetteer is None):
        raise ConfigError(
            f"{path}: city rail_geojson and rail_gazetteer must be configured together"
        )
    renumbering_year, renumbering_note = _renumbering(city, path)
    block_size = city.get("address_block_size", 100)
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        raise ConfigError(
            f"{path}: address_block_size must be a positive integer, got {block_size!r}"
        )
    rename_source = _rename_source(city, path, base)
    address_grid = _address_grid(city, path)
    alias_sweep_skip = _alias_sweep_skip(city, path)
    raw_gated = city.get("gated_fraction", DEFAULT_GATED_FRACTION)
    gated_fraction = _config_number(raw_gated, "gated_fraction", path, positive=True)
    if gated_fraction > 1.0:
        raise ConfigError(
            f"{path}: gated_fraction is a FRACTION of a volume's sheets (0, 1], got {raw_gated!r}"
        )
    raw_volumes = raw.get("volumes", {})
    if not isinstance(raw_volumes, dict):
        raise ConfigError(f"{path}: volumes must be a table")
    volumes: dict[str, VolumeConfig] = {}
    for raw_vid, v in raw_volumes.items():
        vid = _config_volume_id(raw_vid, "volume identifier", path)
        if not isinstance(v, dict):
            raise ConfigError(f"{path}: volume {vid} must be a table")
        volumes[vid] = _load_volume(
            vid,
            v,
            path,
            base,
            city_escalation=city_escalation,
            city_annotation=city_annotation,
            city_annotation_variant=city_annotation_variant,
            city_channels=city_channels,
        )
    cfg = CityConfig(
        name=name,
        centerlines_path=centerlines,
        aliases_dir=aliases_dir,
        centerlines_from_osm=centerlines_from_osm,
        community_areas_path=_optional_respath(city, "community_areas", path, base),
        loc_query=city.get("loc_query"),
        centerline_name_property=city.get("centerline_name_property", "street_nam"),
        centerline_type_property=city.get("centerline_type_property", "street_typ"),
        escalation_model=city_escalation.model,
        escalation_variant=city_escalation.variant,
        escalation_models=city_escalation.models,
        escalation_variants=city_escalation.variants,
        annotation_model=city_annotation,
        annotation_variant=city_annotation_variant,
        evidence_channels=city_channels,
        rail_geojson_path=rail_geojson,
        rail_gazetteer_path=rail_gazetteer,
        renumbering_table_path=_optional_respath(city, "renumbering_table", path, base),
        renumbering_year=renumbering_year,
        renumbering_note=renumbering_note,
        address_block_size=block_size,
        loc_catalog_path=_optional_respath(city, "loc_catalog", path, base),
        rename_source=rename_source,
        address_grid=address_grid,
        alias_sweep_skip=dict(alias_sweep_skip),
        gated_fraction=gated_fraction,
        volumes=volumes,
    )
    _validate_model_refs(cfg, path)
    return cfg


def _load_volume(
    vid: str,
    v: dict[str, Any],
    path: Path,
    base: Path,
    *,
    city_escalation: EscalationResolution,
    city_annotation: str,
    city_annotation_variant: str | None,
    city_channels: tuple[str, ...],
) -> VolumeConfig:
    """Validate one ``[volumes.<vid>]`` table against the city-level defaults.

    Validation order and message text are contracts (runpolicy: "preserving
    CLI validation order and text").
    """
    # before the generic unknown-key error: a retired key deserves the reason
    # it is gone, not a list that reads like a typo
    _reject_retired_keys(path, v, f"volume {vid}")
    unknown = set(v) - VOLUME_KEYS
    if unknown:
        raise ConfigError(f"{path}: volume {vid} has unknown keys {sorted(unknown)}")
    bbox: tuple[float, float, float, float] | None = None
    if "bounds_bbox" in v:
        raw_bbox = v["bounds_bbox"]
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise ConfigError(f"{path}: volume {vid} bounds_bbox must have 4 numbers")
        west, south, east, north = (
            _config_number(value, f"volume {vid} bounds_bbox", path) for value in raw_bbox
        )
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ConfigError(
                f"{path}: volume {vid} bounds_bbox must be ordered west, south, east, north "
                "within longitude/latitude ranges"
            )
        bbox = (west, south, east, north)
    bounds_from = None
    if "bounds_from" in v:
        bounds_from = _config_volume_id(v["bounds_from"], f"volume {vid} bounds_from", path)
    raw_areas = v.get("bounds_areas", [])
    require_str_list(
        raw_areas,
        f"{path}: volume {vid} bounds_areas must be a list of non-empty names",
        stripped=True,
    )
    scale = (
        _config_number(v["scale_m_per_px"], f"volume {vid} scale_m_per_px", path, positive=True)
        if "scale_m_per_px" in v
        else None
    )
    rotation = (
        _config_number(v["rotation_deg"], f"volume {vid} rotation_deg", path)
        if "rotation_deg" in v
        else None
    )
    if (scale is None) != (rotation is None):
        raise ConfigError(
            f"{path}: volume {vid} scale_m_per_px and rotation_deg must be configured together"
        )
    if "addresses_post_1909" in v:
        raise ConfigError(
            f"{path}: volume {vid} uses the removed key addresses_post_1909 — "
            f"rename it to addresses_modern (same tri-state semantics)"
        )
    # A quoted boolean must not silently change the declared era.
    era = _require_bool(v, "addresses_modern", f"volume {vid}", path)
    rescue = _require_bool(v, "quadrant_rescue", f"volume {vid}", path, default=False)
    content_masks = _require_bool(v, "content_masks", f"volume {vid}", path, default=False)
    exempt_raw = v.get("content_mask_exempt", [])
    require_str_list(
        exempt_raw,
        f"{path}: volume {vid} content_mask_exempt must be a list of page-id "
        f'strings (e.g. ["57", "59"]), got {exempt_raw!r}',
    )
    if exempt_raw and not content_masks:
        # the key only excuses pages FROM the hull style; without the
        # declaration it would silently do nothing
        raise ConfigError(f"{path}: volume {vid} content_mask_exempt requires content_masks = true")
    overview_raw = v.get("overview_pages", [])
    require_str_list(
        overview_raw,
        f"{path}: volume {vid} overview_pages must be a list of page-id "
        f'strings (e.g. ["cbd1", "cbd2"]), got {overview_raw!r}',
    )
    for p in overview_raw:
        # the same narrow page grammar review uses: canonical ids only, so a
        # typo ("pcbd1", "CBD1") fails here instead of silently declaring a
        # page that no result record will ever match
        if not valid_review_page(p):
            raise ConfigError(
                f"{path}: volume {vid} overview_pages entry {p!r} is not a "
                f"canonical page id (digits with an optional letter suffix, "
                f"or a known named sheet id in lower case)"
            )
    # resolve_escalation owns the presence-based inheritance rule
    esc = resolve_escalation(v, f"volume {vid}", path, city=city_escalation)
    return VolumeConfig(
        identifier=vid,
        page_scale_multiples=_page_scale_multiples(v, path, vid),
        bounds_bbox=bbox,
        bounds_from_counterpart=bounds_from,
        bounds_areas=tuple(raw_areas),
        scale_m_per_px=scale,
        rotation_deg=rotation,
        quadrant_rescue=rescue,
        content_masks=content_masks,
        content_mask_exempt=tuple(exempt_raw),
        overview_pages=tuple(dict.fromkeys(overview_raw)),
        escalation_model=esc.model,
        escalation_variant=esc.variant,
        escalation_models=esc.models,
        escalation_variants=esc.variants,
        annotation_model=_annotation_voice(v, f"volume {vid}", city_annotation, path),
        annotation_variant=(
            model_variant(v, "annotation_variant", f"volume {vid}", path)
            if "annotation_variant" in v
            else None
            if "annotation_model" in v
            else city_annotation_variant
        ),
        # presence-based, NOT `or`: `evidence_channels = []` on a volume is an
        # explicit CANCEL of the city-wide list, exactly like `escalation_models = []`
        evidence_channels=(
            _channels(v, f"volume {vid}", path) if "evidence_channels" in v else city_channels
        ),
        addresses_modern=era,
        renumbering_table_path=(
            _respath(v["renumbering_table"], f"volume {vid} renumbering_table", path, base)
            if "renumbering_table" in v
            else None
        ),
    )


def _validate_model_refs(cfg: CityConfig, path: Path) -> None:
    """Reject an unusable model reference at CONFIG-LOAD time, not at spend time.

    Every model string resolves through
    :func:`autogeoref.annotate.providers.parse_model_ref`, the same routing the stages do
    later, so a mistyped provider, a bare name belonging to another provider,
    or a too-small model fails here — before a run has spent a call. Validation
    only: the stages CANONICALIZE the names they key caches on, because a CLI
    flag and a script both reach a stage without passing through a config file.
    """
    _check_model_block(cfg, "city", path)
    for vid, vol in cfg.volumes.items():
        _check_model_block(vol, f"volume {vid}", path)


def _check_model_ref(model: str | None, variant: str | None, where: str, path: Path) -> None:
    if not model:
        return
    from ..annotate.providers import VARIANT_PROVIDERS, parse_model_ref

    try:
        ref = parse_model_ref(model)
    except AnnotateError as exc:  # ModelQualityError is an AnnotateError
        raise ConfigError(f"{path}: {where} model {model!r}: {exc}") from exc
    if variant is not None and ref.provider not in VARIANT_PROVIDERS:
        allowed = ", ".join(sorted(VARIANT_PROVIDERS))
        raise ConfigError(
            f"{path}: {where} variant {variant!r} requires one of {allowed}, not {ref.provider!r}"
        )


def _check_model_block(cfg: CityConfig | VolumeConfig, prefix: str, path: Path) -> None:
    """One table's model references: the two singletons, then the ladder."""
    from ..annotate.providers import canonical_model

    _check_model_ref(
        cfg.escalation_model, cfg.escalation_variant, f"{prefix} escalation_model", path
    )
    _check_model_ref(
        cfg.annotation_model, cfg.annotation_variant, f"{prefix} annotation_model", path
    )
    for model, variant in zip(cfg.escalation_models, cfg.escalation_variants, strict=True):
        _check_model_ref(model, variant, f"{prefix} escalation_models", path)
    if len({canonical_model(model) for model in cfg.escalation_models}) != len(
        cfg.escalation_models
    ):
        raise ConfigError(f"{path}: {prefix} escalation_models cannot repeat a model")
