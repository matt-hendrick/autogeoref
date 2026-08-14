"""What a city and a volume declare, as frozen dataclasses.

Everything downstream shares these; only `load` builds one. That direction is
the whole reason for the split — the loader's helpers are exclusively its own,
and a model that imported the loader would be a cycle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..annotate.providers import DEFAULT_MODEL as DEFAULT_ANNOTATION_MODEL
from ..budget import DEFAULT_GATED_FRACTION

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from ..address_grid import AddressGrid
    from ..alias.source import RenameSource


class ConfigError(ValueError):
    """Bad or incomplete city configuration."""


#: The evidence channels a config may declare (see VolumeConfig.evidence_channels).
#: "corroboration" is deliberately NOT here: it is unconditional — it needs no
#: model spend and no city data, it always runs, and making it nameable would
#: imply it could be switched off.
EVIDENCE_CHANNELS = ("junction", "addresses")

# Drift hazard: the volume loop reads exactly these keys and the unknown-keys
# check refuses the rest, so a key added to one side and not the other is
# either silently ignored or wrongly refused. Checked field-by-field against
# `CityConfig.volume()`: every key the loop reads is listed, every key listed
# is read. `addresses_post_1909` exists solely so the volume loop can give the
# removed spelling its rename error.
VOLUME_KEYS = frozenset(
    {
        "addresses_modern",
        "addresses_post_1909",
        "annotation_model",
        "annotation_variant",
        "bounds_areas",
        "bounds_bbox",
        "bounds_from",
        "content_mask_exempt",
        "content_masks",
        "escalation_model",
        "escalation_variant",
        "escalation_models",
        "escalation_variants",
        "evidence_channels",
        "overview_pages",
        "page_scale_multiples",
        "quadrant_rescue",
        "renumbering_table",
        "rotation_deg",
        "scale_m_per_px",
    }
)


@dataclass(frozen=True)
class EscalationResolution:
    """A config table's four escalation fields, after inheritance is resolved.

    Built by :func:`resolve_escalation`. ``VolumeConfig`` stores the same four
    fields and its ladder/tiers readers delegate back here, so the derivation
    from fields to spend-ordered tiers has exactly one home.
    """

    model: str | None = None
    variant: str | None = None
    models: tuple[str, ...] = ()
    variants: tuple[str | None, ...] = ()

    def ladder(self) -> tuple[str, ...]:
        """Tiers in spend order; a single model is a 1-tier ladder."""
        if self.models:
            return self.models
        return (self.model,) if self.model else ()

    def tiers(self) -> tuple[tuple[str, str | None], ...]:
        """Model/effort pairs in spend order."""
        models = self.ladder()
        if self.models:
            variants = self.variants or (None,) * len(models)
            return tuple(zip(models, variants, strict=True))
        return ((self.model, self.variant),) if self.model else ()


@dataclass(frozen=True)
class VolumeConfig:
    """Per-volume knowledge: bounds source and (optionally) pinned constants."""

    identifier: str
    #: exactly one bounds source is used, in this priority order
    bounds_bbox: tuple[float, float, float, float] | None = None
    bounds_from_counterpart: str | None = None
    bounds_areas: tuple[str, ...] = ()
    #: Recorded volume constants; None = derive (two-pass). FRAME CONTRACT:
    #: ``scale_m_per_px`` is EPSG:3857 metres per full-resolution pixel — the
    #: frame the derivation, the gates, the rescue linear part, and the review
    #: seed all read. A pin taken off a printed scale bar at scan DPI is TRUE
    #: ground m/px and must be divided by cos(latitude) first.
    scale_m_per_px: float | None = None
    rotation_deg: float | None = None
    #: OPT-IN: also attempt rescue at +90/180/270 deg (quadrant-rotated scans).
    #: Off by default — a capability extension, not a gate change; every rescue
    #: gate still applies at each orientation.
    quadrant_rescue: bool = False
    #: OPT-IN: bake this volume's masks from each sheet's colored-content HULL
    #: instead of the padded colour box a regular sheet gets — a declaration
    #: that the volume's sheets each detail one block inside a mostly-blank
    #: frame. Never inferred from pixels (`docs/INTERNALS.md` says why).
    content_masks: bool = False
    #: Page ids excused from ALL colour-derived masking: they keep page
    #: rectangles. The escape hatch for a sparse-but-fully-drawn sheet, where a
    #: colour bound would chop legitimately drawn ground. Declared per page,
    #: never detected — no pixel statistic separates sparse-but-full from
    #: genuinely one-block. The bake's auto-exemption means the same thing.
    content_mask_exempt: tuple[str, ...] = ()
    #: NAMED pages printed at a different scale, as a MULTIPLE of the volume's
    #: own measured scale: {"cbd1": 4.0} = "this page is 200 ft to the inch in
    #: a 50 ft/in book". Never a metres-per-pixel value. RE-CENTERS the scale
    #: window; it does not widen it and is not a scale search. Named pages are
    #: excluded from the pass-1 median. See `docs/ADDING-A-CITY.md`.
    page_scale_multiples: Mapping[str, float] = field(default_factory=dict)
    #: Page ids of the volume's district-scale OVERVIEW sheets. The single
    #: owner of the overview class: their masks clip to the inlier-GCP hull,
    #: their paint bakes into a separate artifact nothing serves, and they are
    #: withdrawn from ``stage_seam``'s tie set. Vouching is NOT affected.
    #: Declared, never detected — the ids can be plain numerals.
    overview_pages: tuple[str, ...] = ()
    #: Single-tier escalation model, ALREADY RESOLVED against the city default
    #: at load time: a volume naming either escalation key inherits neither, so
    #: None means "no single-tier model", never "look at the city". Read
    #: escalation_ladder(), not this.
    escalation_model: str | None = None
    #: Reasoning variant paired with ``escalation_model`` when that singular
    #: spelling is used.
    escalation_variant: str | None = None
    #: Escalation ladder, cheapest tier first; overrides escalation_model. The
    #: stage runs by DEFAULT wherever this resolves to a non-empty ladder. An
    #: explicit ``escalation_models = []`` on a volume CANCELS the city ladder;
    #: ``--no-escalate`` skips the stage for one run. Falls back to the city
    #: ladder when the volume omits the key.
    escalation_models: tuple[str, ...] = ()
    #: Provider-owned reasoning variants paired positionally with
    #: ``escalation_models``. None means the provider default for that tier.
    escalation_variants: tuple[str | None, ...] = ()
    #: The model that reads each sheet's street labels ONCE, in the annotate
    #: stage — the volume's PRIMARY annotation voice and the biggest single
    #: spender. A bare name means Anthropic; a ``provider:`` prefix is an
    #: enforced promise. Sonnet-class minimum is enforced at load: small models
    #: hallucinate street names, and this is the read every later stage trusts.
    annotation_model: str = DEFAULT_ANNOTATION_MODEL
    #: Provider-owned reasoning variant for the primary annotation read.
    annotation_variant: str | None = None
    #: The volume's printed-address era for the addresses channel: True = the
    #: printed numbers ARE the modern numbers, False = the volume predates the
    #: city's renumbering and numbers need the published table (until it is
    #: acquired the channel abstains), None = undeclared (also abstains).
    #: Never guess a conversion era.
    addresses_modern: bool | None = None
    #: The evidence channels verified-accept may hear from, ALREADY RESOLVED
    #: against the city default at load time (like escalation_models).
    #: Declared => junction-verify and verified-accept run BY DEFAULT; absent
    #: => they stay opt-in behind their flags, which keeps a city that never
    #: mentions channels inert. Naming "addresses" buys no model reads, but it
    #: is the only channel that may REFUTE. ``--no-verify`` skips them once.
    evidence_channels: tuple[str, ...] = ()
    #: OPT-IN per-volume renumbering conversion, overriding the city table. A
    #: city can have renumbered its districts on different dates, so volumes
    #: from one district convert through a table the rest must NOT use.
    #: None = use the city table.
    renumbering_table_path: Path | None = None

    def _resolved_escalation(self) -> EscalationResolution:
        """The four escalation fields, re-wrapped for the one ladder/tiers rule."""
        return EscalationResolution(
            model=self.escalation_model,
            variant=self.escalation_variant,
            models=self.escalation_models,
            variants=self.escalation_variants,
        )

    def escalation_ladder(self) -> tuple[str, ...]:
        """Escalation tiers in spend order; a single model is a 1-tier ladder."""
        return self._resolved_escalation().ladder()

    def escalation_tiers(self) -> tuple[tuple[str, str | None], ...]:
        """Escalation model/effort pairs in spend order."""
        return self._resolved_escalation().tiers()


def city_slug(name: str) -> str:
    """``'Springfield (Ill.)'`` -> ``'springfield-ill'`` — stable cache-file naming."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@dataclass(frozen=True)
class CityConfig:
    name: str
    centerlines_path: Path
    aliases_dir: Path
    #: True when ``centerlines`` was NOT configured: ``centerlines_path`` is
    #: the per-city OSM cache file, created/extended by ``autogeoref run``
    #: (the OSM default — see the module docstring). False = BYO file used
    #: verbatim, never written.
    centerlines_from_osm: bool = False
    community_areas_path: Path | None = None
    loc_query: str | None = None
    centerline_name_property: str = "street_nam"
    centerline_type_property: str = "street_typ"
    #: City-wide default escalation model (volume config overrides).
    escalation_model: str | None = None
    escalation_variant: str | None = None
    #: City-wide default escalation ladder (volume config overrides).
    escalation_models: tuple[str, ...] = ()
    escalation_variants: tuple[str | None, ...] = ()
    #: City-wide primary annotation voice (volume config overrides). See
    #: VolumeConfig.annotation_model.
    annotation_model: str = DEFAULT_ANNOTATION_MODEL
    annotation_variant: str | None = None
    #: City-wide evidence channels (volume config overrides; see
    #: VolumeConfig.evidence_channels). Empty = the channels stay opt-in.
    evidence_channels: tuple[str, ...] = ()
    #: BYO rail reference geometry (GeoJSON or a cached Overpass JSON),
    #: mirroring the centerlines contract; enables rail-crossing rescue
    #: anchors for v2 annotations. Must be configured with rail_gazetteer_path.
    rail_geojson_path: Path | None = None
    #: Era-label to modern-rail-group bindings for rail rescue anchors. The
    #: channel fails closed without this partner to avoid exhaustive pairings.
    rail_gazetteer_path: Path | None = None
    #: The city's renumbering conversion (RenumberingEntry JSON list). Numbers
    #: outside the table abstain.
    renumbering_table_path: Path | None = None
    #: The YEAR that table took effect. Read by ONE caller — `autogeoref era`,
    #: which proposes a volume's `addresses_modern` from its LOC edition year
    #: and makes a human confirm it. CONFIG, never a constant in the code, and
    #: the pipeline never reads it: a city's calendar belongs to the city, and
    #: inferring an era from an edition year would bake one city's into every
    #: city. This key lets a TOOL do that arithmetic, then ask.
    renumbering_year: int | None = None
    #: Operator guidance shown beside the console's address-era stanza: which
    #: book converts which district, where a city renumbered its districts on
    #: different dates. Free text for humans; the pipeline never reads it.
    #: None = a generic line built from ``renumbering_year`` when that is set.
    renumbering_note: str | None = None
    #: House numbers per city block — the addresses channel's block-level
    #: tolerances derive from it (address_channel.addr_tol_numbers and the
    #: renumbering table's contradiction tolerance, addresses.RenumberingTable
    #: .convert). 100 is the common US convention.
    address_block_size: int = 100
    #: The city's LOC catalog dump (the ``viewer.sources.loc_titles`` format) — the
    #: default ``--loc-catalog`` for `autogeoref era`, `autogeoref publish`,
    #: and the queue console when the flag is not given. None = era demands
    #: the flag; publication and the console just omit the years.
    loc_catalog_path: Path | None = None
    #: The city's PINNED documented street-rename list, read by the alias
    #: proposer (:mod:`autogeoref.alias.source`). None = this city has no
    #: source configured, and the alias sweep then SCANS and REPORTS its
    #: volumes without ever proposing an entry for them: the tool degrades to
    #: visibility, never to invention.
    rename_source: RenameSource | None = None
    #: The city's house-number grid, used ONLY by the alias proposer's numeral
    #: corroboration (:mod:`autogeoref.address_grid`). None = no numeral check,
    #: which shrinks the proposer's auto-write tier rather than loosening it.
    #: The placement pipeline never reads this.
    address_grid: AddressGrid | None = None
    #: Volumes the alias sweep must never propose for, mapped to WHY. The
    #: escape hatch for volumes the match tripwire flags for a reason that is
    #: not an alias gap (an index sheet with two reads, a fairgrounds special);
    #: the reason string is quoted in the sweep report so the list cannot rot
    #: into folklore.
    alias_sweep_skip: Mapping[str, str] = field(default_factory=dict)
    #: Gated-pool fraction for the console's call estimate.
    gated_fraction: float = DEFAULT_GATED_FRACTION
    volumes: dict[str, VolumeConfig] = field(default_factory=dict)

    def volume(self, identifier: str) -> VolumeConfig:
        vol = self.volumes.get(identifier)
        if vol is not None:
            return vol
        # unknown volume: defaults (two-pass derivation) + city-level knobs
        return VolumeConfig(
            identifier=identifier,
            escalation_model=self.escalation_model,
            escalation_variant=self.escalation_variant,
            escalation_models=self.escalation_models,
            escalation_variants=self.escalation_variants,
            annotation_model=self.annotation_model,
            annotation_variant=self.annotation_variant,
            evidence_channels=self.evidence_channels,
        )

    def aliases_path(self, identifier: str) -> Path:
        return self.aliases_dir / f"aliases-{identifier}.json"


def era_undeclared(city: CityConfig, vol: VolumeConfig) -> bool:
    """The volume declares no address era, on a city that RENUMBERED its houses.

    The one owner of that question, read twice: ``cli._cmd_run`` REFUSES to
    start a run on it where the addresses channel is on, and ``console`` uses
    it to say which backlog volumes are not actually startable. Says nothing
    about the era itself — undeclared means MODERN, which is right for cities
    that never renumbered and a live hazard on a pre-renumbering volume.
    Deciding which is the operator's job.
    """
    return city.renumbering_table_path is not None and vol.addresses_modern is None
