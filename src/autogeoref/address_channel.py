"""The ADDRESSES evidence channel for verified-accept.

Cross-model consensus address numerals vote on a provisional placement:
support, refute, or abstain. This module owns the vote (:func:`address_vote`),
its offline diagnostics, the tuning constants, and the sidecar readers that
feed them; the stage that consumes the vote is
:mod:`autogeoref.verified_accept`, where this is the ONLY channel that may
REFUTE.

The vote is BLOCK-level, it BUYS NO MODEL READS (it votes off caches already
on disk), and a volume printed before its city's renumbering must declare its
era or the channel abstains. `docs/INTERNALS.md` § The Addresses Channel
states each of those rules and why.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shapely.geometry import LineString, Point
from shapely.ops import transform as shapely_transform

from .addresses import (
    ADDR_TOL_BLOCK_RATIO,
    BLOCK_SIZE,
    AddressNumeral,
    RenumberingTable,
    Side,
    address_range_sides,
    consensus_numerals,
    line_coords,
    modern_numeral,
)
from .affine import TO_3857, apply_affine
from .annotate.failures import AnnotateError
from .annotate.providers import canonical_model, model_from_cache_key
from .annotate.schema import numerals_from_raw
from .centerlines import centerline_key
from .frames import rotate_bbox
from .names import Aliases, normalize

if TYPE_CHECKING:
    from .era import AddressEra

logger = logging.getLogger(__name__)


#: Minimum in-block consensus numerals for an addresses YES vote, and the
#: floor below which the channel abstains rather than refutes. Measured
#: basis: consensus numerals are 91-98% externally valid, so a correctly
#: placed sheet with a handful of votable numerals puts nearly all of them
#: in-block; 3 in-block plus a strict majority is far below what a correct
#: placement produces and far above what a one-block-shifted one can.
MIN_NUMERALS = 3
#: Perpendicular tolerance, EPSG:3857 meters, implied position to the
#: nearest segment of the hinted street; splits the truth and one-block-shift
#: populations cleanly.
PERP_TOL_M = 50.0
#: Mixed-evidence abstain band: a YES additionally requires EVERY votable numeral's address
#: diff within this multiple of the along-street tolerance. An in-block majority coexisting
#: with numerals hundreds of numbers off is contradictory evidence, not support — and such a
#: YES can survive a large perpendicular displacement, the parallel-street blind flank.
ADDR_BAND_FACTOR = 2.0


def addr_tol_numbers(block_size: int = BLOCK_SIZE) -> float:
    """Along-street tolerance in HOUSE NUMBERS for a city's block size.

    Block-level by design: 0.75 of the block is THE calibrated invariant
    (:data:`autogeoref.addresses.ADDR_TOL_BLOCK_RATIO`, re-exported here);
    pass ``CityConfig.address_block_size`` for a city whose blocks are not
    the common 100-numbers convention.
    """
    return ADDR_TOL_BLOCK_RATIO * block_size


#: The default tolerance (100-number blocks): the measured calibration above.
ADDR_TOL_NUMBERS = addr_tol_numbers()


#: One street segment ready for voting: 3857 geometry + per-side ranges.
_Segment = tuple[LineString, dict[Side, tuple[int, int]]]


def _implied_point_3857(
    coef: Any, bbox: tuple[float, float, float, float], small_to_full: float
) -> tuple[float, float]:
    """Placement-implied world position of an annotation-frame bbox center."""
    cx = (bbox[0] + bbox[2]) / 2 * small_to_full
    cy = (bbox[1] + bbox[3]) / 2 * small_to_full
    return apply_affine(coef, cx, cy)


def _street_segments(
    centerline_features: list[dict[str, Any]],
    aliases: Aliases | None,
    name_property: str = "street_nam",
    type_property: str = "street_typ",
) -> dict[str, list[_Segment]]:
    """Normalized street name -> addressable segments (3857 line + side ranges).

    Keys come from :func:`autogeoref.centerlines.centerline_key` — the ONE
    key rule shared with :class:`~autogeoref.centerlines.CenterlineIndex`,
    numbered PLACE/COURT twins included (found by the 1919 golden testbed:
    p52's numerals all front 37th Place and were invisible without it). The
    name/type property names are configurable exactly like the index's — a
    city whose centerlines use different fields would otherwise silently
    produce zero votable segments.
    """
    out: dict[str, list[_Segment]] = {}
    for f in centerline_features:
        props = f.get("properties") or {}
        key = centerline_key(props, aliases, name_property, type_property)
        if key is None:
            continue
        coords = line_coords(f.get("geometry") or {})
        if coords is None:
            continue
        sides = address_range_sides(props)
        if not sides:
            continue
        line = shapely_transform(TO_3857.transform, LineString(coords))
        out.setdefault(key, []).append((line, sides))
    return out


def _numeral_in_block(
    segments: list[_Segment],
    pt: Point,
    value: int,
    perp_tol_m: float,
    addr_tol: float,
) -> tuple[bool, float, float]:
    """One numeral's in-block test at the implied position.

    Uses the segment of the hinted street NEAREST the implied point (repeated ranges along a
    street resolve to the local block), then checks BOTH terms (see module docstring):
    perpendicular distance and the block-level difference between the printed value and the
    interpolated house number at the projected position. Sides matching the value's parity are
    preferred (odd numbers belong to the odd frontage); if neither side matches, both are
    considered. Returns ``(in_block, perp_distance_m, address_diff)``.
    """
    line, sides = min(segments, key=lambda s: s[0].distance(pt))
    perp = float(line.distance(pt))
    fraction = float(line.project(pt, normalized=True))
    parity = value % 2
    matched = {s: r for s, r in sides.items() if r[0] % 2 == parity or r[1] % 2 == parity} or sides
    diff = min(
        abs((f_add + fraction * (t_add - f_add)) - value) for f_add, t_add in matched.values()
    )
    return perp <= perp_tol_m and diff <= addr_tol, perp, diff


def _sidecar_numerals(annotations_dir: Path, page: str) -> dict[str, list[AddressNumeral]]:
    """Per-model numeral readings for one page, keyed by MODEL identity.

    Two file shapes feed the channel and NEITHER is bought here: the escalation ladder's tier
    caches and the ``v2`` sidecars left by the retired producer. This read path outlives that
    producer deliberately — deleting it would demote the accepts standing on sidecars already on
    disk. BOTH name components are cache keys that may encode a reasoning variant, so both
    decode back to a model identity and are then CANONICALIZED: without that, one model could
    reach the two-voice floor by agreeing with itself in the only channel allowed to REFUTE.
    ``v2`` wins a key collision, and an unparseable name is kept AS READ rather than dropped.
    """
    out: dict[str, list[AddressNumeral]] = {}

    def _identity(cache_key: str) -> str:
        model = model_from_cache_key(cache_key)
        try:
            return canonical_model(model)
        except AnnotateError:
            # unknown provider / a tier that no longer passes the model gate:
            # keep the decoded name so a retired reading is still one voice
            return model

    def _load(f: Path, model: str) -> None:
        if model in out or model.endswith(".failed"):
            return
        try:
            raw = json.loads(f.read_text())
        except json.JSONDecodeError:
            logger.warning("unreadable v2 cache %s, skipping", f)
            return
        numerals = list(numerals_from_raw(raw))
        if numerals:
            out[model] = numerals

    for f in sorted(annotations_dir.glob(f"p{page}.v2.*.json")):
        cache_key = f.name.removeprefix(f"p{page}.v2.").removesuffix(".json")
        # belt and braces: every real ".failed" key already survives decoding
        # unchanged (no encoded key base64-decodes with a ".failed" tail, and the
        # v1 branch needs "%3A"), so _load's own guard would catch it — but the
        # marker convention is load-bearing enough not to rest on that.
        _load(f, cache_key if cache_key.endswith(".failed") else _identity(cache_key))
    for f in sorted(annotations_dir.glob(f"p{page}.escalated.*.json")):
        cache_key = f.name.removeprefix(f"p{page}.escalated.").removesuffix(".json")
        if cache_key.endswith(".failed"):
            continue
        _load(f, _identity(cache_key))
    return out


def _numerals_in_source_frame(
    numerals: Iterable[AddressNumeral], info: Mapping[str, Any]
) -> list[AddressNumeral]:
    """Turn numerals read on the UPRIGHT small back into the SOURCE frame.

    The annotator (and the escalation ladder) read whatever small prep wrote; when prep
    normalized orientation that image is upright and the manifest entry carries
    ``rotation_applied``, while the record affine every vote is scored against stays in the
    source scan frame. Rotating the bboxes back by ``360 - rotation_applied`` is the numeral-
    side twin of :func:`autogeoref.sheet_inputs.annotation_in_source_frame` (streets) and
    :func:`autogeoref.junction_snap.extraction_in_source_frame` (junctions). Without the key
    this is the identity.
    """
    rotation = int(info.get("rotation_applied", 0))
    if not rotation:
        return list(numerals)
    # prep records small_size in the UPRIGHT (as-written) frame
    upright_size = (int(info["small_size"][0]), int(info["small_size"][1]))
    back = (360 - rotation) % 360
    return [replace(n, bbox=rotate_bbox(n.bbox, back, upright_size)) for n in numerals]


@dataclass(frozen=True)
class AddressVoteConfig:
    """The addresses channel's tuning knobs — the seven always travel together."""

    perp_tol_m: float = PERP_TOL_M
    addr_tol: float = ADDR_TOL_NUMBERS
    min_numerals: int = MIN_NUMERALS
    name_property: str = "street_nam"
    type_property: str = "street_typ"
    address_block_size: int = BLOCK_SIZE
    band_factor: float = ADDR_BAND_FACTOR


@dataclass(frozen=True)
class _NumeralTally:
    """One pass over the consensus numerals: counts for the vote AND the ladder."""

    votable: int
    in_block: int
    diagnostics: list[dict[str, float]]
    street_hint_count: int
    era_failures: int
    unmatched_segments: int


def _tally_numerals(
    consensus: list[AddressNumeral],
    segments_by_street: dict[str, list[_Segment]],
    coef: Any,
    small_to_full: float,
    table: RenumberingTable | None,
    aliases: Aliases | None,
    cfg: AddressVoteConfig,
) -> _NumeralTally:
    """THE observation loop: which numerals are votable, and which are in-block.

    Both :func:`address_vote` (the production verdict) and
    :func:`address_vote_diagnostics` (the offline abstention labels) read
    through this one loop, so the diagnostic can never drift from what the
    vote actually observed.
    """
    votable = 0
    in_block = 0
    diagnostics: list[dict[str, float]] = []
    street_hint_count = 0
    era_failures = 0
    unmatched_segments = 0
    for numeral in consensus:
        if not numeral.street_hint:
            continue
        street_hint_count += 1
        value = modern_numeral(
            numeral.street_hint,
            numeral.value,
            table,
            aliases,
            block_size=cfg.address_block_size,
        )
        if value is None:
            # pre-renumbering number with no conversion: honest abstention
            era_failures += 1
            continue
        segments = segments_by_street.get(normalize(numeral.street_hint, aliases))
        if not segments:
            unmatched_segments += 1
            continue
        px, py = _implied_point_3857(coef, numeral.bbox, small_to_full)
        ok, perp, diff = _numeral_in_block(
            segments, Point(px, py), value, cfg.perp_tol_m, cfg.addr_tol
        )
        votable += 1
        diagnostics.append({"perp_m": round(perp, 1), "addr_diff": round(diff, 1)})
        if ok:
            in_block += 1
    return _NumeralTally(
        votable=votable,
        in_block=in_block,
        diagnostics=diagnostics,
        street_hint_count=street_hint_count,
        era_failures=era_failures,
        unmatched_segments=unmatched_segments,
    )


def _vote_from_tally(
    tally: _NumeralTally, detail: dict[str, Any], cfg: AddressVoteConfig
) -> tuple[bool | None, dict[str, Any]]:
    """The vote arms over a tally (see :func:`address_vote` for the rules)."""
    if tally.votable >= cfg.min_numerals and tally.in_block == 0:
        return False, detail
    if tally.in_block >= cfg.min_numerals and 2 * tally.in_block > tally.votable:
        worst = max((n["addr_diff"] for n in tally.diagnostics), default=0.0)
        if worst > cfg.band_factor * cfg.addr_tol:
            # mixed evidence: an in-block majority coexisting with a numeral
            # hundreds of numbers off supports nothing either way
            detail["abstained"] = f"numeral {worst:.0f} numbers off exceeds the abstain band"
            return None, detail
        return True, detail
    return None, detail


def address_vote(
    per_model: Mapping[str, Iterable[AddressNumeral]],
    coef: Any,
    small_to_full: float,
    centerline_features: list[dict[str, Any]],
    aliases: Aliases | None,
    era: AddressEra,
    renumbering: RenumberingTable | None = None,
    *,
    segments_by_street: dict[str, list[_Segment]] | None = None,
    config: AddressVoteConfig | None = None,
) -> tuple[bool | None, dict[str, Any]]:
    """The addresses channel's vote on one placement.

    Returns ``(vote, detail)``: ``True`` (supports), ``False`` (refutes) or ``None`` (abstains),
    plus the audit detail recorded on the result. Only >=2-model consensus numerals participate.
    YES needs ``config.min_numerals`` in-block, a strict majority of the votable set, and no
    votable numeral beyond the mixed-evidence abstain band; NO needs ``config.min_numerals``
    votable and ZERO in-block; anything thinner abstains. ``config`` carries the tuning knobs
    and defaults to the module's. `docs/INTERNALS.md` § The Addresses Channel states the
    rules and their basis.
    """
    cfg = config if config is not None else AddressVoteConfig()
    detail: dict[str, Any] = {"models": sorted(per_model)}
    if era == "unknown":
        detail["skipped"] = "address era unknown (declare addresses_modern in config)"
        return None, detail
    if len([m for m, nums in per_model.items() if list(nums)]) < 2:
        detail["skipped"] = "fewer than 2 models with numeral readings"
        return None, detail
    table = None if era == "modern" else (renumbering or RenumberingTable())
    consensus = consensus_numerals({m: list(n) for m, n in per_model.items()}, aliases)
    if segments_by_street is None:
        segments_by_street = _street_segments(
            centerline_features, aliases, cfg.name_property, cfg.type_property
        )
    tally = _tally_numerals(consensus, segments_by_street, coef, small_to_full, table, aliases, cfg)
    detail.update(
        consensus_numerals=len(consensus),
        votable=tally.votable,
        in_block=tally.in_block,
        perp_tol_m=cfg.perp_tol_m,
        addr_tol=cfg.addr_tol,
        numerals=tally.diagnostics,
    )
    return _vote_from_tally(tally, detail, cfg)


def _classify_abstention(
    observations: Mapping[str, Any], detail: Mapping[str, Any], min_numerals: int
) -> str:
    """The exclusive abstention ladder over one page's observations.

    Eight of the nine classes; the ninth (``address_era_unknown``) is decided
    by the caller before any observation exists to classify.
    """
    if observations["successful_distinct_models"] < 2:
        return "fewer_than_2_successful_distinct_model_readings"
    if observations["consensus_numerals"] == 0:
        return "no_cross_model_consensus_numeral"
    if observations["consensus_with_street_hint"] == 0:
        return "consensus_numerals_without_street_hint"
    if observations["era_conversion_failures"]:
        return "no_usable_era_conversion"
    if observations["unmatched_addressable_segments"]:
        return "no_matching_addressable_centerline_segment"
    if observations["votable"] < min_numerals:
        return "fewer_than_3_votable_numerals"
    if "abstained" in detail:
        return "mixed_contradictory_evidence"
    return "enough_votable_insufficient_in_block_support"


def address_vote_diagnostics(
    per_model: Mapping[str, Iterable[AddressNumeral]],
    coef: Any,
    small_to_full: float,
    centerline_features: list[dict[str, Any]],
    aliases: Aliases | None,
    era: AddressEra,
    *,
    successful_models: Collection[str],
    renumbering: RenumberingTable | None = None,
    segments_by_street: dict[str, list[_Segment]] | None = None,
    config: AddressVoteConfig | None = None,
) -> tuple[bool | None, dict[str, Any], str | None]:
    """Return the production vote plus an exclusive offline abstention class.

    This deliberately calls :func:`address_vote` for the actual verdict, and
    observes through the same :func:`_tally_numerals` loop the vote used, so
    the diagnostic can neither alter a production vote nor drift from what it
    measured — only retain why an abstention occurred.
    """
    cfg = config if config is not None else AddressVoteConfig()
    vote, detail = address_vote(
        per_model,
        coef,
        small_to_full,
        centerline_features,
        aliases,
        era,
        renumbering,
        segments_by_street=segments_by_street,
        config=cfg,
    )
    observations: dict[str, Any] = {
        "successful_distinct_models": len(set(successful_models)),
        "models_with_numerals": len([nums for nums in per_model.values() if list(nums)]),
        "consensus_numerals": 0,
        "consensus_with_street_hint": 0,
        "era_conversion_failures": 0,
        "unmatched_addressable_segments": 0,
        "votable": 0,
        "in_block": 0,
        "worst_addr_diff": None,
    }
    if era != "unknown":
        table = None if era == "modern" else (renumbering or RenumberingTable())
        if segments_by_street is None:
            segments_by_street = _street_segments(
                centerline_features, aliases, cfg.name_property, cfg.type_property
            )
        consensus = consensus_numerals({m: list(n) for m, n in per_model.items()}, aliases)
        tally = _tally_numerals(
            consensus, segments_by_street, coef, small_to_full, table, aliases, cfg
        )
        observations["consensus_numerals"] = len(consensus)
        observations["consensus_with_street_hint"] = tally.street_hint_count
        observations["era_conversion_failures"] = tally.era_failures
        observations["unmatched_addressable_segments"] = tally.unmatched_segments
        observations["votable"] = tally.votable
        observations["in_block"] = tally.in_block
        observations["worst_addr_diff"] = max(
            (n["addr_diff"] for n in tally.diagnostics), default=None
        )
    detail["diagnostic_counts"] = observations
    if vote is not None:
        return vote, detail, None
    if era == "unknown":
        return vote, detail, "address_era_unknown"
    return vote, detail, _classify_abstention(observations, detail, cfg.min_numerals)


__all__ = [
    "ADDR_BAND_FACTOR",
    "ADDR_TOL_BLOCK_RATIO",
    "ADDR_TOL_NUMBERS",
    "MIN_NUMERALS",
    "PERP_TOL_M",
    "AddressVoteConfig",
    "addr_tol_numbers",
    "address_vote",
    "address_vote_diagnostics",
]
