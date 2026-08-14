"""The >=2-independent-verifiers acceptance path for provisional placements.

Every ``REJECTED (rescue revoked...)`` record carries a provisional placement
the strict gates refused to commit on its own evidence. Three INDEPENDENT
channels — corroboration, junction, addresses — can each cast at most one vote
on it. Acceptance requires >= :data:`MIN_CHANNELS` yes votes and NO refute;
refuting is the addresses channel's alone.

Acceptance can rise ONLY through the named status this writes, e.g.
``OK (verified: junction+addresses)``. Only revoked-prefix records are ever
considered, so no other funnel count can move.

`docs/INTERNALS.md` states each channel's shape and why the junction
veto is gone; re-measure that question before widening a channel.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .address_channel import (
    AddressVoteConfig,
    _numerals_in_source_frame,
    _Segment,
    _sidecar_numerals,
    _street_segments,
    addr_tol_numbers,
    address_vote,
)
from .corroborate import corroborations, is_corroborated_near
from .paths import VolumePaths, iter_results, small_sheet_entry, write_result
from .seam import SheetFit, sheet_fit_from_result
from .volume import REVOKED_PREFIX, status_verified
from .vouchers import VouchNodes, committed_vouch_nodes

if TYPE_CHECKING:
    from .addresses import RenumberingTable
    from .era import AddressEra
    from .names import Aliases

logger = logging.getLogger(__name__)

#: Independent channels required to accept (the keystone contract).
MIN_CHANNELS = 2
#: Canonical channel order for the status string.
CHANNELS = ("corroboration", "junction", "addresses")


def _warpable(fit: SheetFit) -> bool:
    """The record's full GCP set (synthetics included) determines an affine.

    A promotion turns a flagged record into a served one, and serving refits
    the transform from the RAW GCPs (gdalwarp poly1) — a set that is
    degenerate on EITHER side produces an unsolvable transform that blocks
    the whole volume's warp stage. The rescue serializer emits synthetic
    corners exactly to prevent this, so with a healthy record this test is a
    no-op tripwire; it exists for records written before the serializer
    tested the world side.
    """
    pts = np.array([[g[0], g[1]] for g in fit.gcps])
    wpts = np.array([[g[2], g[3]] for g in fit.gcps])
    if len(fit.gcps) < 3:
        return False
    for arr in (pts, wpts):
        centered = arr - arr.mean(axis=0)
        sv = np.linalg.svd(centered, compute_uv=False)
        if sv[0] == 0 or sv[-1] / sv[0] < 1e-2:
            return False
    return True


@dataclass(frozen=True)
class _Addresses:
    """Everything the addresses channel needs from the city, gathered once."""

    centerline_features: list[dict[str, Any]]
    aliases: Aliases | None
    era: AddressEra
    renumbering: RenumberingTable | None
    config: AddressVoteConfig


def _page_votes(
    page: str,
    r: dict[str, Any],
    info: Mapping[str, Any],
    fit: SheetFit,
    nodes: VouchNodes,
    annotations: Path,
    allowed: set[str],
    segments_cache: dict[str, list[_Segment]] | None,
    *,
    addresses: _Addresses,
) -> tuple[dict[str, bool | None], dict[str, Any], dict[str, list[_Segment]] | None]:
    """The three channel votes on one provisional page.

    Returns ``(votes, detail, segments_cache)`` — the per-street segment index
    is built lazily on the first page that can use it and threaded back so the
    volume pays for it once.
    """
    votes: dict[str, bool | None] = {}
    detail: dict[str, Any] = {}
    votes["corroboration"] = True if is_corroborated_near(corroborations(fit, nodes)) else None
    js = r.get("junction_snap")
    # SUPPORT or ABSTAIN, never refute — enforced HERE and not only at the
    # producer, because a result record can carry a baked `"supports": false`
    # from the old binary verdict. Reading that verbatim would let it cast a
    # veto the channel is no longer allowed to cast, so anything that is not
    # an explicit True abstains.
    recorded_junction = js.get("supports") if isinstance(js, dict) else None
    votes["junction"] = True if recorded_junction is True and "junction" in allowed else None
    per_model = (
        {
            model: _numerals_in_source_frame(numerals, info)
            for model, numerals in _sidecar_numerals(annotations, page).items()
        }
        if "addresses" in allowed
        else {}  # muted: an undeclared channel does not vote off stale sidecars
    )
    cfg = addresses.config
    if (
        addresses.era != "unknown"
        and segments_cache is None
        and sum(1 for nums in per_model.values() if nums) >= 2
    ):
        segments_cache = _street_segments(
            addresses.centerline_features, addresses.aliases, cfg.name_property, cfg.type_property
        )
    votes["addresses"], detail["addresses"] = address_vote(
        per_model,
        fit.coef,
        1.0 / float(info["scale"]),
        addresses.centerline_features,
        addresses.aliases,
        addresses.era,
        addresses.renumbering,
        segments_by_street=segments_cache,
        config=cfg,
    )
    return votes, detail, segments_cache


def _decide(
    votes: dict[str, bool | None],
    fit: SheetFit,
    detail: dict[str, Any],
    min_channels: int,
    page: str,
) -> tuple[bool, list[str], list[str]]:
    """The acceptance verdict over one page's votes: ``(accept, yes, refuted)``."""
    yes = [c for c in CHANNELS if votes[c] is True]
    refuted = [c for c in CHANNELS if votes[c] is False]
    accept = len(yes) >= min_channels and not refuted
    if accept and not _warpable(fit):
        accept = False
        detail["unwarpable_gcps"] = True
        logger.warning(
            "p%s: verified-accept blocked — the recorded GCP set is degenerate and "
            "cannot be refit by the warper (serving would fail on the whole volume). "
            "Re-run rescue so the serializer emits synthetic corners for it.",
            page,
        )
    return accept, yes, refuted


def _record_verdict(
    r: dict[str, Any],
    rp: Path,
    page: str,
    status: str,
    votes: dict[str, bool | None],
    detail: dict[str, Any],
    accept: bool,
    yes: list[str],
    refuted: list[str],
) -> dict[str, Any]:
    """Write the audit block (and, on accept, the promotion) onto the result."""
    record = {
        "votes": {c: votes[c] for c in CHANNELS},
        "accepted": accept,
        **detail,
    }
    if refuted:
        logger.info("p%s: verified-accept blocked by refuting channel(s) %s", page, refuted)
    if accept:
        record["previous_status"] = status
        r["status"] = status_verified(yes)
        logger.info("p%s: VERIFIED ACCEPT via %s", page, "+".join(yes))
    r["verified_accept"] = record
    write_result(rp, r)
    return record


def _warn_silent_channels(
    allowed: set[str], scored: dict[str, dict[str, Any]], annotations: Path
) -> None:
    """Report a DECLARED channel that abstained on every provisional page."""
    # A DECLARED channel that said nothing on a SINGLE page is a config or data
    # fault wearing the costume of a clean funnel. Corroboration is exempt: a
    # volume whose committed neighbours simply do not touch its flagged sheets
    # is an ordinary, honest state.
    for channel in sorted(allowed - {"corroboration"}):
        if scored and not any(rec["votes"][channel] is not None for rec in scored.values()):
            logger.warning(
                "the %s channel is DECLARED but abstained on ALL %d provisional pages — "
                "it produced no evidence either way, so verified-accept effectively ran "
                "with one fewer channel. Check its inputs before reading this funnel as a "
                "measurement of it (junction: the [cv] extra + readable smalls; addresses: "
                "%s/p<N>.escalated.*.json tier caches or p<N>.v2.*.json sidecars from >=2 "
                "DISTINCT models — nothing buys these on demand since the consensus "
                "producer was cut, so a quiet channel usually means the ladder never "
                "escalated these pages — and the volume's declared address era)",
                channel,
                len(scored),
                annotations,
            )


def stage_verified_accept(
    paths: VolumePaths,
    centerline_features: list[dict[str, Any]],
    aliases: Aliases | None,
    address_era: AddressEra = "unknown",
    renumbering: RenumberingTable | None = None,
    vouch_nodes: VouchNodes | None = None,
    min_channels: int = MIN_CHANNELS,
    config: AddressVoteConfig | None = None,
    channels: Collection[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Accept provisional placements confirmed by >=2 independent channels.

    Runs AFTER corroborate and junction-verify: it consumes the committed voucher
    pool and the recorded advisory ``junction_snap`` verdicts, and a missing verdict
    abstains. Non-revoked records are byte-untouched. ``channels`` is the
    optional-channel allow-list (None = all may vote); ``min_channels`` is the
    acceptance contract and must not go below 2; ``config`` carries the addresses
    knobs, its ``addr_tol`` re-derived from the block size. Returns
    ``{page: verified_accept_record}`` — see `docs/INTERNALS.md`.
    """
    cfg = config if config is not None else AddressVoteConfig()
    addresses = _Addresses(
        centerline_features=centerline_features,
        aliases=aliases,
        era=address_era,
        renumbering=renumbering,
        config=replace(cfg, addr_tol=addr_tol_numbers(cfg.address_block_size)),
    )
    allowed = set(CHANNELS) if channels is None else ({"corroboration"} | set(channels))
    manifest = json.loads(Path(paths.manifest).read_text())
    nodes = committed_vouch_nodes(paths) if vouch_nodes is None else vouch_nodes
    # the per-street segment index is volume-wide; build it once, lazily
    # (only pages with >=2 sidecar models ever need it)
    segments_cache: dict[str, list[_Segment]] | None = None
    scored: dict[str, dict[str, Any]] = {}
    accepted: list[str] = []
    for page, r, rp in iter_results(paths):
        status = str(r.get("status", ""))
        if not status.startswith(REVOKED_PREFIX):
            continue
        # no rotation skip: _numerals_in_source_frame turns numerals read on an
        # orientation-normalized (upright) small back into the record affine's
        # source frame, so rotated scans keep every channel
        entry = small_sheet_entry(
            paths,
            manifest,
            page,
            stage="verified-accept",
            require_image=False,
            skip_rotated=False,
        )
        if entry is None:
            continue
        info, _img = entry
        fit = sheet_fit_from_result(page, r)
        if fit is None:
            continue

        votes, detail, segments_cache = _page_votes(
            page,
            r,
            info,
            fit,
            nodes,
            Path(paths.annotations),
            allowed,
            segments_cache,
            addresses=addresses,
        )
        accept, yes, refuted = _decide(votes, fit, detail, min_channels, page)
        scored[page] = _record_verdict(r, rp, page, status, votes, detail, accept, yes, refuted)
        if accept:
            accepted.append(page)
    logger.info(
        "verified-accept: %d provisional pages scored, %d accepted%s",
        len(scored),
        len(accepted),
        f" ({', '.join('p' + p for p in accepted)})" if accepted else "",
    )
    _warn_silent_channels(allowed, scored, Path(paths.annotations))
    return scored


__all__ = [
    "CHANNELS",
    "MIN_CHANNELS",
    "stage_verified_accept",
]
