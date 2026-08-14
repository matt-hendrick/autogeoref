"""Placement-stage construction for one resolved `RunPolicy`.

prep -> annotate -> match -> revoke-stale -> rescue -> seam -> corroborate ->
report, with the back half appended when the run asks for it.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING

from ..dag import Stage
from ..paths import sheet_images
from ..stages.corroborate import stage_corroborate
from ..stages.match import stage_match
from ..stages.report import stage_report
from ..stages.rescue import stage_rescue, stage_revoke_shared_street_rescues
from ..stages.seam import stage_seam
from ..volume_constants import resolve_constants
from .backhalf import build_back_half_stages

if TYPE_CHECKING:
    from ..runcontext import RunContext
    from ..runpolicy import RunPolicy

logger = logging.getLogger(__name__)


# Operator guidance raised by _run_annotate when sheets could not be read.
_UNREAD_SHEETS_GUIDANCE = (
    "{volume}: {count} sheet(s) could not be READ, and a volume "
    "placed without them is a volume that is quietly short: {unread}.\n"
    "  Each carries a p<N>.failed.json marker in {annotations} recording why. A "
    "marker is NOT an annotation and is never counted as a read — no budget was "
    "charged for a call that did not land.\n"
    "  Transient failure (timeout, 429, a malformed response)? DELETE the marker(s) "
    "and re-run: only those pages are re-read.\n"
    "  Genuinely unreadable sheet? Re-run with --allow-failed-reads to place the "
    "volume WITHOUT them — it will say so, loudly, and the funnel will be short by "
    "exactly that many sheets."
)


def _run_prep(ctx: RunContext) -> None:
    # Full-res scans -> 2000px smalls + the manifest, which is the ONLY valid
    # small<->full pixel-frame conversion. Raises UnrecognizedSheetError on a
    # region image nothing can name, which is a sheet about to be dropped in
    # silence (prep.py) — the run stops there rather than annotating a volume
    # that is quietly short a sheet.
    from ..prep import prep_volume

    paths = ctx.paths
    result = prep_volume(paths.regions, paths.sheets)
    logger.info("prep: %s", result.summary())
    for name, kind in sorted(result.skipped.items()):
        logger.info("prep: %s carries no map (%s); not a page", name, kind)


def _run_annotate(ctx: RunContext) -> None:
    # The read every later stage consumes: ~1 model call per uncached sheet,
    # and the pipeline's largest single spender. It runs by default;
    # --no-annotate is the opt-out for a
    # volume whose annotations are already on disk (every golden fixture).
    from ..annotate_volume import ReadIdentity
    from ..annotate_volume import annotate_volume as run_annotation

    args = ctx.args
    paths = ctx.paths
    vol = ctx.vol
    result = run_annotation(
        paths,
        args.volume,
        identity=ReadIdentity(vol.annotation_model, vol.annotation_variant),
        limit=args.limit,
        jobs=args.annotate_jobs,
        reread_unattributed=args.reread_unattributed,
    )
    if not result.unread or args.allow_failed_reads:
        if result.unread:
            # ASKED for, so allowed — but never quiet. This volume is SHORT, and
            # every number downstream (the funnel, the scale/rotation medians,
            # the report) is computed over the sheets that survived.
            logger.warning(
                "%s: PLACING A VOLUME THAT IS SHORT %d SHEET(S) (--allow-failed-reads): "
                "%s. Their reads did not land; they are absent from the funnel and from "
                "the volume constants derived from it.",
                args.volume,
                len(result.unread),
                ", ".join(result.unread),
            )
        return
    # Failure markers suppress automatic re-spending, so continuing would omit
    # pages from every later stage.
    raise RuntimeError(
        _UNREAD_SHEETS_GUIDANCE.format(
            volume=args.volume,
            count=len(result.unread),
            unread=", ".join(result.unread),
            annotations=paths.annotations,
        )
    )


def _run_street_index(ctx: RunContext) -> None:
    """Fill ``ctx.index_windows``, read by the rescue stage (log-only check)."""
    # Street-index priors spend model budget and are a log-only rescue
    # plausibility check, so they run as a stage.
    from ..street_index import index_priors, read_index

    args = ctx.args
    paths = ctx.paths
    entries = read_index(args.street_index, cache_dir=paths.root / "index")
    ctx.index_windows = index_priors(
        entries,
        ctx.clipped_features,
        ctx.index.aliases,
        renumbering=ctx.index_renumbering,
        address_block_size=ctx.city.address_block_size,
    )
    logger.info("street-index priors: %d page windows", len(ctx.index_windows))


def _cv_missing() -> bool:
    """Is the optional ``[cv]`` extra absent? (cv2 -> junction_snap, orient.)

    A SPEC check, not ``except ImportError`` around the stage call: a broken
    install (``libGL.so.1``, a skimage/numpy ABI mismatch) also raises
    ImportError, and reporting THAT as "the extra is not installed" would send
    the operator to `uv sync --extra cv`, which cannot fix it. Absent = we skip
    and say so; present-but-broken = a real failure, surfaced as one.
    """
    return importlib.util.find_spec("cv2") is None


def _run_escalate(ctx: RunContext, policy: RunPolicy) -> None:
    # Evidence-gated re-annotation of REJECTED pages up a cheap-first model
    # ladder (spends model budget; autogeoref.escalate). Default-ON wherever
    # a ladder resolves.
    #
    # It needs the [cv] extra because drawn junctions gate which rejected
    # pages are worth re-reading.
    args = ctx.args
    if _cv_missing():
        # skipping is honest: escalation is ADDITIVE (a page it cannot re-read
        # stays honestly REJECTED). An explicit --escalate ASKED for the spend,
        # so that caller gets an error rather than a silent no-op.
        if args.escalate or args.escalate_model:
            raise SystemExit(
                "--escalate: the [cv] extra is not installed, and escalation "
                "needs it to gate which pages are worth re-reading — "
                "`uv sync --extra cv`"
            )
        logger.warning(
            "escalation skipped: the [cv] extra is not installed, so the "
            "drawn-junction evidence gate cannot run; rejected pages stay flagged "
            "(`uv sync --extra cv` to enable it)"
        )
        return
    from ..escalate import stage_escalate
    from ..volume import constraints_from_constants

    paths = ctx.paths
    vol = ctx.vol
    # constants: config pin, else what stage_match derived and persisted
    pins = resolve_constants(paths, vol)
    if pins is None:
        raise RuntimeError(
            f"{vol.identifier}: escalate needs pinned scale/rotation "
            f"(none in config and no usable {paths.constants} — run match first)"
        )
    # ladder resolved and validated non-empty at the top of _cmd_run
    try:
        stage_escalate(
            paths,
            ctx.index,
            constraints_from_constants(*pins),
            policy.escalation_models,
            variants=policy.escalation_variants,
            page_scale_multiples=vol.page_scale_multiples,
            jobs=args.annotate_jobs,
        )
    except OSError as exc:
        # The backend could not even be LAUNCHED. Timeouts and bad responses
        # are already the retry path inside the stage; this is the environment.
        # Escalation is ADDITIVE — a page it cannot re-read stays honestly
        # REJECTED — so it must not sink a run that would otherwise rescue,
        # corroborate and report. An explicit --escalate DID ask for the spend:
        # fail loudly for that caller instead of silently placing nothing.
        if args.escalate or args.escalate_model:
            raise
        logger.warning(
            "escalation skipped: model backend could not be launched (%s); "
            "rejected pages stay flagged",
            exc,
        )


def _run_junction_verify(ctx: RunContext) -> None:
    # ADVISORY: records junction_snap verdicts on rescue-family results,
    # never changes a status (autogeoref.verify). Needs the [cv] extra.
    args = ctx.args
    if _cv_missing():
        # The stage is default-ON where a city declares the channel, so an
        # absent extra must not sink a run that would otherwise rescue,
        # corroborate and report — verified-accept simply hears one fewer
        # channel, and `channels_heard` below records that it was not merely
        # quiet. WARN, never swallow; a caller who NAMED the channel gets an
        # error rather than a shrug.
        if args.verify_junctions or args.verified_accept:
            raise SystemExit(
                "--verify-junctions / --verified-accept: the [cv] extra is not "
                "installed and the junction channel cannot run — `uv sync --extra cv`"
            )
        logger.warning(
            "junction channel skipped: the [cv] extra is not installed; "
            "verified-accept will run with one fewer channel "
            "(`uv sync --extra cv` to enable it)"
        )
        return
    from ..verify import stage_junction_verify

    stage_junction_verify(ctx.paths, ctx.centerline_features, ctx.bounds)


def _run_verified_accept(ctx: RunContext, policy: RunPolicy) -> None:
    # >=2-independent-verifiers acceptance for provisional (revoked)
    # placements; consumes the recorded junction verdicts + committed
    # vouch pool + v2 numeral caches (autogeoref.verified_accept)
    from ..address_channel import AddressVoteConfig
    from ..verified_accept import stage_verified_accept

    city = ctx.city
    era = ctx.era
    stage_verified_accept(
        ctx.paths,
        ctx.clipped_features,
        ctx.index.aliases,
        address_era=era,
        renumbering=ctx.renumbering if era == "renumbered" else None,
        # the segment index must key exactly like the centerline index
        config=AddressVoteConfig(
            name_property=city.centerline_name_property,
            type_property=city.centerline_type_property,
            address_block_size=city.address_block_size,
        ),
        # the resolved allow-list, so an UNDECLARED channel does not vote off
        # evidence a previous run left on disk. An explicit flag names its own
        # channel; corroboration is unconditional and never listed.
        channels=policy.allowed_channels,
    )


def build_stages(ctx: RunContext, policy: RunPolicy) -> list[Stage]:
    """Build the ordered placement and optional serving plan for one run."""
    args = ctx.args
    vol = ctx.vol
    paths = ctx.paths
    # Eagerly populated by cli before this plan is built, preserving its failure
    # timing and keeping the stage functions limited to stage work.
    index = ctx.index

    # listed once, here: the prep stage declares these as its inputs (see below)
    region_images = sheet_images(paths.regions)

    return [
        # PREP IS A STAGE, not a thing you remember to do first. Safe in the DAG because
        # prep_sheet is idempotent and the orientation policy is STICKY per volume. Inputs
        # are the IMAGE FILES, never the regions/ directory: a scan rewritten in place does
        # not touch the parent's mtime, so a directory-keyed check would keep a manifest
        # describing the OLD image. `enabled` on the image list, not the directory: a
        # fixture volume ships a manifest and NO scans, and must skip prep, not fail it.
        Stage(
            name="prep",
            run=lambda: _run_prep(ctx),
            inputs=region_images,
            outputs=[paths.manifest],
            enabled=lambda: bool(region_images),
        ),
        # BEFORE match, AFTER prep: prep writes the manifest and the smalls this
        # reads, and match has nothing to fit until this has run. No outputs are
        # declared (the targets are per-page, and the stage is internally cached:
        # an already-annotated volume costs zero calls), so it can never be
        # fresh-skipped into silently doing nothing.
        Stage(
            name="annotate",
            run=lambda: _run_annotate(ctx),
            inputs=[paths.manifest],
            outputs=[],
            enabled=lambda: not args.no_annotate,
        ),
        Stage(
            name="match",
            run=lambda: stage_match(paths, index, vol),
            inputs=[paths.manifest],
            outputs=[],  # per-page targets; stage itself is idempotent
        ),
        Stage(
            name="escalate",
            run=lambda: _run_escalate(ctx, policy),
            # default ON when a ladder resolves; --no-escalate is the opt-out
            enabled=lambda: policy.run_escalation,
        ),
        Stage(
            name="revoke-stale",
            run=lambda: stage_revoke_shared_street_rescues(paths, index.aliases),
        ),
        Stage(
            name="street-index",
            run=lambda: _run_street_index(ctx),
            enabled=lambda: args.street_index is not None,
        ),
        Stage(
            name="rescue",
            run=lambda: stage_rescue(
                paths,
                index,
                vol,
                rail_index=ctx.rail_index,
                index_windows=ctx.index_windows,
                bounds=ctx.bounds,
            ),
        ),
        # seam BEFORE corroborate: corroboration measures node agreement
        # against committed neighbors, and the seam solve improves those
        # positions first — the frame in which the recorded volume was QA'd
        Stage(
            name="seam",
            run=lambda: stage_seam(paths, overview_pages=vol.overview_pages),
        ),
        Stage(name="corroborate", run=lambda: stage_corroborate(paths)),
        Stage(
            name="junction-verify",
            run=lambda: _run_junction_verify(ctx),
            enabled=lambda: policy.run_junction,
        ),
        # after corroborate + junction-verify: consumes both evidence streams.
        # No addresses-channel PRODUCER stage sits between them — that channel
        # reads the escalation ladder's tier caches and any sidecar on disk.
        Stage(
            name="verified-accept",
            run=lambda: _run_verified_accept(ctx, policy),
            enabled=lambda: policy.run_verified,
        ),
        Stage(
            name="report",
            run=lambda: stage_report(
                paths,
                args.volume,
                tiles_root=args.tiles,
                # the serve-staleness note reads the directory THIS city
                # publishes into, which is not the same one for every city
                city_toml=args.city,
                overview_pages=vol.overview_pages,
            ),
        ),
        # the back half (opt-in --warp): committed placements -> servable
        # imagery. warp/mask are internally idempotent per sheet; mosaic
        # rewrites its VRTs only on content change so the tile stage's
        # mtime freshness fires only when a COG or mask really moved.
        *build_back_half_stages(
            args,
            paths,
            content_masks=vol.content_masks,
            content_mask_exempt=vol.content_mask_exempt,
            overview_pages=vol.overview_pages,
            page_scale_multiples=vol.page_scale_multiples,
            enabled=lambda: policy.warp,
        ),
    ]
