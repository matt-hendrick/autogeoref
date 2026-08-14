"""`run` and `prep`: taking a volume from scans to placements.

`run` is prep -> annotate -> match -> revoke-stale -> rescue -> seam ->
corroborate -> report, as an idempotent, dual-marker DAG, and it is the biggest
model spender here. With ``--warp`` the back half joins the same DAG; with
``--warp-only`` it is the only thing that runs. `prep` is the pre-spend look.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..cli_context import fail
from ..config.load import load_city_config
from ..dag import Runner
from ..paths import VolumeBusyError, VolumePaths, volume_lock
from ..run_inputs import NoBoundsSourceError, resolve_bounds
from ..runcontext import RunContext
from ..validation import (
    nonnegative_int,
    positive_int,
    tile_zoom,
    volume_argument,
    volume_id,
)

if TYPE_CHECKING:
    from ..annotate_volume import ReadIdentity
    from ..config.model import CityConfig, VolumeConfig

logger = logging.getLogger(__name__)


def _run_back_half(args: argparse.Namespace, paths: VolumePaths) -> int:
    """warp -> mask -> mosaic -> tile, and NOTHING else (`--warp-only`).

    `--warp` runs the back half after the placement stages, which is right when
    you are placing a volume and WRONG as a way to serve an already-reviewed
    one: the placement stages carry no output targets, so they all re-execute.
    Escalation and the evidence channels would fire again, and a page they
    promote out of the flagged pool would be warped and tiled IN THE SAME
    INVOCATION — a placement no human ever saw becoming public. So serving is
    its own path, consuming committed records and never revisiting one.
    """
    from ..runplan.backhalf import build_back_half_stages

    # The only city-config facts the back half consumes: the volume's declared
    # mask style (VolumeConfig.content_masks / content_mask_exempt), its
    # declared overview pages (VolumeConfig.overview_pages), and its declared
    # per-page scale multiples (mask QA's window re-check only).
    # Everything model- or placement-facing stays out of this path.
    vol = load_city_config(args.city).volume(args.volume)
    stages = build_back_half_stages(
        args,
        paths,
        content_masks=vol.content_masks,
        content_mask_exempt=vol.content_mask_exempt,
        overview_pages=vol.overview_pages,
        page_scale_multiples=vol.page_scale_multiples,
    )
    results = Runner(paths.markers).execute(stages, dry_run=args.dry_run, force=args.force)
    for r in results:
        print(f"{r.name}: {r.status}" + (f" ({r.error})" if r.error else ""))
    return 1 if any(r.status == "failed" for r in results) else 0


def _annotation_estimate(
    paths: VolumePaths,
    volume: str,
    identity: ReadIdentity,
    limit: int | None,
    reread_unattributed: bool = False,
    *,
    escalation_tiers: int,
    gated_fraction: float,
) -> str:
    """What a run will spend on model calls, said out loud before the stages run.

    Not before ALL spend: on a volume with no declared bounds source,
    `bounds_bootstrap.derive_bounds` has already read its sample by the time
    this prints, and those reads are counted here as cached. Printed by
    `_cmd_run` rather than by the stage, because the runner never enters a stage
    body under ``--dry-run`` — and "``--dry-run`` prints the call count and
    spends nothing" is the whole budget gate on the biggest spender.
    """
    from ..annotate_volume import NotPreppedError, plan, unprepped_summary

    try:
        return plan(
            paths,
            volume,
            identity=identity,
            limit=limit,
            reread_unattributed=reread_unattributed,
            escalation_tiers=escalation_tiers,
            gated_fraction=gated_fraction,
        ).summary()
    except NotPreppedError:
        return unprepped_summary(
            paths,
            volume,
            identity.model,
            identity.variant,
            limit,
            escalation_tiers=escalation_tiers,
            gated_fraction=gated_fraction,
        )


def _cmd_run(args: argparse.Namespace) -> int:
    from ..runpolicy import RunPolicy

    try:
        args.volume = volume_id(args.volume)
    except ValueError as exc:
        raise SystemExit(f"invalid volume: {exc}") from exc
    warp_only = RunPolicy.is_warp_only(args)
    paths = VolumePaths(root=args.work / args.volume)
    body = _run_back_half if warp_only else _run_placement
    if args.dry_run:
        # a dry run is genuinely side-effect-free — the runner never enters a
        # stage body and writes no markers — so it must not exclude (or be
        # excluded by) a real owner of the volume
        return body(args, paths)
    try:
        # the ONE owner of this volume tree for the whole operation: taken
        # before any model or GDAL work, so a second run / prep / review-apply
        # fails here instead of duplicating billable reads or interleaving
        # result, marker, and mask writes with ours
        with volume_lock(paths, operation="run --warp-only" if warp_only else "run"):
            return body(args, paths)
    except VolumeBusyError as exc:
        raise SystemExit(f"error: {exc}") from exc


def _resolve_or_derive_bounds(
    args: argparse.Namespace, city: CityConfig, vol: VolumeConfig, paths: VolumePaths
) -> tuple[float, float, float, float]:
    try:
        return resolve_bounds(city, vol, args.viewer_manifest)
    except NoBoundsSourceError as exc:
        # NOTHING is declared — a declared source that failed to load stays a
        # loud ConfigError — so derived bounds are next: a previous run's
        # persisted derivation, else the bootstrap. A dry run spends NOTHING,
        # and a run whose flags already promised a capped or zero annotation
        # spend bootstraps from CACHED reads only, so what this prints stays
        # what it spends.
        from ..bounds import BoundsError
        from ..bounds_bootstrap import SAMPLE_PAGES, derive_bounds, persisted_bounds

        persisted = persisted_bounds(paths)
        if persisted is not None:
            return persisted
        if args.dry_run:
            raise SystemExit(
                f"{exc} — a real run DERIVES bounds here, from <= {SAMPLE_PAGES} "
                "sampled sheets (reads land in the annotation cache the run replays "
                "free); --dry-run spends nothing, so it stops instead"
            ) from exc
        spend = args.limit is None and not args.no_annotate
        try:
            return derive_bounds(paths, args.volume, city, vol, spend=spend)
        except BoundsError as derive_exc:
            # a refusal with instructions, not a bug — no traceback
            raise SystemExit(str(derive_exc)) from derive_exc


def _run_placement(args: argparse.Namespace, paths: VolumePaths) -> int:
    """The placement pipeline for one run invocation (everything but --warp-only)."""
    from ..runpolicy import RunPolicy

    city = load_city_config(args.city)
    vol = city.volume(args.volume)
    policy = RunPolicy.resolve(args, city, vol)
    _preflight_run_inputs(args, city, vol, policy)
    bounds = _resolve_or_derive_bounds(args, city, vol, paths)
    logger.info("%s: bounds %s", args.volume, [round(b, 4) for b in bounds])
    if city.centerlines_from_osm and not args.dry_run:
        # no centerlines configured -> the OSM default: make the per-city
        # cache cover this volume's bounds (one polite Overpass fetch when
        # uncovered, zero network after).
        # Skipped under --dry-run: a dry run spends NOTHING — no model
        # budget (G2 finding 5) and no network fetch either.
        from ..osm import ensure_city_centerlines

        ensure_city_centerlines(city.centerlines_path, bounds)
    ctx = RunContext(args=args, city=city, vol=vol, paths=paths, bounds=bounds)
    # Load shared required inputs before constructing the stage plan.
    _ = ctx.index
    _ = ctx.rail_index

    from ..runplan.placement import build_stages

    stages = build_stages(ctx, policy)
    # The budget, out loud, before the first stage runs — including under
    # --dry-run, which is the only way to ask "what would this cost?" without
    # paying it. See _annotation_estimate.
    if not args.no_annotate:
        from ..annotate_volume import ReadIdentity

        print(
            _annotation_estimate(
                paths,
                args.volume,
                ReadIdentity(vol.annotation_model, vol.annotation_variant),
                args.limit,
                args.reread_unattributed,
                escalation_tiers=len(vol.escalation_ladder()),
                gated_fraction=city.gated_fraction,
            )
        )
    runner = Runner(paths.markers)
    results = runner.execute(stages, dry_run=args.dry_run, force=args.force)
    failed = [r for r in results if r.status == "failed"]
    for r in results:
        print(f"{r.name}: {r.status}" + (f" ({r.error})" if r.error else ""))
    if not failed and not args.dry_run:
        report = paths.root / "report.md"
        if report.exists():
            print()
            print(report.read_text())
    return 1 if failed else 0


def _preflight_run_inputs(
    args: argparse.Namespace, city: CityConfig, vol: VolumeConfig, policy: Any
) -> None:
    """Reject required placement inputs before bounds resolution, fetches, or model spend."""

    def require_file(path: Path, label: str) -> None:
        if not path.is_file():
            raise SystemExit(f"{label} must be an existing file: {path}")

    if not city.centerlines_from_osm:
        require_file(city.centerlines_path, "configured centerlines")
    if args.street_index is not None:
        require_file(args.street_index, "--street-index")
    if args.viewer_manifest is not None:
        require_file(args.viewer_manifest, "--viewer-manifest")
    if vol.bounds_from_counterpart is not None and args.viewer_manifest is None:
        raise SystemExit(
            f"{args.volume}: bounds_from={vol.bounds_from_counterpart} needs --viewer-manifest"
        )
    if vol.bounds_areas:
        if city.community_areas_path is None:
            raise SystemExit(f"{args.volume}: bounds_areas needs city community_areas")
        require_file(city.community_areas_path, "configured community_areas")
    if city.rail_geojson_path is not None:
        require_file(city.rail_geojson_path, "configured rail_geojson")
        if city.rail_gazetteer_path is None:
            raise SystemExit("configured rail_geojson requires rail_gazetteer")
        require_file(city.rail_gazetteer_path, "configured rail_gazetteer")
    renumbering = vol.renumbering_table_path or city.renumbering_table_path
    if (
        "addresses" in policy.allowed_channels
        and vol.addresses_modern is False
        and renumbering is not None
    ):
        require_file(renumbering, "configured renumbering_table")
        from ..addresses import RenumberingTable

        try:
            RenumberingTable.from_json(renumbering)
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(
                f"configured renumbering_table is invalid: {renumbering}: {exc}"
            ) from exc
    for model in args.escalate_model or ():
        from ..annotate import failures, providers

        try:
            providers.parse_model_ref(model)
        except failures.AnnotateError as exc:
            raise SystemExit(f"--escalate-model {model!r}: {exc}") from exc
    policy.warn_unavailable_model_clis(vol.annotation_model if not args.no_annotate else None)


def _cmd_prep(args: argparse.Namespace) -> int:
    """Prep one volume and print the reconciliation — the pre-spend check.

    `run` preps too (it is the first stage). This exists to let you LOOK before
    committing roughly one model call per sheet: the manifest count is
    legitimately lower than the region-file count, so only the reconciliation
    can tell you whether a sheet was skipped by design or dropped by accident.
    """
    from ..prep import prep_volume

    paths = VolumePaths(root=args.work / args.volume)
    if not paths.regions.is_dir():
        print(f"{args.volume}: no regions/ under {paths.root} — nothing to prep", file=sys.stderr)
        return 1
    try:
        # prep rewrites smalls and the manifest — the frame every cached
        # annotation lives in — so it takes the same volume ownership a run does
        with volume_lock(paths, operation="prep"):
            result = prep_volume(
                paths.regions, paths.sheets, normalize_orientation=not args.no_normalize_orientation
            )
    except VolumeBusyError as exc:
        return fail(exc)
    print(result.summary())
    for name, kind in sorted(result.skipped.items()):
        print(f"  skipped (no map): {name}  [{kind}]")
    print(f"manifest: {paths.manifest}")
    return 0


def add_run_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    parents: dict[str, argparse.ArgumentParser],
) -> None:
    run = sub.add_parser(
        "run",
        help="place one volume: scans -> smalls -> annotations -> placements. Spends "
        "model budget (annotate, escalate); --dry-run prints what it would spend and "
        "spends nothing",
        parents=[parents["work_root"]],
    )
    run.add_argument("volume", type=volume_argument)
    run.add_argument("--city", type=Path, required=True)
    run.add_argument(
        "--tiles",
        type=Path,
        default=Path("deploy/tiles"),
        help="serving root, used only to note when committed records outdate the served bake",
    )
    run.add_argument("--viewer-manifest", type=Path, default=None)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument(
        "--no-annotate",
        action="store_true",
        help="skip the annotate stage — the volume's annotations/ are already on disk "
        "(every golden fixture ships them). A volume with NO annotations then has "
        "nothing to match, and `match` halts saying so rather than reporting ok over "
        "zero sheets",
    )
    run.add_argument(
        "--limit",
        type=nonnegative_int,
        default=None,
        help="cap the annotate stage at N model calls (budget control). The cap is "
        "applied BEFORE the count is printed, so the number said out loud is the "
        "number spent; the rest of the volume stays uncached and a later run resumes it",
    )
    run.add_argument(
        "--reread-unattributed",
        action="store_true",
        help="re-read pages whose only annotation predates model identity (a bare "
        "p<N>.json with no cache records), writing keyed reads under the configured "
        "model. Off by default: legacy reads are reused as-is, so a cache-layout "
        "migration never becomes unplanned spend",
    )
    run.add_argument(
        "--annotate-jobs",
        type=positive_int,
        default=1,
        help="pages annotated and escalated concurrently (default 1). Keep it small — this is a "
        "metered backend, and a budget limit hit by one worker still lets the calls "
        "already in flight finish, so a large pool overshoots the limit further",
    )
    run.add_argument(
        "--allow-failed-reads",
        action="store_true",
        help="place the volume even though some sheets could not be READ. Default is to "
        "HALT: a failed read leaves a marker, the marker makes the page invisible to every "
        "later stage, and the volume would be placed and served one sheet short in silence "
        "(this is how _017/_018 lost sheets for months). Delete the markers to retry those "
        "pages instead; use this only for a sheet that is genuinely unreadable",
    )
    run.add_argument(
        "--verify-junctions",
        action="store_true",
        help="the junction channel is ON by default wherever the city TOML declares "
        "evidence_channels; pass this to force the advisory junction-snap verdicts "
        "on a config that does not (needs the [cv] extra)",
    )
    run.add_argument(
        "--verified-accept",
        action="store_true",
        help="accept provisional (revoked) placements confirmed by >=2 independent "
        "evidence channels — ON by default wherever the city TOML declares "
        "evidence_channels; pass this to force it (implies junction-verify). NOTE the "
        "addresses channel is NOT forced on with it: it votes only where the city TOML "
        'declares "addresses", and it reads cached numerals rather than buying any',
    )
    run.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the evidence channels (junction-verify, verified-accept) even "
        "where the city TOML declares them — spends no model budget on provisional "
        "pages",
    )
    run.add_argument(
        "--street-index",
        type=Path,
        default=None,
        help="volume street-index page image: read it (tiled; cached under "
        "<work>/<volume>/index; uncached tiles spend model budget) and use the "
        "per-page windows as a log-only rescue plausibility check",
    )
    run.add_argument(
        "--escalate",
        action="store_true",
        help="escalation is ON by default wherever a ladder is configured; pass "
        "this only to assert that it runs — it errors when no ladder is "
        "configured (config escalation_models / escalation_model)",
    )
    run.add_argument(
        "--no-escalate",
        action="store_true",
        help="skip the escalation stage even when a ladder is configured "
        "(spends no model budget on REJECTED pages)",
    )
    run.add_argument(
        "--escalate-model",
        action="append",
        default=None,
        help="override the escalation ladder; repeat the flag for multiple "
        "tiers, cheapest first — frontier tiers run only after earlier tiers "
        "fail (implies --escalate)",
    )
    run.add_argument(
        "--warp",
        action="store_true",
        help="run the back half after report: warp committed sheets (full-res "
        "scans under <work>/<volume>/regions/) to COGs, detect+heal sheet "
        "masks, mosaic, and pack <volume>.pmtiles (needs system GDAL)",
    )
    run.add_argument(
        "--warp-only",
        action="store_true",
        help="run ONLY the back half (warp/mask/mosaic/tile) on a volume that has "
        "already been placed and reviewed. Unlike --warp it does not re-run the "
        "placement stages — those carry no output targets, so they always re-execute, "
        "and a page the evidence channels promoted on that pass would be baked in the "
        "same breath, unreviewed. This is what `queue --track serve` runs",
    )
    run.add_argument(
        "--max-zoom",
        type=tile_zoom,
        default=None,
        help="max web-mercator tile zoom for --warp (default 20; lower it "
        "for a quick preview bake)",
    )
    run.set_defaults(func=_cmd_run)


def add_prep_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    parents: dict[str, argparse.ArgumentParser],
) -> None:
    prep = sub.add_parser(
        "prep",
        help="downsample a volume's full-res scans to smalls + manifest, and print the "
        "reconciliation (region images -> addressable pages + map-less plates). `run` "
        "does this too; run it alone to LOOK before spending ~1 model call per sheet",
        parents=[parents["work_root"]],
    )
    prep.add_argument("volume", type=volume_argument)
    prep.add_argument(
        "--no-normalize-orientation",
        action="store_true",
        help="write smalls in the SOURCE frame (default: upright — measured median "
        "RMSE 7.03 -> 4.77 m). Ignored on a volume already prepped the other way: "
        "the policy is sticky, because cached annotations live in that frame",
    )
    prep.set_defaults(func=_cmd_prep)
