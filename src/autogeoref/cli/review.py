"""`review`: three disjoint programs behind one flag surface.

``--render`` writes static ghost composites, ``--apply`` materializes saved
sidecars, and the default serves the localhost review UI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..cli_context import apply_reviews_locked, build_review_app, fail
from ..config.load import load_city_config
from ..errors import ReviewError
from ..paths import VolumeBusyError, VolumePaths
from ..validation import port, volume_argument

if TYPE_CHECKING:
    from ..config.model import CityConfig


def _cmd_review(args: argparse.Namespace) -> int:
    """Three disjoint programs share this command's flag surface."""
    city = load_city_config(args.city)
    if args.render is not None:
        return _review_render(args, city)
    if args.apply:
        return _review_apply(args)
    return _review_serve(args, city)


def _review_render(args: argparse.Namespace, city: CityConfig) -> int:
    """--render: batch static ghost composites, no server."""
    if args.apply:
        print("error: --render and --apply are mutually exclusive", file=sys.stderr)
        return 2
    if not args.volume:
        print("error: --render requires --volume", file=sys.stderr)
        return 2
    from ..review.app import review_queue
    from ..review.render import render_ghost_composite

    paths = VolumePaths(root=args.work / args.volume)
    if args.pages:
        pages = [p.strip() for p in args.pages.split(",") if p.strip()]
    else:
        pages = [
            e["page"]
            for e in review_queue(paths, args.volume, include_ok=args.all)
            if e["has_placement"]
        ]
    if not pages:
        print(f"no renderable pages for {args.volume}")
        return 1
    summary: dict[str, Any] = {}
    failed = False
    for page in pages:
        try:
            entry = render_ghost_composite(
                paths, args.volume, page, city.centerlines_path, args.render
            )
        except ReviewError as exc:
            print(f"p{page}: SKIPPED — {exc}", file=sys.stderr)
            failed = True
            continue
        summary[page] = entry
        print(
            f"p{page}: {entry['status']} — {entry['n_gcps']} GCPs, "
            f"{entry['centerlines_drawn']} centerlines"
        )
    out_json = args.render / f"{args.volume}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")
    return 1 if failed else 0


def _review_apply(args: argparse.Namespace) -> int:
    """--apply: materialize saved sidecars, no server."""
    volumes = (
        [args.volume]
        if args.volume
        else sorted(
            p.parent.name
            for p in args.work.glob("*/review")
            if p.is_dir() and any(p.glob("p*.json"))
        )
    )
    if not volumes:
        print(f"no review sidecars under {args.work}")
        return 1
    failed = False
    for volume in volumes:
        try:
            summary = apply_reviews_locked(
                VolumePaths(root=args.work / volume), volume, do_warp=not args.no_warp
            )
        except VolumeBusyError as exc:
            # busy contract on cli_context.apply_reviews_locked — skip THIS
            # volume, keep going
            fail(exc)
            failed = True
            continue
        print(
            f"{volume}: applied {len(summary['applied'])}, "
            f"already applied {len(summary['already_applied'])}, "
            f"skipped {len(summary['skipped'])}, "
            f"warped {len(summary['warped'])}, "
            f"masks written {len(summary['masks_written'])}"
        )
        for w in summary["warnings"]:
            print(f"  WARNING: {w}")
            failed = failed or "changed since review" in w
        if summary["rerun_hint"]:
            print(f"  NEXT: {summary['rerun_hint']}")
    return 1 if failed else 0


def _review_serve(args: argparse.Namespace, city: CityConfig) -> int:
    """Default: the localhost review & adjust UI."""
    from ..review.server import serve

    app = build_review_app(
        args, city, volumes=[args.volume] if args.volume else [], include_ok=args.all
    )
    if not app.volumes:
        print(f"no volumes with results under {args.work}")
        return 1
    serve(app, port=args.port)
    return 0


def add_review_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    parents: dict[str, argparse.ArgumentParser],
) -> None:
    rev = sub.add_parser(
        "review",
        help="reviewer review & adjust UI (localhost-only): ghost overlays for the "
        "flagged pool, verdict/edit sidecars, and --apply to materialize them "
        "as reviewer-verified results (+ mask dry-run + re-warp)",
        parents=[parents["work_root"]],
    )
    rev.add_argument("--city", type=Path, required=True)
    rev.add_argument("--volume", type=volume_argument, default=None, help="limit to one volume")
    rev.add_argument("--port", type=port, default=8765)
    rev.add_argument(
        "--all",
        action="store_true",
        help="queue every sheet, not just the flagged pool (e.g. for mask fixes)",
    )
    rev.add_argument(
        "--apply",
        action="store_true",
        help="materialize saved sidecars: reviewer-verified results, cutline "
        "dry-runs for edited masks, re-warp (no server started)",
    )
    rev.add_argument(
        "--render",
        type=Path,
        default=None,
        metavar="DIR",
        help="batch fallback when no browser is available: render static ghost "
        "composites (centerlines + GCP ties through the recorded placement) "
        "into DIR and exit; requires --volume. The review UI is the QA medium "
        "of record",
    )
    rev.add_argument(
        "--pages",
        default=None,
        help="with --render: comma-separated page ids (default: the queued pool)",
    )
    rev.add_argument(
        "--no-warp",
        action="store_true",
        help="with --apply: write results/masks but skip the re-warp",
    )
    rev.add_argument(
        "--ui",
        type=Path,
        default=None,
        help="review UI directory (default: the packaged UI, autogeoref/review_ui)",
    )
    rev.add_argument(
        "--vendor",
        type=Path,
        default=Path("viewer/vendor"),
        help="MapLibre vendor directory (shared with the viewer)",
    )
    rev.set_defaults(func=_cmd_review)
