"""Read-only views of a finished tree: `report`, `score`, `allmaps`, `status`, `dashboard`.

None of these runs a pipeline stage or spends model budget. `score` grades a
finished placement and reports; nothing in the product reads what it writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ..cli_context import display_catalog, fail, ground_truth_root, status_rows
from ..config.load import load_city_config
from ..paths import VolumeBusyError, VolumePaths, volume_lock
from ..scoring import median_rmse
from ..stages.report import stage_report
from ..validation import volume_argument


def _cmd_report(args: argparse.Namespace) -> int:
    paths = VolumePaths(root=args.work / args.volume)
    # --city only feeds the declared-overview report row; the report is
    # otherwise config-free, so the flag stays optional
    overview_pages: tuple[str, ...] = ()
    if args.city is not None:
        overview_pages = load_city_config(args.city).volume(args.volume).overview_pages
    else:
        # never silently rewrite a declared volume's report with the overview
        # row zeroed — the run/queue paths resolve the declaration from
        # config, and one artifact must not carry two answers
        report_path = paths.root / "report.json"
        if report_path.is_file():
            try:
                existing = json.loads(report_path.read_text())
            except (OSError, ValueError):
                existing = {}
            if existing.get("overview_committed"):
                print(
                    f"error: {args.volume}'s report records declared overview sheets; "
                    "pass --city so the rebuilt report keeps that row",
                    file=sys.stderr,
                )
                return 1
    stage_report(
        paths,
        args.volume,
        tiles_root=args.tiles,
        city_toml=args.city,
        overview_pages=overview_pages,
    )
    print((paths.root / "report.md").read_text())
    return 0


def _score_summary(volume: str, payload: dict[str, Any], directories: list[Path]) -> None:
    """What the pass found, for a human — sources, spread, and named outliers."""
    sources = payload["sources"]
    if not sources:
        print(
            f"{volume}: no ground-truth export in "
            f"{', '.join(str(d) for d in directories)} — nothing has checked this volume, "
            "and no sidecar was written"
        )
        return
    for source in sources:
        marker = " (0-byte marker)" if source["empty_marker"] else ""
        print(f"{source['path']}: {source['pinned_pages']} pinned page(s){marker}")
    pages = payload["pages"]
    if not pages:
        print(f"{volume}: checked, and no accepted placement has a pinned counterpart")
        return
    scores = {page: entry["rmse_vs_human_m"] for page, entry in pages.items()}
    print(
        f"{volume}: {len(scores)} scored accept(s), median {median_rmse(scores):.2f} m, "
        f"max {max(scores.values()):.2f} m"
    )
    # named, never counted: a bare number is not something anyone can go and
    # look at, and looking is the only thing this output is for
    over = sorted(
        ((p, v) for p, v in scores.items() if v > payload["gate_m"]), key=lambda kv: -kv[1]
    )
    if over:
        print(
            f"beyond the {payload['gate_m']:g} m commit gate (they still serve): "
            + ", ".join(f"p{p} {v:.2f} m" for p, v in over)
        )
    seam = payload.get("seam")
    if seam:
        print(
            f"seam solve vs human pins: median {seam['gt_median_before_m']:.3f} m -> "
            f"{seam['gt_median_after_m']:.3f} m over {seam['n_sheets']} sheet(s) "
            f"({seam['verdict']})"
        )


def _cmd_score(args: argparse.Namespace) -> int:
    """Grade a finished volume's placements against human ground truth."""
    from ..score_pass import score_volume

    paths = VolumePaths(root=args.work / args.volume)
    for path, what in ((paths.results, "results"), (paths.manifest, "sheet manifest")):
        if not path.exists():
            print(
                f"error: {args.volume} has no {what} ({path} is absent) — place the volume "
                "first; scoring reads a finished tree and never runs a stage",
                file=sys.stderr,
            )
            return 1
    directories = args.ground_truth or [args.fixtures / "ground-truth"]
    try:
        # the same ownership every mutating operation on a volume tree takes:
        # grading half-written records during a place leg measures nothing
        with volume_lock(paths, operation="score"):
            payload = score_volume(paths, args.volume, directories)
    except VolumeBusyError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    _score_summary(args.volume, payload, directories)
    if payload["sources"]:
        print(f"wrote {paths.scores}")
    return 0


def _cmd_allmaps(args: argparse.Namespace) -> int:
    """Read-only export: committed placements -> IIIF Georeference Annotation."""
    from ..allmaps import AnnotationError, export_volume
    from ..loc import LOCClient, sheet_iiif_services

    if args.item_json is not None:
        item = json.loads(args.item_json.read_text())
    else:
        item = LOCClient(cache_dir=args.cache).item(args.volume)
    paths = VolumePaths(root=args.work / args.volume)
    try:
        page = export_volume(paths, page_services=sheet_iiif_services(item))
    except AnnotationError as exc:
        return fail(exc)
    text = json.dumps(page, indent=1)
    if args.out is not None:
        args.out.write_text(text)
        print(f"wrote {args.out} ({len(page['items'])} sheets)")
    else:
        print(text)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    from ..status import SERVE_STALE
    from ..status_render import format_table, status_json

    ground_truth = ground_truth_root(args)
    rows = status_rows(args)
    if args.stale:
        stale = [r for r in rows if r.serve_stale == SERVE_STALE]
        if args.json:
            print(status_json(stale), end="")
        else:
            for r in stale:
                print(r.volume)
        return 0
    roots = {
        "work": args.work,
        "fixtures": args.fixtures,
        "tiles": args.tiles,
        "ground-truth": ground_truth,
    }
    print(status_json(rows) if args.json else format_table(rows, roots=roots), end="")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    from ..dashboard import build_coverage, coverage_json, render_html
    from ..viewer.config import load_viewer_config

    ground_truth = ground_truth_root(args)
    city = load_city_config(args.city)
    coverage = build_coverage(
        status_rows(args),
        city,
        load_viewer_config(args.city),
        ground_truth_dir=ground_truth,
        loc_catalog=display_catalog(args.loc_catalog, city),
        viewer_manifest=args.viewer_manifest,
    )
    if args.json:
        print(coverage_json(coverage))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # stamped, because a coverage page with no date is the exact failure mode this
    # project keeps hitting: a stale state claim that reads as a current one
    today = datetime.now().astimezone().date().isoformat()
    args.out.write_text(render_html(coverage, generated=today), encoding="utf-8")
    totals = coverage.totals
    print(
        f"wrote {args.out} ({totals.volumes} volumes; "
        f"{sum(1 for r in coverage.rows if r.processed_here)} processed by this pipeline)"
    )
    return 0


def add_report_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    parents: dict[str, argparse.ArgumentParser],
) -> None:
    rep = sub.add_parser(
        "report", help="rebuild the per-volume report", parents=[parents["work_root"]]
    )
    rep.add_argument("volume", type=volume_argument)
    rep.add_argument(
        "--city",
        type=Path,
        default=None,
        help="city config; supplies the volume's declared overview pages for their report row",
    )
    rep.add_argument(
        "--tiles",
        type=Path,
        default=Path("deploy/tiles"),
        help="serving root, used only to note when committed records outdate the served bake",
    )
    rep.set_defaults(func=_cmd_report)


def add_score_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    parents: dict[str, argparse.ArgumentParser],
) -> None:
    score = sub.add_parser(
        "score",
        help="grade a PLACED volume against volunteer ground truth, into "
        "work/<volume>/results-scores.json. Reads a finished tree and runs no "
        "stage: a score can take a placement out of served evidence later, and "
        "can never put one in",
        parents=[parents["work_root"]],
    )
    score.add_argument("volume", type=volume_argument)
    score.add_argument("--fixtures", type=Path, default=Path("fixtures"))
    score.add_argument(
        "--ground-truth",
        type=Path,
        action="append",
        default=None,
        help="volunteer GCP exports (default: <fixtures>/ground-truth). Repeatable: a "
        "volume whose pins live in a second corpus is graded against both in one pass, "
        "and where two carry the same page the FIRST wins",
    )
    score.add_argument("--json", action="store_true")
    score.set_defaults(func=_cmd_score)


def add_allmaps_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    parents: dict[str, argparse.ArgumentParser],
) -> None:
    allm = sub.add_parser(
        "allmaps",
        help="export committed placements as an IIIF Georeference Annotation (Allmaps)",
        parents=[parents["work_root"]],
    )
    allm.add_argument("volume", type=volume_argument)
    allm.add_argument("--out", type=Path, help="write here instead of stdout")
    allm.add_argument(
        "--item-json", type=Path, help="local LOC item JSON instead of the cached client"
    )
    allm.add_argument("--cache", type=Path, default=Path("cache/loc"))
    allm.set_defaults(func=_cmd_allmaps)


def add_status_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    parents: dict[str, argparse.ArgumentParser],
) -> None:
    st = sub.add_parser(
        "status",
        help="per-volume state read off the filesystem: sheets on disk, human ground truth, "
        "this repo's funnel, the archived baseline, and what is being served (read-only)",
        parents=[parents["state_roots"]],
    )
    st.add_argument(
        "--city",
        type=Path,
        default=None,
        help="city TOML: names the deploy/tiles/ directories this city serves from "
        "(viewer.serving_dirs). Without it only the default one is scanned, and a "
        "city publishing under its own name reads as never baked",
    )
    st.add_argument("--json", action="store_true")
    st.add_argument(
        "--stale",
        action="store_true",
        help="only volumes whose served autogeoref archive is older than a committed "
        "record — plain volume names, one per line, for scripting the serve queue. "
        "Never-baked volumes are NOT listed; they read `no bake` in the table",
    )
    st.set_defaults(func=_cmd_status)


def add_dashboard_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    parents: dict[str, argparse.ArgumentParser],
) -> None:
    dash = sub.add_parser(
        "dashboard",
        help="static coverage & provenance page: what this pipeline has placed and "
        "published, by era and by neighborhood, and what is left to run. "
        "Read-only; NOT an accuracy report (see `autogeoref report`)",
        parents=[
            parents["dashboard_required"],
            parents["state_roots"],
            parents["dashboard_catalog_root"],
        ],
    )
    dash.add_argument(
        "--viewer-manifest",
        type=Path,
        default=None,
        help="viewer manifest: supplies the published footprint of served volumes, "
        "used to locate a volume nobody has pinned",
    )
    dash.add_argument("--out", type=Path, default=Path("deploy/dashboard.html"))
    dash.add_argument("--json", action="store_true", help="emit the data, render nothing")
    dash.set_defaults(func=_cmd_dashboard)
