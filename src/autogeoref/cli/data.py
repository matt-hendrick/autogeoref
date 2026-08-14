"""City-data production: `era`, `alias-sweep`, `discover`.

These write configuration and alias files rather than placements. None spends
model budget; `discover` is the only one that reaches the network, through the
cached LOC client.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ..config.load import load_city_config


def _cmd_era(args: argparse.Namespace) -> int:
    """Propose each volume's address era from the LOC year; confirm; write the TOML."""
    from .. import era as era_mod
    from ..viewer.sources import loc_titles

    city = load_city_config(args.city)
    catalog_path = args.loc_catalog or city.loc_catalog_path
    if catalog_path is None:
        print(
            "error: no LOC catalog — pass --loc-catalog or declare loc_catalog in "
            f"{args.city}. The catalog is where each volume's edition YEAR comes from, "
            "and era proposes nothing without one.",
            file=sys.stderr,
        )
        return 2
    catalog = loc_titles(catalog_path, city.name)
    proposals = [era_mod.propose(city, catalog, v) for v in args.volumes]

    refused = [p for p in proposals if p.refusal]
    todo = [p for p in proposals if p.ok and p.declared is None]
    for p in proposals:
        print(era_mod.render(p, city))
        print()

    # whether --yes may cover these proposals is era policy, not CLI plumbing
    if args.yes:
        refusal = era_mod.refuse_untrusted(todo)
        if refusal is not None:
            print(f"error: {refusal}", file=sys.stderr)
            return 1

    if not todo:
        # A run of pure refusals is a FAILURE, not a no-op: the operator asked for a
        # declaration and did not get one, and the volume still will not start.
        return 1 if refused else 0

    if not args.yes:
        print(f"Write {len(todo)} declaration(s) to {args.city}? [y/N] ", end="", flush=True)
        answer = sys.stdin.readline().strip().lower()
        if answer not in ("y", "yes"):
            print("nothing written")
            return 1

    for p in todo:
        era_mod.declare(args.city, p)
        print(f"{p.volume}: wrote addresses_modern = {str(bool(p.modern)).lower()}")
    if any(p.modern is False for p in todo):
        print()
        print(
            "One or more volumes predate the renumbering. CHECK WHICH BOOK each needs "
            "before running it — a Loop volume converts through the 1911 register, not the "
            "city table (see [volumes.sanborn01790_017] for the shape)."
        )
    return 1 if refused else 0


def _cmd_alias_sweep(args: argparse.Namespace) -> int:
    """Scan volumes for alias gaps, auto-write the clean tier, report the rest."""
    from ..alias.sweep import render_report, run_sweep
    from ..fixture_sums import update_sums

    # Cache siblings (p<N>.v2.<model>.json, p<N>.escalated.*) have no manifest
    # entry by design, and this command loads every annotated volume — the
    # loader's per-file warning would bury the report under thousands of lines.
    logging.getLogger("autogeoref.sheet_inputs").setLevel(logging.ERROR)
    city = load_city_config(args.city)
    from ..viewer.layout import city_manifest

    declared = args.viewer_manifest or city_manifest(city.name)
    viewer_manifest = declared if declared.is_file() else None
    result = run_sweep(
        city,
        args.city,
        args.work,
        volumes=args.volumes or None,
        viewer_manifest=viewer_manifest,
        force=args.force,
        dry_run=args.dry_run,
    )
    report = render_report(result, dry_run=args.dry_run)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print(report)
    print(f"report: {args.report}")

    if not args.dry_run and result.written:
        written = [city.aliases_path(o.volume) for o in result.written]
        changed = update_sums(args.sums, Path.cwd(), written)
        if changed:
            print(f"{args.sums}: refreshed {len(changed)} line(s)")
        elif not args.sums.is_file():
            # Not an error: a checkout without the manifest (or a --sums path
            # deliberately pointed elsewhere) still gets its alias files.
            print(f"{args.sums}: absent, not updated")
    # An aborted volume is a FAILURE, not a note: the validator refused a table
    # the tier rules produced, and that is a defect in the proposal, not in the
    # data. Skips and held entries are the normal, healthy outcome.
    return 1 if result.aborted else 0


def _cmd_discover(args: argparse.Namespace) -> int:
    from ..loc import LOCClient
    from ..paths import atomic_write_text
    from ..viewer.sources import loc_titles

    client = LOCClient(cache_dir=args.cache)
    results = client.catalog_results(args.query)
    census = [client.parse_result(item) for item in results]
    digitized = [v for v in census if v.digitized]
    print(f"{len(census)} volumes, {len(digitized)} digitized:")
    for v in census:
        flag = "+" if v.digitized else "-"
        print(f"  {flag} {v.item_id}  {v.title} ({v.date})")
    if args.out is not None:
        atomic_write_text(args.out, json.dumps(results, indent=1))
        # what the file is FOR is titles and years, and an item with neither
        # contributes nothing — so say how many of them there are rather than
        # leaving the operator to find out at publish time
        labelled = loc_titles(args.out, "x")
        print(f"wrote {args.out} ({len(results)} items, {len(labelled)} with a usable year)")
    return 0


def add_era_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    parents: dict[str, argparse.ArgumentParser],
) -> None:
    era_p = sub.add_parser(
        "era",
        help="declare a volume's printed-address era: read its year from the LOC catalog, "
        "propose addresses_modern, and write it to the city TOML after you confirm. "
        "REFUSES where the catalog has no year — it never guesses an era",
        parents=[parents["catalog_root"]],
    )
    era_p.add_argument("volumes", nargs="+")
    era_p.add_argument("--city", type=Path, required=True)
    era_p.add_argument(
        "--yes",
        action="store_true",
        help="skip the confirm. The confirm is the POINT — it is where a human sees "
        "'1896 -> pre-1909 -> false' and catches a wrong year before it becomes a wrong "
        "veto. Use it only when you have already read the proposals",
    )
    era_p.set_defaults(func=_cmd_era)


def add_alias_sweep_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    parents: dict[str, argparse.ArgumentParser],
) -> None:
    sweep = sub.add_parser(
        "alias-sweep",
        help="close historic street-name alias gaps: scan annotated volumes off the "
        "match tripwire, propose entries from the city's pinned rename source, "
        "auto-write only the evidence-clean tier, and report the held tier. "
        "Zero model spend, no network; never re-places and never bakes",
        parents=[parents["work_root"]],
    )
    sweep.add_argument("volumes", nargs="*", help="default: every annotated volume under --work")
    sweep.add_argument("--city", type=Path, required=True)
    sweep.add_argument(
        "--viewer-manifest",
        type=Path,
        default=None,
        help="viewer manifest supplying bounds_from footprints (default: "
        "viewer/<city-slug>/manifest.json), exactly as a run resolves "
        "them; ignored when absent (bounds provenance decides value-in-bounds verdicts, "
        "so a volume whose footprint comes from a counterpart cannot be scanned without it)",
    )
    sweep.add_argument(
        "--report",
        type=Path,
        default=Path("work/scratch/alias-sweep-report.md"),
        help="where the sweep report is written",
    )
    sweep.add_argument(
        "--sums",
        type=Path,
        default=Path("FIXTURE-SHA256SUMS"),
        help="fixture integrity manifest to refresh for the files written "
        "(only those lines; pass an absent path to skip)",
    )
    sweep.add_argument(
        "--dry-run",
        action="store_true",
        help="scan, propose and report, writing NO alias file, marker or manifest line",
    )
    sweep.add_argument(
        "--force",
        action="store_true",
        help="re-sweep volumes that already carry an alias-sweep marker. The city's "
        "alias_sweep_skip list still wins: that is a declaration, not a cache",
    )
    sweep.set_defaults(func=_cmd_alias_sweep)


def add_discover_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    _parents: dict[str, argparse.ArgumentParser],
) -> None:
    disc = sub.add_parser("discover", help="LOC catalog census for a city query")
    disc.add_argument("query")
    disc.add_argument("--cache", type=Path, default=Path("cache/loc"))
    disc.add_argument(
        "--out",
        type=Path,
        default=None,
        help="also write the catalog to this path — the `loc_catalog` file a city "
        "config points at, which is where a published layer's title, year and "
        "volume number come from. Without one a first publish labels the layer "
        "with its bare identifier and gives it no era",
    )
    disc.set_defaults(func=_cmd_discover)
