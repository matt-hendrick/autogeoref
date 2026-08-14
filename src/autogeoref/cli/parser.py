"""The shared flag groups, and the one parser every command hangs off.

This is the only module that knows the whole command set. Each command module
owns its own declaration and handler and imports nothing from here, so the
dependency runs one way.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import add_alias_sweep_parser, add_discover_parser, add_era_parser
from .queue import add_queue_parser
from .report import (
    add_allmaps_parser,
    add_dashboard_parser,
    add_report_parser,
    add_score_parser,
    add_status_parser,
)
from .review import add_review_parser
from .run import add_prep_parser, add_run_parser
from .viewer import add_deploy_bundle_parser, add_publish_parser, add_viewer_manifest_parser


def _shared_parents() -> dict[str, argparse.ArgumentParser]:
    """Flag groups shared across subcommands, each declared exactly once."""
    # Every command that touches the work tree shares one --work declaration.
    work_root = argparse.ArgumentParser(add_help=False)
    work_root.add_argument("--work", type=Path, default=Path("work"))

    # These commands report or serve one filesystem state, so their roots cannot drift.
    state_roots = argparse.ArgumentParser(add_help=False, parents=[work_root])
    state_roots.add_argument(
        "--fixtures",
        type=Path,
        default=Path("fixtures"),
        help="archived funnels, shown as the baseline to beat",
    )
    state_roots.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="volunteer GCP exports (default: <fixtures>/ground-truth); pinned pages are "
        "what makes a volume scoreable",
    )
    state_roots.add_argument(
        "--tiles",
        type=Path,
        default=Path("deploy/tiles"),
        help="serving root; its autogeoref/ subdirectory is the provenance of what is served",
    )

    catalog_root = argparse.ArgumentParser(add_help=False)
    catalog_root.add_argument(
        "--loc-catalog",
        type=Path,
        default=None,
        help="LOC catalog dump for the city — supplies each "
        "candidate's edition YEAR, which is what tells you whether a volume predates "
        "the city's renumbering. Context for your declaration, never a substitute: "
        "the engine does not infer an address era from a year "
        "(default: the city TOML's loc_catalog key)",
    )
    dashboard_catalog_root = argparse.ArgumentParser(add_help=False)
    dashboard_catalog_root.add_argument(
        "--loc-catalog",
        type=Path,
        default=None,
        help="LOC catalog dump for the city — the source "
        "of each volume's year, and so of the era breakdown. Without it every volume "
        "reports its era as unknown (default: the city TOML's loc_catalog key)",
    )

    # Parent order controls usage order; dashboard's required city must lead its roots.
    dashboard_required = argparse.ArgumentParser(add_help=False)
    dashboard_required.add_argument("--city", type=Path, required=True)

    return {
        "work_root": work_root,
        "state_roots": state_roots,
        "catalog_root": catalog_root,
        "dashboard_catalog_root": dashboard_catalog_root,
        "dashboard_required": dashboard_required,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autogeoref")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    parents = _shared_parents()
    # listed in help in this order
    for add_command in (
        add_run_parser,
        add_prep_parser,
        add_report_parser,
        add_score_parser,
        add_allmaps_parser,
        add_queue_parser,
        add_era_parser,
        add_alias_sweep_parser,
        add_status_parser,
        add_dashboard_parser,
        add_discover_parser,
        add_viewer_manifest_parser,
        add_publish_parser,
        add_review_parser,
        add_deploy_bundle_parser,
    ):
        add_command(sub, parents)
    return parser
