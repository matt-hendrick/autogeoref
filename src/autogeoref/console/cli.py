"""The ``autogeoref queue`` command: mutations, drains, and every console surface."""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
import time
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..cli_context import (
    build_review_app,
    console_context,
    fail,
    ground_truth_root,
    publication_config,
)
from ..config.load import load_city_config
from ..config.model import ConfigError
from ..queue import run as queue_run
from ..queue import store as queue_store
from ..queue.command import DrainContext
from ..queue.render import render_text
from ..queue.store import TRACKS, QueueError
from .actions import ConsoleActions
from .backlog import candidates
from .payload import BoardCache, board, board_fingerprint
from .server import serve
from .text import VIEWER_URL, render_candidates

if TYPE_CHECKING:
    from ..viewer.publish import PublicationConfig

logger = logging.getLogger(__name__)


def _viewer_manifest(args: argparse.Namespace) -> Path | None:
    """The page manifest this console reads: the flag, else the city's own page.

    None without a city — the read-only board has no way to know whose page it
    would be, and every consumer here treats a missing manifest as context it
    simply does not have.
    """
    if args.viewer_manifest is not None:
        manifest: Path = args.viewer_manifest
        return manifest
    if args.city is None:
        return None
    from ..viewer.layout import city_manifest

    return city_manifest(load_city_config(args.city).name)


def _candidates_command(args: argparse.Namespace) -> str:
    """The `queue --candidates` invocation matching how the board was started.

    The board's blocked pool needs the per-volume TOML stanzas, which live in the
    text view; this is the command that prints them, with the roots this server is
    actually reading — not a guess, and not a hardcoded city.
    """
    cmd = ["autogeoref", "queue", "--candidates"]
    for flag, value in (
        ("--work", args.work),
        ("--fixtures", args.fixtures),
        ("--ground-truth", args.ground_truth),
        ("--tiles", args.tiles),
        ("--city", args.city),
        ("--loc-catalog", args.loc_catalog),
        ("--viewer-manifest", _viewer_manifest(args)),
    ):
        if value is not None:
            cmd += [flag, str(value)]
    return shlex.join(cmd)


def _era_command(args: argparse.Namespace) -> str | None:
    """The `autogeoref era` invocation for this server's city, minus the volumes.

    The board's blocked card appends the blocked volume ids to this prefix — those
    are already in the page's payload; the city TOML path and catalog path are not,
    and must come from the server for the same reason as `_candidates_command`.
    None without a city, but so is the blocked pool: the era check needs a config.
    """
    if args.city is None:
        return None
    cmd = ["autogeoref", "era", "--city", str(args.city)]
    if args.loc_catalog is not None:
        cmd += ["--loc-catalog", str(args.loc_catalog)]
    return shlex.join(cmd)


def _do_mutations(args: argparse.Namespace, track: str) -> None:
    """``--add`` / ``--remove``: the queue writes, before any run or view."""
    if args.add:
        for volume in args.add:
            entry = queue_store.add(args.work, volume, track, then_serve=not args.review)
            # Name where it actually GOES next. A fetch entry promotes to place,
            # and only from there to serve — printing "will promote to serve" on
            # a fetch enqueue would describe two hops as one.
            if entry.track == "serve":
                note = ""
            elif not entry.then_serve:
                note = " (stops at needs-review)"
            elif entry.track == "fetch":
                note = " (will promote to place, then serve)"
            else:
                note = " (will promote to serve)"
            print(f"queued {entry.volume} on the {entry.track} queue{note}")
    if args.remove:
        for volume in args.remove:
            # no --track means both: a volume you are pulling out of the
            # queue is one you want gone, not one you want half-gone
            n = queue_store.remove(args.work, volume, args.track)
            print(f"removed {n} entr{'y' if n == 1 else 'ies'} for {volume}")


def _do_run(args: argparse.Namespace) -> None:
    """``--run``: drain one queue (``--track``) or every queue in parallel."""
    if args.city is None:
        raise QueueError("--run needs --city")
    publication = publication_config(args, manifest=args.viewer_manifest)
    # A bare `--run` drains EVERY queue IN PARALLEL; `--run --track ...`
    # drains one. Both take per-track locks, so this composes with a
    # separately-started drain — the second just loses the race.
    if args.track:
        # fetch takes None, not `--place-lanes`: run_queue forces it to one
        # lane (LOC conduct) and would log an override nobody asked for.
        lanes = {"fetch": None, "place": args.place_lanes, "serve": args.serve_lanes}[args.track]
        done = queue_run.run_queue(
            DrainContext(
                work=args.work,
                city=args.city,
                extra=args.run_arg or (),
                nice=args.nice,
                publication=publication,
                stop_on_failure=args.stop_on_failure,
            ),
            track=args.track,
            lanes=lanes,
        )
        empty_msg = f"{args.track} queue: nothing to do"
    else:
        done = queue_run.run_all(
            args.work,
            args.city,
            place_lanes=args.place_lanes,
            serve_lanes=args.serve_lanes,
            extra=args.run_arg or (),
            nice=args.nice,
            publication=publication,
            stop_on_failure=args.stop_on_failure,
        )
        empty_msg = "/".join(TRACKS) + " queues: nothing to do"
    if not done:
        print(empty_msg)


def _cmd_candidates(args: argparse.Namespace) -> int:
    """``--candidates``: the text backlog, derived fresh from the tree."""
    rows, city, catalog = console_context(args)
    print(
        render_candidates(candidates(rows, work=args.work, city=city, catalog=catalog), city),
        end="",
    )
    return 0


def _board_payload(args: argparse.Namespace, actions: ConsoleActions) -> dict[str, Any]:
    # Rebuilt from the TREE, nothing captured at startup: an operator leaves
    # this page open while a drain runs, and a snapshot would freeze the
    # backlog. The city TOML is re-read too, so declaring an address era
    # unblocks a volume on the next refresh rather than after a restart. The
    # BoardCache skips the rebuild only while the tree's fingerprint is
    # unchanged.
    rows, city, catalog = console_context(args)
    return board(
        work=args.work,
        rows=rows,
        city=city,
        catalog=catalog,
        viewer_url=args.viewer_url or VIEWER_URL,
        # the page prints this back at you rather than composing it: the
        # HTML must not know a city's name or where its TOML lives
        candidates_command=_candidates_command(args),
        era_command=_era_command(args),
        can_act=actions.can_act,
        viewer_manifest=_viewer_manifest(args),
    )


def _board_freshness(args: argparse.Namespace, catalog_file: Path | None) -> object:
    return board_fingerprint(
        work=args.work,
        fixtures=args.fixtures,
        tiles=args.tiles,
        ground_truth=ground_truth_root(args),
        config_files=[
            p for p in (args.city, catalog_file, _viewer_manifest(args)) if p is not None
        ],
    )


def _cmd_serve(args: argparse.Namespace) -> int:
    """``--serve``: the acting console — or the read-only board without a city."""
    # Every act needs a city: a drain shells out to `autogeoref run --city`, and
    # the review pane needs the city's centerlines and volume constants to draw
    # a ghost overlay at all. Without one it still serves the board — read-only,
    # and saying so on the page rather than offering buttons that would 409.
    console_publication: PublicationConfig | None = None
    if args.city is not None:
        console_publication = publication_config(args, manifest=args.viewer_manifest)
    actions = ConsoleActions(
        work=args.work,
        city=args.city,
        publication=console_publication,
        nice=args.nice,
        place_lanes=args.place_lanes,
        serve_lanes=args.serve_lanes,
    )

    # The fallback catalog is a payload input, so the fingerprint must watch
    # it too — or an edited year never invalidates the cache. Resolved once
    # to keep freshness stat-only; a TOML edit re-pointing loc_catalog still
    # invalidates through the TOML itself, which is always watched.
    catalog_file = args.loc_catalog
    if catalog_file is None and args.city is not None:
        with suppress(ConfigError, OSError):
            catalog_file = load_city_config(args.city).loc_catalog_path

    review_app = None
    if args.city is not None:
        # the shared shape; the console mounts the whole tree, no volume filter
        review_app = build_review_app(args, load_city_config(args.city))
    serve(
        args.work,
        port=args.port,
        build_board=BoardCache(
            partial(_board_payload, args, actions), partial(_board_freshness, args, catalog_file)
        ),
        actions=actions,
        review_app=review_app,
    )
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    """``--watch``: clear + reprint until ctrl-c."""
    # No curses and no new dependency: clear + reprint. Every number is
    # re-read off the work tree each pass, so this view is correct even if
    # the runner is a separate process, or dead.
    try:
        while True:
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write(render_text(args.work))
            sys.stdout.write(f"\n  refreshed {time.strftime('%H:%M:%S')} — ctrl-c to exit\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def main(args: argparse.Namespace) -> int:
    """The `autogeoref queue` command — the console's every surface, in one entry.

    `cli.queue` owns the parser and dispatches here; everything the queue command
    renders or serves (candidates, the text board, the acting console) is this
    package's concern.
    """
    # DEFAULT enqueue is the PLACE queue, end to end: a place entry with then_serve,
    # which promotes itself to the serve queue when placed (queue.store.add's docstring).
    # `--review` turns the promotion off and parks it at needs-review instead.
    track = args.track or "place"
    try:
        _do_mutations(args, track)
        if args.run:
            _do_run(args)
    except QueueError as exc:
        return fail(exc)

    # Without a city config there is NO address-era check, and on a city that
    # renumbered that is most of the backlog: every blocked volume renders as ready
    # and `autogeoref run` refuses it on its first line. Both console surfaces say
    # so — this one on stderr, the board with a banner (`board`'s `era_check`).
    # Silence here would make the console confidently wrong about the one question
    # it exists to answer.
    if (args.candidates or args.serve) and args.city is None:
        print(
            "warning: no --city, so the address-era check is OFF — a volume listed as "
            "runnable may still be REFUSED by `autogeoref run` for an undeclared era.",
            file=sys.stderr,
        )

    if args.candidates:
        return _cmd_candidates(args)
    if args.serve:
        return _cmd_serve(args)
    if args.watch:
        return _cmd_watch(args)
    print(render_text(args.work), end="")
    return 0
