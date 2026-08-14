"""`queue`: the work queue, whose board, drain and console all live in `console`."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..validation import port, positive_float, positive_int, volume_argument


def _cmd_queue(args: argparse.Namespace) -> int:
    # everything the queue command renders or serves is a console concern
    from ..console.cli import main as console_main

    return console_main(args)


def add_queue_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    parents: dict[str, argparse.ArgumentParser],
) -> None:
    # the queue owns what a track IS; a second spelling here would drift from it
    from ..queue.store import TRACKS

    que = sub.add_parser(
        "queue",
        help="work queue: one --add takes a volume from LOC to the map (fetch, place, "
        "then serve, in one drain). --track fetch / place / serve address a single leg. "
        "No args = show the board",
        parents=[parents["state_roots"], parents["catalog_root"]],
    )
    que.add_argument("--city", type=Path, default=None, help="required by --run")
    que.add_argument("--exports", type=Path, default=Path("exports"))
    que.add_argument(
        "--cache",
        type=Path,
        default=Path("cache/loc"),
        help="LOC client cache; a fetch drain downloads through it and a serve drain "
        "reads each volume's item document from it for the researcher exports",
    )
    que.add_argument(
        "--track",
        choices=TRACKS,
        default=None,
        help="which QUEUE to add to / run. Default enqueue is `place` end-to-end (it "
        "promotes itself to serve when placed); default --run drains EVERY queue in "
        "parallel. `fetch` downloads a volume's jp2 masters from LOC and promotes it to "
        "place — the one track that needs no work tree yet. `serve` bakes a volume that "
        "is already placed. On --remove, omitting it removes the volume from EVERY queue",
    )
    que.add_argument(
        "--add", action="append", type=volume_argument, metavar="VOLUME", help="enqueue a volume"
    )
    que.add_argument(
        "--remove", action="append", type=volume_argument, metavar="VOLUME", help="drop a volume"
    )
    que.add_argument(
        "--review",
        action="store_true",
        help="place this volume but DON'T promote it to serve — park it at needs-review. "
        "OPT-IN: ghost-overlay review is a diagnostic, not a gate, and a volume may be "
        "served with no human look. Ask for it when a funnel looks strange",
    )
    que.add_argument(
        "--run", action="store_true", help="drain the queue(s): all in parallel, or --track one"
    )
    que.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="abort on the first FAILED entry, leaving everything that had not started "
        "queued. What a rate-limited run needs: without it a model limit fails volume 1, "
        "then volume 2 starts and hits the same wall, and the drain burns a doomed attempt "
        "on every volume in the queue. On a bare --run it stops ALL the queues, so a limit "
        "does not leave a download lane going for hours; work already in flight finishes. "
        "Re-run after the limit resets to pick up what was left queued — the entry that "
        "FAILED is terminal and needs a re-add, or it silently leaves the batch",
    )
    que.add_argument(
        "--place-lanes",
        type=positive_int,
        default=1,
        help="volumes the PLACE queue drains at once (default 1: model-bound, and one at a "
        "time keeps the spend legible — two would write one volume's annotation cache twice)",
    )
    que.add_argument(
        "--serve-lanes",
        type=positive_int,
        default=None,
        help="volumes the SERVE queue bakes at once (default: derived from CPU, capped — see "
        "queue.run.default_serve_lanes; NOT cpu_count, because GDAL is already multithreaded and "
        "baking is memory-heavy). Raise it if you have measured your box",
    )
    que.add_argument(
        "--run-arg",
        action="append",
        metavar="ARG",
        help="extra argument passed through to `autogeoref run` (repeatable), e.g. "
        "--run-arg --no-escalate",
    )
    que.add_argument("--nice", type=int, default=10, help="niceness for spawned runs (0=off)")
    que.add_argument("--watch", action="store_true", help="re-render the board until ctrl-c")
    que.add_argument("--interval", type=positive_float, default=5.0, help="--watch refresh seconds")
    que.add_argument(
        "--candidates",
        action="store_true",
        help="what you could START today, on both tracks, with a sheet count and a "
        "labelled call ESTIMATE — paste the printed command back to enqueue it. Pass "
        "--city to have it check which volumes a run would REFUSE (undeclared address era)",
    )
    que.add_argument(
        "--serve",
        action="store_true",
        help="THE admin console on localhost: runnable / running / needs you / served, and "
        "it RUNS them — enqueue a volume, start and stop a drain, review a flagged sheet "
        "against its ghost overlay, send it to serve. With --city it acts; without one it "
        "is a read-only board (a run needs a city, and so does a ghost overlay)",
    )
    que.add_argument("--port", type=port, default=8766, help="--serve port")
    que.add_argument(
        "--vendor",
        type=Path,
        default=Path("viewer/vendor"),
        help="MapLibre vendor directory for --serve's review pane (shared with the viewer)",
    )
    que.add_argument(
        "--viewer-manifest",
        type=Path,
        default=None,
        help="viewer manifest rebuilt after each successful serve publication "
        "(default: viewer/<city-slug>/manifest.json)",
    )
    que.add_argument(
        "--viewer-url",
        default=None,  # console.text.VIEWER_URL, resolved lazily — see console.cli.main
        help="where the SERVED column links to (scripts/serve_viewer.py's port)",
    )
    que.set_defaults(func=_cmd_queue)
