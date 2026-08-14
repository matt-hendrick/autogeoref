"""The child command for one queue leg: `autogeoref run`, or the LOC fetcher."""

from __future__ import annotations

import logging
import shlex
import subprocess
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .store import QueueEntry, QueueError, log_path

if TYPE_CHECKING:
    from ..viewer.publish import PublicationConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DrainContext:
    """The values a drain threads verbatim through every frame.

    ``run_all`` -> ``run_queue`` -> ``_execute`` -> ``_run_leg`` -> ``_command``
    used to re-spell this list at every hop; one frozen record carries it once.
    Per-worker knobs (``track``, ``lanes``, ``follow_while``, ``poll_s``) stay
    parameters — they differ between the three workers, so they are not shared
    context. ``abort`` is the one stop shared across workers (see :func:`.run.run_all`).
    """

    work: Path
    city: Path
    extra: Sequence[str] = ()
    nice: int = 10
    publication: PublicationConfig | None = None
    stop_on_failure: bool = False
    abort: threading.Event = field(default_factory=threading.Event)


#: Flags the queue itself configures on every child run. Overriding an identity
#: flag would leave the queue holding state and locks for one tree while the
#: child acts on another; overriding a mode flag would change what the leg IS,
#: baking a second pass's promotions or reporting placement for a run that did
#: no work. Behaviour flags such as ``--no-escalate`` still pass.
_QUEUE_OWNED_ARGS = ("--city", "--work", "--viewer-manifest", "--warp", "--warp-only", "--dry-run")
#: Same default as `cli`'s `queue --cache`, for a fetch leg run without a
#: publication configuration to name one.
_LOC_CACHE = Path("cache/loc")


def _manifest_for(ctx: DrainContext) -> Path:
    """The page manifest a child run rebuilds: the publication's, else this
    city's own — ``viewer/<city-slug>/manifest.json``. Never a site-wide path,
    or a drain of city B would rewrite city A's page."""
    if ctx.publication is not None:
        return ctx.publication.manifest
    from ..config.load import load_city_config
    from ..viewer.layout import city_manifest

    return city_manifest(load_city_config(ctx.city).name)


def _reject_owned_overrides(extra: Sequence[str]) -> None:
    """Refuse passthrough arguments that override queue-owned identity or mode.

    Matches the bare flag, its ``--flag=value`` form, and any ``--`` prefix of an
    owned flag — the child parser abbreviates (argparse ``allow_abbrev``), so
    ``--viewer=x`` would land on ``--viewer-manifest`` if only exact names were
    checked.
    """
    smuggled = sorted(
        {
            owned
            for arg in extra
            for owned in _QUEUE_OWNED_ARGS
            if (head := arg.split("=", 1)[0]).startswith("--")
            and len(head) > 2
            and owned.startswith(head)
        }
    )
    if smuggled:
        raise QueueError(
            f"{', '.join(smuggled)}: --run-arg cannot override what the queue itself "
            "configures. The queue owns each child's identity (--city, --work, "
            "--viewer-manifest) and mode (--warp, --warp-only, --dry-run): overriding "
            "identity would leave it tracking one work tree while the child acts on "
            "another, and overriding mode would re-run the funnel inside a place leg, "
            "bake an unplaced volume, or report placement for a --dry-run that did "
            "nothing. Per-run behavior flags such as --no-escalate still pass through."
        )


def autogeoref_bin() -> str:
    """Absolute path to THIS interpreter's ``autogeoref``, else the bare name.

    A bare ``"autogeoref"`` is not enough. The documented way to run this repo is
    ``.venv/bin/autogeoref …``, which does NOT put ``.venv/bin`` on PATH — so the
    drain would spawn a command the shell cannot find and fail every volume with
    a FileNotFoundError that says nothing about the venv. The console script sits
    next to the interpreter running us, so resolve it there; the bare name stays
    as the fallback for an installed-on-PATH environment.
    """
    candidate = Path(sys.executable).with_name("autogeoref")
    return str(candidate) if candidate.exists() else "autogeoref"


#: Where the fetch leg's implementation lives, relative to the repo root. THE
#: only implementation of the fetch (its own docstring says so), so the queue
#: shells out to it rather than growing a second acquisition path.
FETCH_SCRIPT = Path("scripts") / "fetch_loc_volume.py"


def fetch_script() -> Path:
    """Absolute path to ``scripts/fetch_loc_volume.py`` in this checkout.

    Resolved from the package root, not the process's cwd (a drain is started
    from wherever the operator happens to be, and the console starts one
    detached) and not this module's ``__file__`` (which moves when a module
    does; the package root does not). Unlike :func:`autogeoref_bin` there is no
    installed fallback — ``scripts/`` is not packaged in the wheel — so a
    missing script is a refusal naming what it looked for, not a
    ``FileNotFoundError`` on every fetch entry.
    """
    from importlib.resources import files as resource_files

    package = Path(str(resource_files("autogeoref"))).resolve()
    candidate = package.parents[1] / FETCH_SCRIPT
    if not candidate.is_file():
        raise QueueError(
            f"the fetch queue needs {FETCH_SCRIPT} and it is not at {candidate}. The fetch "
            "track runs from a source checkout; `scripts/` is not part of the installed "
            "package."
        )
    return candidate


def _command(leg: str, entry: QueueEntry, ctx: DrainContext) -> list[str]:
    """The command for ONE LEG of a queue entry — `autogeoref run`, or the fetcher.

    The queue shells out to the same command a human would type. The FETCH leg
    is the one exception: acquisition lives in :data:`FETCH_SCRIPT` and
    ``--run-arg`` does NOT reach it, though the extras are still validated.

    The serve leg is ``--warp-only``, NOT ``--warp``, even though one entry runs
    both legs back to back: the placement stages declare no output targets, so
    ``--warp`` would re-run the whole funnel before tiling, re-spending budget.
    """
    _reject_owned_overrides(ctx.extra)
    if leg == "fetch":
        cache = ctx.publication.loc_cache if ctx.publication is not None else _LOC_CACHE
        return [
            sys.executable,
            str(fetch_script()),
            entry.volume,
            "--work",
            str(ctx.work),
            "--cache",
            str(cache),
        ]
    # allowed extras go FIRST; the queue-owned arguments come after them so the
    # configured values win argparse's last-flag-wins even if a new override
    # form ever slips past the rejection above
    cmd = [autogeoref_bin(), "run", entry.volume, *ctx.extra]
    cmd += ["--city", str(ctx.city), "--work", str(ctx.work)]
    cmd += ["--viewer-manifest", str(_manifest_for(ctx))]
    if leg == "serve":
        cmd.append("--warp-only")
    return cmd


def _run_leg(leg: str, entry: QueueEntry, ctx: DrainContext) -> int:
    """Run one leg to completion; return its exit code. Log per (volume, leg)."""
    lp = log_path(ctx.work, entry.volume, leg)
    lp.parent.mkdir(parents=True, exist_ok=True)
    cmd = _command(leg, entry, ctx)
    if ctx.nice:
        cmd = ["nice", "-n", str(ctx.nice), *cmd]
    entry.log = str(lp)
    logger.info("%s [%s]: %s", entry.volume, leg, shlex.join(cmd))
    with lp.open("w") as fh:
        fh.write(f"$ {shlex.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    return proc.returncode
