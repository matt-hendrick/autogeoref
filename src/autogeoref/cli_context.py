"""Shared plumbing for the two operator surfaces: the ``cli`` package and ``console/``.

The CLI commands and the console grew the same expressions independently — the
ground-truth fallback, the four status roots, the publication roots, catalog
resolution, the error print. Each is stated once here; the callers keep their
genuine behavioural differences (which this module's docstrings name rather
than smooth over).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Heavy modules (status pulls mask.qa/PIL; viewer.publish pulls the
# publication stack) stay function-local below so `import autogeoref.cli`
# keeps its per-command lazy-import discipline.
from .config.load import load_city_config

if TYPE_CHECKING:
    from .config.model import CityConfig
    from .paths import VolumePaths
    from .review.app import ReviewApp
    from .status import VolumeStatus
    from .viewer.publish import PublicationConfig

logger = logging.getLogger(__name__)


def fail(exc: object) -> int:
    """The operator-surface error contract: ``error: <exc>`` on stderr, exit code 1."""
    print(f"error: {exc}", file=sys.stderr)
    return 1


def ground_truth_root(args: argparse.Namespace) -> Path:
    """The ground-truth dir: the flag, else ``<fixtures>/ground-truth``.

    The same default ``status.build_status`` states in params form; the two
    spellings must stay identical.
    """
    root: Path = args.ground_truth or args.fixtures / "ground-truth"
    return root


def status_rows(args: argparse.Namespace) -> list[VolumeStatus]:
    """``build_status`` over the four roots every state-derived command shares.

    With a ``--city`` it also reads that city's ``viewer.serving_dirs``, so a
    city publishing into its own ``deploy/tiles/`` directory is reported as
    served rather than as never baked.
    """
    from .status import build_status
    from .viewer.config import SERVING_DIRS, load_viewer_config

    city_toml: Path | None = getattr(args, "city", None)
    serving_dirs = load_viewer_config(city_toml).serving_dirs if city_toml else SERVING_DIRS
    return build_status(
        work=args.work,
        fixtures=args.fixtures,
        tiles=args.tiles,
        ground_truth=ground_truth_root(args),
        serving_dirs=serving_dirs,
    )


def packaged_ui(name: str) -> Path:
    """A UI directory shipped inside the package — works from a wheel, not just a checkout."""
    from importlib.resources import files as resource_files

    return Path(str(resource_files("autogeoref") / name))


def display_catalog(explicit: Path | None, city: CityConfig) -> Path | None:
    """Resolve the LOC catalog PATH for a display surface (titles, era chips).

    The rule, stated once for both operator surfaces (func:`console_context` follows it too,
    returning the loaded dict instead of the path): the explicit flag fails loudly downstream —
    the operator named that file, and silence would hide their typo. The config fallback
    (``loc_catalog`` in the city TOML) only warns: catalog data is display context, and a stale
    config path must not take a read-only command or the console down with it. ``cli._cmd_era``
    is the deliberate exception — era proposes nothing without a catalog, so a missing one is a
    hard error there, not a warning.
    """
    if explicit is not None:
        return explicit
    path = city.loc_catalog_path
    if path is None:
        return None
    from .viewer.sources import loc_titles

    try:
        loc_titles(path, city.name)
    except Exception as exc:  # noqa: BLE001 - an unusable catalog is omitted, not fatal
        logger.warning("city loc_catalog %s unusable, omitted: %s", path, exc)
        return None
    return path


def publication_config(
    args: argparse.Namespace, *, manifest: Path | None, loc_item: Path | None = None
) -> PublicationConfig:
    """Publication roots with the city's display catalog as the default.

    The two callers differ by exactly the parameters: the queue command's
    manifest flag is ``--viewer-manifest`` where ``publish``'s is ``--manifest``,
    and only ``publish`` takes a pre-fetched LOC item (``--item-json``). Either
    flag left unset resolves to this city's own page — ``viewer/<city-slug>/`` —
    and the archives land in the first directory the city's ``serving_dirs``
    declares, which is the one this pipeline writes.
    """
    from .viewer.config import load_viewer_config
    from .viewer.layout import VIEWER_ROOT, city_manifest
    from .viewer.publish import PublicationConfig

    city = load_city_config(args.city)
    viewer = load_viewer_config(args.city)
    # only the two viewer commands declare a viewer root; the queue and the
    # console publish into the conventional one
    viewer_root: Path = getattr(args, "viewer", None) or VIEWER_ROOT
    return PublicationConfig(
        work=args.work,
        city_toml=args.city,
        tiles_root=args.tiles,
        manifest=manifest or city_manifest(city.name, viewer_root),
        viewer_root=viewer_root,
        serve_dir=viewer.serving_dirs[0],
        loc_catalog=args.loc_catalog or city.loc_catalog_path,
        exports_root=args.exports,
        loc_item=loc_item,
        loc_cache=args.cache,
    )


def console_context(args: argparse.Namespace) -> tuple[list[Any], Any, dict[str, Any]]:
    """``(status rows, city config | None, LOC catalog meta)`` for the console views.

    The city config is what makes the backlog honest — without it the console cannot know that a
    run would REFUSE this volume for want of an address era (`config.era_undeclared`), and it
    would happily hand you an `--add` line for a volume that dies on its first line. So `--city`
    is not decoration here; when it is absent the console says what it cannot check rather than
    implying it did. The catalog defaults from the city TOML's ``loc_catalog`` when the flag is
    not passed, under :func:`display_catalog`'s rule — flag fails loudly, config fallback warns
    — except that this returns the loaded years dict, not the path.
    """
    rows = status_rows(args)
    city = load_city_config(args.city) if args.city else None
    catalog: dict[str, Any] = {}
    if city is not None:
        from .viewer.sources import loc_titles

        if args.loc_catalog is not None:
            catalog = loc_titles(args.loc_catalog, city.name)
        elif city.loc_catalog_path is not None:
            try:
                catalog = loc_titles(city.loc_catalog_path, city.name)
            except Exception as exc:  # noqa: BLE001 - optional display data, never the console
                logger.warning(
                    "city loc_catalog %s unusable, years omitted: %s",
                    city.loc_catalog_path,
                    exc,
                )
    return rows, city, catalog


def build_review_app(args: argparse.Namespace, city: CityConfig, **selection: Any) -> ReviewApp:
    """The review surface both operator commands mount: one work root, the packaged UI.

    ``review`` narrows with ``volumes=`` / ``include_ok=`` and honors its
    ``--ui`` override; the queue console mounts the whole tree (its parser has
    no ``--ui``, so the ``getattr`` resolves to the packaged UI there, always).
    """
    # deferred: the review model pulls in numpy/pyproj, and neither command
    # should pay for a map renderer before it knows it will draw one
    from .review.app import ReviewApp

    ui = getattr(args, "ui", None)
    return ReviewApp(
        work=args.work,
        city=city,
        ui_dir=ui if ui is not None else packaged_ui("review_ui"),
        vendor_dir=args.vendor,
        **selection,
    )


def apply_reviews_locked(paths: VolumePaths, volume: str, *, do_warp: bool) -> dict[str, Any]:
    """``review.apply_reviews``, with ``VolumeBusyError`` left for the caller's surface.

    apply_reviews takes the volume lock, and a held one means a run or prep owns the tree right
    now. Both operator surfaces call through here and answer the refusal their own way: the CLI
    prints the error and skips THAT volume, the console 409s the request. ``do_warp`` differs
    deliberately too — the CLI warps unless ``--no-warp``; the console passes ``do_warp=False``
    because an HTTP request is no place to spend ten minutes in gdalwarp, and the volume's serve
    leg warps from these very records moments later.
    """
    from .review.apply import apply_reviews

    return apply_reviews(paths, volume, do_warp=do_warp)
