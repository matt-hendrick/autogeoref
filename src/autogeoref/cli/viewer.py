"""What gets served: `viewer-manifest`, `publish`, `deploy-bundle`.

Each consumes an archive something else already baked; none runs GDAL or a
pipeline stage.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ..cli_context import display_catalog, fail, publication_config
from ..config.load import load_city_config
from ..paths import VolumePaths
from ..stages.report import stage_report
from ..viewer.layout import TILES_ROOT, VIEWER_ROOT


def _cmd_viewer_manifest(args: argparse.Namespace) -> int:
    from ..config.model import ConfigError
    from ..viewer.config import load_viewer_config
    from ..viewer.coverage import SheetFootprints
    from ..viewer.layout import city_manifest, city_tiles, refresh_cities
    from ..viewer.manifest import AreaSource, build_manifest, no_layers_note, write_manifest
    from ..viewer.stories import stage_story_assets

    city = load_city_config(args.city)
    viewer = load_viewer_config(args.city)
    out = args.out or city_manifest(city.name, args.viewer)
    # Defaulted rather than left to the operator: omitting `--pmtiles` used to
    # collect ZERO volumes, and the story gate then failed naming a stop that
    # is in fact covered — an error that reads as a story bug and is not one.
    pmtiles_dirs = args.pmtiles or (city_tiles(viewer.serving_dirs[0], args.tiles),)
    manifest = build_manifest(
        city.name,
        viewer,
        out_path=out,
        pmtiles_dirs=pmtiles_dirs,
        # `publish` rebuilds this manifest through the same fallback, so
        # building without one here would drop the titles a later publish
        # restores
        loc_catalog=display_catalog(args.loc_catalog, city),
        areas=AreaSource(city.community_areas_path),
        footprints=SheetFootprints(args.exports),
    )
    # An empty manifest is a legal object, but not a page: it publishes an
    # atlas with no layers and — until the page learned to say so — nothing on
    # screen explaining why. Refused here rather than in `build_manifest`,
    # which a bare city legitimately calls before it has baked anything.
    if not manifest["volumes"]:
        return fail(ConfigError(no_layers_note(pmtiles_dirs)))
    write_manifest(manifest, out)
    stage_story_assets(viewer.stories, out.parent)
    refresh_cities(args.viewer)
    print(f"wrote {out} ({len(manifest['volumes'])} volumes)")
    return 0


def _cmd_publish(args: argparse.Namespace) -> int:
    """Publish an already-baked archive without running any pipeline stage."""
    from ..queue.publish import settle_published
    from ..viewer.publish import PublicationError, publish_owed, publish_volume

    city = load_city_config(args.city)
    # Asked BEFORE the publish, because the publish clears it. A marker here means
    # this command IS the documented recovery for a drain that baked and died, and
    # the stranded queue row it left behind is this command's to close.
    was_owed = publish_owed(args.volume, args.work) is not None
    config = publication_config(args, manifest=args.manifest, loc_item=args.item_json)
    try:
        destination = publish_volume(args.volume, config, source=args.source)
    except PublicationError as exc:
        return fail(exc)
    # the volume report may still carry the serve-staleness note this publish
    # just resolved; refresh it so the artifact stops asking for a serve pass
    try:
        stage_report(
            VolumePaths(root=args.work / args.volume),
            args.volume,
            tiles_root=args.tiles,
            city_toml=args.city,
            overview_pages=city.volume(args.volume).overview_pages,
        )
    except (OSError, ValueError) as exc:
        print(f"warning: published, but the report refresh failed: {exc}", file=sys.stderr)
    if was_owed and settle_published(args.work, args.volume) is not None:
        print(f"closed the stranded serve queue entry for {args.volume}")
    print(f"published {args.volume}: {destination}; rebuilt {config.manifest}")
    return 0


def _cmd_deploy_bundle(args: argparse.Namespace) -> int:
    from ..config.model import city_slug
    from ..viewer.deploy import (
        CARD_FILE,
        SITE_URL_ENV,
        TOKEN_ENV,
        DeployError,
        build_deploy_bundle,
    )

    slug = city_slug(load_city_config(args.city).name)
    out = args.out or Path("deploy") / slug
    token = args.mapbox_token or os.environ.get(TOKEN_ENV) or None
    site_url = args.site_url or os.environ.get(SITE_URL_ENV) or None
    try:
        bundle = build_deploy_bundle(
            args.viewer,
            out,
            args.tiles_base_url,
            city=slug,
            mapbox_token=token,
            site_url=site_url,
        )
    except DeployError as exc:
        return fail(exc)
    print(f"wrote {out} ({len(bundle['volumes'])} volumes)")
    if site_url is None:
        print(
            f"note: no site URL (--site-url / {SITE_URL_ENV}), so the bundle ships "
            "no canonical link, no og:url, no sitemap.xml and no share-card image",
            file=sys.stderr,
        )
    # said either way. A card is only ever looked at by somebody else's
    # crawler, so a misplaced or misnamed one is otherwise a silent no-op: the
    # deploy succeeds and the link unfurls without a picture.
    if not (args.viewer / slug / CARD_FILE).is_file():
        print(
            f"note: no share card at {args.viewer / slug / CARD_FILE}, so the pages "
            "declare a small card and no image",
            file=sys.stderr,
        )
    if token is None:
        print(
            f"note: no Mapbox token (--mapbox-token / {TOKEN_ENV}), so the deployed "
            "page will report that address search is unavailable",
            file=sys.stderr,
        )
    print(
        "upload: every file in the bundle directory (the page files, vendor/, "
        "manifest.json, the crawler files and any share card) to static hosting; "
        ".pmtiles files to the tiles base URL"
    )
    return 0


def add_viewer_manifest_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    parents: dict[str, argparse.ArgumentParser],
) -> None:
    vman = sub.add_parser(
        "viewer-manifest",
        help="generate the static viewer's manifest.json (volumes, era chips, "
        "and the [viewer] site block from the city TOML)",
        parents=[parents["catalog_root"]],
    )
    vman.add_argument("--city", type=Path, required=True)
    vman.add_argument(
        "--pmtiles",
        type=Path,
        action="append",
        default=None,
        help="directory of .pmtiles archives, each named by its volume "
        "identifier. Repeatable, earlier "
        "directories win (default: <tiles>/<the city's first serving_dirs entry>). "
        "Its NAME must appear in the city's viewer.serving_dirs "
        "or the build is refused: every layer is published under the city's one "
        "credit line, so an undeclared directory would publish someone else's "
        "georeferencing as this project's",
    )
    vman.add_argument("--tiles", type=Path, default=TILES_ROOT)
    vman.add_argument("--viewer", type=Path, default=VIEWER_ROOT)
    vman.add_argument(
        "--out",
        type=Path,
        default=None,
        help="manifest path (default: <viewer>/<city-slug>/manifest.json). One "
        "page per city: every input to this file is one city's config, so a "
        "shared path would let a second city overwrite the first city's page",
    )
    vman.add_argument(
        "--exports",
        type=Path,
        default=Path("exports"),
        help="researcher exports root: the placed-sheet extents the story "
        "coverage gate checks a stop's camera against. A volume with no export "
        "tree is judged on its published envelope instead, which cannot see a "
        "hole inside it",
    )
    vman.set_defaults(func=_cmd_viewer_manifest)


def add_publish_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    parents: dict[str, argparse.ArgumentParser],
) -> None:
    publish = sub.add_parser(
        "publish",
        help="publish an existing work-tree PMTiles archive, rewrite the volume's "
        "researcher exports (exports/<volume>/), and rebuild the viewer manifest; "
        "does not run GDAL or pipeline stages",
        parents=[parents["work_root"], parents["catalog_root"]],
    )
    publish.add_argument("volume")
    publish.add_argument("--city", type=Path, required=True)
    publish.add_argument("--tiles", type=Path, default=TILES_ROOT)
    publish.add_argument("--viewer", type=Path, default=VIEWER_ROOT)
    publish.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="this city's page manifest (default: <viewer>/<city-slug>/manifest.json)",
    )
    publish.add_argument(
        "--source",
        type=Path,
        default=None,
        help="archive to publish (default: <work>/<volume>/<volume>.pmtiles)",
    )
    publish.add_argument("--exports", type=Path, default=Path("exports"))
    publish.add_argument(
        "--item-json", type=Path, help="local LOC item JSON instead of the cached client"
    )
    publish.add_argument("--cache", type=Path, default=Path("cache/loc"))
    publish.set_defaults(func=_cmd_publish)


def add_deploy_bundle_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    _parents: dict[str, argparse.ArgumentParser],
) -> None:
    from ..viewer.deploy import SITE_URL_ENV

    bundle = sub.add_parser(
        "deploy-bundle",
        help="build the publishable static copy of ONE city's viewer (strips "
        "local paths, rewrites PMTiles references to the public tiles base URL)",
    )
    bundle.add_argument(
        "tiles_base_url",
        help="public base URL where the .pmtiles files will live (no trailing slash)",
    )
    bundle.add_argument("--city", type=Path, required=True)
    bundle.add_argument("--viewer", type=Path, default=VIEWER_ROOT)
    bundle.add_argument(
        "--out",
        type=Path,
        default=None,
        help="bundle directory (default: deploy/<city-slug>). One bundle per "
        "city: the manifest sits beside the page files, so a shared directory "
        "would let one city's upload replace another's",
    )
    bundle.add_argument(
        "--mapbox-token",
        default=None,
        help="Mapbox PUBLIC token (pk.…) restricted to the deployment's URL, "
        "written into the bundle's config.js to turn address search on. "
        "Prefer AUTOGEOREF_MAPBOX_TOKEN, which keeps it out of shell history. "
        "Without one the deployed page reports search unavailable",
    )
    bundle.add_argument(
        "--site-url",
        default=None,
        help="public base URL the PAGE is served at (e.g. https://example.com). "
        "Fills og:url, the canonical link and the sitemap. Prefer "
        f"{SITE_URL_ENV}. Without one the share card still carries a title and "
        "a description, and the tags that need an absolute URL are left out "
        "rather than guessed",
    )
    bundle.set_defaults(func=_cmd_deploy_bundle)
