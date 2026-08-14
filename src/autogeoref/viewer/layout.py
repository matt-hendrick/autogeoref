"""Where one city's published viewer files live.

Every input to a manifest is one city's config — the site block, the era
buckets, the geocoder, the basemap, the credit line — so the manifest is named
for that city and not for the site. Publishing a second city then cannot
overwrite the first city's page.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..config.model import ConfigError, city_slug
from ..paths import atomic_write_text

logger = logging.getLogger(__name__)

#: The local viewer directory: the hand-written page files, plus one
#: subdirectory per published city.
VIEWER_ROOT = Path("viewer")

#: Archive root the publish and manifest commands share.
TILES_ROOT = Path("deploy/tiles")

MANIFEST_NAME = "manifest.json"

#: The page's index of the cities published beside it. The page reads it to
#: decide which manifest to load when the URL names no city.
CITIES_NAME = "cities.json"


def city_dir(city_name: str, viewer_root: Path = VIEWER_ROOT) -> Path:
    """``<viewer_root>/<city-slug>/`` — one city's publishable set.

    Refuses a name that slugs to nothing (no ASCII letters or digits): the
    empty slug would collapse the path back to the site-wide one this naming
    exists to remove, and two cities would silently share it.
    """
    slug = city_slug(city_name)
    if not slug:
        raise ConfigError(
            f"city name {city_name!r} has no letters or digits to name its page "
            f"directory with — give [city].name an ASCII form"
        )
    return viewer_root / slug


def city_manifest(city_name: str, viewer_root: Path = VIEWER_ROOT) -> Path:
    """``<viewer_root>/<city-slug>/manifest.json``, the default for one city."""
    return city_dir(city_name, viewer_root) / MANIFEST_NAME


def city_tiles(serve_dir: str, tiles_root: Path = TILES_ROOT) -> Path:
    """The archive directory a city publishes into.

    ``serve_dir`` is the FIRST entry of the city's ``viewer.serving_dirs``: the
    one this pipeline writes. Later entries are foreign archives the city
    vouches for and never writes, so they are not a publish target.
    """
    return tiles_root / serve_dir


def refresh_cities(viewer_root: Path = VIEWER_ROOT) -> Path:
    """Rewrite ``<viewer_root>/cities.json`` from the manifests on disk.

    Derived rather than merged: a city whose directory is gone leaves the index
    on the next build, and a damaged index cannot outlive one. An unreadable
    manifest is skipped — the index is navigation, and one bad file must not
    take the others off the page.
    """
    if not viewer_root.is_dir():
        # no page files there, so no page to index — writing one would leave a
        # stray file in whatever directory the caller happened to be in
        return viewer_root / CITIES_NAME
    entries = []
    for manifest in sorted(viewer_root.glob(f"*/{MANIFEST_NAME}")):
        slug = manifest.parent.name
        try:
            site = json.loads(manifest.read_text(encoding="utf-8")).get("site") or {}
        except (OSError, ValueError) as exc:
            logger.warning("%s: unreadable, left out of the city index: %s", manifest, exc)
            continue
        name = site.get("name") if isinstance(site, dict) else None
        entries.append(
            {
                "slug": slug,
                "name": str(name or slug),
                "manifest": f"{slug}/{MANIFEST_NAME}",
            }
        )
    entries.sort(key=lambda entry: entry["name"])
    return atomic_write_text(viewer_root / CITIES_NAME, json.dumps({"cities": entries}, indent=1))
