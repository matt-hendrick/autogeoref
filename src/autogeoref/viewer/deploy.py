"""Static deploy bundle: a publishable viewer copy with public tile URLs."""

from __future__ import annotations

import json
import logging
import re
import shutil
from html import escape, unescape
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from ..paths import atomic_write_text
from .layout import CITIES_NAME, MANIFEST_NAME
from .stories import ASSETS_DIR

logger = logging.getLogger(__name__)


class DeployError(RuntimeError):
    """The deploy bundle cannot be built or would leak local infrastructure."""


#: Substrings that must never appear in a published manifest. Inherited from
#: the original bundle script (which caught a real private-IP leak); the scan
#: is over the whole manifest text, safety before precision.
LEAK_TOKENS = ("localhost", "192.168.", "172.", "titiler", "8008", "wsl.localhost")

#: Every loose file the two pages need, declared rather than globbed. The
#: bundle is what a visitor loads and there is no build step to notice a gap,
#: so a missing one is a hard error here instead of a blank page in a browser.
PAGE_FILES = (
    "index.html",
    "404.html",
    "app.css",
    "lib.js",
    "app.js",
    "walkthrough.html",
    "walkthrough.css",
    "walkthrough.js",
)

#: The two pages that carry share-card metadata, and the path each is served
#: at. ``404.html`` is not among them: it asks not to be indexed, and a share
#: card for a page that says "there is no page here" is worth nothing.
SOCIAL_PAGES = {"index.html": "", "walkthrough.html": "walkthrough"}

#: Where a page's generated head metadata is spliced in. The pages are
#: city-fact-free by contract, and a share card is nothing but city facts.
SOCIAL_MARKER = "<!--SOCIAL-->"

#: A page's own title and description, read back out of it. The walkthrough
#: writes both itself — they are project facts — and the deploy mirrors them
#: rather than restating them, so the copy has one home.
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.DOTALL)
DESCRIPTION_RE = re.compile(r'<meta name="description" content="([^"]*)"\s*/?>')

#: The tab, bookmark and home-screen icons both pages link. Separate from
#: ``PAGE_FILES`` because that list is the hand-written sources the page checks
#: read as text, and these are artwork; the bundle copies both the same way.
ICON_FILES = (
    "favicon.ico",
    "favicon.svg",
    "apple-touch-icon.png",
    "safari-pinned-tab.svg",
)

#: Read by the Pages platform at deploy time and never served to a visitor, so
#: it is neither a page file nor artwork. Its own tuple for the same reason
#: ``ICON_FILES`` has one: the bundle must carry it, and the page-file contract
#: does not describe it.
PLATFORM_FILES = ("_headers",)

#: The walkthrough's rendered plates and the JSON that captions them.
WALKTHROUGH_DIR = "walkthrough"

#: The share card, staged beside a city's manifest, the shape story assets use.
#: PER CITY and optional: the pages are shared by every city, so one card at the
#: viewer root would show one city's map to a reader sharing another's. A city
#: with none gets a `summary` card. JPEG because it is a photograph of a scan,
#: and its size decides whether an unfurler waits for it.
CARD_FILE = "og.jpg"

#: The smallest a card may be. Readers drop a large-card image below roughly
#: 300x157 and fall back to the small card — which draws the empty frame the
#: large card was gated to avoid, by a second route. Half the 1200x630 the
#: renderer draws, so a deliberate half-size card still passes.
CARD_MIN = (600, 315)

#: The deploy-time config the page loads before its own scripts, and the one
#: page file this GENERATES rather than copies — which is what keeps the token
#: out of the repository. Deliberately not in ``PAGE_FILES``: every member of
#: that list must exist to be copied, and this one is written either way.
CONFIG_FILE = "config.js"

#: Read when ``--mapbox-token`` is not passed. An environment variable keeps
#: the token out of shell history and out of the process list.
TOKEN_ENV = "AUTOGEOREF_MAPBOX_TOKEN"

#: Read when ``--site-url`` is not passed. Beside ``TOKEN_ENV`` because the
#: deploy scripts already carry their settings in the environment.
SITE_URL_ENV = "AUTOGEOREF_SITE_URL"

#: A Mapbox public token's alphabet. The check is what makes interpolating the
#: token into a JS string literal safe — nothing here needs escaping.
PUBLIC_TOKEN_RE = re.compile(r"pk\.[A-Za-z0-9._-]+")


def mapbox_config_js(token: str | None) -> str:
    """``config.js`` for the bundle: the search token, or an empty one.

    A public ``pk.`` token is the only accepted shape. A secret ``sk.`` token
    carries account write access and would be published verbatim, so it is
    refused rather than shipped.
    """
    if token is None:
        return (
            "/* No Mapbox token was supplied to deploy-bundle, so the deployed\n"
            "   page reports that address search is unavailable. */\n"
            'window.MAPBOX_TOKEN = "";\n'
        )
    token = token.strip()
    # case-insensitive: a mistyped `SK.` is still a secret token, and the
    # generic message below would not say why it must not be published
    if token.lower().startswith("sk."):
        raise DeployError(
            "SAFETY STOP: that is a Mapbox SECRET token (sk.) and the bundle "
            "publishes it verbatim — use a public pk. token restricted to the "
            "deployment's URL"
        )
    if not PUBLIC_TOKEN_RE.fullmatch(token):
        raise DeployError(f"mapbox token must be a public pk.<payload> token, got {token[:8]!r}…")
    return f'window.MAPBOX_TOKEN = "{token}";\n'


def public_tiles_base(tiles_base_url: str) -> str:
    """Validated ``http(s)://host[:port][/prefix]`` root for public archive
    URLs. The old ``startswith("http")`` check admitted pseudo-schemes,
    credentials, query strings, and fragments — each of which concatenates
    into a syntactically live but wrong public URL. The optional path prefix
    is normalized to single slashes with no trailing slash."""
    parts = urlsplit(tiles_base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise DeployError(f"tiles base URL must be http(s) with a hostname, got {tiles_base_url!r}")
    try:
        _ = parts.port
    except ValueError as exc:
        raise DeployError(f"tiles base URL has an invalid port: {tiles_base_url!r}") from exc
    if parts.username is not None or parts.password is not None:
        raise DeployError(f"tiles base URL must not embed credentials: {tiles_base_url!r}")
    if parts.query or parts.fragment:
        raise DeployError(f"tiles base URL must not carry a query or fragment: {tiles_base_url!r}")
    prefix = "/".join(segment for segment in parts.path.split("/") if segment)
    root = f"{parts.scheme}://{parts.netloc}"
    return f"{root}/{prefix}" if prefix else root


def _tile_url(tiles_base: str, archive_path: str) -> str:
    """Join a validated base and one archive's basename structurally."""
    return f"{tiles_base}/{quote(Path(archive_path).name)}"


def public_site_url(site_url: str | None) -> str | None:
    """Validated ``http(s)://host[:port][/prefix]`` root the page is served at,
    with no trailing slash. ``None`` passes through: the absolute-URL tags are
    then left out rather than guessed, because a wrong ``og:url`` sends every
    reader who shares the page to somewhere else."""
    if site_url is None:
        return None
    parts = urlsplit(site_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise DeployError(f"site URL must be http(s) with a hostname, got {site_url!r}")
    if parts.username is not None or parts.password is not None:
        raise DeployError(f"site URL must not embed credentials: {site_url!r}")
    if parts.query or parts.fragment:
        raise DeployError(f"site URL must not carry a query or fragment: {site_url!r}")
    prefix = "/".join(segment for segment in parts.path.split("/") if segment)
    root = f"{parts.scheme}://{parts.netloc}"
    return f"{root}/{prefix}" if prefix else root


def _tag(name: str, value: str, *, attr: str = "name") -> str:
    """One meta tag with its content escaped for an attribute."""
    return f'<meta {attr}="{escape(name, quote=True)}" content="{escape(value, quote=True)}">'


def _social_html(
    title: str,
    description: str,
    site_url: str | None,
    path: str,
    card: tuple[int, int] | None = None,
) -> str:
    """The head metadata for one page: its real title, a description, and the
    share-card tags a crawler reads.

    ``path`` is where the page is served under the site root (``""`` for the
    atlas). ``card`` is the staged image's pixel size, or None.

    The LARGE card is claimed only with an image AND an absolute URL:
    `og:image` must be absolute or readers drop it, and claiming the large card
    without a usable image draws an empty frame. One condition, both tags.
    """
    lines = [f"<title>{escape(title)}</title>"]
    if description:
        lines.append(_tag("description", description))
    lines.append(_tag("og:type", "website", attr="property"))
    lines.append(_tag("og:title", title, attr="property"))
    if description:
        lines.append(_tag("og:description", description, attr="property"))
    large = bool(card and site_url)
    lines.append(_tag("twitter:card", "summary_large_image" if large else "summary"))
    if site_url:
        url = f"{site_url}/{path}" if path else f"{site_url}/"
        lines.append(_tag("og:url", url, attr="property"))
        lines.append(f'<link rel="canonical" href="{escape(url, quote=True)}">')
    if large and card:
        lines.append(_tag("og:image", f"{site_url}/{CARD_FILE}", attr="property"))
        # the dimensions let a reader lay the card out before the bytes land
        lines.append(_tag("og:image:width", str(card[0]), attr="property"))
        lines.append(_tag("og:image:height", str(card[1]), attr="property"))
    return "\n".join(lines)


def _page_social(
    name: str,
    text: str,
    site: dict[str, Any],
    site_url: str | None,
    card: tuple[int, int] | None,
) -> str:
    """The metadata for one page file, from whichever side owns its copy.

    The atlas page's title and description are city facts and come from the
    manifest. The walkthrough's are project facts and it writes them itself, so
    they are read back out of it — restating them here would give the same
    sentence two homes and let them drift apart.
    """
    if name == "index.html":
        parts = [str(site.get(key) or "").strip() for key in ("name", "kicker")]
        title = " — ".join(part for part in parts if part) or "Atlas"
        description = str(site.get("dek") or "").strip()
    else:
        # unescaped on the way in, because it is read back out of MARKUP and
        # `_social_html` escapes what it is given. Without this an ampersand in
        # the page's own copy ships as `&amp;amp;` and unfurls literally.
        found = TITLE_RE.search(text)
        title = unescape(found.group(1)).strip() if found else ""
        described = DESCRIPTION_RE.search(text)
        description = unescape(described.group(1)).strip() if described else ""
    if not title:
        raise DeployError(f"{name} has no title to publish — the share card would be blank")
    return _social_html(title, description, site_url, SOCIAL_PAGES[name], card)


def _write_page(
    source: Path,
    target: Path,
    site: dict[str, Any],
    site_url: str | None,
    card: tuple[int, int] | None,
) -> None:
    """Copy one page file, splicing its generated metadata in on the way.

    The marker is required rather than optional: a page that lost it would
    deploy with the placeholder title and nothing would notice until the link
    was shared.
    """
    if source.name not in SOCIAL_PAGES:
        # byte for byte: a script or a stylesheet has nothing spliced into it,
        # and a text round-trip would rewrite its line endings on a host whose
        # newline is not the one it was written with
        shutil.copy(source, target)
        return
    text = source.read_text(encoding="utf-8")
    social = _page_social(source.name, text, site, site_url, card)
    # A page's own title and description are the INPUT to the block above; the
    # block is the output and carries both, so the originals come out. Leaving
    # either in ships it twice, and a reader of two takes the first — which for
    # the atlas page is the placeholder title this exists to replace.
    text = TITLE_RE.sub("", text, count=1)
    text = DESCRIPTION_RE.sub("", text, count=1)
    atomic_write_text(target, text.replace(SOCIAL_MARKER, social, 1))


def _robots_txt(site_url: str | None) -> str:
    """``robots.txt`` for the bundle.

    Written even with no site URL: most of its value here is being a real file
    at a real path, so the platform stops answering that path with the atlas
    page and a crawler stops reading HTML as a robots directive.
    """
    lines = ["User-agent: *", "Allow: /"]
    if site_url:
        lines.append(f"Sitemap: {site_url}/sitemap.xml")
    return "\n".join(lines) + "\n"


def _sitemap_xml(site_url: str) -> str:
    """``sitemap.xml`` naming the two pages a reader can land on."""
    urls = "".join(
        f"<url><loc>{escape(f'{site_url}/{path}' if path else f'{site_url}/')}</loc></url>"
        for path in SOCIAL_PAGES.values()
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{urls}</urlset>\n"
    )


def _card_size(source_dir: Path) -> tuple[int, int] | None:
    """A city's share card measured, or ``None`` when it ships none.

    Reading only, so it runs with the other pre-flight checks before the first
    write: a refusal partway through leaves a half-built bundle.

    Shipping none is the common case, not an error. An unreadable or undersized
    one is: both put an `og:image` on every page that a reader cannot show, and
    the only thing that ever looks at a card is somebody else's crawler, so
    nothing downstream would report it.
    """
    card = source_dir / CARD_FILE
    if not card.is_file():
        return None
    from PIL import Image, UnidentifiedImageError  # lazy: only a card needs it

    try:
        with Image.open(card) as image:
            # `open` parses the header and decodes nothing, so a truncated file
            # measures fine and fails in the reader. `load` is what reads it.
            image.load()
            size = (int(image.width), int(image.height))
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise DeployError(f"{card} is not a readable image: {exc}") from exc
    if size[0] < CARD_MIN[0] or size[1] < CARD_MIN[1]:
        raise DeployError(
            f"{card} is {size[0]}x{size[1]}, under the {CARD_MIN[0]}x{CARD_MIN[1]} a large "
            "card needs — a reader that rejects it draws the empty frame this exists to avoid"
        )
    return size


def _write_bundle_files(
    viewer_dir: Path,
    source_dir: Path,
    out_dir: Path,
    copied: tuple[str, ...],
    site: dict[str, Any],
    site_root: str | None,
    config_js: str,
    card: tuple[int, int] | None = None,
) -> None:
    """Every loose file the bundle serves, other than the manifests and vendor.

    Three kinds: the pages, which have their head metadata spliced in on the
    way through; the artwork and the platform's own files, copied byte for
    byte; and the two crawler files, which exist only in the bundle.
    """
    for name in copied:
        if name in PAGE_FILES:
            _write_page(viewer_dir / name, out_dir / name, site, site_root, card)
        else:
            shutil.copy(viewer_dir / name, out_dir / name)
    # only when the pages will name it. A card with no site URL is a card no
    # tag can reach, and shipping it puts unreferenced bytes in the bundle —
    # the same condition the tags are gated on, so the two cannot disagree.
    if card is not None and site_root:
        shutil.copy(source_dir / CARD_FILE, out_dir / CARD_FILE)
    # generated, never copied: this is what keeps the token out of the tree
    atomic_write_text(out_dir / CONFIG_FILE, config_js)
    # Real files at the paths a crawler asks for. Their ABSENCE is the defect:
    # with no 404.html in the bundle the platform answers every unmatched path
    # with the atlas page under a 200, so /robots.txt read as HTML and a
    # crawler reported the site as having an invalid one.
    atomic_write_text(out_dir / "robots.txt", _robots_txt(site_root))
    if site_root:
        atomic_write_text(out_dir / "sitemap.xml", _sitemap_xml(site_root))


def assert_no_leaks(text: str) -> None:
    """The safety gate: refuse a bundle manifest naming local infrastructure."""
    for token in LEAK_TOKENS:
        if token in text:
            raise DeployError(f"SAFETY STOP: {token!r} found in deploy manifest — do not upload")


def build_deploy_bundle(
    viewer_dir: Path,
    out_dir: Path,
    tiles_base_url: str,
    *,
    city: str,
    mapbox_token: str | None = None,
    site_url: str | None = None,
) -> dict[str, Any]:
    """Static, publishable copy of ONE city's viewer: the page files + vendor
    assets + that city's manifest, whose PMTiles references are rewritten from
    local relative paths to ``tiles_base_url``. The .pmtiles files are uploaded
    separately.

    ``city`` is the slug naming ``<viewer_dir>/<city>/manifest.json``, put
    beside the page. ``mapbox_token`` is written into the bundle's
    ``config.js``, replacing the token-free placeholder; without one the
    deployed page reports that address search is unavailable."""
    tiles_base = public_tiles_base(tiles_base_url)
    site_root = public_site_url(site_url)
    config_js = mapbox_config_js(mapbox_token)  # before any write: a bad token stops here
    source_dir = viewer_dir / city
    manifest = json.loads((source_dir / MANIFEST_NAME).read_text(encoding="utf-8"))

    volumes: list[dict[str, Any]] = []
    for volume in manifest["volumes"]:
        v = dict(volume)
        if "pmtiles" in v:
            v["pmtiles"] = _tile_url(tiles_base, v["pmtiles"])
        volumes.append(v)

    if not any("pmtiles" in v for v in volumes):
        raise DeployError(
            "no per-volume pmtiles in the manifest — nothing to publish. Bake at "
            "least one PMTiles archive, regenerate the manifest, then re-run this."
        )

    # only volumes with something to show belong in the public district list
    published = [v for v in volumes if "pmtiles" in v]
    skipped = sorted({str(v["era"]) for v in volumes if v not in published and v.get("era")})
    if skipped:
        logger.info("eras without a pmtiles file are left out of the bundle: %s", skipped)

    out: dict[str, Any] = {"volumes": published}
    if manifest.get("site"):
        site = dict(manifest["site"])
        # the basemap archive is served from the same public bucket as the
        # atlas ones; the manifest's local relative path would 404 on the
        # deployed site (and MapLibre reports that as an empty basemap)
        basemap = site.get("basemap") or {}
        if basemap.get("pmtiles"):
            basemap = dict(basemap)
            basemap["pmtiles"] = _tile_url(tiles_base, basemap["pmtiles"])
            site["basemap"] = basemap
        # A manifest on disk outlives the code that wrote it. `provider` named
        # a geocoder nothing ever read, and publishing it now would advertise
        # the opposite of the rule the page follows.
        if isinstance(site.get("geocoder"), dict) and "provider" in site["geocoder"]:
            geocoder = {k: v for k, v in site["geocoder"].items() if k != "provider"}
            site["geocoder"] = geocoder
        out["site"] = site

    text = json.dumps(out, indent=1)
    assert_no_leaks(text)

    copied = (*PAGE_FILES, *ICON_FILES, *PLATFORM_FILES)
    missing = [name for name in copied if not (viewer_dir / name).is_file()]
    if missing:
        raise DeployError(f"viewer directory is missing these files: {', '.join(missing)}")
    # with the file check and before the first write: a marker refusal partway
    # through would otherwise leave a bundle directory half built
    unmarked = [
        name
        for name in SOCIAL_PAGES
        if SOCIAL_MARKER not in (viewer_dir / name).read_text(encoding="utf-8")
    ]
    if unmarked:
        raise DeployError(
            f"these pages have lost their {SOCIAL_MARKER} marker: {', '.join(unmarked)}"
        )
    # measured here rather than at the copy: whether a card exists decides what
    # every page's tags claim, and an unreadable one must stop the build before
    # it has written anything
    card = _card_size(source_dir)

    atomic_write_text(out_dir / MANIFEST_NAME, text)
    # A one-entry index, so the deployed page resolves its city exactly the way
    # the local one does instead of having a second code path for the case.
    atomic_write_text(
        out_dir / CITIES_NAME,
        json.dumps(
            {
                "cities": [
                    {
                        "slug": city,
                        "name": str((out.get("site") or {}).get("name") or city),
                        "manifest": MANIFEST_NAME,
                    }
                ]
            },
            indent=1,
        ),
    )
    _write_bundle_files(
        viewer_dir, source_dir, out_dir, copied, out.get("site") or {}, site_root, config_js, card
    )
    if (viewer_dir / "vendor").is_dir():
        shutil.copytree(viewer_dir / "vendor", out_dir / "vendor", dirs_exist_ok=True)
    # story assets are staged beside the city's manifest under one conventional
    # name, so this needs to know nothing about the city that configured them
    if (source_dir / ASSETS_DIR).is_dir():
        shutil.copytree(source_dir / ASSETS_DIR, out_dir / ASSETS_DIR, dirs_exist_ok=True)
    # walkthrough plates are committed to the repo, not staged per city
    if (viewer_dir / WALKTHROUGH_DIR).is_dir():
        shutil.copytree(viewer_dir / WALKTHROUGH_DIR, out_dir / WALKTHROUGH_DIR, dirs_exist_ok=True)
    logger.info("wrote %s (%d volumes)", out_dir, len(published))
    return out
