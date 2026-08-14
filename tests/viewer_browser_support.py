"""The servable viewer fixture the browser tests load, and how they load it.

Built with no GDAL and no network: solid-colour PNGs packed into a PMTiles
archive with Pillow, and a basemap served by the same loopback server the page
is. Shared by the browser modules so each keeps only its own assertions.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from autogeoref.tiles import pack_pmtiles
from browser import ROOT, PageLoad, load_page, serve

VIEWER_DIR = ROOT / "viewer"

#: The fixture city. Deliberately nowhere near Chicago: the viewer carries no
#: city fact, and a fixture that happened to match the real one would hide it.
PROBE_CENTER = (-71.0589, 42.3601)
PROBE_BOUNDS = [-71.13, 42.31, -70.99, 42.41]

#: The fixture city's slug, and the directory its manifest lives in — the
#: shipping layout is one manifest per city under the page directory.
PROBE_SLUG = "probeville"

#: The atlas archive as the MANIFEST names it, chosen so the two bases
#: DISAGREE: from `viewer/<slug>/manifest.json` this is `viewer/archives/`,
#: from the page at `viewer/index.html` it would be `/archives/`, where nothing
#: is served. A path with enough `..` to reach the server root resolves the
#: same under both — the clamp swallows the difference — and would leave a
#: page-relative regression invisible to every assertion here.
ATLAS_ARCHIVE = "../archives/probe.pmtiles"
ARCHIVE_DIR = "viewer/archives"

#: The same archive named from a manifest sitting BESIDE the page, which is the
#: index-less layout.
ATLAS_ARCHIVE_BESIDE_PAGE = "archives/probe.pmtiles"

#: The overlay is down, which means a TILE PAINTED — or the page waited out its
#: 20s deadline. NOT merely that `drawAtlas` ran and added the sources, which is
#: what this used to mean. A fixture whose imagery never loads therefore waits
#: that deadline out rather than proceeding: give one an `until` of its own.
DREW = 'document.getElementById("loading").classList.contains("done")'

#: The overlay's other resting state: no atlas to draw, and a reason on screen.
CHOOSING = 'document.getElementById("loading").classList.contains("choose")'

#: The atlas sources the page actually put on the map, keyed by volume id —
#: what DREW, as distinct from what the district list enumerates.
SOURCES = "JSON.stringify(Object.keys(window.beforeMap.getStyle().sources))"

#: The credit line naming the basemap.
CREDITED = 'document.getElementById("footer").textContent.includes("OpenStreetMap")'

#: Long enough that the credit, which a working basemap publishes in well under
#: a second, has had every chance to appear.
SETTLE_S = 6.0


def tile_xy(lng: float, lat: float, z: int) -> tuple[int, int]:
    n = 2**z
    lat_r = math.radians(lat)
    return (
        int((lng + 180.0) / 360.0 * n),
        int((1.0 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2.0 * n),
    )


def _atlas_archive(out: Path, work: Path) -> None:
    """A real PMTiles archive over the probe city. No GDAL: `pack_pmtiles` needs none."""
    xyz = work / "xyz"
    for z in (10, 11, 12, 13):
        x, y = tile_xy(*PROBE_CENTER, z)
        for dx in (0, 1):
            for dy in (0, 1):
                folder = xyz / str(z) / str(x + dx)
                folder.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (256, 256), (198, 122, 58)).save(folder / f"{y + dy}.png")
    pack_pmtiles(xyz, out, tile_ext="png")


#: A vector tile with one empty layer, ten bytes. MapLibre reports it as a
#: LOADED tile, which is all these tests need — and it means a vector basemap
#: fixture costs no GDAL, no planet extract and nothing gitignored.
EMPTY_MVT = bytes([0x1A, 0x08, 0x78, 0x02, 0x0A, 0x01, 0x78, 0x28, 0x80, 0x20])


def _vector_archive(out: Path) -> None:
    """A minimal MVT PMTiles archive over the probe city.

    `pack_pmtiles` writes raster tile types only, so this uses the writer
    directly — the tile type is what tells MapLibre how to read the payload.
    """
    from pmtiles.tile import Compression, TileType, zxy_to_tileid
    from pmtiles.writer import Writer

    corners = [
        (z, *tile_xy(*PROBE_CENTER, z), dx, dy)
        for z in range(1, 15)
        for dx in (0, 1)
        for dy in (0, 1)
    ]
    entries = [(zxy_to_tileid(z, x + dx, y + dy), EMPTY_MVT) for z, x, y, dx, dy in corners]
    entries.sort(key=lambda entry: entry[0])
    with out.open("wb") as handle:
        writer = Writer(handle)  # type: ignore[no-untyped-call]
        for tileid, blob in entries:
            writer.write_tile(tileid, blob)  # type: ignore[no-untyped-call]
        writer.finalize(  # type: ignore[no-untyped-call]
            {
                "tile_type": TileType.MVT,
                "tile_compression": Compression.NONE,
                "min_zoom": 1,
                "max_zoom": 14,
                "min_lon_e7": int(PROBE_BOUNDS[0] * 1e7),
                "min_lat_e7": int(PROBE_BOUNDS[1] * 1e7),
                "max_lon_e7": int(PROBE_BOUNDS[2] * 1e7),
                "max_lat_e7": int(PROBE_BOUNDS[3] * 1e7),
                "center_zoom": 12,
                "center_lon_e7": int(PROBE_CENTER[0] * 1e7),
                "center_lat_e7": int(PROBE_CENTER[1] * 1e7),
            },
            {"generator": "autogeoref-test"},
        )


def manifest_for(basemap_tiles: str | None) -> dict[str, Any]:
    site: dict[str, Any] = {
        "name": "Probeville",
        "title": "Probeville {era} / Now",
        "footer_source_html": "Probe fixture",
        "home_point": list(PROBE_CENTER),
    }
    if basemap_tiles is not None:
        site["basemap"] = {
            "type": "raster",
            "tiles": basemap_tiles,
            "attribution": "© OpenStreetMap contributors",
            "maxzoom": 19,
        }
    return {
        "volumes": [
            {
                "id": "probe_001",
                "era": "1890s",
                "year": 1894,
                "volume_number": "1",
                "label": "Probe District",
                "bounds": PROBE_BOUNDS,
                "pmtiles": ATLAS_ARCHIVE,
            }
        ],
        "site": site,
    }


def vector_manifest(archive: str) -> dict[str, Any]:
    """The shape a real deployment uses: a vendored style over a hosted archive."""
    manifest = manifest_for(None)
    manifest["site"]["basemap"] = {
        "type": "vector",
        "pmtiles": archive,
        "styles": {
            "atlas": "vendor/basemap/style-grayscale.json",
            "now": "vendor/basemap/style-light.json",
        },
        "attribution": "© OpenStreetMap contributors, Protomaps",
    }
    return manifest


#: A hostname the page must treat as a public deployment. `.test` is reserved
#: for exactly this, so it can never become somebody's real site.
PUBLIC_HOST = "atlas.example.test"


def build_viewer_bundle(root: Path) -> Path:
    """Populate ``root`` as a servable viewer: the real page files, a real
    PMTiles archive over the probe city, a raster basemap tile and a vector
    basemap archive. `conftest`'s ``viewer_bundle`` fixture is the caller."""
    shutil.copytree(VIEWER_DIR, root / "viewer")
    (root / ARCHIVE_DIR).mkdir(parents=True, exist_ok=True)
    _atlas_archive(root / ARCHIVE_DIR / "probe.pmtiles", root)
    base = root / "base"
    base.mkdir()
    Image.new("RGB", (256, 256), (34, 34, 44)).save(base / "tile.png")
    # No favicon is synthesized at the served root: the page links its own,
    # so the browser never falls back to asking for /favicon.ico.
    _vector_archive(root / "basemap.pmtiles")
    return root


def city_index(*cities: tuple[str, str]) -> dict[str, Any]:
    """The index the page reads to resolve which manifest to load."""
    return {
        "cities": [
            {"slug": slug, "name": name, "manifest": f"{slug}/manifest.json"}
            for slug, name in cities
        ]
    }


def publish(
    root: Path, manifest: dict[str, Any], *, slug: str = PROBE_SLUG, index: bool = True
) -> None:
    """Put one city's manifest where a published viewer keeps it.

    ``index=False`` leaves out ``cities.json``: one manifest beside the page and
    no index, which is what a hand-copied directory looks like. A deploy bundle
    is NOT this — it writes its own one-entry index.
    """
    city_dir = root / "viewer" / slug
    city_dir.mkdir(parents=True, exist_ok=True)
    if index:
        (city_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "viewer" / "cities.json").write_text(
            json.dumps(city_index((slug, "Probeville"))), encoding="utf-8"
        )
    else:
        (root / "viewer" / "cities.json").unlink(missing_ok=True)
        (root / "viewer" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def manifest_with_many_districts(count: int = 30) -> dict[str, Any]:
    """The probe manifest with more districts than any panel can show at once.

    The extra entries carry no archive: the district INDEX lists every volume
    in a selected era whether or not its imagery is published, which is what
    this fixture is about and what keeps it from needing 30 PMTiles files.
    """
    manifest = manifest_for("/base/tile.png")
    first = manifest["volumes"][0]
    for number in range(2, count + 1):
        entry = {k: v for k, v in first.items() if k != "pmtiles"}
        entry["id"] = f"probe_{number:03d}"
        entry["volume_number"] = str(number)
        entry["label"] = f"Probe District {number}"
        manifest["volumes"].append(entry)
    return manifest


def load(
    root: Path,
    manifest: dict[str, Any],
    *,
    until: str = DREW,
    settle_s: float = 0.0,
    capture: dict[str, str] | None = None,
    query: str = "",
    fragment: str = "",
    viewport: tuple[int, int] | None = None,
    host_alias: str | None = None,
) -> PageLoad:
    """``host_alias`` serves the same bundle under a public-looking hostname,
    for the decisions the page makes from ``location.hostname``."""
    publish(root, manifest)
    with serve(root) as base_url:
        if host_alias:
            base_url = base_url.replace("127.0.0.1", host_alias, 1)
        return load_page(
            f"{base_url}/viewer/index.html{query}{fragment}",
            until=until,
            settle_s=settle_s,
            capture=capture,
            viewport=viewport,
            host_alias=host_alias,
        )


def captured_json(page: PageLoad, key: str) -> Any:
    """One captured expression, parsed back from the JSON it serialized.

    ``PageLoad.captured`` is ``dict[str, object]`` because a capture may return
    any JSON value; the callers that ask a page to `JSON.stringify` something
    read it here, and the assertion names the key when a capture came back as
    something other than the string it promised.
    """
    value = page.captured[key]
    assert isinstance(value, str), f"{key} captured a {type(value).__name__}, not a string"
    return json.loads(value)


def links(selector: str) -> str:
    """The anchors under ``selector``, as the browser PARSED them. Escaped
    markup renders the same text, so only the DOM tells the two apart."""
    return (
        f'JSON.stringify([...document.querySelectorAll("{selector} a")]'
        '.map(a => [a.getAttribute("href"), a.textContent]))'
    )


def visibility(layer: str) -> str:
    return (
        f'window.beforeMap.getLayer("{layer}") ? '
        f'window.beforeMap.getLayoutProperty("{layer}", "visibility") : "absent"'
    )
