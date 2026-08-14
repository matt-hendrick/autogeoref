"""Convert cached Overpass responses into matcher-shaped GeoJSON.

Configured centerlines remain first-class. OSM-derived centerlines omit
address ranges, so address evidence abstains. Intersection topology is not
in the response either and is recovered geometrically: two streets sharing a
vertex are split there and given ``fnode_id``/``tnode_id``, so the junction
channel reads OSM through the same properties as a bring-your-own file. Network
access is limited to the injectable, cached Overpass fetch path and is never
used in tests.
"""

from __future__ import annotations

import fcntl
import json
import logging
import re
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from .centerlines import centerline_key
from .paths import atomic_write_text

logger = logging.getLogger(__name__)

Bounds = tuple[float, float, float, float]

#: One whole street way: its matcher properties and its 4326 lng/lat run.
Ways = list[tuple[dict[str, Any], list[list[float]]]]

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

#: Fetch bboxes are expanded by this margin so the cached data always
#: out-covers every consumer's own clip margin (clip_features_4326 clips at
#: 0.02 deg — see REQUIRED_MARGIN_DEG).
FETCH_MARGIN_DEG = 0.03

#: Revision of the emitted feature properties, recorded in every cache file.
#: A cache recording an older revision is rebuilt with one fetch rather than
#: read: revision 2 added the derived junction topology, and reading a
#: revision-1 file would leave the junction channel permanently blind on a
#: city that looks fully cached.
CENTERLINE_SCHEMA_VERSION = 2

#: Decimal places a vertex is rounded to for its node key. 1e-7 deg is ~1 cm —
#: the precision Overpass already emits, so two ways sharing an OSM node always
#: agree. Two OSM nodes stacked at one coordinate would merge, which is what a
#: shared node looks like anyway.
NODE_KEY_DECIMALS = 7

#: Coverage demanded of the cache before a volume runs against it: the
#: volume bounds expanded by the widest consumer clip margin
#: (``geometry.clip_features_4326``'s 0.02 deg default). Must stay
#: strictly below FETCH_MARGIN_DEG or a just-fetched bbox would not cover
#: its own volume.
REQUIRED_MARGIN_DEG = 0.02


class OSMFetchError(RuntimeError):
    """Overpass answered with something other than usable JSON content."""


#: OSM ``highway`` classes that are never street centerlines for matching
#: (sidewalks, alleys, trails); excluded in the Overpass query and again,
#: defensively, when converting a cached response that was fetched broader.
EXCLUDED_HIGHWAYS = frozenset({"footway", "cycleway", "path", "service", "steps"})

#: Trailing name token -> ``street_typ`` code. Keys cover spelled-out and
#: abbreviated OSM forms so hand-edited extracts round-trip.
SUFFIX_CODES = {
    "STREET": "ST",
    "ST": "ST",
    "AVENUE": "AVE",
    "AVE": "AVE",
    "AV": "AVE",
    "BOULEVARD": "BLVD",
    "BLVD": "BLVD",
    "PLACE": "PL",
    "PL": "PL",
    "COURT": "CT",
    "CT": "CT",
    "DRIVE": "DR",
    "DR": "DR",
    "ROAD": "RD",
    "RD": "RD",
    "TERRACE": "TER",
    "TER": "TER",
    "PARKWAY": "PKWY",
    "PKWY": "PKWY",
    "LANE": "LN",
    "LN": "LN",
    "SQUARE": "SQ",
    "SQ": "SQ",
    "WAY": "WAY",
    "EXPRESSWAY": "EXPY",
    "EXPY": "EXPY",
    "HIGHWAY": "HWY",
    "HWY": "HWY",
    "PLAZA": "PLZ",
    "PLZ": "PLZ",
    "CRESCENT": "CRES",
    "ROW": "ROW",
}

_DIRECTIONS = frozenset({"N", "S", "E", "W", "NORTH", "SOUTH", "EAST", "WEST"})

_PARENTHETICAL = re.compile(r"\([^)]*\)")

_CITY_LOCKS: dict[Path, threading.Lock] = {}
_CITY_LOCKS_LOCK = threading.Lock()

#: Spelled-out tokens the official schema abbreviates mid-name (verified
#: against the official data: ``ST LAWRENCE``, ``DR MARTIN LUTHER KING JR``
#: — never SAINT/DOCTOR/JUNIOR). Applied per token AFTER the trailing type
#: pop, so a trailing ``Drive`` is a type code and a leading ``Doctor`` is
#: an honorific, never confused.
_TOKEN_ABBREV = {"SAINT": "ST", "DOCTOR": "DR", "JUNIOR": "JR"}


def split_osm_name(name: str) -> tuple[str, str]:
    """Split an OSM street name into matcher name and type fields.

    Uppercases, drops punctuation and parentheticals, pops ONE trailing direction token (the
    official schema's ``suf_dir``, which trails the type), pops ONE trailing type token into the
    ``street_typ`` code, abbreviates the honorifics the schema never spells out, then strips ONE
    leading direction token — exactly one, mirroring the single ``pre_dir`` field, and never
    down to nothing, since the direction word can BE the name. A name without a recognized
    trailing type keeps every token and gets ``''``.
    """
    n = name.upper().replace("'", "").replace("’", "")  # noqa: RUF001 — curly quote intended
    n = _PARENTHETICAL.sub(" ", n)
    for ch in ".,()/&-":
        n = n.replace(ch, " ")
    toks = [t for t in n.split(" ") if t]
    # trailing direction after the type is the official suf_dir, not the name
    if len(toks) >= 3 and toks[-1] in _DIRECTIONS and toks[-2] in SUFFIX_CODES:
        toks = toks[:-1]
    typ = ""
    if len(toks) >= 2 and toks[-1] in SUFFIX_CODES:
        typ = SUFFIX_CODES[toks[-1]]
        toks = toks[:-1]
    toks = [_TOKEN_ABBREV.get(t, t) for t in toks]
    if len(toks) > 1 and toks[0] in _DIRECTIONS:
        toks = toks[1:]
    return " ".join(toks), typ


def overpass_query(bounds_4326: Bounds, timeout_s: int = 300) -> str:
    """Overpass QL for named streets in a (minlon, minlat, maxlon, maxlat) bbox.

    One query per city, run once and cached forever (fetch-script conduct).
    Excludes the never-a-street classes server-side; ``out geom`` returns
    per-way coordinate runs in stable way-id order (deterministic downstream
    candidate order, mirroring ``CenterlineIndex``'s reliance on source
    order).
    """
    minx, miny, maxx, maxy = bounds_4326
    excluded = "|".join(sorted(EXCLUDED_HIGHWAYS))
    bbox = f"{miny},{minx},{maxy},{maxx}"  # Overpass order: south,west,north,east
    return (
        f"[out:json][timeout:{timeout_s}];\n"
        f'way["highway"]["name"]["highway"!~"^({excluded})$"]({bbox});\n'
        "out geom;\n"
    )


def node_key(lon: float, lat: float) -> str:
    """Stable vertex identity for the junction graph: rounded ``lon,lat``."""
    return f"{lon:.{NODE_KEY_DECIMALS}f},{lat:.{NODE_KEY_DECIMALS}f}"


def _usable_ways(data: dict[str, Any]) -> Ways:
    """Filter Overpass elements to (properties, coordinates) per usable way.

    Skips: non-way elements, degenerate geometry, unnamed ways, excluded
    highway classes (defense in depth vs a broader cached fetch), and
    ``area=yes`` ways (a named pedestrian square outlines a polygon, not a
    centerline). Source order is preserved.
    """
    ways: Ways = []
    for el in data.get("elements", []):
        if el.get("type") != "way":
            continue
        pts = el.get("geometry")
        if not pts or len(pts) < 2:
            continue
        tags = el.get("tags") or {}
        name = tags.get("name")
        if not name or tags.get("highway") in EXCLUDED_HIGHWAYS or tags.get("area") == "yes":
            continue
        street_nam, street_typ = split_osm_name(name)
        if not street_nam:
            continue
        props = {
            "street_nam": street_nam,
            "street_typ": street_typ,
            "osm_id": el.get("id"),
            "osm_name": name,
            "highway": tags.get("highway"),
        }
        ways.append((props, [[p["lon"], p["lat"]] for p in pts]))
    return ways


def _streets_by_node(ways: Ways) -> dict[str, set[str]]:
    """Node key -> the distinct street identities whose ways touch it.

    Identity is :func:`autogeoref.centerlines.centerline_key`, the one key rule
    every consumer shares, so a numbered PLACE/COURT twin counts as the
    different street it is rather than as its numbered neighbour continuing.
    """
    names: dict[str, set[str]] = {}
    for props, coords in ways:
        key = centerline_key(props) or ""
        for node in {node_key(c[0], c[1]) for c in coords}:
            names.setdefault(node, set()).add(key)
    return names


def cut_ways_into_features(ways: Ways) -> list[dict[str, Any]]:
    """Node whole ways into matcher-shape features carrying the junction graph.

    A way is CUT at every interior vertex it shares with a DIFFERENTLY-named
    way, and each piece keyed by its endpoint coordinates as
    ``fnode_id``/``tnode_id`` — the graph Overpass does not ship. One way can
    therefore emit several features, all with the same ``osm_id``, contiguous
    and in order. Cutting is a property of the whole set, so a caller merging
    two fetches must re-cut the union rather than splice the outputs.
    """
    streets_at = _streets_by_node(ways)
    features: list[dict[str, Any]] = []
    for props, coords in ways:
        keys = [node_key(c[0], c[1]) for c in coords]
        mine = centerline_key(props) or ""
        # A shared vertex is a junction only when another STREET meets here:
        # two same-named ways touching end to end are one street continuing,
        # and cutting there would invent a degree-3 node out of nothing.
        cuts = [i for i in range(1, len(coords) - 1) if streets_at[keys[i]] - {mine}]
        starts = [0, *cuts]
        ends = [*cuts, len(coords) - 1]
        for a, b in zip(starts, ends, strict=True):
            features.append(
                {
                    "type": "Feature",
                    "properties": {**props, "fnode_id": keys[a], "tnode_id": keys[b]},
                    "geometry": {"type": "LineString", "coordinates": coords[a : b + 1]},
                }
            )
    return features


def ways_from_features(features: list[dict[str, Any]]) -> Ways:
    """Stitch emitted features back into whole ways — the inverse of the cut.

    Pieces of one way are contiguous and in order and adjacent pieces share the
    cut vertex, so a run of equal ``osm_id`` rejoins exactly. Features written
    before the cut existed are one piece each and pass through unchanged.
    """
    ways: Ways = []
    for f in features:
        props = dict(f["properties"])
        props.pop("fnode_id", None)
        props.pop("tnode_id", None)
        coords = [list(c) for c in f["geometry"]["coordinates"]]
        if ways and ways[-1][0].get("osm_id") is not None and ways[-1][0] == props:
            ways[-1][1].extend(coords[1:])
            continue
        ways.append((props, coords))
    return ways


def overpass_to_centerline_features(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Overpass JSON (``out geom``) -> matcher-shape GeoJSON features.

    Emits LineString features carrying ``street_nam``/``street_typ`` (the split
    name), provenance (``osm_id``, ``osm_name``, ``highway``) and the
    ``fnode_id``/``tnode_id`` the junction channel reads. See
    :func:`cut_ways_into_features` for the cut rule. Source order is preserved.
    """
    return cut_ways_into_features(_usable_ways(data))


def centerline_geojson(data: dict[str, Any]) -> dict[str, Any]:
    """Overpass JSON -> a FeatureCollection ready for ``centerlines =`` in a city TOML."""
    return {
        "type": "FeatureCollection",
        "schema_version": CENTERLINE_SCHEMA_VERSION,
        "features": overpass_to_centerline_features(data),
    }


def expand_bounds(bounds_4326: Bounds, margin_deg: float) -> Bounds:
    minx, miny, maxx, maxy = bounds_4326
    return (minx - margin_deg, miny - margin_deg, maxx + margin_deg, maxy + margin_deg)


def _covers(outer: Bounds, inner: Bounds) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _city_lock(path: Path) -> threading.Lock:
    """Return the in-process half of a per-city cache lock."""
    key = path.resolve()
    with _CITY_LOCKS_LOCK:
        return _CITY_LOCKS.setdefault(key, threading.Lock())


@contextmanager
def _locked_city_cache(path: Path) -> Iterator[None]:
    """Serialize cache read/merge/publish across threads and processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with _city_lock(path), lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _load_city_cache(path: Path) -> tuple[list[dict[str, Any]], list[list[float]], int | None]:
    """Read a cache into (features, fetched bboxes, schema version).

    A torn or malformed file is an empty rebuild; a version the caller has to
    act on is returned rather than decided here.
    """
    if not path.exists():
        return [], [], CENTERLINE_SCHEMA_VERSION
    try:
        cached = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("%s: unparseable OSM cache (%s); rebuilding", path, exc)
        return [], [], CENTERLINE_SCHEMA_VERSION
    version = cached.get("schema_version")
    return (
        cached.get("features", []),
        cached.get("fetched_bboxes", []),
        version if isinstance(version, int) else None,
    )


def _publish_city_cache(
    path: Path, features: list[dict[str, Any]], fetched: list[list[float]]
) -> None:
    atomic_write_text(
        path,
        json.dumps(
            {
                "type": "FeatureCollection",
                "schema_version": CENTERLINE_SCHEMA_VERSION,
                "fetched_bboxes": fetched,
                "features": features,
            }
        ),
    )


def check_overpass_response(data: dict[str, Any]) -> dict[str, Any]:
    """Reject Overpass responses that report failure INSIDE a 200 body.

    Overpass signals query timeouts and runtime errors as HTTP 200 plus a
    ``remark`` field, with ``elements`` empty or silently truncated. Caching
    such a response would poison a cached-forever file (coverage recorded
    over missing data), so every path that consumes a response — the
    runtime default fetch, injected fetchers, replayed raw files, the
    prefetch script — must pass it through this check.
    """
    remark = data.get("remark")
    if remark and ("error" in str(remark).lower() or "timed out" in str(remark).lower()):
        raise OSMFetchError(f"Overpass remark reports failure: {remark}")
    return data


def fetch_overpass(query: str, timeout_s: float = 360.0) -> dict[str, Any]:
    """ONE Overpass request — the module's single network choke point.

    GET only, https, overpass-api.de only (the URL is a module constant, not
    a parameter), honest User-Agent, no redirect following. Never called
    from tests: every caller accepts an injectable ``fetch`` instead.
    """
    import httpx  # runtime-only: the no-network test paths never import it

    from .loc import USER_AGENT

    url = f"{OVERPASS_URL}?{urlencode({'data': query})}"
    logger.info("Overpass fetch: %s", query.replace("\n", " "))
    resp = httpx.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout_s,
        follow_redirects=False,
    )
    if resp.status_code != 200:
        raise OSMFetchError(f"Overpass answered HTTP {resp.status_code}: {resp.text[:500]}")
    result: dict[str, Any] = resp.json()
    return check_overpass_response(result)


def ensure_city_centerlines(
    path: Path,
    bounds_4326: Bounds,
    fetch: Callable[[str], dict[str, Any]] | None = None,
) -> Path:
    """Make the per-city OSM centerline cache cover ``bounds_4326``; return ``path``.

    The cache carries two extra top-level members: ``schema_version`` and ``fetched_bboxes``,
    every bbox ever fetched into it. A volume whose margin-expanded bounds sit inside a recorded
    bbox runs with zero network; an uncovered one costs exactly ONE Overpass query, merged in
    source order with any way it returned taken from the new response, so covered reruns are
    byte-stable. Since a cached-forever file must never be silently wrong, a failed response or
    a fetch yielding no streets raises instead of caching, while an unparseable or outdated file
    is rebuilt with a warning. One lock holds read, coverage check, fetch, merge and publish.
    """
    with _locked_city_cache(path):
        # Reload after acquiring the lock: another volume may have published
        # new coverage while this caller waited.
        features, fetched, version = _load_city_cache(path)
        if features and version != CENTERLINE_SCHEMA_VERSION:
            # An older file holds the same streets, only uncut; the graph is
            # recoverable from its own geometry, so upgrade in place. Refetching
            # would be both network a cached-forever file has already spent and
            # a way to strand an offline machine's prefetched copy.
            logger.warning(
                "%s: OSM cache is schema v%s; re-deriving the junction graph in place",
                path,
                version,
            )
            features = cut_ways_into_features(ways_from_features(features))
            _publish_city_cache(path, features, fetched)
        needed = expand_bounds(bounds_4326, REQUIRED_MARGIN_DEG)
        if any(len(b) == 4 and _covers((b[0], b[1], b[2], b[3]), needed) for b in fetched):
            return path
        fetch_bbox = expand_bounds(bounds_4326, FETCH_MARGIN_DEG)
        data = check_overpass_response((fetch or fetch_overpass)(overpass_query(fetch_bbox)))
        fetched_ways = _usable_ways(data)
        if not features and not fetched_ways:
            # a city volume bbox with zero named streets is a failed fetch in
            # every realistic case; caching it would permanently blind the city
            raise OSMFetchError(
                f"Overpass returned no usable named-street ways for bbox {list(fetch_bbox)}; "
                "refusing to cache an empty street set (prefetch manually via "
                "scripts/fetch_osm_centerlines.py if this area is genuinely empty)"
            )
        # Merge and re-cut as WHOLE ways. Overpass returns a way's full geometry
        # whenever it clips the bbox, so a long way spanning two fetches would
        # otherwise keep whichever response's cuts landed last and lose every
        # junction the other one saw — permanently, since coverage is recorded.
        refetched = {
            props["osm_id"] for props, _ in fetched_ways if props.get("osm_id") is not None
        }
        kept = [w for w in ways_from_features(features) if w[0].get("osm_id") not in refetched]
        merged = cut_ways_into_features([*kept, *fetched_ways])
        logger.info(
            "%s: fetched bbox %s -> %d ways (%d total, %d features)",
            path.name,
            [round(b, 4) for b in fetch_bbox],
            len(fetched_ways),
            len(kept) + len(fetched_ways),
            len(merged),
        )
        _publish_city_cache(path, merged, [*fetched, list(fetch_bbox)])
    return path
