"""OSM centerline support (autogeoref.osm): name shim + Overpass converter.

No network anywhere here — the converter runs on a small hand-checked
Overpass sample committed under tests/data/ (a test asset, NOT a frozen
fixture; the manifest-pinned fixture tree is untouched).
"""

import json
import threading
from pathlib import Path

import pytest

from autogeoref.addresses import AddressNumeral, match_address
from autogeoref.centerlines import CenterlineIndex
from autogeoref.junction_snap import world_from_centerlines
from autogeoref.names import normalize
from autogeoref.osm import (
    CENTERLINE_SCHEMA_VERSION,
    EXCLUDED_HIGHWAYS,
    centerline_geojson,
    node_key,
    overpass_query,
    overpass_to_centerline_features,
    split_osm_name,
)

SAMPLE = Path(__file__).parent / "data" / "osm_overpass_sample.json"


# Real OSM Chicago `name` tags (cross-checked against the cached ref-volume
# fetch, cache/osm-centerlines-chicago-refvol*.geojson) -> the official
# Chicago (street_nam, street_typ) split the matcher was validated on.
OSM_NAME_SPLITS = [
    ("West Madison Street", "MADISON", "ST"),
    ("North State Street", "STATE", "ST"),
    ("South Michigan Avenue", "MICHIGAN", "AVE"),
    ("West Garfield Boulevard", "GARFIELD", "BLVD"),
    ("North Lake Shore Drive", "LAKE SHORE", "DR"),
    ("South Dr. Martin Luther King Jr. Drive", "DR MARTIN LUTHER KING JR", "DR"),
    ("West 31st Street", "31ST", "ST"),
    ("East 31st Place", "31ST", "PL"),
    ("West Cermak Road", "CERMAK", "RD"),
    ("North Milwaukee Avenue", "MILWAUKEE", "AVE"),
    # the direction word IS the name (official street_nam='NORTH')
    ("West North Avenue", "NORTH", "AVE"),
    ("North Broadway", "BROADWAY", ""),
    ("West Fullerton Parkway", "FULLERTON", "PKWY"),
    ("North Clark Street", "CLARK", "ST"),
    ("South Wabash Avenue", "WABASH", "AVE"),
    ("West St. Paul Avenue", "ST PAUL", "AVE"),
    ("North Hermitage Avenue", "HERMITAGE", "AVE"),
    ("South Ashland Avenue", "ASHLAND", "AVE"),
    ("West Diversey Parkway", "DIVERSEY", "PKWY"),
    ("East Wacker Drive", "WACKER", "DR"),
    ("West Roosevelt Road", "ROOSEVELT", "RD"),
    ("North Damen Avenue", "DAMEN", "AVE"),
    ("South Archer Avenue", "ARCHER", "AVE"),
    ("North Elston Avenue", "ELSTON", "AVE"),
    ("West Belmont Avenue", "BELMONT", "AVE"),
    ("East 47th Street", "47TH", "ST"),
    ("South Prairie Avenue", "PRAIRIE", "AVE"),
    ("West Schiller Street", "SCHILLER", "ST"),
    ("North Clybourn Avenue", "CLYBOURN", "AVE"),
    ("West Terra Cotta Place", "TERRA COTTA", "PL"),
    # exactly ONE leading direction strips (single pre_dir field):
    ("South South Shore Drive", "SOUTH SHORE", "DR"),
    # no recognized trailing type token: every token kept
    ("Avenue L", "AVENUE L", ""),
    ("South Avenue L", "AVENUE L", ""),
    ("North Lincoln Avenue", "LINCOLN", "AVE"),
    ("West Armitage Avenue", "ARMITAGE", "AVE"),
    # -- from the cached ref-volume fetch (all verbatim OSM name tags) --
    ("South Cottage Grove Avenue", "COTTAGE GROVE", "AVE"),
    ("South Stony Island Avenue", "STONY ISLAND", "AVE"),
    ("South Halsted Street", "HALSTED", "ST"),
    ("South LaSalle Street", "LASALLE", "ST"),
    ("South Hyde Park Boulevard", "HYDE PARK", "BLVD"),
    ("West Winneconna Parkway", "WINNECONNA", "PKWY"),
    ("East Midway Plaisance", "MIDWAY PLAISANCE", ""),
    ("Chicago Skyway", "CHICAGO SKYWAY", ""),
    ("East 71st Street", "71ST", "ST"),
    ("West 23rd Place", "23RD", "PL"),
    ("South East End Avenue", "EAST END", "AVE"),
    ("South South Chicago Avenue", "SOUTH CHICAGO", "AVE"),
    ("Stevenson Expressway", "STEVENSON", "EXPY"),
    ("South DuSable Lake Shore Drive", "DUSABLE LAKE SHORE", "DR"),
    # honorific/saint tokens: the official schema abbreviates
    # (ST LAWRENCE / DR MARTIN LUTHER KING JR in the official data)
    ("South Saint Lawrence Avenue", "ST LAWRENCE", "AVE"),
    (
        "South Doctor Martin Luther King Junior Drive",
        "DR MARTIN LUTHER KING JR",
        "DR",
    ),
    # parenthetical variants collapse onto the plain name's split
    ("Dan Ryan Expressway (Local)", "DAN RYAN", "EXPY"),
    # trailing direction after the type = official suf_dir, not the name
    ("South Lake Park Avenue West", "LAKE PARK", "AVE"),
    ("South Park Shore East Court", "PARK SHORE EAST", "CT"),
]


@pytest.mark.parametrize(("osm_name", "nam", "typ"), OSM_NAME_SPLITS)
def test_split_osm_name(osm_name: str, nam: str, typ: str) -> None:
    assert split_osm_name(osm_name) == (nam, typ)


# What the matcher actually joins: a printed Sanborn-style label (annotation
# side, through names.normalize) against the index key built from the shim's
# street_nam/street_typ (CenterlineIndex side). Both routes must converge.
PRINTED_VS_OSM = [
    ("W. MADISON ST.", "West Madison Street"),
    ("GARFIELD BOUL.", "West Garfield Boulevard"),
    ("31ST PL.", "East 31st Place"),
    ("31ST ST", "West 31st Street"),
    ("ST. LAWRENCE AVE.", "South Saint Lawrence Avenue"),
    ("COTTAGE GROVE AVE", "South Cottage Grove Avenue"),
    ("HALSTED ST", "South Halsted Street"),
    ("STONY ISLAND AV", "South Stony Island Avenue"),
    ("N. CLARK ST", "North Clark Street"),
]


@pytest.mark.parametrize(("printed", "osm_name"), PRINTED_VS_OSM)
def test_split_matches_annotation_side_normalization(printed: str, osm_name: str) -> None:
    from autogeoref.names import _NUMERIC_ORDINAL

    nam, typ = split_osm_name(osm_name)
    key = normalize(nam)
    # CenterlineIndex.__init__'s numbered PLACE/COURT twin rule
    if typ in {"PL", "CT"} and _NUMERIC_ORDINAL.match(key):
        key = f"{key} {typ}"
    assert key == normalize(printed)


def test_overpass_query_shape() -> None:
    q = overpass_query((-87.7, 41.8, -87.6, 41.9))
    assert q.startswith("[out:json]")
    # Overpass bbox order is south,west,north,east
    assert "(41.8,-87.7,41.9,-87.6)" in q
    for cls in EXCLUDED_HIGHWAYS:
        assert cls in q
    assert 'way["highway"]["name"]' in q
    assert "out geom;" in q


def test_converter_sample() -> None:
    data = json.loads(SAMPLE.read_text())
    feats = overpass_to_centerline_features(data)
    # 7 usable ways; the named footway, unnamed way, area=yes plaza,
    # one-point way, and node element are all skipped. Madison and State are
    # each cut into two at the vertex they share, so they emit two features.
    assert [f["properties"]["osm_id"] for f in feats] == [
        101,
        101,
        102,
        102,
        103,
        104,
        105,
        106,
        107,
    ]
    first = feats[0]
    assert first["properties"]["street_nam"] == "MADISON"
    assert first["properties"]["street_typ"] == "ST"
    assert first["properties"]["osm_name"] == "West Madison Street"
    assert first["geometry"]["type"] == "LineString"
    assert first["geometry"]["coordinates"][0] == [-87.66, 41.8819]
    by_id = {f["properties"]["osm_id"]: f for f in feats}
    assert by_id[104]["properties"]["street_typ"] == "PL"
    fc = centerline_geojson(data)
    assert fc["type"] == "FeatureCollection"
    assert fc["schema_version"] == CENTERLINE_SCHEMA_VERSION
    assert len(fc["features"]) == len(feats)


def test_converter_derives_the_junction_graph() -> None:
    """Overpass carries no node ids; a shared vertex becomes one."""
    feats = overpass_to_centerline_features(json.loads(SAMPLE.read_text()))
    crossing = node_key(-87.6281, 41.8819)
    # Madison is cut at the crossing: two pieces meeting there, and the piece
    # geometry still runs from the way's own start to the cut.
    madison = [f for f in feats if f["properties"]["osm_id"] == 101]
    assert [(f["properties"]["fnode_id"], f["properties"]["tnode_id"]) for f in madison] == [
        (node_key(-87.66, 41.8819), crossing),
        (crossing, node_key(-87.62, 41.8819)),
    ]
    assert madison[0]["geometry"]["coordinates"] == [[-87.66, 41.8819], [-87.6281, 41.8819]]
    # ...and the graph the junction channel reads has exactly that one node,
    # degree 4 (two Madison pieces + two State pieces).
    world = world_from_centerlines(feats)
    assert len(world.nodes_3857) == 1
    assert world.node_degrees.tolist() == [4]


def test_the_cut_uses_the_shared_street_key_not_the_bare_name() -> None:
    """A numbered PLACE twin is a different street; a type spelling is not.

    ``street_nam`` alone cannot tell them apart — `31ST PL` and `31ST ST` both
    reduce to `31ST` — so the cut asks `centerline_key`, the one key rule the
    index and the address channel already share.
    """

    def crossing(name_a: str, name_b: str) -> int:
        data = {
            "elements": [
                {
                    "type": "way",
                    "id": 1,
                    "tags": {"highway": "residential", "name": name_a},
                    "geometry": [
                        {"lat": 41.838, "lon": -87.62},
                        {"lat": 41.838, "lon": -87.61},
                        {"lat": 41.838, "lon": -87.60},
                    ],
                },
                {
                    "type": "way",
                    "id": 2,
                    "tags": {"highway": "residential", "name": name_b},
                    "geometry": [
                        {"lat": 41.837, "lon": -87.61},
                        {"lat": 41.838, "lon": -87.61},
                    ],
                },
            ]
        }
        return len(world_from_centerlines(overpass_to_centerline_features(data)).nodes_3857)

    # different streets that happen to share a numbered name -> a real junction
    assert crossing("East 31st Street", "East 31st Place") == 1
    # one street whose type is spelled two ways -> not a junction
    assert crossing("Brentwood Road", "Brentwood Drive") == 0


def test_same_named_ways_meeting_end_to_end_are_not_a_junction() -> None:
    """One street continuing through a vertex must not become a degree-3 node."""
    data = {
        "elements": [
            {
                "type": "way",
                "id": 1,
                "tags": {"highway": "residential", "name": "Walkup Avenue"},
                "geometry": [
                    {"lat": 42.24, "lon": -88.33},
                    {"lat": 42.24, "lon": -88.32},
                    {"lat": 42.24, "lon": -88.31},
                ],
            },
            {
                "type": "way",
                "id": 2,
                "tags": {"highway": "residential", "name": "Walkup Ave"},
                "geometry": [
                    {"lat": 42.24, "lon": -88.32},
                    {"lat": 42.25, "lon": -88.32},
                ],
            },
        ]
    }
    feats = overpass_to_centerline_features(data)
    assert [f["properties"]["osm_id"] for f in feats] == [1, 2]  # no cut
    assert len(world_from_centerlines(feats).nodes_3857) == 0


def test_centerline_index_roundtrip() -> None:
    """Converted features drop straight into CenterlineIndex (default properties)."""
    feats = overpass_to_centerline_features(json.loads(SAMPLE.read_text()))
    index = CenterlineIndex(feats)
    # the PLACE/COURT twin contract: 31st St and 31st Pl index under
    # distinct keys (different parallel streets half a block apart)
    assert "31ST" in index.by_name
    assert "31ST PL" in index.by_name
    assert index.by_name["31ST"] is not index.by_name["31ST PL"]
    # a real crossing resolves geometrically, keyed by annotation-side labels
    pts = index.intersections(normalize("W. MADISON ST."), normalize("STATE ST"))
    assert len(pts) == 1
    assert pts[0] == pytest.approx((-87.6281, 41.8819))


def test_address_channel_abstains_on_osm() -> None:
    """OSM ways carry no address-range fields -> the channel abstains (no crash)."""
    feats = overpass_to_centerline_features(json.loads(SAMPLE.read_text()))
    numeral = AddressNumeral(value=3117, bbox=(0.0, 0.0, 10.0, 10.0), street_hint=None)
    assert match_address(numeral, "W MADISON ST", feats) == []


# ---------------------------------------------------------------------------
# OSM as the default centerline source
# ---------------------------------------------------------------------------


def test_overpass_endpoint_conduct() -> None:
    """GET-only https to overpass-api.de — the URL is a constant, not a knob."""
    from autogeoref.osm import OVERPASS_URL

    assert OVERPASS_URL == "https://overpass-api.de/api/interpreter"


def test_config_without_centerlines_defaults_to_osm(tmp_path: Path) -> None:
    from autogeoref.config.load import load_city_config

    cfg_path = tmp_path / "city.toml"
    cfg_path.write_text('[city]\nname = "Vireo City"\naliases_dir = "aliases"\n')
    cfg = load_city_config(cfg_path)
    assert cfg.centerlines_from_osm is True
    assert cfg.centerlines_path == tmp_path / "cache" / "osm-centerlines-vireo-city.geojson"

    # osm_cache_dir overrides the cache location
    cfg_path.write_text(
        '[city]\nname = "Vireo City"\naliases_dir = "aliases"\nosm_cache_dir = "elsewhere"\n'
    )
    assert load_city_config(cfg_path).centerlines_path.parent == tmp_path / "elsewhere"

    # a configured centerlines file is BYO: used verbatim, flag off
    cfg_path.write_text(
        '[city]\nname = "Vireo City"\naliases_dir = "aliases"\ncenterlines = "own.geojson"\n'
    )
    cfg = load_city_config(cfg_path)
    assert cfg.centerlines_from_osm is False
    assert cfg.centerlines_path == tmp_path / "own.geojson"


def test_ensure_city_centerlines_fetch_cache_merge(tmp_path: Path) -> None:
    from autogeoref.osm import (
        FETCH_MARGIN_DEG,
        ensure_city_centerlines,
        expand_bounds,
    )

    sample: dict[str, object] = json.loads(SAMPLE.read_text())
    calls: list[str] = []

    def fake_fetch(query: str) -> dict[str, object]:
        calls.append(query)
        return sample

    def never_fetch(query: str) -> dict[str, object]:
        raise AssertionError("covered bounds must not fetch")

    path = tmp_path / "cache" / "osm-centerlines-x.geojson"
    bounds = (-87.66, 41.83, -87.60, 41.89)

    # first run: one fetch, matcher-shape file, coverage recorded
    ensure_city_centerlines(path, bounds, fetch=fake_fetch)
    assert len(calls) == 1
    fc = json.loads(path.read_text())
    assert fc["schema_version"] == CENTERLINE_SCHEMA_VERSION
    assert fc["fetched_bboxes"] == [list(expand_bounds(bounds, FETCH_MARGIN_DEG))]
    assert {f["properties"]["street_nam"] for f in fc["features"]} >= {"MADISON", "31ST"}
    # the file drops straight into CenterlineIndex
    assert "MADISON" in CenterlineIndex(fc["features"]).by_name

    # covered rerun (exact and slightly-shifted-but-inside): zero network
    ensure_city_centerlines(path, bounds, fetch=never_fetch)
    shifted = (bounds[0] + 0.005, bounds[1] + 0.005, bounds[2], bounds[3])
    ensure_city_centerlines(path, shifted, fetch=never_fetch)

    # uncovered bounds: exactly one more fetch, merged by osm_id
    n_before = len(fc["features"])
    far = (-87.66, 41.70, -87.60, 41.73)
    ensure_city_centerlines(path, far, fetch=fake_fetch)
    assert len(calls) == 2
    fc2 = json.loads(path.read_text())
    assert len(fc2["fetched_bboxes"]) == 2
    # the same sample came back: every way re-taken from it, nothing duplicated.
    # The merge key is the WAY, not the feature — a cut way's pieces share an id.
    assert len(fc2["features"]) == n_before
    assert [f["properties"]["osm_id"] for f in fc2["features"]] == [
        f["properties"]["osm_id"] for f in fc["features"]
    ]


def test_ensure_never_caches_a_failed_or_empty_fetch(tmp_path: Path) -> None:
    """Overpass reports timeouts as HTTP 200 + a `remark` with truncated
    elements; caching that (or a zero-street result) would poison a
    cached-forever file. Both must raise and leave no file behind."""
    from autogeoref.osm import OSMFetchError, ensure_city_centerlines

    path = tmp_path / "osm.geojson"
    bounds = (-87.66, 41.83, -87.60, 41.89)

    remark = {"remark": "runtime error: Query timed out in 'query' at line 2", "elements": []}
    with pytest.raises(OSMFetchError, match="remark"):
        ensure_city_centerlines(path, bounds, fetch=lambda _q: remark)
    assert not path.exists()

    with pytest.raises(OSMFetchError, match="no usable named-street ways"):
        ensure_city_centerlines(path, bounds, fetch=lambda _q: {"elements": []})
    assert not path.exists()


def test_ensure_rebuilds_a_corrupt_cache(tmp_path: Path) -> None:
    """A torn write must not brick every future run of the city."""
    from autogeoref.osm import ensure_city_centerlines

    sample: dict[str, object] = json.loads(SAMPLE.read_text())
    path = tmp_path / "osm.geojson"
    path.write_text('{"type": "FeatureCollection", "feat')  # torn write
    bounds = (-87.66, 41.83, -87.60, 41.89)
    calls: list[str] = []

    def fake_fetch(query: str) -> dict[str, object]:
        calls.append(query)
        return sample

    ensure_city_centerlines(path, bounds, fetch=fake_fetch)
    assert len(calls) == 1
    fc = json.loads(path.read_text())  # parseable again
    assert fc["features"] and len(fc["fetched_bboxes"]) == 1


def test_merge_retakes_a_way_the_new_response_also_returned(tmp_path: Path) -> None:
    """Node ids are per-response, so a stale copy of a re-fetched way must not survive.

    Keeping it would leave that way uncut where the ways cut against it in the
    new response now meet it — a junction silently missing from the graph.
    """
    from autogeoref.osm import ensure_city_centerlines

    def way(way_id: int, name: str, coords: list[tuple[float, float]]) -> dict[str, object]:
        return {
            "type": "way",
            "id": way_id,
            "tags": {"highway": "residential", "name": name},
            "geometry": [{"lon": x, "lat": y} for x, y in coords],
        }

    # first fetch sees Main alone; the second also sees the Oak that crosses it
    main = way(1, "Main Street", [(-88.34, 42.24), (-88.32, 42.24), (-88.30, 42.24)])
    oak = way(2, "Oak Street", [(-88.32, 42.23), (-88.32, 42.24), (-88.32, 42.25)])
    responses = [{"elements": [main]}, {"elements": [main, oak]}]
    path = tmp_path / "osm.geojson"

    ensure_city_centerlines(path, (-88.33, 42.235, -88.31, 42.245), fetch=lambda _q: responses[0])
    first = json.loads(path.read_text())["features"]
    assert [f["properties"]["osm_id"] for f in first] == [1]  # nothing to cut against
    assert len(world_from_centerlines(first).nodes_3857) == 0

    ensure_city_centerlines(path, (-88.36, 42.20, -88.28, 42.28), fetch=lambda _q: responses[1])
    merged = json.loads(path.read_text())["features"]
    # Main was re-taken and is now cut, so the crossing is a real degree-4 node
    assert [f["properties"]["osm_id"] for f in merged] == [1, 1, 2, 2]
    world = world_from_centerlines(merged)
    assert len(world.nodes_3857) == 1
    assert world.node_degrees.tolist() == [4]


def test_merge_keeps_junctions_the_new_response_could_not_see(tmp_path: Path) -> None:
    """A later, narrower fetch must not cost the earlier one its junctions.

    Overpass returns a way's FULL geometry whenever it clips the bbox, so a long
    way spanning two fetches comes back in both. Cutting it against only the
    second response would drop every crossing the first one saw — permanently,
    because the first bbox stays recorded as covered.
    """
    from autogeoref.osm import ensure_city_centerlines

    def way(way_id: int, name: str, coords: list[tuple[float, float]]) -> dict[str, object]:
        return {
            "type": "way",
            "id": way_id,
            "tags": {"highway": "residential", "name": name},
            "geometry": [{"lon": x, "lat": y} for x, y in coords],
        }

    # Main runs the length of the map and comes back in BOTH responses; Oak
    # crosses it in the north, Elm in the south, and neither fetch sees both.
    main = way(1, "Main Street", [(-88.34, 42.24), (-88.32, 42.24), (-88.30, 42.24)])
    oak = way(2, "Oak Street", [(-88.32, 42.24), (-88.32, 42.25)])
    elm = way(3, "Elm Street", [(-88.30, 42.24), (-88.30, 42.23)])
    path = tmp_path / "osm.geojson"

    ensure_city_centerlines(
        path, (-88.33, 42.238, -88.31, 42.246), fetch=lambda _q: {"elements": [main, oak]}
    )
    assert len(world_from_centerlines(json.loads(path.read_text())["features"]).nodes_3857) == 1

    ensure_city_centerlines(
        path, (-88.31, 42.228, -88.29, 42.236), fetch=lambda _q: {"elements": [main, elm]}
    )
    world = world_from_centerlines(json.loads(path.read_text())["features"])
    assert len(world.nodes_3857) == 2  # the Oak crossing survived the second fetch


def test_ensure_upgrades_a_pre_schema_cache_without_touching_the_network(tmp_path: Path) -> None:
    """A cache with no node ids holds the same streets; the graph is in its geometry.

    Refetching would spend network a cached-forever file has already spent, and
    would strand a machine that got its copy from an offline prefetch.
    """
    from autogeoref.osm import ensure_city_centerlines

    def never_fetch(query: str) -> dict[str, object]:
        raise AssertionError("an in-place upgrade must not fetch")

    path = tmp_path / "osm.geojson"
    bounds = (-87.66, 41.83, -87.60, 41.89)
    # a pre-schema file: one feature per way, no node ids, coverage recorded
    old = {
        "type": "FeatureCollection",
        "fetched_bboxes": [[-88.0, 41.0, -87.0, 42.0]],
        "features": [
            {
                "type": "Feature",
                "properties": {"street_nam": "MADISON", "street_typ": "ST", "osm_id": 101},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-87.66, 41.88], [-87.63, 41.88], [-87.60, 41.88]],
                },
            },
            {
                "type": "Feature",
                "properties": {"street_nam": "STATE", "street_typ": "ST", "osm_id": 102},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-87.63, 41.87], [-87.63, 41.88], [-87.63, 41.89]],
                },
            },
        ],
    }
    path.write_text(json.dumps(old))

    ensure_city_centerlines(path, bounds, fetch=never_fetch)

    fc = json.loads(path.read_text())
    assert fc["schema_version"] == CENTERLINE_SCHEMA_VERSION
    assert fc["fetched_bboxes"] == [[-88.0, 41.0, -87.0, 42.0]]  # coverage survives
    assert all("fnode_id" in f["properties"] for f in fc["features"])
    world = world_from_centerlines(fc["features"])
    assert len(world.nodes_3857) == 1  # the crossing the uncut file could not express
    assert world.node_degrees.tolist() == [4]


def test_concurrent_ensure_preserves_each_volume_coverage(tmp_path: Path) -> None:
    """Concurrent volumes must reload and merge, not overwrite each other's cache."""
    from autogeoref.osm import FETCH_MARGIN_DEG, ensure_city_centerlines, expand_bounds

    path = tmp_path / "cache" / "osm.geojson"
    bounds = [(-87.66, 41.83, -87.60, 41.89), (-87.66, 41.70, -87.60, 41.73)]
    sample: dict[str, object] = json.loads(SAMPLE.read_text())
    start = threading.Barrier(2)
    errors: list[BaseException] = []

    def fetch(_: str) -> dict[str, object]:
        return sample

    def write_cache(volume_bounds: tuple[float, float, float, float]) -> None:
        try:
            start.wait()
            ensure_city_centerlines(path, volume_bounds, fetch=fetch)
        except BaseException as exc:  # communicate failures from the worker thread
            errors.append(exc)

    threads = [
        threading.Thread(target=write_cache, args=(volume_bounds,)) for volume_bounds in bounds
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert not errors
    cache = json.loads(path.read_text())
    assert {tuple(bbox) for bbox in cache["fetched_bboxes"]} == {
        expand_bounds(bound, FETCH_MARGIN_DEG) for bound in bounds
    }
