"""The viewer's address search: which geocoder, and whether to search at all.

Two tiers for one rule. ``viewer/lib.js``'s ``chooseGeocoder`` is executed in
node — a source grep would pass on a bug — and the page is then loaded by a
browser under a public-looking hostname, which is the only way to see what a
deployed visitor is actually offered.

Mapbox on a configured token; Nominatim only on a dev host, because OSMF's
public instance forbids a deployment's search traffic; nothing at all on a
public host with no token.

Needs ``node`` and a headless browser on PATH; without them these skip.
"""

from __future__ import annotations

import json
from pathlib import Path

from js_support import viewer
from viewer_browser_support import PUBLIC_HOST, captured_json
from viewer_browser_support import load as _load
from viewer_browser_support import manifest_for as _manifest

#: A configured city: the suffix appended to a bare query, and the bias box as
#: west,south,east,north.
GEO = {"suffix": ", Springfield, IL", "bbox": [-90.0, 39.0, -89.0, 40.0]}

#: Shaped like a real public token: `pk.` and a base64url payload.
TOKEN = "pk.eyJ1IjoidGVzdCJ9.abc-DEF_123"


def js(*values: object) -> str:
    """``values`` as a comma-joined JS argument list."""
    return ", ".join(json.dumps(v) for v in values)


def test_a_configured_token_sends_every_search_to_mapbox() -> None:
    """Token first, on any host: a deployment that owns a geocoder uses it.

    Note what a developer gets from this on localhost — a token URL-restricted
    to production is REJECTED there, so the local box reports the geocoder as
    unavailable. That is the honest answer, and why the failed-request note is
    kept distinct from "No match".
    """
    for host in ("atlas.example.com", "localhost"):
        got = viewer(f"L.chooseGeocoder('12 Main St', {js(GEO)}, {js(TOKEN)}, {js(host)})")
        assert got["provider"] == "mapbox", host
        assert got["url"].startswith("https://api.mapbox.com/search/geocode/v6/forward?q=")
        # the configured suffix and bias box ride the query
        assert "q=12%20Main%20St%2C%20Springfield%2C%20IL&" in got["url"]
        assert "&bbox=-90,39,-89,40" in got["url"]
        assert f"access_token={TOKEN}" in got["url"]


def test_the_mapbox_endpoint_is_v6_and_not_the_retired_v5_one() -> None:
    """The v5 `mapbox.places` endpoint is legacy, and the two answer in
    different shapes. Asserted as an ABSENCE as well as a presence: a revert to
    v5 that left `geocodeHit` reading v6 keys would send a working request and
    report "No match" for every valid address, which no other test here would
    catch."""
    got = viewer(f"L.chooseGeocoder('12 Main St', {js(GEO)}, {js(TOKEN)}, 'atlas.example.com')")
    assert "geocoding/v5" not in got["url"]
    assert "mapbox.places" not in got["url"]


def test_a_public_host_with_no_token_sends_no_search_at_all() -> None:
    """The release blocker this rule exists for. OSMF's public instance forbids
    a deployment's search traffic and the failure mode is a block — a search box
    that then answers "No match" forever, blaming the atlas. So the fallback is
    a dev-host convenience and nothing else."""
    public = viewer(f"L.chooseGeocoder('12 Main St', {js(GEO)}, '', 'atlas.example.com')")
    assert public["provider"] is None
    assert "search" in public["reason"].lower()
    for host in ("localhost", "127.0.0.1", "[::1]", ""):
        local = viewer(f"L.chooseGeocoder('12 Main St', {js(GEO)}, null, {js(host)})")
        assert local["provider"] == "nominatim", host
        assert local["url"].startswith("https://nominatim.openstreetmap.org/search")
        # viewbox is west,north,east,south — the other order boxes the ocean
        assert "&bounded=1&viewbox=-90,40,-89,39" in local["url"]


def test_an_unconfigured_geocoder_block_still_geocodes_as_typed() -> None:
    """A city that configures no suffix or bias box gets a plain query rather
    than `undefined` spliced into the URL."""
    got = viewer("L.chooseGeocoder('12 Main St', null, null, 'localhost')")
    assert got["url"].endswith("&q=12%20Main%20St")
    assert "viewbox" not in got["url"]


def test_each_providers_answer_is_read_in_its_own_shape() -> None:
    mapbox = {
        "features": [
            {
                "geometry": {"type": "Point", "coordinates": [-89.65, 39.8]},
                "properties": {
                    "full_address": "12 Main St, Springfield",
                    "name": "12 Main St",
                },
            }
        ]
    }
    assert viewer(f"L.geocodeHit('mapbox', {js(mapbox)})") == {
        "lngLat": [-89.65, 39.8],
        "name": "12 Main St, Springfield",  # the full address, not the short name
    }
    nominatim = [{"lon": "-89.65", "lat": "39.8", "display_name": "12, Main St, Springfield, IL"}]
    assert viewer(f"L.geocodeHit('nominatim', {js(nominatim)})") == {
        "lngLat": [-89.65, 39.8],
        "name": "12, Main St, Springfield",  # first three parts only
    }


def test_a_malformed_geocoder_answer_reads_as_no_match_rather_than_throwing() -> None:
    """A blocked or truncated response arrives as a body that is not the
    documented shape. Throwing there leaves the note stuck on "searching…" with
    the reason only in the console."""
    for payload in ("null", "{}", "[]", '{"features": []}', '[{"lon": "x", "lat": "y"}]'):
        assert viewer(f"L.geocodeHit('mapbox', {payload})") is None, payload
        assert viewer(f"L.geocodeHit('nominatim', {payload})") is None, payload


def test_a_v5_shaped_answer_no_longer_reads_as_a_hit() -> None:
    """The half of the v6 move that a shape test cannot state on its own. A
    v5 body carries the point in `center` and the label in `place_name`; if
    either were still read, a half-finished revert would look like it worked."""
    v5 = {"features": [{"center": [-89.65, 39.8], "place_name": "12 Main St, Springfield"}]}
    assert viewer(f"L.geocodeHit('mapbox', {js(v5)})") is None


#: The search box and its note, as the page left them for a reader.
SEARCH_STATE = (
    'JSON.stringify({note: document.getElementById("search-note").textContent, '
    'disabled: document.getElementById("address").disabled})'
)


def test_a_public_host_with_no_token_turns_the_search_box_off_and_says_why(
    viewer_bundle: Path,
) -> None:
    """The same rule, on a page served as a public host.

    A deployed bundle with no token used to fall through to OSMF's instance for
    every visitor's search. The box has to be off and say so, and the reader has
    to learn that without typing an address first.
    """
    page = _load(
        viewer_bundle,
        _manifest("/base/tile.png"),
        host_alias=PUBLIC_HOST,
        capture={"search": SEARCH_STATE},
    )
    state = captured_json(page, "search")
    assert state["disabled"] is True, "a public page offered a search it cannot run"
    assert "search" in state["note"].lower(), state["note"]
    assert page.console == (), f"the page complained: {page.console}"


def test_a_dev_host_keeps_the_search_box_usable(viewer_bundle: Path) -> None:
    """The other half: the fallback is still there where it is in policy, so
    the gate cannot be satisfied by disabling search everywhere."""
    page = _load(viewer_bundle, _manifest("/base/tile.png"), capture={"search": SEARCH_STATE})
    state = captured_json(page, "search")
    assert state["disabled"] is False
    assert state["note"] == ""


def test_a_deployed_token_leaves_the_search_box_on_for_a_public_visitor(
    viewer_bundle: Path,
) -> None:
    """The configuration a release actually ships, which neither test above
    covers: public host AND a token. The bundle's own `config.js` is what turns
    search back on, so this is also the only check that the file the page loads
    reaches the decision at all."""
    config = viewer_bundle / "viewer" / "config.js"
    was = config.read_text(encoding="utf-8")
    config.write_text(f'window.MAPBOX_TOKEN = "{TOKEN}";\n', encoding="utf-8")
    try:
        page = _load(
            viewer_bundle,
            _manifest("/base/tile.png"),
            host_alias=PUBLIC_HOST,
            capture={"search": SEARCH_STATE, "token": "JSON.stringify(window.MAPBOX_TOKEN)"},
        )
    finally:
        config.write_text(was, encoding="utf-8")
    assert captured_json(page, "token") == TOKEN, "config.js never reached the page"
    state = captured_json(page, "search")
    assert state["disabled"] is False, "a token was deployed and search stayed off"
    assert state["note"] == ""
    assert page.console == (), f"the page complained: {page.console}"


def test_a_null_or_blank_coordinate_is_not_the_gulf_of_guinea() -> None:
    """`Number(null)` and `Number("")` are both 0, and 0,0 is a real place off
    west Africa. A body with a null coordinate must read as no match, not fly
    the reader into the ocean."""
    for coords in ("[null, null]", '["", ""]', "[null, 39.8]", "[{}, []]"):
        payload = f'{{"features": [{{"geometry": {{"coordinates": {coords}}}}}]}}'
        assert viewer(f"L.geocodeHit('mapbox', {payload})") is None, coords
    for pair in ('{"lon": null, "lat": null}', '{"lon": "", "lat": ""}'):
        assert viewer(f"L.geocodeHit('nominatim', [{pair}])") is None, pair
    # a genuine zero is still a hit: null island is not the same as "no answer"
    assert viewer('L.geocodeHit("mapbox", {"features":[{"geometry":{"coordinates":[0,0]}}]})') == {
        "lngLat": [0, 0],
        "name": "",
    }
