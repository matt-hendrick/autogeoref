"""Which city a page load resolves to, and what a dateless layer draws.

Two features of one shape: the page is told which manifest to load, and a
volume the catalog gave no year still becomes a layer. Both were invisible to
every assertion available before — the tier-3 failure they fix was a page
headed with one city's name drawing another city's sheets, and a served
archive the page never added a source for.

The pure decisions are run in node; what actually drew needs the browser.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from browser import load_page, serve
from js_support import viewer
from viewer_browser_support import (
    ATLAS_ARCHIVE_BESIDE_PAGE,
    CHOOSING,
    DREW,
    PROBE_CENTER,
    PROBE_SLUG,
    SOURCES,
    captured_json,
    city_index,
)
from viewer_browser_support import links as _links
from viewer_browser_support import load as _load
from viewer_browser_support import manifest_for as _manifest
from viewer_browser_support import publish as _publish
from viewer_browser_support import visibility as _visibility

#: The characters the page renders in an era label \u2014 a merged decade run is
#: joined with an en dash, separate runs with a middle dot. Escapes, so an
#: ASCII stand-in cannot creep into the expected value.
EN_DASH = "\u2013"
MIDDLE_DOT = " \u00b7 "


def test_a_volume_with_no_era_joins_one_undated_group() -> None:
    """A catalog-less first publish writes `year: null`, so `era` is null. The
    layer is still a layer: it gets a group like any other, and every caller
    downstream keys on a string that is now always present."""
    assert viewer("L.eraOf({id: 'v', era: '1890s'})") == "1890s"
    assert viewer("L.eraOf({id: 'v', era: null})") == "undated"
    assert viewer("L.eraOf({id: 'v'})") == "undated"
    # "" is what a manifest writes for a year it has but cannot bucket; it is
    # no more a group name than null is
    assert viewer("L.eraOf({id: 'v', era: ''})") == "undated"
    assert viewer("L.UNDATED") == "undated"


def test_undated_sorts_last_among_dated_eras() -> None:
    """`undated` parses to NaN, so leaving it to the year comparison would put
    it wherever the engine's sort happened to leave it. A dated corpus keeps
    its exact order and the odd one out does not head the chip row."""
    assert viewer("['1890s', 'undated', '1950', '1910s'].sort(L.compareErasNewestFirst)") == [
        "1950",
        "1910s",
        "1890s",
        "undated",
    ]
    # the all-undated corpus a one-volume city gets, and the dated one Chicago
    # has, both unchanged by the new branch
    assert viewer("['undated'].sort(L.compareErasNewestFirst)") == ["undated"]
    assert viewer("['1890s', '1950'].sort(L.compareErasNewestFirst)") == ["1950", "1890s"]


def test_the_undated_label_reads_as_itself_and_trails_the_dated_ones() -> None:
    """It is not a decade, so it never merges into a run, and the selection
    label puts it after the years for the same reason the chip row does."""
    assert viewer("L.selectionLabel(['undated'])") == "undated"
    assert viewer("L.selectionLabel(['undated', '1890s'])") == f"1890s{MIDDLE_DOT}undated"
    # the dated-only labels are unchanged by the new comparator
    assert viewer("L.selectionLabel(['1910s', '1890s', '1900s'])") == f"1890s{EN_DASH}1910s"


CITIES = json.dumps(
    {
        "cities": [
            {"slug": "alpha", "name": "Alpha", "manifest": "alpha/manifest.json"},
            {"slug": "beta", "name": "Beta", "manifest": "beta/manifest.json"},
        ]
    }
)
ONE_CITY = json.dumps({"cities": [{"slug": "solo", "name": "Solo", "manifest": "solo/x.json"}]})


def test_a_link_naming_a_city_loads_that_citys_manifest() -> None:
    """One page, one city. Which one is a fact about the LINK, not about the
    page files, so two cities can share one set of them."""
    assert viewer(f"L.chooseCity({CITIES}, '?city=beta').manifest") == "beta/manifest.json"
    # against the index, never string-built from the query: a slug the index
    # does not list resolves to no path at all
    assert viewer(f"L.chooseCity({CITIES}, '?city=../secret').manifest") is None
    assert viewer(f"L.chooseCity({CITIES}, '?city=gone').error") == "gone"
    # ...and that answer still offers the ones that ARE here
    assert viewer(f"L.chooseCity({CITIES}, '?city=gone').choose.length") == 2


def test_a_visit_naming_no_city_gets_the_only_one_or_the_list() -> None:
    """A default is honest while there is one city and a lie once there are
    two, so the answer follows the index rather than a configured favourite."""
    assert viewer(f"L.chooseCity({ONE_CITY}, '').manifest") == "solo/x.json"
    assert viewer(f"L.chooseCity({CITIES}, '').manifest") is None
    assert viewer(f"L.chooseCity({CITIES}, '').choose.length") == 2


def test_an_empty_or_malformed_index_never_resolves_to_a_manifest() -> None:
    """A page that draws neither a map nor a reason is the failure this whole
    resolution exists to stop, so every degenerate index lands on `error`."""
    for index in ("{}", '{"cities": []}', '{"cities": "nope"}', "null"):
        picked = viewer(f"L.chooseCity({index}, '')")
        assert picked.get("manifest") is None, index
        assert "error" in picked, index
    # an entry missing the path it would be loaded from is not an entry
    half = json.dumps({"cities": [{"slug": "a", "name": "A"}]})
    assert viewer(f"L.chooseCity({half}, '').manifest") is None


def test_two_cities_share_one_page_and_each_draws_its_own(viewer_bundle: Path) -> None:
    """One set of page files, two manifests, and the link says which.

    The defect this replaces was a page headed with one city's name drawing
    another city's sheets, which no assertion about the page's own DOM could
    have caught — so both loads are checked for the title AND the layer.
    """
    root = viewer_bundle
    _publish(root, _manifest("/base/tile.png"))
    other = _manifest("/base/tile.png")
    other["site"]["name"] = "Otherton"
    other["site"]["title"] = "Otherton {era} / Now"
    other["volumes"][0]["id"] = "other_001"
    other["volumes"][0]["label"] = "Other District"
    (root / "viewer" / "otherton").mkdir(exist_ok=True)
    (root / "viewer" / "otherton" / "manifest.json").write_text(json.dumps(other), encoding="utf-8")
    (root / "viewer" / "cities.json").write_text(
        json.dumps(city_index((PROBE_SLUG, "Probeville"), ("otherton", "Otherton"))),
        encoding="utf-8",
    )

    with serve(root) as base_url:
        for slug, title, layer in (
            (PROBE_SLUG, "Probeville 1890s / Now", "probe_001"),
            ("otherton", "Otherton 1890s / Now", "other_001"),
        ):
            page = load_page(
                f"{base_url}/viewer/index.html?city={slug}",
                until=DREW,
                capture={"sources": SOURCES},
            )
            assert f"<title>{title}</title>" in page.dom
            assert layer in captured_json(page, "sources"), (
                f"{slug} drew {page.captured['sources']}"
            )
            assert page.console == (), f"{slug} complained: {page.console}"


def test_a_visit_naming_no_city_is_offered_the_ones_there_are(viewer_bundle: Path) -> None:
    """Never a blank page. With more than one city and nothing in the URL, the
    overlay says so and links to each — the atlas is not silently picked."""
    root = viewer_bundle
    _publish(root, _manifest("/base/tile.png"))
    (root / "viewer" / "otherton").mkdir(exist_ok=True)
    (root / "viewer" / "otherton" / "manifest.json").write_text(
        json.dumps(_manifest("/base/tile.png")), encoding="utf-8"
    )
    (root / "viewer" / "cities.json").write_text(
        json.dumps(city_index((PROBE_SLUG, "Probeville"), ("otherton", "Otherton"))),
        encoding="utf-8",
    )
    with serve(root) as base_url:
        page = load_page(
            f"{base_url}/viewer/index.html",
            until=CHOOSING,
            capture={"links": _links("#loading")},
        )
    assert captured_json(page, "links") == [
        [f"?city={PROBE_SLUG}", "Probeville"],
        ["?city=otherton", "Otherton"],
    ]
    assert "Choose an atlas" in page.element('id="loading"')


def test_a_link_naming_a_city_that_is_not_here_says_so(viewer_bundle: Path) -> None:
    """A forwarded link outlives a city's slug. The reader gets a reason and
    the list, not an empty pane."""
    _publish(viewer_bundle, _manifest("/base/tile.png"))
    with serve(viewer_bundle) as base_url:
        page = load_page(
            f"{base_url}/viewer/index.html?city=nowhere",
            until=CHOOSING,
            capture={"links": _links("#loading")},
        )
    assert "nowhere" in page.element('id="loading"')
    assert captured_json(page, "links") == [[f"?city={PROBE_SLUG}", "Probeville"]]


def test_a_directory_with_no_index_loads_the_manifest_beside_the_page(
    viewer_bundle: Path,
) -> None:
    """A hand-copied directory: page files and a single manifest, no index at
    all. It must still draw — the index is how a MULTI-city viewer resolves,
    not a requirement. (A deploy bundle writes its own one-entry index and does
    not take this path.)"""
    manifest = _manifest("/base/tile.png")
    # the manifest is beside the page now, so it names the archive from there
    manifest["volumes"][0]["pmtiles"] = ATLAS_ARCHIVE_BESIDE_PAGE
    _publish(viewer_bundle, manifest, index=False)
    with serve(viewer_bundle) as base_url:
        page = load_page(f"{base_url}/viewer/index.html", until=DREW, capture={"sources": SOURCES})
    assert "probe_001" in captured_json(page, "sources")
    (viewer_bundle / "viewer" / "manifest.json").unlink()


def test_a_clone_that_has_published_nothing_says_exactly_that(viewer_bundle: Path) -> None:
    """The first thing a stranger sees, before they have run anything.

    No index and no manifest is not a fault — it is a viewer working correctly
    over an empty tree — so it must not report itself as one. Measured in the
    cold-clone container, where this is the literal first page load.
    """
    root = viewer_bundle
    (root / "viewer" / "cities.json").unlink(missing_ok=True)
    (root / "viewer" / "manifest.json").unlink(missing_ok=True)
    with serve(root) as base_url:
        page = load_page(f"{base_url}/viewer/index.html", until=CHOOSING)
    overlay = page.element('id="loading"')
    assert "No atlas is published here yet" in overlay, overlay
    assert "manifest.json" not in overlay, "a missing file is not a fault to report"


def test_a_manifest_that_is_not_one_says_so_instead_of_hanging(viewer_bundle: Path) -> None:
    """JSON that parses is not yet a manifest.

    Without the shape check the page threw on its first loop over `volumes` and
    the overlay reached NEITHER `done` nor `choose` — stuck on "loading the
    atlas…" for as long as the visitor waited. That is the blank page every
    other branch here exists to avoid, reached by the one input nothing
    validated.
    """
    page = _load(viewer_bundle, {"site": {"name": "Probeville"}}, until=CHOOSING)
    assert "is not a viewer manifest" in page.element('id="loading"')


def test_a_manifest_with_no_layers_says_so(viewer_bundle: Path) -> None:
    """`volumes: []` is well-formed and empty. The builder refuses to WRITE one,
    but a manifest already on disk outlives the code that wrote it."""
    page = _load(viewer_bundle, {"volumes": [], "site": {"name": "Probeville"}}, until=CHOOSING)
    assert "No layers are published" in page.element('id="loading"')


def test_an_unreadable_index_is_not_reported_as_a_missing_manifest(
    viewer_bundle: Path,
) -> None:
    """ABSENT and UNREADABLE are different answers.

    A 5xx on the index used to fall through to "could not load manifest.json" —
    naming a file the visitor never asked for, in a directory whose manifests
    are all fine, and hiding the actual fault.
    """
    root = viewer_bundle
    _publish(root, _manifest("/base/tile.png"))
    (root / "viewer" / "cities.json").write_text("{ truncated", encoding="utf-8")
    with serve(root) as base_url:
        page = load_page(f"{base_url}/viewer/index.html", until=CHOOSING)
    overlay = page.element('id="loading"')
    assert "list of atlases" in overlay, overlay
    assert "manifest.json" not in overlay, overlay
    (root / "viewer" / "cities.json").unlink()


def test_a_stops_image_resolves_against_the_manifest_not_the_page(
    viewer_bundle: Path,
) -> None:
    """Story images are staged BESIDE THE MANIFEST, under `story-assets/`, and
    the manifest names them relatively — so they resolve against the manifest,
    exactly as its archive paths do.

    This is the one place the two bases visibly differ, and the assertion is
    `naturalWidth`: an `img` whose src 404s still has the element, the alt text
    and the layout, so only the decoded pixels tell the two apart.
    """
    root = viewer_bundle
    staged = root / "viewer" / PROBE_SLUG / "story-assets"
    staged.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 12), (200, 80, 40)).save(staged / "wharf.png")
    manifest = _manifest("/base/tile.png")
    manifest["site"]["stories"] = [
        {
            "id": "probe-story",
            "title": "A story",
            "stops": [
                {
                    "id": "wharf",
                    "title": "The wharf",
                    "camera": {"center": list(PROBE_CENTER), "zoom": 13},
                    "media": [{"src": "story-assets/wharf.png", "alt": "the wharf"}],
                }
            ],
        }
    ]
    page = _load(
        root,
        manifest,
        fragment="#story=probe-story&stop=wharf",
        until='document.querySelector("#story figure img")',
        capture={
            "shot": "JSON.stringify((() => { const i ="
            ' document.querySelector("#story figure img");'
            " return [i.naturalWidth, i.naturalHeight, i.src]; })())"
        },
    )
    width, height, src = captured_json(page, "shot")
    assert (width, height) == (24, 12), f"the stop's image did not load: {src}"
    assert f"/viewer/{PROBE_SLUG}/story-assets/wharf.png" in src


CHIPS = 'JSON.stringify([...document.querySelectorAll("#eras .era")].map(e => e.textContent))'


def test_a_volume_with_no_era_still_draws_under_one_undated_chip(viewer_bundle: Path) -> None:
    """A city publishing its first volume with no LOC catalog gets `year: null`,
    so `era` is null. The page used to return before adding the source: the
    archive was served, the manifest listed it, and the atlas was not there,
    with nothing on screen saying why. It draws now, and it is selectable."""
    manifest = _manifest("/base/tile.png")
    manifest["volumes"][0]["era"] = None
    manifest["volumes"][0]["year"] = None
    page = _load(viewer_bundle, manifest, capture={"sources": SOURCES, "chips": CHIPS})

    assert 'class="done"' in page.element('id="loading"'), "the atlas never drew"
    assert "probe_001" in captured_json(page, "sources"), "the dateless layer was dropped"
    assert captured_json(page, "chips") == ["undated"]
    assert 'id="vol-ct">1<' in page.element('id="vol-ct"')
    assert page.console == (), f"the page complained: {page.console}"


def test_the_undated_chip_switches_its_layer_off_and_back_on(viewer_bundle: Path) -> None:
    """The chip is a control, not a caption: an undated group has to toggle
    like a dated one, or the layer is visible and unmanageable."""
    manifest = _manifest("/base/tile.png")
    manifest["volumes"][0]["era"] = None
    page = _load(
        viewer_bundle,
        manifest,
        capture={
            "run": f"""
            (() => {{
              const chip = document.querySelector("#eras .era");
              chip.click();
              const off = {_visibility("probe_001")};
              chip.click();
              return [off, {_visibility("probe_001")}];
            }})()
            """
        },
    )
    assert page.captured["run"] == ["none", "visible"]


def test_a_mixed_manifest_keeps_every_volume_and_trails_the_undated_chip(
    viewer_bundle: Path,
) -> None:
    """The case a Chicago-scale corpus hits if one volume ever loses its year.
    Both chips render, the dated one heads the row, and NO volume is missing
    from the page — the district list only shows the selected era, so the count
    is asserted with both chips on."""
    manifest = _manifest("/base/tile.png")
    manifest["volumes"].append(
        {**manifest["volumes"][0], "id": "probe_002", "era": None, "year": None, "label": "Undated"}
    )
    manifest["site"]["default_eras"] = ["1890s", "undated"]
    page = _load(viewer_bundle, manifest, capture={"sources": SOURCES, "chips": CHIPS})

    assert captured_json(page, "chips") == ["1890s", "undated"]
    sources = captured_json(page, "sources")
    assert {"probe_001", "probe_002"} <= set(sources), f"a volume is missing: {sources}"
    listed = page.element('id="volumes"')
    assert "Probe District" in listed and "Undated" in listed
    assert 'id="vol-ct">2<' in page.element('id="vol-ct"')
    assert page.console == (), f"the page complained: {page.console}"
