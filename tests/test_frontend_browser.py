"""Every frontend surface, loaded by a browser and asked what it drew.

Everything else about these pages is checked without running them. This is the
one tier that catches `the page loaded and drew nothing` — a renamed export, a
script tag in the wrong order, a throw on the first line of boot.

The servable fixture is `viewer_browser_support`; which manifest a page load
resolves to, and what an undated volume draws, are `test_frontend_cities`.
What the page says about where its pixels came from is
`test_viewer_credits_browser`.

Needs a headless browser on PATH; without one these skip.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from autogeoref.local_server import loopback_server
from browser import ROOT, PageLoad, load_page
from viewer_browser_support import (
    ATLAS_ARCHIVE,
    DREW,
    PROBE_CENTER,
    SETTLE_S,
    SOURCES,
    VIEWER_DIR,
)
from viewer_browser_support import load as _load
from viewer_browser_support import manifest_for as _manifest
from viewer_browser_support import manifest_with_many_districts as _manifest_with_many_districts
from viewer_browser_support import visibility as _visibility


def test_the_viewer_draws_its_atlas_with_nothing_but_this_host(viewer_bundle: Path) -> None:
    """The whole chain, run: manifest -> era chips -> a PMTiles source -> the
    loading overlay clearing. Every string asserted here is the page's own
    rendered output, and DNS is blocked, so nothing off this machine is
    involved in producing any of it."""
    page = _load(viewer_bundle, _manifest("/base/tile.png"))

    assert "<title>Probeville 1890s / Now</title>" in page.dom
    # `done` is set by drawAtlas, which cannot run before a real source is added
    assert 'class="done"' in page.element('id="loading"')
    volumes = page.element('id="vol-ct"')
    assert 'id="vol-ct">1<' in volumes
    assert "Probe District" in volumes
    assert "Vol. 1" in volumes
    assert "1894" in volumes
    assert ">1890s<" in page.element('id="plate-era"')
    # not just error-severity: the public page is meant to be silent, and it
    # can be, because everything it needs is in the bundle
    assert page.console == (), f"the page complained: {page.console}"


def test_a_manifest_carrying_a_stale_era_pmtiles_key_still_draws(viewer_bundle: Path) -> None:
    """The manifest on disk outlives the code that wrote it.

    A manifest published before citywide era layers were retired still carries
    `era_pmtiles`. Nothing may read it now, and a page that tripped over it
    would draw nothing at all — so the districts must draw exactly as they do
    without it.
    """
    manifest = _manifest("/base/tile.png")
    manifest["era_pmtiles"] = {"1890s": ATLAS_ARCHIVE}
    page = _load(viewer_bundle, manifest, capture={"sources": SOURCES})

    assert 'class="done"' in page.element('id="loading"'), "the atlas never drew"
    sources = json.loads(page.captured["sources"])
    assert "probe_001" in sources, f"the district was dropped: {sources}"
    assert "era-1890s" not in sources, "the retired citywide layer was added"
    assert page.console == (), f"the page complained: {page.console}"


def test_a_district_carrying_a_stale_provenance_field_still_draws(viewer_bundle: Path) -> None:
    """The manifest on disk outlives the code that wrote it.

    A manifest published before `provenance` was dropped still carries it, and
    a rule keyed on that field would treat the two vintages differently. Nothing
    may read it, so both draw the same.
    """
    manifest = _manifest("/base/tile.png")
    manifest["volumes"][0]["provenance"] = "autogeoref"
    page = _load(viewer_bundle, manifest, capture={"sources": SOURCES})
    assert "probe_001" in json.loads(page.captured["sources"])


def _manifest_two_drawn_districts() -> dict:
    """Two districts, both with an archive on the map — so switching one off
    has something to leave behind."""
    manifest = _manifest("/base/tile.png")
    second = dict(manifest["volumes"][0])
    second["id"] = "probe_002"
    second["volume_number"] = "2"
    second["label"] = "Second District"
    manifest["volumes"].append(second)
    return manifest


def test_one_district_switches_off_without_taking_its_neighbour(viewer_bundle: Path) -> None:
    """The control the panel gained, driven: it hides that district's layer and
    only that one, it says so in the row, and clicking it again brings the
    layer back. Asked of the MAP, not of the row's styling."""
    page = _load(
        viewer_bundle,
        _manifest_two_drawn_districts(),
        capture={
            # read BEFORE anything is clicked: the list is first built while the
            # map has no layers at all, and a control that only appears once
            # something incidental rebuilds it is a control that may never appear
            "controls_at_draw": 'document.querySelectorAll("#volumes .vol-eye").length',
            "run": f"""
            (() => {{
              const eye = () => document.querySelectorAll("#volumes .vol-eye")[0];
              const before = [{_visibility("probe_001")}, {_visibility("probe_002")}];
              eye().click();
              const off = [{_visibility("probe_001")}, {_visibility("probe_002")}];
              const row = document.querySelectorAll("#volumes .vol")[0];
              const marked = row.classList.contains("off");
              const pressed = eye().getAttribute("aria-pressed");
              const nameOff = eye().getAttribute("aria-label");
              const focused = document.activeElement === eye();
              const box = eye().getBoundingClientRect();
              const showsAll = Boolean(document.querySelector("#volumes .vol-all"));
              const stillListed = document.querySelectorAll("#volumes .vol").length;
              eye().click();
              return {{
                before: before, off: off, marked: marked, pressed: pressed,
                names: [nameOff, eye().getAttribute("aria-label")],
                focused: focused, target: [Math.round(box.width), Math.round(box.height)],
                showsAll: showsAll, stillListed: stillListed,
                back: [{_visibility("probe_001")}, {_visibility("probe_002")}],
                url_after: location.href,
              }};
            }})()
            """,
        },
    )

    assert page.captured["controls_at_draw"] == 2, "the districts have no switch when the map draws"
    run = page.captured["run"]
    assert run["before"] == ["visible", "visible"]
    assert run["off"] == ["none", "visible"], "hid the wrong district, or both"
    assert run["marked"] is True
    # the name says what the control is, `aria-pressed` which way it is set
    assert run["pressed"] == "false", "a switched-off district still reads as pressed"
    assert run["names"] == ["Probe District on the map", "Probe District on the map"], (
        "the accessible name changes with the state, so it names an action and not a thing"
    )
    assert run["stillListed"] == 2, "a switched-off district left the list it is switched on from"
    # switching rebuilds the list, which takes the activated button out of the
    # document: a keyboard reader would be dropped back at the top of the page
    assert run["focused"] is True, "the keyboard lost the control it just used"
    assert min(run["target"]) >= 24, f"pointer target below the floor: {run['target']}"
    assert run["showsAll"] is True
    assert run["back"] == ["visible", "visible"], "the district did not come back"
    assert "off=" not in run["url_after"], f"an empty key on a plain link: {run['url_after']}"
    assert page.console == (), f"the page complained: {page.console}"


def test_show_all_brings_back_every_switched_off_district(viewer_bundle: Path) -> None:
    """The only way back for a district hidden in an era the reader then
    switched off — so it is driven, not merely found in the DOM: both layers
    return, the control goes with the last hidden district, and the link stops
    carrying a key that no longer describes anything."""
    page = _load(
        viewer_bundle,
        _manifest_two_drawn_districts(),
        capture={
            "run": f"""
            (() => {{
              document.querySelectorAll("#volumes .vol-eye").forEach(b => b.click());
              const hiddenBoth = [{_visibility("probe_001")}, {_visibility("probe_002")}];
              const urlHidden = location.href;
              document.querySelector("#volumes .vol-all").click();
              return {{
                hiddenBoth: hiddenBoth,
                urlHidden: urlHidden,
                drawn: [{_visibility("probe_001")}, {_visibility("probe_002")}],
                dimmedRows: document.querySelectorAll("#volumes .vol.off").length,
                stillOffers: Boolean(document.querySelector("#volumes .vol-all")),
                url: location.href,
              }};
            }})()
            """,
        },
    )

    run = page.captured["run"]
    assert run["hiddenBoth"] == ["none", "none"]
    assert "off=probe_001,probe_002" in run["urlHidden"]
    assert run["drawn"] == ["visible", "visible"], "Show all left a district switched off"
    assert run["dimmedRows"] == 0
    assert run["stillOffers"] is False, "still offering to show all when nothing is hidden"
    assert "off=" not in run["url"], f"the link still carries the key: {run['url']}"


def test_a_district_with_no_published_imagery_has_no_switch(viewer_bundle: Path) -> None:
    """The switch is offered where there is a layer to switch. A volume listed
    without imagery is in the index — that is the point of the index — but it
    has nothing this control could do."""
    manifest = _manifest_two_drawn_districts()
    del manifest["volumes"][1]["pmtiles"]  # listed, not published

    page = _load(
        viewer_bundle,
        manifest,
        capture={
            "rows": 'document.querySelectorAll("#volumes .vol").length',
            "switches": 'document.querySelectorAll("#volumes .vol-eye").length',
            "which": '[...document.querySelectorAll("#volumes .vol-eye")]'
            ".map(b => b.dataset.volume)",
        },
    )

    assert page.captured["rows"] == 2
    assert page.captured["switches"] == 1
    assert page.captured["which"] == ["probe_001"]


def test_a_switched_off_district_survives_its_era_and_travels_in_the_link(
    viewer_bundle: Path,
) -> None:
    """Two independent switches: turning the era off and on again must not
    quietly restore a district the reader hid. And the link has to describe
    what the copier is looking at, so `off=` rides the same hash as the rest of
    the view — the assertion is that a link built by the page reproduces it."""
    page = _load(
        viewer_bundle,
        _manifest_two_drawn_districts(),
        capture={
            "run": f"""
            (() => {{
              document.querySelectorAll("#volumes .vol-eye")[0].click();
              const era = document.querySelector("#eras .era");
              era.click();                                   // the era off...
              const dark = {_visibility("probe_001")};
              era.click();                                   // ...and back on
              return {{
                dark: dark,
                afterEra: [{_visibility("probe_001")}, {_visibility("probe_002")}],
                url: location.href,
              }};
            }})()
            """
        },
    )

    run = page.captured["run"]
    assert run["dark"] == "none"
    assert run["afterEra"] == ["none", "visible"], "the era switch un-hid a hidden district"
    # the exact value: a substring would also pass on a link hiding both
    assert "off=probe_001&" in run["url"] or run["url"].endswith("off=probe_001"), (
        f"the link does not carry exactly the hidden district: {run['url']}"
    )


def test_a_link_opens_with_its_districts_already_switched_off(viewer_bundle: Path) -> None:
    """The other end of that link, and the reason the ids are filtered: one the
    manifest no longer lists must not hide anything, or a stale link would
    black out a district that has nothing to do with it."""
    page = _load(
        viewer_bundle,
        _manifest_two_drawn_districts(),
        fragment="#off=probe_001,gone_999",
        capture={
            "drawn": f"[{_visibility('probe_001')}, {_visibility('probe_002')}]",
            "hidden_rows": 'document.querySelectorAll("#volumes .vol.off").length',
        },
    )

    assert page.captured["drawn"] == ["none", "visible"]
    assert page.captured["hidden_rows"] == 1, "a stale id in the link hid something"


def test_a_story_link_carries_what_is_switched_off_behind_it(viewer_bundle: Path) -> None:
    """A story stop keeps whatever the reader switched off behind it, and the
    address bar inside a story carries `off`. So the link the page WRITES there
    has to be one it can READ back — the story branch returns early, and the
    hidden districts have to be applied before it does, or the most forwarded
    link this site produces silently un-hides them for the recipient."""
    manifest = _manifest_two_drawn_districts()
    manifest["site"]["stories"] = _manifest_with_a_story()["site"]["stories"]

    page = _load(
        viewer_bundle,
        manifest,
        query="?stories=1",
        fragment="#off=probe_001&story=probe-story&stop=wharf",
        until=f'{DREW} && document.querySelector("#story h2")',
        capture={
            "in_story": f"[{_visibility('probe_001')}, {_visibility('probe_002')}]",
            "after_exit": f"""
            (() => {{
              {_CLICK.format("story-exit")};
              return {{
                drawn: [{_visibility("probe_001")}, {_visibility("probe_002")}],
                url: location.href,
              }};
            }})()
            """,
        },
    )

    assert page.captured["in_story"] == ["none", "visible"], "the story link un-hid a district"
    after = page.captured["after_exit"]
    assert after["drawn"] == ["none", "visible"], "leaving the story un-hid it"
    assert "off=probe_001" in after["url"], f"leaving the story dropped the key: {after['url']}"


def test_a_story_keeps_its_controls_in_reach_in_a_short_panel(viewer_bundle: Path) -> None:
    """Back/Next are pinned to the bottom of the scroll region, so a stop with
    more prose than the panel is tall cannot push them out of it."""
    manifest = _manifest_with_a_story()
    manifest["site"]["stories"][0]["stops"][0]["body_html"] = "<p>Long prose.</p>" * 60
    page = _load(
        viewer_bundle,
        manifest,
        query="?stories=1",
        fragment="#story=probe-story&stop=wharf",
        viewport=(1280, 700),
        until=f'{DREW} && document.querySelector("#story h2")',
        capture={
            "fit": """
            (() => {
              const panel = document.getElementById('panel').getBoundingClientRect();
              const body = document.getElementById('panel-body');
              const overflows = body.scrollHeight > body.clientHeight + 1;
              body.scrollTop = 0;
              const top = document.getElementById('story-nav').getBoundingClientRect();
              body.scrollTop = body.scrollHeight;
              const end = document.getElementById('story-nav').getBoundingClientRect();
              return {
                overflows: overflows,
                pinnedAtTop: top.bottom <= panel.bottom + 1,
                pinnedAtEnd: end.bottom <= panel.bottom + 1,
              };
            })()
            """
        },
    )

    fit = page.captured["fit"]
    assert fit["overflows"] is True, "the fixture's prose no longer overruns the panel"
    assert fit["pinnedAtTop"] is True, "Back/Next start out below the panel"
    assert fit["pinnedAtEnd"] is True, "Back/Next left the panel when the prose scrolled"


def test_each_stop_opens_at_its_own_top(viewer_bundle: Path) -> None:
    """A reader reaches Next at the BOTTOM of a stop — that is where the
    control is. The stop it opens has to start at its own beginning.

    The panel body is the scroll region now, not the prose, so the reset that
    used to do this (`storyPanel.scrollTop`) moved with it: without one, Next
    lands the reader past the end of the new stop, looking at its sources.
    Leaving the story is the same problem one layer out — the panel behind it
    would still be scrolled into the middle of the district list.
    """
    # 30 districts, so the panel behind the story has somewhere to be scrolled
    # TO: with one district the list is shorter than the panel and the browser
    # clamps the scroll back to the top on its own, proving nothing
    manifest = _manifest_with_many_districts(30)
    manifest["site"]["stories"] = _manifest_with_a_story()["site"]["stories"]
    for stop in manifest["site"]["stories"][0]["stops"]:
        stop["body_html"] = "<p>Long prose.</p>" * 40
    page = _load(
        viewer_bundle,
        manifest,
        query="?stories=1",
        fragment="#story=probe-story&stop=wharf",
        viewport=(1280, 700),
        until=f'{DREW} && document.querySelector("#story h2")',
        capture={
            "run": f"""
            (() => {{
              const panel = () => document.getElementById('panel').getBoundingClientRect();
              const body = document.getElementById('panel-body');
              const heading = () => document.querySelector('#story h2').getBoundingClientRect();
              body.scrollTop = body.scrollHeight;          // read to the end of the stop
              {_CLICK.format("story-next")};
              const p = panel(), h = heading();
              const headingInside = h.top >= p.top - 1 && h.bottom <= p.bottom + 1;
              body.scrollTop = body.scrollHeight;
              {_CLICK.format("story-exit")};
              const first = document.querySelector('#volumes .vol').getBoundingClientRect();
              return {{
                headingInside: headingInside,
                nextStop: document.querySelector('#story-entry') && null,
                scrollAfterExit: body.scrollTop,
                firstRowBelowPanelTop: first.top >= panel().top - 1,
              }};
            }})()
            """
        },
    )

    run = page.captured["run"]
    assert run["headingInside"] is True, "Next opened the stop already scrolled past its title"
    assert run["scrollAfterExit"] == 0, "left a story with the panel scrolled into the list"
    assert run["firstRowBelowPanelTop"] is True


def _manifest_with_a_story(basemap_tiles: str | None = "/base/tile.png") -> dict:
    """The probe manifest plus one two-stop story over the probe city."""
    manifest = _manifest(basemap_tiles)
    manifest["site"]["stories"] = [
        {
            "id": "probe-story",
            "title": "What the probe city cleared",
            "dek": "Two stops.",
            "stops": [
                {
                    "id": "wharf",
                    "title": "The wharf",
                    "body_html": "<p>Prose.</p>",
                    "camera": {"center": list(PROBE_CENTER), "zoom": 13},
                },
                {
                    "id": "yards",
                    "title": "The yards",
                    "camera": {"center": [PROBE_CENTER[0] + 0.01, PROBE_CENTER[1]], "zoom": 13},
                },
            ],
        }
    ]
    return manifest


#: What the panel offers, read from the loaded page rather than the serialized
#: DOM: `#story-entry` is present in the markup either way and it is the `on`
#: class and the buttons inside it that say whether a visit was offered one.
STORY_UI = {
    "entry_on": 'document.getElementById("story-entry").classList.contains("on")',
    "entry_buttons": 'document.querySelectorAll("#story-entry .story-open").length',
    "stop_title": 'document.querySelector("#story h2") ? '
    'document.querySelector("#story h2").textContent : ""',
    "nav_on": 'document.getElementById("story-nav").classList.contains("on")',
}


def test_a_plain_visit_is_offered_no_story_at_all(viewer_bundle: Path) -> None:
    """A configured story is not automatically a published one: with no opt-in
    in the link, the panel that loads has no entry list, no buttons and no
    story nav — the district index gets that room instead."""
    page = _load(viewer_bundle, _manifest_with_a_story(), capture=STORY_UI)

    assert page.captured["entry_on"] is False
    assert page.captured["entry_buttons"] == 0
    assert page.captured["nav_on"] is False
    assert "Probe District" in page.element('id="vol-ct"')  # the panel still works
    assert page.console == (), f"the page complained: {page.console}"


def test_the_query_param_offers_the_story_list(viewer_bundle: Path) -> None:
    """The other half: with the opt-in present the entry list is built from the
    manifest, so the absence above is the gate and not a broken feature."""
    page = _load(viewer_bundle, _manifest_with_a_story(), query="?stories=1", capture=STORY_UI)

    assert page.captured["entry_on"] is True
    assert page.captured["entry_buttons"] == 1
    assert "What the probe city cleared" in page.dom


def test_a_story_permalink_opens_without_the_query_param(viewer_bundle: Path) -> None:
    """The site produced these links. Gating the ENTRY LIST must not break the
    permalink: `#story=` opens its stop with no opt-in in the query string."""
    page = _load(
        viewer_bundle,
        _manifest_with_a_story(),
        fragment="#story=probe-story&stop=yards",
        until=f'{DREW} && document.querySelector("#story h2")',
        capture=STORY_UI,
    )

    assert page.captured["stop_title"] == "The yards"
    assert page.captured["nav_on"] is True
    assert page.captured["entry_buttons"] == 1


#: Clicking, from the harness: one expression drives the whole sequence and
#: returns what each step produced, so nothing the page does on its own can run
#: between a click and the reading it is checked against. A chained form failed
#: once under a loaded suite and could not be reproduced afterwards, here or by
#: review; this shape removes the question rather than answering it.
_STOP_TITLE = 'document.querySelector("#story h2").textContent'
_CLICK = 'document.getElementById("{0}").click()'
_OPEN_FIRST_STORY = 'document.querySelector("#story-entry .story-open").click()'


def test_the_controls_work_and_leaving_a_story_keeps_the_way_back(viewer_bundle: Path) -> None:
    """The reader who followed a permalink drives the story and then leaves it.

    Next and Leave must work — they are wired for every configured story, never
    behind the entry-list gate — and the opt-in that reader expressed BY
    following the link is written into the link, so the page they are left
    looking at is one they can reload, bookmark or copy and still find the
    story. Without that the escape hatch lasts exactly as long as the story:
    leaving clears the `story` key.
    """
    page = _load(
        viewer_bundle,
        _manifest_with_a_story(),
        fragment="#story=probe-story&stop=wharf",
        until=f'{DREW} && document.querySelector("#story h2")',
        capture={
            "run": f"""
            (() => {{
              const opened_at = {_STOP_TITLE};
              const url_in_story = location.href;
              {_CLICK.format("story-next")};
              const after_next = {_STOP_TITLE};
              {_CLICK.format("story-exit")};
              return {{
                opened_at: opened_at,
                url_in_story: url_in_story,
                after_next: after_next,
                still_in_story: document.body.classList.contains("story"),
                entry_after_exit: {STORY_UI["entry_on"]},
                url_after_exit: location.href,
              }};
            }})()
            """
        },
    )

    run = page.captured["run"]
    assert run["opened_at"] == "The wharf"
    assert run["after_next"] == "The yards", "Next did nothing"
    assert run["still_in_story"] is False, "Leave the story did nothing"
    assert run["entry_after_exit"] is True
    assert "stories=1" in run["url_in_story"]
    url = run["url_after_exit"]
    assert "stories=1" in url, f"left a story with no way back into it: {url}"
    assert "story=probe-story" not in url  # the free view is republished
    assert page.console == (), f"a story control threw: {page.console}"


def test_re_entering_a_story_writes_the_opt_in_once(viewer_bundle: Path) -> None:
    """Entering a story records the opt-in in the address bar, and a reader
    enters one as often as they like. The link must carry the key ONCE however
    many times that happens — `Copy a link to this view` hands out whatever is
    in the address bar.

    Two things hold it, and this is the end-to-end assertion over both: the
    write REPLACES a key it finds (`queryWrite`, covered by its own test), and
    the guard reads the live query string rather than the page-load one.
    """
    page = _load(
        viewer_bundle,
        _manifest_with_a_story(),
        fragment="#story=probe-story&stop=wharf",
        until=f'{DREW} && document.querySelector("#story h2")',
        capture={
            "urls": f"""
            (() => {{
              const seen = [location.href];               // entered from the permalink
              for (let i = 0; i < 2; i++) {{
                {_CLICK.format("story-exit")};
                {_OPEN_FIRST_STORY};                     // and again from the list
                seen.push(location.href);
              }}
              return seen;
            }})()
            """
        },
    )

    for index, url in enumerate(page.captured["urls"]):
        assert url.count("stories=1") == 1, f"entry {index + 1} left {url}"
    assert page.console == (), f"the page complained: {page.console}"


def test_a_permalink_naming_a_story_the_city_dropped_still_offers_the_list(
    viewer_bundle: Path,
) -> None:
    """Story ids are configuration and configuration is edited. A link naming
    one that is gone is the reader MOST in need of the entry list, so the arm
    turns on for a `#story=` link that is present, not one that resolves."""
    page = _load(
        viewer_bundle, _manifest_with_a_story(), fragment="#story=renamed", capture=STORY_UI
    )

    assert page.captured["stop_title"] == ""  # nothing opened; nothing to open
    assert page.captured["entry_buttons"] == 1
    assert page.captured["entry_on"] is True


def test_an_explicit_off_suppresses_the_list_but_never_the_permalink(
    viewer_bundle: Path,
) -> None:
    """`?stories=0` is the answer that beats the permalink arm — otherwise the
    off values mean nothing on the only visits with another way in. The story
    itself still opens and still drives: a link the site produced always works.
    """
    page = _load(
        viewer_bundle,
        _manifest_with_a_story(),
        query="?stories=0",
        fragment="#story=probe-story&stop=wharf",
        until=f'{DREW} && document.querySelector("#story h2")',
        capture={
            "run": f"""
            (() => {{
              const stop_title = {_STOP_TITLE};
              const entry_buttons = {STORY_UI["entry_buttons"]};
              {_CLICK.format("story-next")};
              const after_next = {_STOP_TITLE};
              {_CLICK.format("story-exit")};
              return {{
                stop_title: stop_title,
                entry_buttons: entry_buttons,
                after_next: after_next,
                entry_after_exit: {STORY_UI["entry_on"]},
                url: location.href,
              }};
            }})()
            """
        },
    )

    run = page.captured["run"]
    assert run["stop_title"] == "The wharf"
    assert run["after_next"] == "The yards"
    assert run["entry_buttons"] == 0
    assert run["entry_after_exit"] is False
    assert "stories=1" not in run["url"], "overruled the link's own answer"
    assert page.console == (), f"a story control threw: {page.console}"


def test_no_basemap_configured_draws_the_atlas_and_says_why(viewer_bundle: Path) -> None:
    """`chooseBasemap` returns bare on a host with nothing configured; this is
    that decision reaching a real page. Served from 127.0.0.1 the dev-only
    raster default applies, and with DNS blocked it loads nothing — so the
    credit must stay off and the atlas must still be there."""
    page = _load(viewer_bundle, _manifest(None), settle_s=SETTLE_S)

    assert 'class="done"' in page.element('id="loading"')
    assert "OpenStreetMap" not in page.element('id="footer"')


# ---------------------------------------------------------------------------
# the operator pages, against their real servers
# ---------------------------------------------------------------------------


@contextmanager
def _running(handler: type) -> Iterator[str]:
    """``handler`` on a loopback port; yields the base URL."""
    with loopback_server(handler, 0) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{httpd.server_address[1]}"
        finally:
            httpd.shutdown()
            thread.join(timeout=10)


#: The review UI is an operator tool on localhost and draws its context over
#: OSM's raster tiles by name, so with DNS blocked that request dies inside the
#: browser — and these two shapes are what dying looks like. Filtering them
#: costs this smoke a genuine fetch failure of the page's own API; what covers
#: that instead is the rendered queue, the loaded sheet, and the server's own
#: record of having painted the overlay.
BASEMAP_UNREACHABLE = ("ERR_NAME_NOT_RESOLVED", "TypeError: Failed to fetch")


def _page_errors(page: PageLoad) -> tuple[str, ...]:
    return tuple(
        m.text
        for m in page.errors
        if not any(shape in m.text for shape in BASEMAP_UNREACHABLE)
        and "favicon.ico" not in m.text
        and "404 (Not Found)" not in m.text
    )


@pytest.fixture
def review_server(tmp_path: Path) -> Iterator[tuple[str, Any]]:
    """The real ``ReviewApp``, over the real page files, on a loopback port."""
    from autogeoref.review.server import ReviewHandler
    from review_support import make_app  # the same tmp work tree the unit tests use

    app, paths = make_app(tmp_path)
    app.ui_dir = ROOT / "src" / "autogeoref" / "review_ui"
    app.vendor_dir = VIEWER_DIR / "vendor"
    # the ghost overlay is a raster of these; without them the page draws a
    # sheet list and no sheet
    for page in ("2", "4", "6"):
        Image.new("RGB", (138, 200), (222, 208, 178)).save(paths.sheets / f"p{page}_small.jpg")
    with _running(type("BoundReviewHandler", (ReviewHandler,), {"app": app})) as base_url:
        yield base_url, app


@pytest.fixture
def console_server(tmp_path: Path) -> Iterator[dict[str, Any]]:
    """The console with NO city — the configuration a bare queue runs in."""
    from autogeoref.console.actions import ConsoleActions
    from autogeoref.console.server import ConsoleRoutes

    board: dict[str, Any] = {
        "entries": [
            {
                "volume": "vol_running",
                "track": "place",
                "status": "running",
                "started": 1,
                "finished": None,
                "note": "",
                "progress": {
                    "pages": 10,
                    "reads": 4,
                    "accepted": 3,
                    "flagged": 1,
                    "stage": "match <b>bold</b>",
                    "stage_status": "ok",
                },
            },
            {
                "volume": "vol_failed",
                "track": "place",
                "status": "failed",
                "started": 1,
                "finished": 2,
                "note": "",
                "progress": {
                    "pages": 10,
                    "reads": 10,
                    "accepted": 0,
                    "flagged": 10,
                    "error": "boom <script>alert(1)</script>",
                    "failed_markers": 2,
                },
            },
        ],
        "served": [{"volume": "vol_served", "accepted": 7, "flagged": 1}],
        "runnable": [
            {
                "volume": "vol_ready",
                "sheets": 12,
                "calls": {"low": 10, "ceiling": 20},
                "track": "place",
                "year": 1894,
                "notes": [],
                "blocked": None,
            }
        ],
        "tracks": ["fetch", "place", "serve"],
        "links": {"viewer": "/viewer/", "candidates": "autogeoref queue --candidates"},
        "context": {
            "vol_running": {"city": "Probeville", "year": 1894, "neighborhoods": ["North"]}
        },
        "can_act": False,
        "drain": {"running": False},
        "spend": {"reads": 14},
    }
    handler = type(
        "BoundConsoleHandler",
        (ConsoleRoutes,),
        {
            "work": tmp_path,
            "console_ui": ROOT / "src" / "autogeoref" / "queue_ui",
            "build_board": staticmethod(lambda: board),
            "actions": ConsoleActions(work=tmp_path, city=None),
            "review_app": None,
        },
    )
    with _running(handler) as base_url:
        yield {"url": base_url, "board": board}


def test_the_review_page_loads_a_sheet_and_paints_its_ghost(
    review_server: tuple[str, Any],
) -> None:
    """The page that writes placements, loaded by something automated for the
    first time.

    The strongest assertion available is the server's own: it records an
    overlay as SHOWN when it serves the ghost raster, and refuses a verdict on
    a sheet it never painted. So a page that drew nothing cannot pass this —
    and neither can one that drew a sheet list and no sheet.
    """
    base_url, app = review_server
    page = load_page(
        f"{base_url}/",
        until='document.getElementById("sheet-title").textContent.startsWith("p")',
        capture={
            "queue": 'document.getElementById("queue").textContent',
            "title": 'document.getElementById("sheet-title").textContent',
            "status": 'document.getElementById("sheet-status").textContent',
            "rows": 'document.querySelectorAll("#queue .qi").length',
            "affine": "typeof ReviewAffine.similarityOps",
        },
    )
    # the flagged pool, from the real review_queue: p4 and p6, never the
    # committed p2
    assert page.captured["rows"] == 2
    assert "p4" in page.captured["queue"] and "p6" in page.captured["queue"]
    assert "p2" not in page.captured["queue"]
    assert page.captured["title"].startswith("p4")
    # the shared script actually loaded, which a page that 404s it would not
    assert page.captured["affine"] == "function"
    # the server's own record: it served the ghost raster for this sheet
    assert app.overlay_shown("volX", "4")
    assert _page_errors(page) == (), _page_errors(page)


def test_the_review_page_renders_a_status_string_as_text_not_markup(
    review_server: tuple[str, Any],
) -> None:
    """The escaping contract, verified as RENDERING rather than as source. A
    result's status is a free string off disk, and the row it lands in is built
    with `innerHTML`; here one carries a tag, and the tag has to be VISIBLE —
    escaped, not executed and not swallowed."""
    base_url, app = review_server
    (app.paths("volX").results / "p6.json").write_text(
        json.dumps({"page": "6", "status": "REJECTED <script>alert(1)</script> & <b>bold</b>"}),
        encoding="utf-8",
    )
    page = load_page(
        f"{base_url}/",
        until='document.querySelectorAll("#queue .qi").length > 0',
        capture={"queue": 'document.getElementById("queue").textContent'},
    )
    assert "<script>alert(1)</script>" in page.captured["queue"]
    assert "<script>alert(1)</script>" not in page.dom
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page.dom
    assert _page_errors(page) == (), _page_errors(page)


def test_the_board_page_fills_every_track_panel_from_the_api(
    console_server: dict[str, Any],
) -> None:
    """A console with no city still has a board — and it must still have its
    script, which is the route the fallthrough would have 404'd."""
    panels = ("runnable", "running", "needs", "served", "fetch", "place", "serve")
    page = load_page(
        f"{console_server['url']}/",
        until='document.querySelectorAll("#place table tbody tr").length > 0',
        capture={name: f'document.getElementById("{name}").textContent' for name in panels}
        | {"nocity": 'document.getElementById("nocity").hidden'},
    )
    # each entry in the panel the operator would look in, and in no other
    assert "vol_running" in page.captured["running"]
    assert "vol_running" in page.captured["place"]
    assert "vol_running" not in page.captured["fetch"]
    assert "vol_running" not in page.captured["serve"]
    assert "vol_ready" in page.captured["runnable"]
    assert "vol_served" in page.captured["served"]
    assert "vol_failed" in page.captured["needs"]
    assert "nothing queued" in page.captured["serve"]  # an empty track still renders
    # the context line, composed by board.js from the payload
    assert "Probeville, 1894 - North" in page.captured["running"]
    # a console with no city says so where the operator is looking, not on a
    # button's hover title
    assert page.captured["nocity"] is False
    assert _page_errors(page) == (), _page_errors(page)


def test_a_board_string_containing_markup_renders_as_text(
    console_server: dict[str, Any],
) -> None:
    """Board text includes marker error strings — tracebacks, straight off
    disk. Verified as rendering: the tag has to be VISIBLE, which means it was
    escaped, not executed and not swallowed."""
    page = load_page(
        f"{console_server['url']}/",
        until='document.getElementById("needs").textContent.includes("boom")',
    )
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page.dom
    assert "<script>alert(1)</script>" not in page.dom
    assert "match &lt;b&gt;bold&lt;/b&gt;" in page.dom
    assert _page_errors(page) == (), _page_errors(page)
