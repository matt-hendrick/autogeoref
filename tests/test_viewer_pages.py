"""The page files carry no city fact, and keep the plumbing a refactor could drop.

Every city fact reaches the page through ``manifest.site``, so a Chicago
string, coordinate or CDN URL in a page file is a defect. These stay source
greps deliberately: they are contracts about what is ABSENT, which no unit test
can express, and the viewer's decisions are executed by the JS tests instead.
The vendored assets are checked for existence and self-hosting.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from autogeoref.viewer.deploy import CONFIG_FILE, ICON_FILES, PAGE_FILES
from viewer_support import (
    NOT_FOUND_HTML,
    VIEWER_CONFIG,
    VIEWER_CSS,
    VIEWER_DIR,
    VIEWER_HTML,
    VIEWER_JS,
    VIEWER_LIB,
    WALK_CSS,
    WALK_HTML,
    WALK_JS,
)


def _font_stacks(style: dict[str, Any]) -> set[str]:
    """Every glyph stack a style can ask for. ``text-font`` is either a plain
    list of names or an expression with the names buried in ``literal``
    branches, so this walks the whole value and keeps the strings that name a
    font: font names carry a space ("Noto Sans Regular"), operators do not."""

    def walk(value: Any) -> Iterator[str]:
        if isinstance(value, str):
            if " " in value:
                yield value
        elif isinstance(value, list):
            for item in value:
                yield from walk(item)

    return {
        stack
        for layer in style["layers"]
        for stack in walk(layer.get("layout", {}).get("text-font") or [])
    }


class TestViewerPageFilesAreCityFactFree:
    """§7.4's generalization check, enforced forever: every city fact must
    come from manifest.site. If one of these trips, someone hardcoded a city
    fact (or a CDN URL) back into a page file.

    These stay source greps on purpose. A grep is right for `no city fact
    leaked back in` and `the CDN did not come back` — contracts about what is
    absent, which no unit test can express. The viewer's *decisions* moved to
    ``viewer/lib.js`` and are run by ``tests/test_frontend_js.py`` instead.
    """

    FORBIDDEN: ClassVar[list[str]] = [
        "hicago",  # Chicago / chicago
        "oldinsurancemaps",  # the volunteer credit is per-era config now
        "OldInsuranceMaps",
        "robey",  # street renames are pipeline-only; the viewer never sees one
        "ROBEY",
        "-87.",  # any Chicago longitude
        "41.8",  # any Chicago latitude (41.62..42.05 band)
        "41.9",
        "42.0",
        "1950",  # default era is config; viewer carries no city-specific suffix
        "State & Madison",
        "unpkg.com",  # CDN-free: everything is vendored
        "fonts.googleapis",
        "gstatic",
    ]

    #: Plumbing that has to survive a refactor, per file. Every token here is a
    #: WIRING fact — a vendored tag, a choke point, a magic number's reason —
    #: which is what a grep is good for. A token that stated a *decision* has
    #: been converted to a test that runs it (``tests/test_frontend_js.py``).
    REQUIRED: ClassVar[dict[str, list[str]]] = {
        "index.html": [
            "vendor/maplibre-gl.js",
            "vendor/maplibre-gl-compare.js",
            "vendor/pmtiles.js",
            "vendor/fonts/fonts.css",
            '<link rel="stylesheet" href="app.css">',
            # before the scripts that read the token it sets
            '<script src="config.js"></script>',
            '<script src="lib.js"></script>',
            '<script src="app.js"></script>',
            # where the deploy writes the real title and the share-card tags.
            # Without it the bundle ships the placeholder title, and nothing
            # downstream notices until someone shares the link.
            "<!--SOCIAL-->",
        ],
        "walkthrough.html": [
            "<!--SOCIAL-->",
            # its own copy is the INPUT the deploy mirrors into the card, so
            # losing either leaves the card describing nothing
            '<meta name="description"',
        ],
        "app.css": [
            "--panel-overlays-map",  # the breakpoint the clamp reads, declared here
            "@media (max-width: 820px)",  # the small-screen layout exists
            ":focus-visible",  # every control keeps a keyboard focus signal
        ],
        "lib.js": [
            "module.exports = ViewerLib",  # runnable by the tests that execute it
        ],
        "config.js": [
            "window.MAPBOX_TOKEN",  # the one name the page reads out of this file
        ],
        "app.js": [
            # NOT the city resolution: which manifest a load picks is a
            # DECISION, and `test_frontend_cities` executes every branch of it
            # in node and in a browser. A grep here would pass on a bug.
            "manifest.site",
            "tileSize: 256",  # the documented 2x-blur bug guard, carried over
            # the geocoder DECISION is lib.js's and run by the JS tests; this
            # is the wiring that hands it the token and the serving host
            "chooseGeocoder(q, GEO, window.MAPBOX_TOKEN, location.hostname)",
            # the basemap DECISION is lib.js's and run by the JS tests; this is
            # the wiring that hands it the host the page is actually served from
            "chooseBasemap(S.basemap, location.hostname, pane)",
            # a basemap is credited by a landed tile, never by a built style
            "watchBasemapSource(before)",
            "default_eras",  # multi-era default from the manifest
            "selectedEras.has(era)",  # era chips are toggles, not radios
            "before.moveLayer",  # chronological restack of overlapping volumes
            # one clamp, and the vendored compare's own choke point routed
            # through it, so the drag and any programmatic set get the same
            # treatment — wiring no unit test of the clamp itself can see.
            # `reportSwipe` rides the same choke point for the same reason: it
            # is where every move of the handle ends up, so the value a screen
            # reader is told cannot drift from where the handle actually is.
            "compare._setPosition = (x) => { setPosition(clampSliderX(x)); reportSwipe(); }",
            # the vertical half of the same clamp: the DECISION is lib.js's and
            # run by the JS tests, and this is the choke point that hands it a
            # measurement. What the layout then does with it is a browser test
            "handleTop({",
            "--panel-overlays-map",  # read from script, declared once in CSS
            # no configured stories means no story UI at all; WHICH visits are
            # then offered the entry list is run by the browser tests
            "S.stories.length",
            "function enterStory(",  # the story panel, driven entirely from config
            "swipeToFraction(",  # a stop's divider goes through the shared clamp
        ],
    }

    #: Every loose file the two pages are made of. Kept beside
    #: ``viewer.deploy.PAGE_FILES``, which is what ships them. The walkthrough
    #: belongs here for the same reason the viewer does: every fact it states
    #: arrives in generated JSON, so a city name in its source is a defect.
    PAGE_FILES: ClassVar[tuple[Path, ...]] = (
        VIEWER_HTML,
        NOT_FOUND_HTML,
        VIEWER_CSS,
        VIEWER_LIB,
        VIEWER_JS,
        WALK_HTML,
        WALK_CSS,
        WALK_JS,
    )

    #: Every hand-written script the page loads, INCLUDING the one the bundle
    #: generates rather than copies. `PAGE_FILES` is the copied set; the checks
    #: about what a served file may contain answer to the loaded set.
    LOADED_FILES: ClassVar[tuple[Path, ...]] = (*PAGE_FILES, VIEWER_CONFIG)

    #: Everything the bundle PUBLISHES as text, which is a wider set than what
    #: the page loads: `_headers` is read by the host and never served, but it
    #: ships, so a city path or a local origin written into it is published
    #: just the same. It is in no other grep, by construction.
    PUBLISHED_FILES: ClassVar[tuple[Path, ...]] = (*LOADED_FILES, VIEWER_DIR / "_headers")

    def test_no_city_facts_or_cdn(self) -> None:
        tripped = {
            path.name: [token for token in self.FORBIDDEN if token in path.read_text("utf-8")]
            for path in self.PUBLISHED_FILES
        }
        tripped = {name: tokens for name, tokens in tripped.items() if tokens}
        assert not tripped, f"city facts / CDN leaked back into the viewer: {tripped}"

    def test_the_page_files_are_the_ones_the_bundle_ships(self) -> None:
        """A page file the deploy bundle does not copy deploys as a blank page,
        and nothing downstream of the copy would notice."""
        assert set(PAGE_FILES) == {path.name for path in self.PAGE_FILES}
        for path in self.PAGE_FILES:
            assert path.is_file(), f"{path.name} is missing"

    @pytest.mark.parametrize("name", sorted(REQUIRED))
    def test_required_plumbing_present(self, name: str) -> None:
        text = (VIEWER_DIR / name).read_text(encoding="utf-8")
        missing = [token for token in self.REQUIRED[name] if token not in text]
        assert not missing, f"viewer {name} lost required plumbing: {missing}"

    def test_the_committed_config_carries_no_token(self) -> None:
        """A committed token is a token nobody rotates, and this file is the
        one place in the tree shaped to hold one. The bundle rewrites it."""
        assert 'window.MAPBOX_TOKEN = "";' in VIEWER_CONFIG.read_text(encoding="utf-8")
        assert CONFIG_FILE not in PAGE_FILES, "the bundle generates config.js, never copies it"
        for path in self.PUBLISHED_FILES:
            assert "pk.ey" not in path.read_text(encoding="utf-8"), path.name

    def test_no_page_file_reaches_a_geocoder_outside_the_gated_decision(self) -> None:
        """The absent half of the Mapbox/Nominatim rule. `chooseGeocoder` is
        the only thing that may name a provider, because it is the only thing
        that checks the host: a second URL built anywhere else would send a
        public deployment's search traffic to OSMF's instance regardless."""
        for path in self.LOADED_FILES:
            if path == VIEWER_LIB:
                continue
            text = path.read_text(encoding="utf-8")
            for host in ("nominatim.openstreetmap.org", "api.mapbox.com"):
                assert host not in text, f"{path.name} builds its own {host} URL"

    def test_blur_bug_comment_carried_over(self) -> None:
        """The tileSize:256 rule exists because declaring 512 stretched
        gdal2tiles' 256px tiles 2x and blurred fine print — keep the why."""
        assert "MUST be 256" in VIEWER_JS.read_text(encoding="utf-8")

    def test_compare_handle_is_centred_on_the_divider(self) -> None:
        """The vendored stylesheet positions the swiper for its own 60px
        handle, so a restyle that sets only `width`/`height` leaves the handle
        beside the divider instead of on it — and the clamp's edge margin then
        protects the wrong pixels, stranding a handle it believes is in reach.
        """
        css = VIEWER_CSS.read_text(encoding="utf-8")
        swiper = css.split(".compare-swiper-vertical {", 1)[1].split("}", 1)[0]
        assert "left:" in swiper and "margin:" in swiper

    def test_attribution_is_not_only_in_the_collapsible_panel(self) -> None:
        """Collapsing the panel must not take the basemap credit with it.

        The licence obliges the credit in every state, so the whole chain has
        to survive a refactor: a sources control outside `#panel`, a toggle
        that reveals it, and the same composed line written to both places.
        """
        html = VIEWER_HTML.read_text(encoding="utf-8")
        script = VIEWER_JS.read_text(encoding="utf-8")
        assert 'document.getElementById("footer").innerHTML = creditLine' in script
        assert 'document.getElementById("attrib-text").innerHTML = creditLine' in script
        body = html.split("<body>", 1)[1]
        panel = body.split('<aside id="panel">', 1)[1].split("</aside>", 1)[0]
        for token in ('<div id="attrib">', 'id="attrib-toggle"', 'id="attrib-text"'):
            assert token in body, f"the sources control lost {token}"
            assert token not in panel, f"{token} moved inside the collapsible panel"
        assert 'attribText.toggleAttribute("hidden", !opening)' in script

    def test_nothing_builds_a_url_search_params_over_the_shared_hash(self) -> None:
        """The absent half of the fragment contract — what
        ``test_writing_one_hash_key_leaves_every_other_key_byte_for_byte``
        cannot see. A `URLSearchParams` round trip re-encodes `/` and `,` and
        turns a bare `#anchor` into `#anchor=`, so a second writer that reached
        for one would corrupt keys the tested writer preserves."""
        for path in self.PAGE_FILES:
            # the comment may name it; nothing may build one
            assert "new URLSearchParams" not in path.read_text(encoding="utf-8")

    def test_an_unchosen_collapse_state_follows_the_layout(self) -> None:
        """The same flag means `header strip` on a docked panel and `gone` on a
        floating one, so a state nobody chose has to be re-inferred when the
        breakpoint is crossed — otherwise a rotation hides the whole panel.
        The one control a docked panel has must also say what it does next.

        The `!active` term is story mode's: following a story link IS a choice
        about the panel, so re-inference must not close one mid-read.
        """
        script = VIEWER_JS.read_text(encoding="utf-8")
        assert (
            "if (!panelChoiceIsExplicit && !active) setPanelCollapsed(!panelOverlaysMap(), false)"
        ) in script
        assert 'panelToggle.setAttribute("aria-label", label)' in script

    def test_a_story_never_hides_its_own_way_out(self) -> None:
        """Story mode hides the panel header to give the prose the height, and
        the header holds the only toggle a docked panel has. Doing that while
        the panel is collapsed — which a resize can do by itself — leaves a
        reader with an empty strip and no control at all."""
        css = VIEWER_CSS.read_text(encoding="utf-8")
        assert "body.story:not(.panel-collapsed) #panel header" in css
        assert "body.story #panel header {" not in css

    def test_the_district_row_puts_every_catalog_field_in_a_text_node(self) -> None:
        """The number, the label and the year are LOC catalog free text, and
        the row that carried them as interpolated markup was safe only by a
        grammar two modules away. `no-unsanitized/property` is what now costs a
        written reason for every markup write on every surface; this is the one
        row where building elements instead is the point."""
        assert "go.append(cell(" in VIEWER_JS.read_text(encoding="utf-8")

    def test_the_viewer_never_suppresses_a_focus_outline(self) -> None:
        """`outline: none` on the two inputs was the defect: a keyboard visitor
        got no signal at all on the opacity slider. Re-adding it anywhere is
        the same defect, and it is an absence no rendering test can see."""
        assert "outline: none" not in VIEWER_CSS.read_text(encoding="utf-8")

    def test_both_pages_link_every_committed_icon(self) -> None:
        """A blank tab and a `/favicon.ico` 404 on every visit was the defect,
        and it is one a browser reports nowhere the tests look.

        The hrefs must stay relative: the local server puts the page under
        `/viewer/` while the bundle is itself the web root, so an absolute path
        is only right for one of them. The `.ico` link has to precede the
        `.svg` one — a browser takes the last format it understands, and the
        reverse order hands every modern browser the raster.
        """
        for name in ICON_FILES:
            path = VIEWER_DIR / name
            assert path.is_file() and path.stat().st_size, f"missing icon asset {name}"
        for page in (VIEWER_HTML, WALK_HTML):
            head = page.read_text(encoding="utf-8").split("</head>", 1)[0]
            for name in ICON_FILES:
                assert f'href="{name}"' in head, f"{page.name} does not link {name}"
            assert 'href="/favicon' not in head, f"{page.name} links an icon from the server root"
            assert head.index('href="favicon.ico"') < head.index('href="favicon.svg"')

    def test_vendored_assets_exist(self) -> None:
        vendor = VIEWER_DIR / "vendor"
        for name in (
            "maplibre-gl.js",
            "maplibre-gl.css",
            "maplibre-gl-compare.js",
            "maplibre-gl-compare.css",
            "pmtiles.js",
            "fonts/fonts.css",
        ):
            assert (vendor / name).is_file(), f"missing vendored asset {name}"
        fonts_css = (vendor / "fonts" / "fonts.css").read_text()
        assert "gstatic" not in fonts_css  # fonts fully local too

    def test_vector_basemap_assets_are_self_hosted(self) -> None:
        """The style's glyphs and sprite must resolve inside the bundle: a
        style straight from `generate_style` points them at protomaps.github.io,
        which puts a third party back in the render path."""
        basemap = VIEWER_DIR / "vendor" / "basemap"
        styles = sorted(basemap.glob("style-*.json"))
        assert styles, "no vendored basemap styles — this test would pass vacuously"
        for style_path in styles:
            style = json.loads(style_path.read_text(encoding="utf-8"))
            assert style["glyphs"].startswith("vendor/basemap/fonts/")
            assert style["sprite"].startswith("vendor/basemap/sprites/")
            assert "://" not in style["glyphs"] and "://" not in style["sprite"]
            # the archive location is a deployment fact, filled from the manifest
            assert style["sources"]["protomaps"]["url"] == ""
            for stack in _font_stacks(style):
                # Every range, not just the Latin ones this city's labels reach:
                # a missing range is a silently unlabelled feature, and first
                # paint on Chicago already asks for 8192-8447. A re-vendor that
                # ships only 0-255 must fail here, not in someone's browser.
                ranges = sorted((basemap / "fonts" / stack).glob("*.pbf"))
                assert len(ranges) == 256, (
                    f"{style_path.name}: glyph stack {stack} has {len(ranges)} of 256 ranges"
                )
            sprite = basemap / "sprites" / f"{Path(style['sprite']).name}.png"
            assert sprite.is_file(), f"{style_path.name} references unvendored {sprite.name}"


def test_the_page_reads_its_manifest_once() -> None:
    """A static deploy's manifest cannot change between page loads, so a re-poll
    finds only what it already has. The timer that did it removed layers and
    sources and dropped entries from the visitor's own `hidden` set, forever, in
    every open tab.

    An ABSENCE, so a grep is right. `setInterval` alone is too narrow — a
    recursive `setTimeout` would slip past — so the manifest read is counted and
    the cache-busting header is named too.
    """
    script = VIEWER_JS.read_text(encoding="utf-8")
    assert "setInterval" not in script
    assert "no-store" not in script
    assert script.count("readJson(manifestHref)") == 1
    assert "manifestDelta" not in VIEWER_LIB.read_text(encoding="utf-8")
