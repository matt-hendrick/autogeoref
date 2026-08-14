"""Where the viewer puts its panel and its divider, at screen shapes that hurt.

Two tiers of the same subject, together because the second is what the first
is for. `viewer/lib.js` decides where the compare handle may sit on each axis,
run here in node against known numbers; the browser half then loads the page at
a phone, a phone on its side and a short window, and asks the layout what it
actually did with those answers.

The wide, comfortable window is `test_frontend_browser`; where a manifest lives
on disk, which is the other sense of the word layout, is `test_viewer_layout`.
Needs ``node`` and a headless browser; without either these skip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from js_support import viewer
from viewer_browser_support import captured_json
from viewer_browser_support import load as _load
from viewer_browser_support import manifest_for as _manifest_for
from viewer_browser_support import manifest_with_many_districts as _many

# ---------------------------------------------------------------------------
# the decisions, run in node
# ---------------------------------------------------------------------------


def test_the_handle_is_held_clear_of_a_floating_panel_and_of_both_edges() -> None:
    """The compare clamp, run rather than grepped: a handle parked behind the
    floating panel can never be grabbed again, and one off either edge is
    half-invisible. A docked panel takes no map width, so it blocks nothing."""
    floating = "{width: 1200, edge: 24, panelRight: 338}"
    docked = "{width: 400, edge: 24, panelRight: null}"
    assert viewer(f"L.clampSlider(10, {floating})") == 362  # out from under the panel
    assert viewer(f"L.clampSlider(600, {floating})") == 600  # already legal, untouched
    assert viewer(f"L.clampSlider(9999, {floating})") == 1176  # held off the right edge
    assert viewer(f"L.clampSlider(2, {docked})") == 24  # only the edge margin applies
    assert viewer(f"L.clampSlider(390, {docked})") == 376


def test_the_clamp_survives_a_panel_wider_than_the_window_and_a_junk_position() -> None:
    """The floor may exceed the ceiling on a narrow window, and there the floor
    has to win: an inverted range or a NaN would strand the handle off-screen,
    which is the one failure the clamp exists to prevent."""
    assert viewer("L.clampSlider(500, {width: 300, edge: 24, panelRight: 400})") == 424
    assert viewer("L.clampSlider(NaN, {width: 1200, edge: 24, panelRight: 338})") == 362
    assert viewer("L.clampSlider(null, {width: 1200, edge: 24, panelRight: null})") == 24


def test_arrow_keys_step_the_divider_and_page_keys_jump_it() -> None:
    """The vendored compare answers a pointer and nothing else, so the keys are
    this project's decision. Up/right raise and down/left lower, because that
    is the direction a screen reader announces a slider against."""
    assert viewer("L.swipeStep('ArrowRight', 0.5, false)") == pytest.approx(0.52)
    assert viewer("L.swipeStep('ArrowUp', 0.5, false)") == pytest.approx(0.52)
    assert viewer("L.swipeStep('ArrowLeft', 0.5, false)") == pytest.approx(0.48)
    assert viewer("L.swipeStep('ArrowDown', 0.5, false)") == pytest.approx(0.48)
    assert viewer("L.swipeStep('PageUp', 0.5, false)") == pytest.approx(0.6)
    assert viewer("L.swipeStep('PageDown', 0.5, false)") == pytest.approx(0.4)
    # shift makes an arrow jump as far as a Page key
    assert viewer("L.swipeStep('ArrowRight', 0.5, true)") == pytest.approx(0.6)
    assert viewer("L.swipeStep('Home', 0.5, false)") == 0
    assert viewer("L.swipeStep('End', 0.5, false)") == 1


def test_a_key_the_divider_does_not_answer_is_left_to_the_page() -> None:
    """`null` is not "no movement" — it is what tells `app.js` to skip its
    `preventDefault`, so Tab still leaves the handle and a typed character
    still reaches the page."""
    for key in ("Tab", "Enter", " ", "a", "Escape", "F5"):
        assert viewer(f"L.swipeStep({key!r}, 0.5, false)") is None, key


def test_a_key_named_like_an_object_member_is_not_a_direction() -> None:
    """The lookup is over arrays for this reason: `{ArrowLeft: 1}['constructor']`
    is a function, which reads as a hit and would slide the divider on a
    keystroke that never named a direction."""
    for key in ("constructor", "toString", "__proto__", "hasOwnProperty"):
        assert viewer(f"L.swipeStep({key!r}, 0.5, false)") is None, key


def test_the_divider_stops_at_the_ends_and_survives_a_junk_position() -> None:
    """0 and 1 are the fractions of the map's WIDTH, not legal pixel positions:
    `clampSlider` holds the handle off the edge afterwards, and pre-clamping
    here would apply the margin twice."""
    assert viewer("L.swipeStep('ArrowLeft', 0.01, false)") == 0
    assert viewer("L.swipeStep('ArrowRight', 0.99, false)") == 1
    assert viewer("L.swipeStep('PageDown', 0.0, false)") == 0
    # a width of zero divides to NaN before this is ever called
    assert viewer("L.swipeStep('ArrowRight', NaN, false)") == pytest.approx(0.52)


#: The handle's own geometry, as `app.js` measures it: half of 44px, and the
#: depth of the era plates and the sources control along the top edge.
RADIUS, HEADROOM = 22, 88


def handle(map_height: float | None, panel_top: float | None) -> object:
    """`handleTop` for a map of ``map_height`` with a panel starting at ``panel_top``.

    ``None`` on either is the measurement the page could not take, which the
    caller passes straight through rather than substituting a number for.
    """
    args = {"mapHeight": map_height, "panelTop": panel_top}
    given = ", ".join(f"{k}: {'null' if v is None else v}" for k, v in args.items())
    return viewer(f"L.handleTop({{{given}, headroom: {HEADROOM}, radius: {RADIUS}}})")


def test_the_handle_moves_into_the_strip_a_docked_panel_leaves() -> None:
    """The vertical half of the same problem. The vendored plugin pins the
    handle halfway down the map; a panel docked across the bottom of a phone
    covers that, and a handle nobody can grab takes the swipe with it. The
    answer is the middle of the strip left over, clear of the chrome along the
    top edge — and `null` where nothing is covering anything, which leaves the
    stylesheet's own 50% in place."""
    assert handle(667, 253) == 171  # midway between the chrome and the sheet
    assert handle(667, 275) == 182  # the strip grows, the handle follows it down
    assert handle(800, 800) is None, "a panel that covers nothing must not move the handle"
    assert handle(800, None) is None, "an unmeasurable panel must not move the handle"
    assert handle(None, 200) is None, "an unmeasurable map must not move the handle"


def test_a_strip_thinner_than_the_handle_puts_it_above_the_panel_not_under_it() -> None:
    """There is no position clear of both the top chrome and the panel here, so
    one of them wins. It has to be the panel: it is opaque across the whole
    width and covers whatever it reaches, while the chrome is two boxes at one
    side that the handle is usually nowhere near. Centring on the strip instead
    puts the whole handle behind the panel, which is the one outcome this
    function exists to prevent."""
    for map_height, panel_top in ((400, 30), (400, 60), (300, 100), (667, 75)):
        top = handle(map_height, panel_top)
        assert isinstance(top, int)
        assert top - RADIUS < panel_top, (
            f"the handle at {top} is entirely behind a panel starting at {panel_top}"
        )
        assert top >= RADIUS, f"the handle at {top} hangs off the top of the map"
        assert top <= map_height - RADIUS, f"the handle at {top} hangs off the bottom"


# ---------------------------------------------------------------------------
# the layouts, loaded in a browser
# ---------------------------------------------------------------------------

#: Scrolls the panel to its end and reports what a reader can reach: how much
#: of it scrolls, how much the masthead above the scroll region took, and
#: whether the compare handle ended up somewhere a thumb could get to it.
PANEL_FIT = """
(() => {
  const panel = document.getElementById('panel').getBoundingClientRect();
  const body = document.getElementById('panel-body');
  const rows = [...document.querySelectorAll('#volumes .vol')];
  const scrolls = body.scrollHeight > body.clientHeight + 1;
  body.scrollTop = body.scrollHeight;
  const foot = document.querySelector('#panel footer').getBoundingClientRect();
  const last = rows[rows.length - 1].getBoundingClientRect();
  const handle = document.querySelector('.compare-swiper-vertical').getBoundingClientRect();
  return {
    rows: rows.length,
    scrolls: scrolls,
    scrollRegion: Math.round(body.clientHeight),
    masthead: Math.round(panel.height - body.clientHeight),
    footerInside: foot.bottom <= panel.bottom + 1,
    lastRowInside: last.top >= panel.top - 1 && last.bottom <= panel.bottom + 1,
    handleClear: handle.right <= panel.left + 1 || handle.left >= panel.right - 1
      || handle.bottom <= panel.top + 1,
    handleOnScreen: handle.top >= 0 && handle.bottom <= window.innerHeight,
    // what a thumb aimed at the middle of the handle would actually press,
    // and how much of the handle is not behind the panel
    handleHit: (() => {
      const hit = document.elementFromPoint(
        Math.round((handle.left + handle.right) / 2),
        Math.round((handle.top + handle.bottom) / 2));
      return hit && hit.classList.contains('compare-swiper-vertical')
        ? 'swiper' : (hit ? (hit.id || hit.tagName) : 'nothing');
    })(),
    handleVisibleFrac: (() => {
      const overlapsX = handle.right > panel.left && handle.left < panel.right;
      const floor = Math.min(handle.bottom, window.innerHeight,
                             overlapsX ? panel.top : Infinity);
      return Math.max(0, floor - Math.max(handle.top, 0)) / handle.height;
    })(),
    panelLeft: Math.round(panel.left),
    panelWidth: Math.round(panel.width),
    mapAbovePanel: Math.round(Math.max(0, panel.top)),
    overlaysMap: getComputedStyle(document.documentElement)
      .getPropertyValue('--panel-overlays-map').trim(),
    pageScrollsX: document.documentElement.scrollWidth > window.innerWidth + 1,
  };
})()
"""


#: The page has reached the exact point the overlay used to come down at:
#: `drawAtlas` renders the district list on the line that used to clear the
#: loading screen. Waiting on this rather than on the overlay is what lets the
#: test below ask a question about the overlay at all.
LISTED = "document.querySelectorAll('#volumes .vol').length > 0"


def test_the_loading_overlay_waits_for_a_tile_not_for_the_style(
    viewer_bundle: Path,
) -> None:
    """The overlay used to be dismissed on `style.load`, which fires when the
    style JSON is parsed — a long way before anything is on screen. On a
    throttled line that left both panes empty grey behind a fully drawn panel,
    which reads as broken rather than loading.

    Nothing here can paint: no basemap is configured and the atlas archive does
    not exist. So the page reaches the old dismissal point, and the overlay has
    to still be up. The console is not asserted empty — a missing archive is
    the point of the fixture, and it complains.
    """
    manifest = _manifest_for(None)
    manifest["volumes"][0]["pmtiles"] = "no-such-archive.pmtiles"
    page = _load(
        viewer_bundle,
        manifest,
        until=LISTED,
        settle_s=1.5,
        capture={"done": 'document.getElementById("loading").classList.contains("done")'},
    )

    assert page.captured["done"] is False, (
        "the loading overlay came down with nothing painted in either pane"
    )


#: Focuses the divider and drives it by key, reporting what actually moved.
#: A synthetic `keydown` on the element, because that is where the handler is
#: bound and the question is whether the handler exists and is wired to the
#: clamped path — not whether the browser routes keystrokes, which it does.
KEYBOARD_SWIPE = """
(() => {
  const swiper = document.querySelector('.compare-swiper-vertical');
  const width = document.getElementById('compare-wrap').getBoundingClientRect().width;
  const at = () => window.compare.currentPosition / width;
  const send = (key, shift) => {
    const ev = new KeyboardEvent('keydown',
      {key: key, shiftKey: !!shift, bubbles: true, cancelable: true});
    swiper.dispatchEvent(ev);
    return ev.defaultPrevented;
  };
  swiper.focus();
  const focused = document.activeElement === swiper;
  const start = at();
  const rightHandled = send('ArrowRight', false);
  const afterRight = at();
  send('ArrowLeft', false);
  const backAgain = at();
  const ignoredHandled = send('a', false);
  const afterIgnored = at();
  send('End', false);
  const atEnd = at();
  const endValueNow = Number(swiper.getAttribute('aria-valuenow'));
  send('Home', false);
  const atHome = at();
  return {
    focusable: swiper.tabIndex === 0,
    role: swiper.getAttribute('role'),
    labelled: Boolean(swiper.getAttribute('aria-label')),
    focused: focused,
    rightHandled: rightHandled,
    ignoredHandled: ignoredHandled,
    movedRight: afterRight - start,
    returned: Math.abs(backAgain - start) < 1e-9,
    ignoredMoved: Math.abs(afterIgnored - backAgain) > 1e-9,
    endBeyondRight: atEnd > afterRight,
    homeBeforeStart: atHome < start,
    // what a screen reader is told, against where the handle actually is
    valueNowMatchesEnd: Math.abs(endValueNow - Math.round(atEnd * 100)) <= 1,
    valueText: swiper.getAttribute('aria-valuetext'),
    // the clamp still owns the ends: End must not park the handle off-screen
    endOnScreen: swiper.getBoundingClientRect().right <= window.innerWidth + 1,
    homeReachable: swiper.getBoundingClientRect().left >= -1,
  };
})()
"""


def test_the_divider_can_be_driven_by_keyboard(viewer_bundle: Path) -> None:
    """The page's central control was pointer-only: the vendored compare binds
    `mousedown` and `touchstart` and nothing else, so a reader without a
    pointer could not move the line between then and now at all.

    Driven rather than grepped, and the two halves that a source read would
    miss are here: an unanswered key must leave `preventDefault` alone, or the
    handle would swallow Tab and trap focus on itself; and Home/End must still
    land somewhere reachable, because the fraction this returns is clamped to
    the layout afterwards and not before."""
    page = _load(viewer_bundle, _many(4), capture={"keys": KEYBOARD_SWIPE})

    keys: Any = page.captured["keys"]
    assert keys["focusable"] is True, "the divider cannot be reached by Tab"
    assert keys["role"] == "slider"
    assert keys["labelled"] is True, "the divider is announced with no name"
    assert keys["focused"] is True
    assert keys["rightHandled"] is True, "ArrowRight did not reach the divider"
    assert keys["movedRight"] > 0, "ArrowRight did not move the divider right"
    assert keys["returned"] is True, "ArrowLeft did not undo ArrowRight"
    assert keys["ignoredHandled"] is False, "a plain letter was swallowed by the divider"
    assert keys["ignoredMoved"] is False, "a plain letter moved the divider"
    assert keys["endBeyondRight"] is True
    assert keys["homeBeforeStart"] is True
    assert keys["valueNowMatchesEnd"] is True, "the announced value is not where the handle is"
    assert keys["valueText"] and "atlas" in keys["valueText"]
    assert keys["endOnScreen"] is True, "End pushed the handle off the right edge"
    assert keys["homeReachable"] is True, "Home pushed the handle off the left edge"
    assert page.console == (), f"the page complained: {page.console}"


#: A story over the probe city, minimal but with two stops — one arrow key has
#: to be able to move BETWEEN them for the collision below to be visible.
def _story_manifest() -> dict[str, Any]:
    manifest = _manifest_for("/base/tile.png")
    centre = manifest["site"]["home_point"]
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
                    "camera": {"center": list(centre), "zoom": 13},
                },
                {
                    "id": "yards",
                    "title": "The yards",
                    "camera": {"center": [centre[0] + 0.01, centre[1]], "zoom": 13},
                },
            ],
        }
    ]
    return manifest


#: With the divider focused inside a story, send one ArrowRight and report what
#: BOTH controls did.
STORY_KEY = """
(() => {
  const swiper = document.querySelector('.compare-swiper-vertical');
  const width = document.getElementById('compare-wrap').getBoundingClientRect().width;
  const stop = () => (document.querySelector('#story h2') || {}).textContent || '';
  const at = () => window.compare.currentPosition / width;
  swiper.focus();
  const beforeStop = stop(), beforePos = at();
  swiper.dispatchEvent(new KeyboardEvent('keydown',
    {key: 'ArrowRight', bubbles: true, cancelable: true}));
  const afterStop = stop(), afterPos = at();
  // Escape is a key the divider does NOT answer, so it must still reach the
  // story and leave it even while the handle holds focus
  swiper.dispatchEvent(new KeyboardEvent('keydown',
    {key: 'Escape', bubbles: true, cancelable: true}));
  return {
    beforeStop, afterStop, movedDivider: afterPos - beforePos,
    leftStory: !document.getElementById('story-nav').classList.contains('on'),
  };
})()
"""


def test_the_divider_keys_do_not_also_drive_the_story(viewer_bundle: Path) -> None:
    """The story listens for its own arrows on `window`, and the divider's
    handler bubbles there. Without `stopPropagation` one ArrowRight nudged the
    divider AND advanced the stop — and a stop that declares a `swipe` then
    overwrote the nudge, so the reader was moved somewhere they never asked to
    go, in the one mode where the divider matters most.

    Escape is asserted in the same breath because the fix must not go too far:
    the divider answers no Escape, so leaving a story has to keep working while
    the handle holds focus."""
    page = _load(
        viewer_bundle,
        _story_manifest(),
        query="?stories=1",
        fragment="#story=probe-story&stop=wharf",
        until='document.querySelector("#story h2")',
        capture={"run": STORY_KEY},
    )

    run: Any = page.captured["run"]
    assert run["beforeStop"] == "The wharf"
    assert run["afterStop"] == "The wharf", "an arrow on the divider advanced the story stop"
    assert run["movedDivider"] > 0, "the arrow did not move the divider"
    assert run["leftStory"] is True, "Escape no longer leaves the story from the divider"
    assert page.console == (), f"the page complained: {page.console}"


def test_the_era_chips_say_which_are_on_in_aria_not_only_in_colour(
    viewer_bundle: Path,
) -> None:
    """The chips are toggles and any subset may be on. The `active` class
    paints that; without `aria-pressed` nothing announces it, so a reader who
    cannot see the colour cannot tell which eras are drawn."""
    page = _load(
        viewer_bundle,
        _many(4),
        capture={
            "chips": (
                "JSON.stringify([...document.querySelectorAll('#eras button')]"
                ".map(b => ({on: b.classList.contains('active'), "
                "pressed: b.getAttribute('aria-pressed')})))"
            )
        },
    )

    chips = captured_json(page, "chips")
    assert chips, "no era chips rendered"
    for chip in chips:
        assert chip["pressed"] == ("true" if chip["on"] else "false"), chip
    assert page.console == (), f"the page complained: {page.console}"


def test_a_short_window_can_still_reach_every_district(viewer_bundle: Path) -> None:
    """The panel's fixed parts come to some 600px. The district index used to
    be the only elastic child, so on a short window it collapsed to its own
    padding — no district visible at all — and the credit line was pushed out
    through the bottom of the card to float on the map.

    Asserted after scrolling the panel to its end: the last of 30 districts is
    inside the card, and so is the credit line the licence obliges. This
    viewport is where the old layout fails BOTH — a taller one hides the footer
    half of the defect."""
    page = _load(viewer_bundle, _many(30), viewport=(1280, 620), capture={"fit": PANEL_FIT})

    fit: Any = page.captured["fit"]
    assert fit["rows"] == 30
    assert fit["scrolls"] is True, "the panel does not scroll, so its overflow is unreachable"
    assert fit["lastRowInside"] is True, "the last district cannot be scrolled into the panel"
    assert fit["footerInside"] is True, "the credit line renders outside the panel"
    assert fit["pageScrollsX"] is False
    assert page.console == (), f"the page complained: {page.console}"


#: The smallest phone still in use, and the one the sheet was measured on. The
#: SMALL one leads: the layout this replaced left it 34px of scroll region
#: against a 136px masthead, while at 375x667 it cleared the same bar by 3px —
#: so a test written only at the larger size passes on the defect.
PHONE_PORTRAIT = [(320, 568), (375, 667)]


@pytest.mark.parametrize(("width", "height"), PHONE_PORTRAIT)
def test_a_phone_sheet_is_not_spent_on_its_own_masthead(
    viewer_bundle: Path, width: int, height: int
) -> None:
    """The masthead above the sheet's scroll region does not scroll, so a
    sheet sized to clear the compare handle by arithmetic was spent entirely
    on it: the chips, the districts and the credit line shared tens of pixels.

    Asserted as a split rather than a pixel count — the part a reader can
    scroll must be at least the part they cannot. The handle is asked for
    separately: the sheet may only grow because script moves the handle into
    the strip above it."""
    page = _load(
        viewer_bundle,
        _many(30),
        viewport=(width, height),
        fragment="#panel=open",
        capture={"fit": PANEL_FIT},
    )

    fit: Any = page.captured["fit"]
    assert fit["overlaysMap"] == "0", "the panel is meant to dock at this width"
    assert fit["rows"] == 30
    assert fit["scrolls"] is True
    assert fit["scrollRegion"] >= fit["masthead"], (
        f"the sheet is mostly masthead: {fit['masthead']}px fixed "
        f"against {fit['scrollRegion']}px a reader can scroll"
    )
    assert fit["lastRowInside"] is True
    assert fit["footerInside"] is True
    assert fit["handleClear"] is True, "the compare handle is behind the sheet"
    assert fit["handleOnScreen"] is True
    assert fit["pageScrollsX"] is False
    assert page.console == (), f"the page complained: {page.console}"


#: Narrow AND short, which no layout here is shaped for: too narrow for the
#: column, too short for the sheet to take a useful share. What it must not do
#: is take a share anyway and swallow the divider.
@pytest.mark.parametrize(("width", "height"), [(400, 260), (400, 300), (480, 320)])
def test_a_window_too_small_for_any_layout_still_leaves_the_divider(
    viewer_bundle: Path, width: int, height: int
) -> None:
    """The sheet's floor is what binds here, and a floor set for comfort takes
    the map instead: at 260px of height a 220px sheet leaves 40px of atlas, and
    the handle — the only way to work the comparison — ends up behind it.

    Two bars, because a panel can fail either of them alone. The map keeps
    enough height to be a comparison at all, and the handle is GRABBABLE rather
    than merely present: asked of the topmost element at the point a thumb
    would land on."""
    page = _load(
        viewer_bundle,
        _many(30),
        viewport=(width, height),
        fragment="#panel=open",
        capture={"fit": PANEL_FIT},
    )

    fit: Any = page.captured["fit"]
    assert fit["mapAbovePanel"] >= 130, (
        f"the panel left {fit['mapAbovePanel']}px of map on a {height}px window"
    )
    assert fit["handleHit"] == "swiper", (
        f"the handle is not the thing under a thumb at its centre: {fit['handleHit']}"
    )
    assert fit["handleVisibleFrac"] >= 0.9, (
        f"only {fit['handleVisibleFrac']:.0%} of the handle clears the panel"
    )
    assert fit["pageScrollsX"] is False
    assert page.console == (), f"the page complained: {page.console}"


#: What the page settled on, against what the same measurement says NOW — and
#: what is left when the script's answer is taken away again.
HANDLE_SETTLED = """
(() => {
  const swiper = document.querySelector('.compare-swiper-vertical');
  const root = document.documentElement;
  const map = document.getElementById('compare-wrap').getBoundingClientRect();
  const panel = document.getElementById('panel').getBoundingClientRect();
  const settled = parseFloat(getComputedStyle(swiper).top);
  const fresh = ViewerLib.handleTop({
    mapHeight: map.height, panelTop: panel.top - map.top, headroom: 88, radius: 22});
  root.style.removeProperty('--handle-top');
  const withoutScript = parseFloat(getComputedStyle(swiper).top);
  const rect = swiper.getBoundingClientRect();
  return {
    settled: settled, fresh: fresh, mapHeight: map.height,
    withoutScript: withoutScript,
    fallbackOnScreen: rect.top >= 0 && rect.bottom <= window.innerHeight,
  };
})()
"""


def test_the_handle_is_placed_from_where_the_panel_ended_up(viewer_bundle: Path) -> None:
    """The panel settles in on a transform, which moves the box this is
    measured from and not the box a resize observer reports — so the only
    reading taken during it was of a panel still arriving, and nothing came
    afterwards to correct it.

    Then the other half: taking the script's answer away must leave the handle
    halfway down the map, which is what the stylesheet says. It is not free —
    a `var()` that substitutes something unusable computes to `auto` and puts
    the handle off the top of the map, not back on the declared fallback."""
    page = _load(
        viewer_bundle,
        _many(30),
        viewport=(320, 568),
        fragment="#panel=open",
        settle_s=1.5,
        capture={"handle": HANDLE_SETTLED},
    )

    got: Any = page.captured["handle"]
    assert abs(got["settled"] - got["fresh"]) <= 1, (
        f"the handle is at {got['settled']} and the settled layout wants {got['fresh']}"
    )
    assert abs(got["withoutScript"] - got["mapHeight"] / 2) <= 1, (
        f"without the property the handle is at {got['withoutScript']}, "
        f"not halfway down a {got['mapHeight']}px map"
    )
    assert got["fallbackOnScreen"] is True
    assert page.console == (), f"the page complained: {page.console}"


def test_a_phone_on_its_side_gets_a_column_and_not_a_letterbox(viewer_bundle: Path) -> None:
    """Height is the scarce axis in landscape, and a bottom sheet spends it
    twice — on the masthead and on the map strip it leaves — so neither half
    is usable. The panel goes back to a column at the leading edge, which puts
    it over the map again, which is what the divider clamp needs to know."""
    page = _load(
        viewer_bundle,
        _many(30),
        viewport=(667, 375),
        fragment="#panel=open",
        capture={"fit": PANEL_FIT},
    )

    fit: Any = page.captured["fit"]
    assert fit["overlaysMap"] == "1", "a column overlays the map and the clamp must be told"
    assert fit["panelLeft"] < 20 and fit["panelWidth"] < 340, "the panel is not a bottom sheet"
    assert fit["scrollRegion"] >= fit["masthead"]
    assert fit["scrolls"] is True
    assert fit["lastRowInside"] is True
    assert fit["footerInside"] is True
    assert fit["handleClear"] is True, "the compare handle is behind the panel"
    assert fit["pageScrollsX"] is False
    assert page.console == (), f"the page complained: {page.console}"
