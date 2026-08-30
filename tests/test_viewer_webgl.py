"""What the viewer does in a browser that will not render it.

Every map on the page is WebGL. A browser can refuse a 3D context while working
otherwise, and the boot used to die on that throw with the overlay still up.

The support probe is a pure decision and runs in node. Whether the page then
says so needs a real load, so those tests want a headless browser.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from js_support import viewer
from viewer_browser_support import load as _load
from viewer_browser_support import manifest_for as _manifest

#: Overlay at rest with a reason on screen, no fade.
SAID = 'document.getElementById("loading").classList.contains("choose")'

#: Overlay faded: something decided the pane is no longer empty.
FADED = 'document.getElementById("loading").classList.contains("done")'

#: Refuse every WebGL context, leaving 2d alone. Installed before the vendored
#: map so nothing holds one when the boot runs.
REFUSE_WEBGL = """<script>
(function () {
  var real = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (name) {
    return /webgl/i.test(name) ? null : real.apply(this, arguments);
  };
})();
</script>
"""

#: A boot failure unrelated to WebGL. `chooseCity` runs after an await, so it
#: rejects the boot rather than breaking the parse.
BREAK_BOOT = """<script>
ViewerLib.chooseCity = function () { throw new Error("injected boot failure"); };
</script>
"""

#: Dropping `style.load` stops `drawAtlas`, so no tile paints — the one state
#: only the deadline ends. Shortening the long timer avoids a 20s wall clock; it
#: is the only timer near that length here or in the vendored map. That value is
#: not the contract under test — where it is ARMED is.
STARVE_THE_ATLAS = """<script>
(function () {
  var on = maplibregl.Map.prototype.on;
  maplibregl.Map.prototype.on = function (type) {
    return type === "style.load" ? this : on.apply(this, arguments);
  };
  var later = window.setTimeout;
  window.setTimeout = function (fn, ms) {
    return later(fn, ms >= 5000 ? 50 : ms);
  };
})();
</script>
"""

MAP_TAG = '<script src="vendor/maplibre-gl.js"></script>'
APP_TAG = '<script src="app.js"></script>'


def _bundle_with(bundle: Path, tmp_path: Path, snippet: str, before: str) -> Path:
    """A private copy of ``bundle`` with ``snippet`` spliced ahead of ``before``.

    Copied because `viewer_bundle` is module-scoped: editing in place would hand
    every sibling test a page that cannot draw.
    """
    root = tmp_path / "bundle"
    shutil.copytree(bundle, root)
    page = root / "viewer" / "index.html"
    html = page.read_text(encoding="utf-8")
    assert before in html, f"the script tag {before!r} moved; this test is stale"
    page.write_text(html.replace(before, snippet + before, 1), encoding="utf-8")
    return root


def test_the_probe_accepts_either_context_and_refuses_a_throw() -> None:
    """`webglAvailable` answers on what the browser hands back.

    WebGL 2 is preferred, WebGL 1 is enough because the map falls back to it,
    and a driver that raises during creation is a refusal.
    """
    assert viewer("L.webglAvailable(function () { return null; })") is False
    assert viewer("L.webglAvailable(function (n) { return n === 'webgl2' ? {} : null; })") is True
    assert viewer("L.webglAvailable(function (n) { return n === 'webgl' ? {} : null; })") is True
    assert viewer("L.webglAvailable(function () { throw new Error('no driver'); })") is False


def test_the_probe_releases_the_context_it_opened() -> None:
    """A probe that kept its context would spend one of the browser's few slots
    to learn nothing. No extension is still a yes."""
    tracked = (
        "(function () {"
        "  var lost = 0;"
        "  var stub = { getExtension: function () {"
        "    return { loseContext: function () { lost++; } };"
        "  } };"
        "  return [L.webglAvailable(function () { return stub; }), lost];"
        "})()"
    )
    assert viewer(tracked) == [True, 1]
    no_extension = (
        "L.webglAvailable(function () { return { getExtension: function () { return null; } }; })"
    )
    assert viewer(no_extension) is True


def test_a_release_that_throws_does_not_change_the_answer() -> None:
    """Releasing is best effort.

    An earlier version wrapped creation and release in one try, so a context
    whose `getExtension` raised was reported as no WebGL at all.
    """
    raises_on_extension = (
        "L.webglAvailable(function () {"
        "  return { getExtension: function () { throw new Error('gone'); } };"
        "})"
    )
    assert viewer(raises_on_extension) is True
    raises_on_lose = (
        "L.webglAvailable(function () {"
        "  return { getExtension: function () {"
        "    return { loseContext: function () { throw new Error('gone'); } };"
        "  } };"
        "})"
    )
    assert viewer(raises_on_lose) is True


def test_a_browser_refusing_webgl_is_told_why(viewer_bundle: Path, tmp_path: Path) -> None:
    """The overlay must rest in a state that explains itself.

    Not `done`: fading leaves a blank page saying nothing, the same dead end by
    another route.
    """
    root = _bundle_with(viewer_bundle, tmp_path, REFUSE_WEBGL, MAP_TAG)
    page = _load(root, _manifest("/base/tile.png"), until=SAID)

    overlay = page.element('id="loading"')
    assert "WebGL" in overlay, f"the overlay never named the cause: {overlay}"
    assert "done" not in overlay, "the overlay faded out over a page with no map"
    assert not page.errors, f"the page threw rather than explaining: {page.errors}"


def test_any_throw_in_the_boot_still_brings_the_overlay_to_rest(
    viewer_bundle: Path, tmp_path: Path
) -> None:
    """The probe only covers the failure we know about.

    One long async function: a throw anywhere skips every remaining line,
    including whatever would have cleared the overlay. This injects one the
    probe cannot see.
    """
    root = _bundle_with(viewer_bundle, tmp_path, BREAK_BOOT, APP_TAG)
    page = _load(root, _manifest("/base/tile.png"), until=SAID)

    overlay = page.element('id="loading"')
    assert "could not start" in overlay, f"the overlay never came to rest: {overlay}"
    assert "done" not in overlay, "the overlay faded out over a page with no map"
    logged = [m.text for m in page.errors]
    assert any("injected boot failure" in text for text in logged), (
        f"the cause was swallowed instead of logged: {logged}"
    )


def test_the_deadline_clears_the_overlay_when_the_atlas_never_draws(
    viewer_bundle: Path, tmp_path: Path
) -> None:
    """The failsafe's job is a page where `drawAtlas` never runs.

    Armed inside `drawAtlas` it needed a loaded style first, so it guarded a
    missing archive but not a map that never got going. Moved back there, no
    timer is scheduled here at all.
    """
    root = _bundle_with(viewer_bundle, tmp_path, STARVE_THE_ATLAS, APP_TAG)
    page = _load(root, _manifest("/base/tile.png"), until=FADED)

    overlay = page.element('id="loading"')
    assert "done" in overlay, f"the overlay never came down: {overlay}"
    assert "choose" not in overlay, "a working browser was told something was wrong"
