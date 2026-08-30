"""How the viewer survives an edge that drops the first range request.

An archive over the CDN's cacheable size limit answers its first range request
with the whole object. pmtiles rejects that, the district goes missing with
nothing on screen, and only a reload brings it back — the second request is
always correct. Worse, pmtiles cancels that response only when it owns the
abort controller, and for tile reads it does not, so the archive arrives in full
for bytes nothing will read.

`readWithRetry` owns both behaviours and runs in node, which is the only tier
that can see whose signal reached the read. The browser tier checks that it is
wired into the real library at all.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from js_support import viewer
from viewer_browser_support import DREW
from viewer_browser_support import load as _load
from viewer_browser_support import manifest_for as _manifest

#: The message pmtiles throws when a range request comes back whole.
BYTE_SERVING = (
    "Server returned no content-length header or content-length exceeding "
    "request. Check that your storage backend supports HTTP Byte Serving."
)

#: Fail the FIRST range request for each archive byte range, then behave. The
#: oversized content-length is what pmtiles rejects; a browser that refuses to
#: set that header leaves none, which it rejects too.
COLD_EDGE = """<script>
(function () {
  window.__headerCalls = 0;
  var real = window.fetch;
  var seen = {};
  window.fetch = function (input, init) {
    var url = String(input && input.url ? input.url : input);
    var head = init && init.headers;
    var range = head && typeof head.get === "function" ? head.get("range") : null;
    var key = url + "|" + range;
    if (range && url.indexOf(".pmtiles") !== -1) {
      if (range === "bytes=0-16383") window.__headerCalls += 1;
      if (!seen[key] || FOREVER) {
        seen[key] = true;
        return Promise.resolve(new Response(new ArrayBuffer(8), {
          status: 200,
          headers: { "content-length": String(4 * 1024 * 1024 * 1024) },
        }));
      }
    }
    return real.apply(this, arguments);
  };
})();
</script>
"""

#: Attempts on the archive header, the one range every load reads exactly once.
#: A global maximum would count MapLibre's own repeat requests for a tile.
HEADER_CALLS = "window.__headerCalls || 0"

MAP_TAG = '<script src="vendor/maplibre-gl.js"></script>'

#: A read that never settles, for driving the abort paths.
HANGS = "function (owned) { seen = owned; return new Promise(function () {}); }"


def _bundle_with(bundle: Path, tmp_path: Path, snippet: str) -> Path:
    """A private copy of ``bundle`` with ``snippet`` spliced ahead of the map.

    Copied because `viewer_bundle` is module-scoped: editing in place would hand
    every sibling test a page whose tiles fail.
    """
    root = tmp_path / "bundle"
    shutil.copytree(bundle, root)
    page = root / "viewer" / "index.html"
    html = page.read_text(encoding="utf-8")
    assert MAP_TAG in html, "the vendored map script tag moved; this test is stale"
    page.write_text(html.replace(MAP_TAG, snippet + MAP_TAG, 1), encoding="utf-8")
    return root


def test_only_a_byte_serving_failure_is_worth_retrying() -> None:
    """The library throws a plain Error, so the message is the discriminator.

    Everything else has to propagate. An ETag mismatch in particular is the
    library's own recovery signal, and retrying it would swallow that.
    """
    assert viewer(f"L.coldRangeFailure(new Error({json.dumps(BYTE_SERVING)}))") is True
    assert viewer("L.coldRangeFailure(new Error('Bad response code: 404'))") is False
    assert viewer("L.coldRangeFailure(new Error('The user aborted a request.'))") is False
    assert viewer("L.coldRangeFailure(new Error('Server returned non-matching ETag'))") is False


def test_the_predicate_survives_anything_a_rejection_can_carry() -> None:
    """A rejected fetch need not carry an Error at all, and a predicate that
    throws inside a catch block would replace the failure with its own."""
    assert viewer("L.coldRangeFailure(null)") is False
    assert viewer("L.coldRangeFailure(undefined)") is False
    assert viewer("L.coldRangeFailure({})") is False
    assert viewer("L.coldRangeFailure('a string')") is False


def test_the_read_is_given_a_signal_of_its_own_and_that_signal_is_aborted() -> None:
    """The whole point: the caller's signal cannot be relied on to cancel.

    pmtiles aborts a rejected response only when it owns the controller, so a
    read handed MapLibre's signal leaves the archive arriving. Handing the read
    a signal this owns, and aborting it on failure, is what stops that — and the
    caller's own signal must NOT be aborted, since the caller still owns it.
    """
    outcome = viewer(
        "(function () {"
        "  var outer = new AbortController();"
        "  var seen = null;"
        "  return L.readWithRetry(function (owned) {"
        "    seen = owned;"
        "    return Promise.reject(new Error('Bad response code: 404'));"
        "  }, outer.signal).then(function () { return 'resolved'; }, function () {"
        "    return [seen !== outer.signal, seen.aborted, outer.signal.aborted];"
        "  });"
        "})()"
    )
    assert outcome == [True, True, False]


def test_the_callers_abort_still_reaches_the_read() -> None:
    """MapLibre cancels tile requests constantly while panning.

    Stock pmtiles hands its signal straight to fetch, so cancelling reaches the
    network. Owning a signal would quietly end that unless the caller's aborts
    relay into it.
    """
    relayed = viewer(
        "(function () {"
        "  var outer = new AbortController();"
        "  var seen = null;"
        "  L.readWithRetry(" + HANGS + ", outer.signal);"
        "  outer.abort();"
        "  return Promise.resolve().then(function () { return seen.aborted; });"
        "})()"
    )
    assert relayed is True


def test_a_signal_already_aborted_stops_the_read_before_it_starts() -> None:
    """A tile cancelled before its read begins must not reach the network."""
    aborted = viewer(
        "(function () {"
        "  var outer = new AbortController();"
        "  outer.abort();"
        "  var seen = null;"
        "  L.readWithRetry(" + HANGS + ", outer.signal);"
        "  return seen.aborted;"
        "})()"
    )
    assert aborted is True


def test_the_relay_is_released_once_the_read_settles() -> None:
    """One listener per read, on a signal MapLibre reuses for the whole request.
    Left attached they accumulate for as long as the page is open."""
    counts = viewer(
        "(function () {"
        "  var added = 0, removed = 0;"
        "  var fake = { aborted: false,"
        "    addEventListener: function () { added += 1; },"
        "    removeEventListener: function () { removed += 1; } };"
        "  return L.readWithRetry(function () { return Promise.resolve('ok'); }, fake)"
        "    .then(function () { return [added, removed]; });"
        "})()"
    )
    assert counts == [1, 1]


def test_a_cold_read_is_retried_once_and_only_a_cold_read() -> None:
    """Two attempts for the failure a second attempt fixes, one for anything
    else, and never a third — a retry that re-entered on its own failure would
    hammer the archive for as long as the page is open."""
    recovered = viewer(
        "(function () {"
        "  var tries = 0;"
        "  return L.readWithRetry(function () {"
        "    tries += 1;"
        "    if (tries === 1) return Promise.reject(new Error(" + json.dumps(BYTE_SERVING) + "));"
        "    return Promise.resolve('bytes');"
        "  }, null).then(function (v) { return [v, tries]; });"
        "})()"
    )
    assert recovered == ["bytes", 2]

    def attempts(message: str) -> object:
        return viewer(
            "(function () {"
            "  var tries = 0;"
            "  return L.readWithRetry(function () {"
            "    tries += 1;"
            "    return Promise.reject(new Error(" + json.dumps(message) + "));"
            "  }, null).then(function () { return 'resolved'; }, function () { return tries; });"
            "})()"
        )

    assert attempts(BYTE_SERVING) == 2, "a cold read was not retried, or was retried twice"
    assert attempts("Bad response code: 404") == 1, "an unrelated failure was retried"


def test_an_edge_that_drops_the_first_range_still_draws_the_atlas(
    viewer_bundle: Path, tmp_path: Path
) -> None:
    """The retry, wired into the real library over the real page.

    The console is the discriminator: without the retry pmtiles reports the
    failure there and the district is missing. The overlay is not — the
    basemap's own tiles bring it down either way.
    """
    root = _bundle_with(viewer_bundle, tmp_path, COLD_EDGE.replace("FOREVER", "false"))
    page = _load(root, _manifest("/base/tile.png"), until=DREW, capture={"header": HEADER_CALLS})

    assert not page.errors, f"the retry did not absorb the failure: {page.errors}"
    assert page.captured["header"] == 2, "the stub never failed a read; the test proved nothing"


def test_a_persistent_failure_is_retried_once_over_the_real_library(
    viewer_bundle: Path, tmp_path: Path
) -> None:
    """The bound holds through pmtiles' own caching and retry paths.

    Counted on the archive header, which one load reads exactly once, so the
    number is the retry and not MapLibre asking for a tile again.
    """
    root = _bundle_with(viewer_bundle, tmp_path, COLD_EDGE.replace("FOREVER", "true"))
    page = _load(
        root,
        _manifest("/base/tile.png"),
        until="(window.__headerCalls || 0) >= 2",
        settle_s=3.0,
        capture={"header": HEADER_CALLS},
    )

    assert page.captured["header"] == 2, "the header read was attempted more than twice"
