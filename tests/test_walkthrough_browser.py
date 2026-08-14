"""The walkthrough page, loaded by a browser and stepped through end to end.

Display-only makes this the whole test. With no behaviour to speak of, what can
break is a plate that does not resolve, and a broken image is invisible in the
DOM, in the console and in a source grep.

Needs a headless browser on PATH; without one these skip.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, cast

import pytest

from browser import ROOT, load_page, serve

VIEWER_DIR = ROOT / "viewer"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The real page files and the committed plates, on a loopback server."""
    root = tmp_path_factory.mktemp("walkthrough-bundle")
    shutil.copytree(VIEWER_DIR, root / "viewer")
    # Nothing is synthesized at the served root: the page links its own icons,
    # so `page.console == ()` below covers the fallback /favicon.ico request too.
    return root


#: Steps through every panel and every state, and reports what each plate drew.
#: `naturalWidth` is the only thing that separates a rendered figure from a
#: broken-image box, and no console message distinguishes them either.
WALK_EVERY_PLATE = """
(async () => {
  const plate = document.getElementById('plate');
  const steps = [...document.querySelectorAll('#steps button')];
  const seen = [];
  for (const step of steps) {
    step.click();
    const states = [...document.querySelectorAll('#states button')];
    for (const state of (states.length ? states : [null])) {
      if (state) state.click();
      await new Promise((done) => {
        if (plate.complete && plate.naturalWidth) return done();
        plate.addEventListener('load', done, {once: true});
        plate.addEventListener('error', done, {once: true});
      });
      const order = document.getElementById('order-note');
      seen.push({
        step: step.textContent,
        state: state ? state.textContent : 'main',
        src: plate.getAttribute('src'),
        width: plate.naturalWidth,
        title: document.getElementById('title').textContent,
        caption: document.getElementById('caption').textContent.length,
        tally: document.getElementById('tally-note').textContent,
        // shown only where the JSON carries one, so both readings are asserted
        note: order.offsetParent === null ? '' : order.textContent,
      });
    }
  }
  return seen;
})()
"""


def test_the_walkthrough_draws_every_panel_and_every_plate(bundle: Path) -> None:
    """Twenty-one panels and every prepared state, stepped through in a browser.

    Display-only makes this cheap and makes it the whole test: with no
    behaviour to speak of, what can break is a plate that does not resolve, and
    that is invisible in the DOM, in the console and in a source grep.
    """
    with serve(bundle) as base_url:
        page = load_page(
            f"{base_url}/viewer/walkthrough.html",
            until='document.body.dataset.ready === "1"',
            capture={
                "seen": WALK_EVERY_PLATE,
                "steps": 'document.querySelectorAll("#steps button").length',
            },
        )

    assert page.captured["steps"] == 21, "the walkthrough lost a panel"
    seen = cast("list[dict[str, Any]]", page.captured["seen"])
    # every assertion below iterates this, so an empty one passes them all. It
    # was empty for a while: the capture returned an un-awaited Promise.
    assert len(seen) >= page.captured["steps"], f"the walk stopped: {seen}"
    broken = [row["src"] for row in seen if not row["width"]]
    assert not broken, f"plates that did not load: {broken}"
    assert len({row["src"] for row in seen}) == len(seen), "two states share one plate"
    assert all(row["title"] for row in seen)
    assert all(row["caption"] > 200 for row in seen), "a panel lost its explanation"
    assert all(row["tally"] for row in seen), "the running tally went silent"
    # a panel's note is the only place it says it is told out of run order, or
    # drawn from a different atlas. It was styled but never rendered for a while,
    # which nothing saw, so both readings are pinned here.
    panels = json.loads((bundle / "viewer" / "walkthrough" / "panels.json").read_text())
    expected = {p["title"]: p["note"] for p in panels["panels"]}
    shown = {row["title"]: row["note"] for row in seen}
    assert shown == expected, "a note is missing from the page, or shown where there is none"
    assert any(expected.values()), "no panel carries a note, so this asserts nothing"
    assert page.console == (), f"the page complained: {page.console}"


def test_a_walkthrough_link_opens_at_its_own_step(bundle: Path) -> None:
    """The page writes these links into the address bar, so it has to read them
    back: a step and, where a panel has prepared states, which one."""
    with serve(bundle) as base_url:
        page = load_page(
            f"{base_url}/viewer/walkthrough.html#step=10&state=handedness",
            until='document.body.dataset.ready === "1"',
            capture={
                "title": 'document.getElementById("title").textContent',
                "src": 'document.getElementById("plate").getAttribute("src")',
                "pressed": '[...document.querySelectorAll("#states button")]'
                '.filter(b => b.getAttribute("aria-pressed") === "true").map(b => b.textContent)',
            },
        )

    assert "say no" in cast("str", page.captured["title"])
    assert "handedness" in cast("str", page.captured["src"])
    assert page.captured["pressed"] == ["Not a mirror image"]


def test_the_walkthrough_offers_the_maps_and_the_source(bundle: Path) -> None:
    """This page is where a reader arrives asking how, so it owes them both
    ways on. The map link is relative and resolved here; the source link is
    read as the browser PARSED it, since escaped markup renders the same text.
    """
    with serve(bundle) as base_url:
        page = load_page(
            f"{base_url}/viewer/walkthrough.html",
            until='document.body.dataset.ready === "1"',
            capture={
                "links": 'JSON.stringify([...document.querySelectorAll("#outro a")]'
                '.map(a => [a.getAttribute("href"), a.textContent]))',
                "maps": 'fetch("index.html").then(r => r.status)',
            },
        )

    hrefs, labels = zip(*json.loads(cast("str", page.captured["links"])), strict=True)
    assert hrefs == ("index.html", "https://github.com/matt-hendrick/autogeoref")
    assert "Read the source" in labels[1]
    assert page.captured["maps"] == 200, "the map link does not resolve on the bundle"
    assert page.console == (), f"the page complained: {page.console}"
