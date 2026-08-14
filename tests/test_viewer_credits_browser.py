"""What the page says about where its pixels came from, loaded by a browser.

Three claims share one composed line and two surfaces: the source of the scans,
who georeferenced them, and the basemap under both. A basemap credit is a
licence obligation, so it has to survive a collapsed panel and must never
appear for a basemap nobody could see — and the colophon beside it is the way
out to the method and to the code, which no manifest supplies.

Needs a headless browser on PATH; without one these skip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from viewer_browser_support import (
    CREDITED,
    DREW,
    PROBE_BOUNDS,
    SETTLE_S,
)
from viewer_browser_support import captured_json as _captured_json
from viewer_browser_support import links as _links
from viewer_browser_support import load as _load
from viewer_browser_support import manifest_for as _manifest
from viewer_browser_support import vector_manifest as _vector_manifest


def test_the_viewer_credits_a_basemap_that_actually_drew(viewer_bundle: Path) -> None:
    """The other half of §6: the credit is owed the moment a tile lands, and a
    fix that never credits anything would satisfy the absence test alone. This
    waits for the credit rather than for the page, so it fails by timing out if
    a working basemap stops being credited."""
    page = _load(viewer_bundle, _manifest("/base/tile.png"), until=f"{DREW} && {CREDITED}")
    footer = page.element('id="footer"')
    assert "Probe fixture" in footer
    assert "basemap © OpenStreetMap contributors" in footer
    assert "basemap © OpenStreetMap contributors" in page.element('id="attrib-text"')


def test_the_credit_renders_as_html_so_it_can_carry_a_link(viewer_bundle: Path) -> None:
    """`optional_credits` is HTML, and the point of that is a link back to the
    method — a page that says "auto-georeferenced" owes the reader a way to go
    and look.

    Both surfaces, because the same string is written twice: the footer, and
    the sources control a collapsed panel leaves behind.
    """
    manifest = _manifest("/base/tile.png")
    manifest["site"]["optional_credits_html"] = (
        'auto-georeferenced by <a href="https://example.invalid/method">this pipeline</a>'
    )
    page = _load(
        viewer_bundle,
        manifest,
        capture={"footer": _links("#footer"), "attrib": _links("#attrib-text")},
    )

    expected = [["https://example.invalid/method", "this pipeline"]]
    assert _captured_json(page, "footer") == expected
    assert _captured_json(page, "attrib") == expected
    assert "&lt;a" not in page.element('id="footer"'), "the markup was escaped, not rendered"


def test_two_eras_credits_keep_their_links(viewer_bundle: Path) -> None:
    """Credit lines are deduped WHOLE, not split on the separator.

    Splitting cut an anchor in half: the opening fragment deduped away against
    another era's, and that era's credit rendered as bare text with its link
    gone, no console error and nothing to see. A credit is HTML now, so the
    join layer has to treat it that way.
    """
    manifest = _manifest("/base/tile.png")
    manifest["volumes"].append(
        {**manifest["volumes"][0], "id": "probe_002", "era": "1900s", "year": 1904}
    )
    manifest["site"]["era_credits"] = {
        "1890s": '<a href="/m">Alpha · Beta</a>',
        "1900s": '<a href="/m">Alpha · Gamma</a>',
    }
    # both chips on: only SELECTED eras contribute a credit, so one era alone
    # could never show the fragments colliding
    manifest["site"]["default_eras"] = ["1890s", "1900s"]
    page = _load(viewer_bundle, manifest, capture={"footer": _links("#footer")})

    assert _captured_json(page, "footer") == [
        ["/m", "Alpha · Beta"],  # oldest era first
        ["/m", "Alpha · Gamma"],
    ]
    assert page.console == (), f"the page complained: {page.console}"


def test_the_new_credits_key_wins_over_the_pre_rename_one(viewer_bundle: Path) -> None:
    """A partially migrated manifest can carry both. The current spelling wins;
    the fallback is a floor, not an override."""
    manifest = _manifest("/base/tile.png")
    manifest["site"]["optional_credits_html"] = "the current spelling"
    manifest["site"]["default_credits_html"] = "the retired spelling"
    footer = _load(viewer_bundle, manifest).element('id="footer"')
    assert "the current spelling" in footer
    assert "the retired spelling" not in footer


def test_a_manifest_written_before_the_credits_rename_still_shows_its_credit(
    viewer_bundle: Path,
) -> None:
    """A published manifest outlives the code that wrote it. A site published
    before `default_credits` became `optional_credits` keeps its credit until it
    is republished — losing it would be silent, which is the whole hazard."""
    manifest = _manifest("/base/tile.png")
    manifest["site"]["default_credits_html"] = "placed by the old spelling"
    page = _load(viewer_bundle, manifest)
    assert "placed by the old spelling" in page.element('id="footer"')


def test_a_city_with_no_credit_publishes_no_credit(viewer_bundle: Path) -> None:
    """Optional means optional. The page must still render — the footer just
    carries the source and basemap lines and says nothing about placement."""
    page = _load(viewer_bundle, _manifest("/base/tile.png"), until=f"{DREW} && {CREDITED}")
    footer = page.element('id="footer"')
    assert "Probe fixture" in footer
    assert "auto-georeferenced" not in footer
    assert page.console == (), f"the page complained: {page.console}"


def test_the_colophon_reaches_the_method_and_the_source(viewer_bundle: Path) -> None:
    """The map page's two ways out, and the one surface that carries them for
    every city — the probe city configures no credit and still gets both.

    The walkthrough href is relative and RESOLVED here, not just read: the two
    roots differ (the local server puts the page under /viewer/, the bundle IS
    the web root), and a rename of the page it points at breaks a link that
    still greps clean.
    """
    page = _load(
        viewer_bundle,
        _manifest("/base/tile.png"),
        capture={
            "links": _links("#colophon"),
            "method": 'fetch("walkthrough.html").then(r => r.status)',
        },
    )

    hrefs, labels = zip(*_captured_json(page, "links"), strict=True)
    assert hrefs == ("walkthrough.html", "https://github.com/matt-hendrick/autogeoref")
    assert "How this map was made" in labels[0]
    assert "Source on GitHub" in labels[1]
    assert page.captured["method"] == 200, "the method link does not resolve on the bundle"
    assert page.console == (), f"the page complained: {page.console}"


def test_a_basemap_that_loaded_no_tile_is_not_credited(viewer_bundle: Path) -> None:
    """The regression test for a false attribution claim.

    The flag was set where the raster style was BUILT — before a single tile
    was requested — so a basemap nobody could reach was still credited in the
    footer. A visitor cannot check the tiles; the credit line is the one part
    of the failure they would actually read. The atlas must still draw, which
    is the other half: a basemap failure may not take the page with it.
    """
    page = _load(viewer_bundle, _manifest("/base/absent-{z}-{x}-{y}.png"), settle_s=SETTLE_S)

    assert 'class="done"' in page.element('id="loading"')
    assert "Probe District" in page.element('id="vol-ct"')
    footer = page.element('id="footer"')
    assert "Probe fixture" in footer
    assert "OpenStreetMap" not in footer, "credited a basemap that loaded nothing"
    # the same line in both places, so a collapsed panel cannot disagree
    assert "OpenStreetMap" not in page.element('id="attrib-text"')


def test_a_vector_basemap_is_credited_only_once_its_archive_answers(
    viewer_bundle: Path,
) -> None:
    """The configuration that actually deploys.

    A vector basemap is a vendored STYLE over a hosted ARCHIVE, and the style
    is always present because it ships in this bundle — so crediting when the
    style parses says nothing about whether a visitor can see a basemap. What a
    deploy gets wrong is the archive path, and `viewer/deploy.py` rewrites that
    one to a public bucket.
    """
    page = _load(
        viewer_bundle,
        _vector_manifest("../basemap.pmtiles"),
        until=(
            f"{DREW} && {CREDITED}"
            " && window.beforeMap.getMaxBounds() !== null"
            " && window.afterMap.getMaxBounds() !== null"
        ),
        capture={
            "before": "window.beforeMap.getMaxBounds().toArray().flat()",
            "after": "window.afterMap.getMaxBounds().toArray().flat()",
        },
    )
    assert "basemap © OpenStreetMap contributors, Protomaps" in page.element('id="footer"')
    # A self-hosted basemap is an EXTRACT, and BOTH panes are held inside its
    # footprint or a zoom-out lands in a void with a city-shaped patch in it.
    # The footprint arrives with the archive's own metadata, so it is not
    # readable where the style loads — which is why the clamp hangs off the
    # same event as the credit and not off `load`, an event a missing archive
    # never fires. (The archive stores degrees as e7 integers.)
    for pane in ("before", "after"):
        assert page.captured[pane] == pytest.approx(PROBE_BOUNDS, abs=1e-6), pane


def test_a_vector_basemap_whose_archive_404s_is_not_credited(viewer_bundle: Path) -> None:
    """Same shape, archive missing — one wrong path in a manifest. The atlas
    still draws; the credit names two third parties whose data is not on the
    page, so it must not appear."""
    page = _load(viewer_bundle, _vector_manifest("../absent-basemap.pmtiles"), settle_s=SETTLE_S)

    assert 'class="done"' in page.element('id="loading"')
    assert "Probe District" in page.element('id="vol-ct"')
    footer = page.element('id="footer"')
    assert "Probe fixture" in footer
    assert "OpenStreetMap" not in footer, "credited a basemap whose archive 404s"
    assert "Protomaps" not in footer
    assert "OpenStreetMap" not in page.element('id="attrib-text"')
