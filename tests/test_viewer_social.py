"""What a crawler, a link unfurler and a 404 get from the deploy bundle.

The pages are city-fact-free by contract, so their title, description and
share-card tags cannot live in the markup and are generated here instead — for
the same reason `config.js` is. `robots.txt`, `sitemap.xml` and `404.html` are
in the bundle rather than the tree because two of them need the site's own URL,
and because the platform answers every unmatched path with `index.html` under a
200 until a `404.html` is present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autogeoref.viewer.deploy import (
    CARD_FILE,
    CARD_MIN,
    SOCIAL_MARKER,
    DeployError,
    build_deploy_bundle,
    public_site_url,
)
from viewer_support import DEPLOY_CITY as CITY
from viewer_support import deploy_viewer_dir


@pytest.fixture
def viewer_dir(tmp_path: Path) -> Path:
    """Its own fixture over the shared builder: a pytest fixture imported from
    another test module reads to the linter as a redefinition."""
    return deploy_viewer_dir(tmp_path)


def _bundle_with_site(
    viewer_dir: Path, out_dir: Path, site: dict[str, object], **kwargs: object
) -> Path:
    """Rebuild the fixture bundle with ``site`` as the manifest's site block."""
    path = viewer_dir / CITY / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["site"] = site
    path.write_text(json.dumps(manifest))
    build_deploy_bundle(viewer_dir, out_dir, "https://tiles.example.com", city=CITY, **kwargs)  # type: ignore[arg-type]
    return out_dir / "index.html"


def test_the_bundle_publishes_a_real_title_and_share_card(viewer_dir: Path, tmp_path: Path) -> None:
    """The page cannot carry these itself — it is city-fact-free by contract,
    and a share card is nothing but city facts. Nor can the script fill them
    in: a crawler reads the markup and never runs it, which is why every link
    to the site unfurled as the placeholder title with no description."""
    page = _bundle_with_site(
        viewer_dir,
        tmp_path / "deploy",
        {"name": "Testville", "kicker": "Fire Insurance Atlas", "dek": "Every pink block."},
        site_url="https://atlas.example.com",
    ).read_text()

    assert "<title>Testville — Fire Insurance Atlas</title>" in page
    assert "<title>Placeholder</title>" not in page, "the placeholder title shipped as well"
    assert page.count("<title>") == 1, "two titles leave the crawler reading the first"
    assert '<meta name="description" content="Every pink block.">' in page
    assert '<meta property="og:title" content="Testville — Fire Insurance Atlas">' in page
    assert '<meta property="og:description" content="Every pink block.">' in page
    assert '<meta name="twitter:card" content="summary">' in page
    assert '<meta property="og:url" content="https://atlas.example.com/">' in page
    assert '<link rel="canonical" href="https://atlas.example.com/">' in page


def test_without_a_site_url_the_absolute_tags_are_left_out_rather_than_guessed(
    viewer_dir: Path, tmp_path: Path
) -> None:
    """A wrong `og:url` sends everyone who shares the page somewhere else, so
    the tags that need one are omitted. The title and description do not need
    one and still ship — most of a share card's value is in those."""
    out = tmp_path / "deploy"
    page = _bundle_with_site(viewer_dir, out, {"name": "Testville", "dek": "A dek."}).read_text()

    assert "<title>Testville</title>" in page
    assert '<meta property="og:description" content="A dek.">' in page
    assert "og:url" not in page
    assert "canonical" not in page
    assert not (out / "sitemap.xml").exists(), "a sitemap of relative URLs is not a sitemap"
    # robots.txt still ships: most of its job here is EXISTING, so the platform
    # stops answering /robots.txt with the atlas page under a 200
    assert (out / "robots.txt").read_text() == "User-agent: *\nAllow: /\n"


def test_a_site_url_names_both_pages_in_the_sitemap_and_robots(
    viewer_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "deploy"
    _bundle_with_site(viewer_dir, out, {"name": "Testville"}, site_url="https://atlas.example.com/")

    assert "Sitemap: https://atlas.example.com/sitemap.xml" in (out / "robots.txt").read_text()
    sitemap = (out / "sitemap.xml").read_text()
    assert "<loc>https://atlas.example.com/</loc>" in sitemap
    assert "<loc>https://atlas.example.com/walkthrough</loc>" in sitemap


def test_the_walkthrough_keeps_its_own_copy_rather_than_being_told_it(
    viewer_dir: Path, tmp_path: Path
) -> None:
    """The walkthrough is about the pipeline, not about a city, so its title
    and description are project facts and live in the page. The deploy mirrors
    them into the card instead of restating them, which is what keeps one
    sentence from having two homes that can drift apart."""
    walk = viewer_dir / "walkthrough.html"
    walk.write_text(
        "<title>How it works</title>\n"
        '<meta name="description" content="A walkthrough.">\n' + SOCIAL_MARKER
    )
    out = tmp_path / "deploy"
    _bundle_with_site(viewer_dir, out, {"name": "Testville"}, site_url="https://atlas.example.com")
    page = (out / "walkthrough.html").read_text()

    assert '<meta property="og:title" content="How it works">' in page
    assert '<meta property="og:description" content="A walkthrough.">' in page
    assert '<meta property="og:url" content="https://atlas.example.com/walkthrough">' in page
    # the page's own copy is the INPUT to the generated block, so it comes out
    # with the title: shipped twice, a reader of two takes the first
    assert page.count('name="description"') == 1, "the description shipped twice"
    assert page.count("<title>") == 1
    # the city's name is not the walkthrough's subject and must not appear
    assert "Testville" not in page


def test_a_page_that_lost_its_marker_is_refused(viewer_dir: Path, tmp_path: Path) -> None:
    """Silently shipping the placeholder is the failure this replaced. It would
    look exactly like a working deploy until someone shared the link."""
    (viewer_dir / "index.html").write_text("<title>Placeholder</title>\n<p>no marker</p>")
    with pytest.raises(DeployError, match="marker"):
        build_deploy_bundle(viewer_dir, tmp_path / "deploy", "https://tiles.example.com", city=CITY)


def test_a_name_with_html_in_it_cannot_break_out_of_its_attribute(
    viewer_dir: Path, tmp_path: Path
) -> None:
    """Every value here is operator-supplied config interpolated into markup."""
    page = _bundle_with_site(
        viewer_dir,
        tmp_path / "deploy",
        {"name": 'Ye "Olde" <b>Towne</b> & Co', "dek": "A & B"},
    ).read_text()

    assert "<b>Towne</b>" not in page
    assert "&lt;b&gt;Towne&lt;/b&gt;" in page
    assert '<meta property="og:description" content="A &amp; B">' in page
    assert 'content="Ye &quot;Olde&quot;' in page


@pytest.mark.parametrize(
    "bad",
    [
        "atlas.example.com",  # no scheme
        "ftp://atlas.example.com",
        "https://user:pw@atlas.example.com",
        "https://atlas.example.com/?a=1",
        "https://atlas.example.com/#frag",
    ],
)
def test_an_unusable_site_url_is_refused_rather_than_published(bad: str) -> None:
    with pytest.raises(DeployError):
        public_site_url(bad)


def test_a_site_url_is_normalized_the_way_the_tiles_base_is() -> None:
    assert public_site_url("https://atlas.example.com/") == "https://atlas.example.com"
    assert public_site_url("https://atlas.example.com//sub//dir/") == (
        "https://atlas.example.com/sub/dir"
    )
    assert public_site_url(None) is None


def test_a_page_that_lost_its_marker_leaves_no_half_built_bundle(
    viewer_dir: Path, tmp_path: Path
) -> None:
    """The refusal is raised with the file checks, before the first write. Any
    later and the operator is left with a bundle directory holding a manifest
    and some of its pages, which uploads as a broken site."""
    (viewer_dir / "walkthrough.html").write_text("<title>No marker</title>")
    out_dir = tmp_path / "deploy"
    with pytest.raises(DeployError, match="marker"):
        build_deploy_bundle(viewer_dir, out_dir, "https://tiles.example.com", city=CITY)

    assert not (out_dir / "manifest.json").exists(), "a refused bundle wrote its manifest anyway"
    assert not (out_dir / "index.html").exists()


def test_copy_read_back_out_of_markup_is_not_escaped_twice(
    viewer_dir: Path, tmp_path: Path
) -> None:
    """The walkthrough's copy is read back out of HTML, where it is already
    entity-encoded, and the tag builder escapes what it is given. Escaping it
    again ships `Maps &amp;amp; Atlases`, which unfurls literally."""
    (viewer_dir / "walkthrough.html").write_text(
        "<title>Maps &amp; Atlases</title>\n"
        '<meta name="description" content="A &amp; B, &quot;quoted&quot;">\n' + SOCIAL_MARKER
    )
    out = tmp_path / "deploy"
    _bundle_with_site(viewer_dir, out, {"name": "Testville"})
    page = (out / "walkthrough.html").read_text()

    assert "<title>Maps &amp; Atlases</title>" in page
    assert "&amp;amp;" not in page, "the copy was escaped twice"
    assert 'content="A &amp; B, &quot;quoted&quot;"' in page


# ---------------------------------------------------------------------------
# the share card
# ---------------------------------------------------------------------------


def _put_card(viewer_dir: Path, size: tuple[int, int] = (1200, 630)) -> Path:
    """Stage a card of ``size`` beside the fixture city's manifest."""
    from PIL import Image

    card = viewer_dir / CITY / CARD_FILE
    Image.new("RGB", size, (200, 180, 160)).save(card, "JPEG")
    return card


def test_a_staged_card_is_shipped_and_claimed_as_a_large_one(
    viewer_dir: Path, tmp_path: Path
) -> None:
    """`og:image` has to be absolute or most readers drop it, so it rides the
    same site URL the canonical link does."""
    _put_card(viewer_dir)
    out = tmp_path / "deploy"
    page = _bundle_with_site(
        viewer_dir, out, {"name": "Testville"}, site_url="https://atlas.example.com"
    ).read_text()

    assert (out / CARD_FILE).is_file(), "the card was not copied into the bundle"
    assert '<meta property="og:image" content="https://atlas.example.com/og.jpg">' in page
    assert '<meta property="og:image:width" content="1200">' in page
    assert '<meta property="og:image:height" content="630">' in page
    assert '<meta name="twitter:card" content="summary_large_image">' in page
    # the walkthrough is shared as often as the atlas and gets the same card
    assert "og:image" in (out / "walkthrough.html").read_text()


def test_a_city_with_no_card_claims_the_small_one(viewer_dir: Path, tmp_path: Path) -> None:
    """The common case. A large card declared without an image renders an empty
    frame in some readers, which is worse than the small card it replaced."""
    out = tmp_path / "deploy"
    page = _bundle_with_site(
        viewer_dir, out, {"name": "Testville"}, site_url="https://atlas.example.com"
    ).read_text()

    assert not (out / CARD_FILE).exists()
    assert "og:image" not in page
    assert '<meta name="twitter:card" content="summary">' in page


def test_a_card_with_no_site_url_is_not_claimed_at_all(viewer_dir: Path, tmp_path: Path) -> None:
    """The case that would otherwise ship a broken card: the image exists, but
    without an absolute URL there is no way to name it, and claiming the large
    card anyway is what draws the empty frame."""
    _put_card(viewer_dir)
    out = tmp_path / "deploy"
    page = _bundle_with_site(viewer_dir, out, {"name": "Testville"}).read_text()

    assert "og:image" not in page
    assert '<meta name="twitter:card" content="summary">' in page


def test_an_unreadable_card_stops_the_build_before_it_writes(
    viewer_dir: Path, tmp_path: Path
) -> None:
    """Nothing downstream would report it: the only thing that ever looks at a
    card is someone else's crawler. And the refusal has to come before the
    first write, or it leaves a half-built bundle that uploads as a broken
    site."""
    (viewer_dir / CITY / CARD_FILE).write_bytes(b"not an image")
    out = tmp_path / "deploy"
    with pytest.raises(DeployError, match="readable image"):
        build_deploy_bundle(viewer_dir, out, "https://tiles.example.com", city=CITY)

    assert not (out / "manifest.json").exists(), "a refused bundle wrote its manifest anyway"
    assert not (out / "index.html").exists()


def test_the_cards_declared_size_is_the_files_own(viewer_dir: Path, tmp_path: Path) -> None:
    """Read from the image rather than assumed, so a card rendered at another
    size cannot be advertised at the wrong one."""
    _put_card(viewer_dir, (800, 418))
    page = _bundle_with_site(
        viewer_dir,
        tmp_path / "deploy",
        {"name": "Testville"},
        site_url="https://atlas.example.com",
    ).read_text()

    assert '<meta property="og:image:width" content="800">' in page
    assert '<meta property="og:image:height" content="418">' in page


def test_a_card_too_small_to_show_is_refused(viewer_dir: Path, tmp_path: Path) -> None:
    """The second route to the empty frame this whole gate exists to avoid. A
    reader drops a large-card image below its own floor and falls back to the
    small card — with the image already claimed, which is what draws the
    frame. Refused rather than downgraded: a card that small is a mistake."""
    _put_card(viewer_dir, (CARD_MIN[0] - 1, CARD_MIN[1]))
    out = tmp_path / "deploy"
    with pytest.raises(DeployError, match="large"):
        build_deploy_bundle(viewer_dir, out, "https://tiles.example.com", city=CITY)

    assert not (out / "manifest.json").exists(), "a refused bundle wrote its manifest anyway"


def test_a_card_at_the_floor_is_accepted(viewer_dir: Path, tmp_path: Path) -> None:
    """The other side of the same bar, so the floor cannot drift up unnoticed."""
    _put_card(viewer_dir, CARD_MIN)
    page = _bundle_with_site(
        viewer_dir,
        tmp_path / "deploy",
        {"name": "Testville"},
        site_url="https://atlas.example.com",
    ).read_text()

    assert f'<meta property="og:image:width" content="{CARD_MIN[0]}">' in page


def test_a_truncated_card_is_caught_rather_than_measured(viewer_dir: Path, tmp_path: Path) -> None:
    """`Image.open` parses a header and decodes nothing, so a half-copied card
    measures perfectly and fails in the reader. A truncated file is the most
    likely way to get exactly the bytes-no-reader-can-decode this refuses."""
    card = viewer_dir / CITY / CARD_FILE
    _put_card(viewer_dir)
    whole = card.read_bytes()
    card.write_bytes(whole[: len(whole) // 3])

    with pytest.raises(DeployError, match="readable image"):
        build_deploy_bundle(viewer_dir, tmp_path / "deploy", "https://tiles.example.com", city=CITY)


def test_a_card_no_tag_can_reach_is_not_shipped(viewer_dir: Path, tmp_path: Path) -> None:
    """With no site URL there is no absolute URL to name it by, so the tags are
    not emitted — and the bytes must not be either, or the bundle carries a
    file nothing references. The copy and the tags answer to one condition."""
    _put_card(viewer_dir)
    out = tmp_path / "deploy"
    page = _bundle_with_site(viewer_dir, out, {"name": "Testville"}).read_text()

    assert "og:image" not in page
    assert not (out / CARD_FILE).exists(), "shipped a card that nothing points at"
