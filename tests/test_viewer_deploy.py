"""The deploy bundle: strip local-only fields, rewrite archive URLs, refuse a leak.

The bundle is the last thing between a local manifest and a public page, so it
copies every page file, rewrites each archive path onto the public tiles base,
and hard-stops on a leak token — the check exists because a private host once
rode out in a merged manifest. An invalid tiles base is refused rather than
joined.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autogeoref.viewer.deploy import (
    CONFIG_FILE,
    ICON_FILES,
    PAGE_FILES,
    PLATFORM_FILES,
    DeployError,
    build_deploy_bundle,
    mapbox_config_js,
    public_tiles_base,
)
from viewer_support import DEPLOY_CITY as CITY
from viewer_support import VIEWER_DIR, deploy_viewer_dir, page_stub

#: Shaped like a real public token: `pk.` and a base64url payload.
PUBLIC_TOKEN = "pk.eyJ1IjoidGVzdCJ9.abc-DEF_123"


@pytest.fixture
def viewer_dir(tmp_path: Path) -> Path:
    return deploy_viewer_dir(tmp_path)


def test_deploy_bundle_strips_and_rewrites(viewer_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "deploy"
    build_deploy_bundle(viewer_dir, out_dir, "https://tiles.example.com/", city=CITY)
    bundle = json.loads((out_dir / "manifest.json").read_text())

    by_id = {v["id"]: v for v in bundle["volumes"]}
    # v1: own pmtiles -> kept, rewritten to the public base
    assert by_id["v1"]["pmtiles"] == "https://tiles.example.com/v1.pmtiles"
    # v2: no archive of its own -> not in the public list
    assert "v2" not in by_id
    # v3: own pmtiles -> kept, rewritten to the public base
    assert by_id["v3"]["pmtiles"] == "https://tiles.example.com/v3.pmtiles"
    assert "era_pmtiles" not in bundle  # citywide era layers are retired
    assert "era_provenance" not in bundle  # provenance is not a manifest field
    assert bundle["site"] == {"name": "Testville"}
    # a stale on-disk rename table is dropped, not carried into the bundle
    assert "street_aliases" not in bundle
    # static assets ride along — every page file, not just the entry point:
    # a bundle missing app.js deploys as a page that draws nothing, and one
    # missing an icon deploys as a blank tab and a 404 on every visit
    for name in (*PAGE_FILES, *ICON_FILES, *PLATFORM_FILES):
        assert (out_dir / name).exists(), f"the bundle dropped {name}"
    assert (out_dir / "vendor" / "maplibre-gl.js").exists()


def test_the_bundle_carries_a_one_entry_city_index(viewer_dir: Path, tmp_path: Path) -> None:
    """The deployed page resolves its city exactly the way the local one does.

    Leaving the index out would work — the page falls back to a manifest beside
    it — but only through a branch the public path would then be the ONLY user
    of. The manifest is named relatively from the bundle root, not by slug: one
    bundle is one city and its manifest sits beside the page.
    """
    out_dir = tmp_path / "deploy"
    build_deploy_bundle(viewer_dir, out_dir, "https://tiles.example.com/", city=CITY)
    assert json.loads((out_dir / "cities.json").read_text()) == {
        "cities": [{"slug": CITY, "name": "Testville", "manifest": "manifest.json"}]
    }


def test_deploy_bundle_refuses_to_ship_a_page_missing_a_file(
    viewer_dir: Path, tmp_path: Path
) -> None:
    """The failure this replaces is silent: without the check, a viewer
    directory with no app.js copies cleanly and deploys as a blank page."""
    (viewer_dir / "app.js").unlink()
    with pytest.raises(DeployError, match=r"app\.js"):
        build_deploy_bundle(
            viewer_dir, tmp_path / "deploy", "https://tiles.example.com/", city=CITY
        )
    assert not (tmp_path / "deploy" / "manifest.json").exists()


def test_deploy_bundle_rewrites_the_basemap_archive(viewer_dir: Path, tmp_path: Path) -> None:
    """The basemap archive ships from the same public bucket as the atlas ones.
    Left as the manifest's local relative path it 404s on the deployed site,
    and MapLibre reports that as an empty basemap, not as an error."""
    manifest = json.loads((viewer_dir / CITY / "manifest.json").read_text())
    manifest["site"]["basemap"] = {
        "type": "vector",
        "pmtiles": "../deploy/tiles/basemap/basemap-20260723.pmtiles",
        "styles": {"atlas": "vendor/basemap/style-grayscale.json", "now": "x.json"},
    }
    (viewer_dir / CITY / "manifest.json").write_text(json.dumps(manifest))
    build_deploy_bundle(viewer_dir, tmp_path / "deploy", "https://tiles.example.com/", city=CITY)
    bundle = json.loads((tmp_path / "deploy" / "manifest.json").read_text())
    basemap = bundle["site"]["basemap"]
    assert basemap["pmtiles"] == "https://tiles.example.com/basemap-20260723.pmtiles"
    # the styles are bundled static assets, so they stay relative
    assert basemap["styles"]["atlas"] == "vendor/basemap/style-grayscale.json"


def test_deploy_bundle_leak_scan_hard_stops(viewer_dir: Path, tmp_path: Path) -> None:
    manifest = json.loads((viewer_dir / CITY / "manifest.json").read_text())
    manifest["site"]["note"] = "served from localhost during QA"
    (viewer_dir / CITY / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DeployError, match="localhost"):
        build_deploy_bundle(viewer_dir, tmp_path / "deploy", "https://tiles.example.com", city=CITY)


def test_a_legacy_titiler_tiles_key_is_refused_not_laundered(
    viewer_dir: Path, tmp_path: Path
) -> None:
    """Nothing in the pipeline writes a volume-level `tiles` key; only a
    hand-written or very old manifest can carry one. `build_deploy_bundle` used
    to pop it silently, which published a repaired copy of a manifest that was
    wrong on disk and left the source wrong. It now refuses.

    The refusal comes from `LEAK_TOKENS`, which is a substring blocklist and not
    a private-host detector: it catches this URL, and it would NOT catch a
    `10.x` host on an unlisted port. Widening it is its own change — `172.`
    already shows the false-positive hazard, since it matches a longitude.
    """
    manifest = json.loads((viewer_dir / CITY / "manifest.json").read_text())
    manifest["volumes"][0]["tiles"] = (
        "http://localhost:8008/cog/tiles/{z}/{x}/{y}.png?url=http%3A%2F%2F10.0.0.5"
    )
    (viewer_dir / CITY / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DeployError, match="localhost"):
        build_deploy_bundle(viewer_dir, tmp_path / "deploy", "https://tiles.example.com", city=CITY)


def test_deploy_bundle_refuses_empty_publish(tmp_path: Path) -> None:
    d = tmp_path / "viewer"
    d.mkdir()
    for name in PAGE_FILES:
        (d / name).write_text(page_stub(name))
    (d / CITY).mkdir()
    (d / CITY / "manifest.json").write_text(json.dumps({"volumes": [{"id": "v", "era": "1950"}]}))
    with pytest.raises(DeployError, match="nothing"):
        build_deploy_bundle(d, tmp_path / "deploy", "https://tiles.example.com", city=CITY)


@pytest.mark.parametrize(
    "bad",
    [
        "/local/path",  # no scheme at all
        "ftp://tiles.example.com",  # non-http(s) scheme
        "httpx://tiles.example.com",  # pseudo-scheme the old startswith("http") admitted
        "http",  # bare token the old check also admitted
        "https://",  # no hostname
        "https://user:secret@tiles.example.com",  # embedded credentials
        "https://tiles.example.com/tiles?v=1",  # query string
        "https://tiles.example.com/tiles#latest",  # fragment
        "https://tiles.example.com:not-a-port/tiles",  # invalid port
    ],
)
def test_deploy_bundle_rejects_invalid_tile_base(
    viewer_dir: Path, tmp_path: Path, bad: str
) -> None:
    with pytest.raises(DeployError, match="tiles base URL"):
        build_deploy_bundle(viewer_dir, tmp_path / "deploy", bad, city=CITY)


@pytest.mark.parametrize(
    ("base", "expected_root"),
    [
        ("https://tiles.example.com", "https://tiles.example.com"),
        ("https://tiles.example.com/", "https://tiles.example.com"),
        (
            "https://tiles.example.com/sanborn/tiles/",
            "https://tiles.example.com/sanborn/tiles",
        ),
        (
            "https://tiles.example.com//sanborn//tiles",
            "https://tiles.example.com/sanborn/tiles",
        ),
        ("http://tiles.example.com:8080", "http://tiles.example.com:8080"),
    ],
)
def test_deploy_bundle_joins_archive_urls_structurally(
    viewer_dir: Path, tmp_path: Path, base: str, expected_root: str
) -> None:
    """Valid roots, with and without a path prefix, produce exact public URLs
    built from the normalized base plus each archive's basename."""
    out_dir = tmp_path / "deploy"
    build_deploy_bundle(viewer_dir, out_dir, base, city=CITY)
    bundle = json.loads((out_dir / "manifest.json").read_text())

    by_id = {v["id"]: v for v in bundle["volumes"]}
    assert by_id["v3"]["pmtiles"] == f"{expected_root}/v3.pmtiles"
    assert by_id["v1"]["pmtiles"] == f"{expected_root}/v1.pmtiles"


def test_public_tiles_base_quotes_archive_basenames() -> None:
    from autogeoref.viewer.deploy import _tile_url

    assert (
        _tile_url(public_tiles_base("https://tiles.example.com/pm/"), "../work/a b.pmtiles")
        == "https://tiles.example.com/pm/a%20b.pmtiles"
    )


def test_deploy_bundle_carries_story_assets_and_scans_the_captions(
    viewer_dir: Path, tmp_path: Path
) -> None:
    """Story content rides inside `site`, so the leak scan already reads every
    caption; the images need the one extra copy step."""
    manifest = json.loads((viewer_dir / CITY / "manifest.json").read_text())
    manifest["site"] = {"name": "X", "stories": [{"id": "a", "title": "A", "stops": []}]}
    (viewer_dir / CITY / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    # staged beside the city's manifest, which is where `stage_story_assets`
    # puts them and where the manifest's relative `src` resolves from
    (viewer_dir / CITY / "story-assets").mkdir()
    (viewer_dir / CITY / "story-assets" / "a.jpg").write_bytes(b"jpeg")

    out = tmp_path / "bundle"
    out.mkdir()
    bundle = build_deploy_bundle(viewer_dir, out, "https://tiles.example.com", city=CITY)
    assert bundle["site"]["stories"][0]["id"] == "a"
    assert (out / "story-assets" / "a.jpg").is_file()

    manifest["site"]["stories"][0]["title"] = "See http://localhost:8008/x"
    (viewer_dir / CITY / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DeployError, match="SAFETY STOP"):
        build_deploy_bundle(viewer_dir, out, "https://tiles.example.com", city=CITY)


def test_the_bundles_config_is_generated_and_never_copied(viewer_dir: Path, tmp_path: Path) -> None:
    """The token reaches the page through the bundle, never through the tree:
    a committed token is a token nobody rotates. So the repository's `config.js`
    is not a bundle input at all — whatever it says, the bundle writes its own.
    """
    assert CONFIG_FILE not in PAGE_FILES  # else a viewer dir without one cannot ship
    (viewer_dir / CONFIG_FILE).write_text('window.MAPBOX_TOKEN = "pk.stale";', encoding="utf-8")
    out = tmp_path / "deploy"
    build_deploy_bundle(
        viewer_dir, out, "https://tiles.example.com", city=CITY, mapbox_token=PUBLIC_TOKEN
    )
    assert (out / CONFIG_FILE).read_text(encoding="utf-8") == (
        f'window.MAPBOX_TOKEN = "{PUBLIC_TOKEN}";\n'
    )


def test_a_viewer_directory_with_no_config_still_bundles(viewer_dir: Path, tmp_path: Path) -> None:
    """`config.js` is generated, so its absence upstream is not a missing page
    file. Listing it as one would make the bundle demand a file it overwrites."""
    (viewer_dir / CONFIG_FILE).unlink(missing_ok=True)
    out = tmp_path / "deploy"
    build_deploy_bundle(
        viewer_dir, out, "https://tiles.example.com", city=CITY, mapbox_token=PUBLIC_TOKEN
    )
    assert (out / CONFIG_FILE).is_file()


def test_a_bundle_with_no_token_ships_an_empty_config_rather_than_no_config(
    viewer_dir: Path, tmp_path: Path
) -> None:
    """A missing `config.js` is a 404 on every visit and an undefined token —
    which used to mean the deployed search silently fell back onto OSMF's
    public instance. An empty one is what makes the page say so instead."""
    out = tmp_path / "deploy"
    build_deploy_bundle(viewer_dir, out, "https://tiles.example.com", city=CITY)
    assert 'window.MAPBOX_TOKEN = "";' in (out / CONFIG_FILE).read_text(encoding="utf-8")


def test_a_secret_token_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    """`sk.` tokens carry account write access and the bundle publishes its
    config verbatim, so this is a leak and not a typo."""
    with pytest.raises(DeployError, match="SAFETY STOP"):
        mapbox_config_js("sk.eyJ1IjoidGVzdCJ9.secret")


@pytest.mark.parametrize(
    "bad",
    [
        "not-a-token",  # no scheme prefix at all
        "pk.",  # prefix with no payload
        'pk.abc";alert(1);//',  # would close the JS string literal it lands in
        "pk.abc\nwindow.evil=1",  # a second statement smuggled in on line two
    ],
)
def test_a_token_that_is_not_a_public_token_is_refused(bad: str) -> None:
    with pytest.raises(DeployError, match="pk"):
        mapbox_config_js(bad)


def test_a_bad_token_stops_the_bundle_before_it_writes_a_manifest(
    viewer_dir: Path, tmp_path: Path
) -> None:
    """Half a bundle is worse than none: the operator uploads what is there."""
    out = tmp_path / "deploy"
    with pytest.raises(DeployError):
        build_deploy_bundle(
            viewer_dir, out, "https://tiles.example.com", city=CITY, mapbox_token="nope"
        )
    assert not (out / "manifest.json").exists()


def test_a_stale_provider_field_is_not_republished(viewer_dir: Path, tmp_path: Path) -> None:
    """`provider` named a geocoder nothing read, and Chicago's said `nominatim`.
    A manifest written before it was dropped is still on disk, and shipping it
    advertises the opposite of the rule the deployed page follows."""
    manifest = json.loads((viewer_dir / CITY / "manifest.json").read_text())
    manifest["site"]["geocoder"] = {"provider": "nominatim", "suffix": ", Testville, IL"}
    (viewer_dir / CITY / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "deploy"
    bundle = build_deploy_bundle(viewer_dir, out, "https://tiles.example.com", city=CITY)
    assert bundle["site"]["geocoder"] == {"suffix": ", Testville, IL"}


def _cache_rules() -> dict[str, str]:
    """``{pattern: Cache-Control value}`` parsed from the committed `_headers`.

    Keyed on the header NAME as well as the path, because a block may carry
    more than one header and a parser that keeps only the last line grades
    whichever happened to be written last.
    """
    rules: dict[str, str] = {}
    pattern: str | None = None
    for raw in (VIEWER_DIR / "_headers").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("/"):
            pattern = line
        elif pattern is not None and ":" in line:
            name, _, value = line.partition(":")
            if name.strip().lower() == "cache-control":
                rules[pattern] = value.strip()
    return rules


def test_no_two_cache_rules_can_match_one_path() -> None:
    """Cloudflare MERGES every matching rule and comma-joins a repeated header,
    so two patterns over one path yield a single Cache-Control holding both
    values — and a cache may honour the first or the most restrictive, which
    silently discards the longer TTL.

    This is the invariant the rest of the policy rests on. A `/*` rule added to
    "state the default" would break both rules below without failing anything
    else, which is exactly why it is asserted rather than commented.
    """
    patterns = list(_cache_rules())

    def matches(pattern: str, path: str) -> bool:
        # the only wildcard this file uses; a splat greedily spans `/` too
        head, star, _tail = pattern.partition("*")
        return path.startswith(head) if star else path == pattern

    # every path the bundle actually serves, plus the roots that would collide
    served = [
        "/index.html",
        "/app.js",
        "/config.js",
        "/manifest.json",
        "/walkthrough.html",
        "/vendor/maplibre-gl.js",
        "/vendor/fonts/some.pbf",
        "/walkthrough/panel-01-the-problem.jpg",
    ]
    for path in served:
        hit = [p for p in patterns if matches(p, path)]
        assert len(hit) <= 1, f"{path} matches {hit}; merged rules discard the longer TTL"


def test_the_cache_policy_pins_only_what_cannot_change() -> None:
    """Only content that never changes under a stable name may be pinned."""
    rules = _cache_rules()

    # vendor is the one place a year is bought, on the promise that an upgrade
    # lands under a NEW filename
    assert rules["/vendor/*"] == "public, max-age=31536000, immutable"

    # the plates are re-rendered in place and one is the README hero, so a year
    # would strand a correction in every warm cache
    assert "immutable" not in rules["/walkthrough/*"]

    # nothing else is pinned, and nothing may quietly join the set
    assert {p for p, v in rules.items() if "immutable" in v} == {"/vendor/*"}


def test_the_token_file_gets_no_long_ttl_from_any_rule() -> None:
    """`config.js` carries the Mapbox token a rotation replaces. Any rule that
    MATCHES it — not merely one spelled `/config.js` — must leave it short, or
    a revoked token outlives its replacement in every warm cache.
    """
    for pattern, value in _cache_rules().items():
        head, star, _ = pattern.partition("*")
        if not ("/config.js".startswith(head) if star else pattern == "/config.js"):
            continue
        seconds = [int(p.split("=", 1)[1]) for p in value.split(",") if "max-age=" in p]
        assert all(s <= 300 for s in seconds), f"{pattern} caches the token for {seconds}s"
