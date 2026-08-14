"""Guided stories: optional by construction, and refused rather than half-rendered.

A city that configures no stories gets no story UI and no manifest key. A city
that configures one has every stop validated by name — era, camera, ids that go
into the permalink verbatim, media schemes, overlay paths that may not leave the
assets directory — because the alternative is a broken story in a reader's
browser. Staging the images is a rename, so a failed copy leaves the previous
set serving.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from autogeoref.config.model import ConfigError
from autogeoref.viewer.config import load_viewer_config, site_dict
from autogeoref.viewer.stories import stage_story_assets
from viewer_support import VIREO_TOML, _stop, _story, _story_world

VIREO_STORIES = VIREO_TOML.parent / "vireo-stories.json"


def test_a_city_with_no_stories_gets_no_stories_key(tmp_path: Path) -> None:
    """The generalization contract: configuring nothing must cost nothing."""
    toml = tmp_path / "quiet.toml"
    toml.write_text('[city]\nname = "Quiet"\n[viewer]\n', encoding="utf-8")
    viewer = load_viewer_config(toml)
    assert viewer.stories is None
    assert "stories" not in site_dict("Quiet", viewer)


def test_vireo_story_round_trips_into_the_site_block() -> None:
    viewer = load_viewer_config(VIREO_TOML)
    assert viewer.stories is not None
    assert viewer.stories.path == VIREO_STORIES.resolve()
    site = site_dict("Port Vireo", viewer)
    story = site["stories"][0]
    assert story["id"] == "harbour-line"
    assert [s["id"] for s in story["stops"]] == ["wharf", "yards"]
    first = story["stops"][0]
    assert first["camera"] == {"center": [172.64, -43.53], "zoom": 16.0}
    assert first["eras"] == ["1905"] and first["swipe"] == 0.55
    assert first["overlay"]["geojson"]["features"][0]["geometry"]["type"] == "LineString"
    assert first["sources"][0]["label"] == "Sheet 4, Vol. 1"
    # optional fields stay absent rather than arriving as nulls
    assert "media" not in first and "overlay" not in story["stops"][1]


@pytest.mark.parametrize(
    ("stop", "message"),
    [
        (_stop(eras=["1907"]), "not an era bucket label"),
        (_stop(camera=None), "camera"),
        (_stop(swipe=1.4), "not a fraction"),
        (_stop(camera={"center": [172.64], "zoom": 16}), "camera.center must be"),
        (_stop(camera={"center": [172.64, -43.53], "zoom": 99}), "camera.zoom"),
        (_stop(title=""), "title must be a non-empty string"),
        (_stop(media=[{"src": "nope.jpg", "alt": "x"}]), "no viewer.stories.assets"),
    ],
)
def test_an_invalid_story_is_refused_by_name(
    tmp_path: Path, stop: dict[str, Any], message: str
) -> None:
    """Refuse the config rather than render a broken story — and say where."""
    if stop["camera"] is None:
        del stop["camera"]
    toml = _story_world(tmp_path, [_story(stop)])
    with pytest.raises(ConfigError, match=message) as caught:
        load_viewer_config(toml)
    assert "s.json" in str(caught.value)


@pytest.mark.parametrize("bad", ["x&panel=closed", "a b", "one#two"])
def test_an_id_that_would_corrupt_the_permalink_is_refused(tmp_path: Path, bad: str) -> None:
    """Ids go into the shared URL fragment verbatim. An `&` or `=` in one
    injects a second key, which the viewer reads as somebody else's setting."""
    toml = _story_world(tmp_path, [_story(_stop(id=bad))])
    with pytest.raises(ConfigError, match="permalink key"):
        load_viewer_config(toml)


def test_duplicate_stop_ids_are_refused(tmp_path: Path) -> None:
    """Stop ids are permalink keys, so a duplicate is an ambiguous link."""
    toml = _story_world(tmp_path, [_story(_stop(), _stop())])
    with pytest.raises(ConfigError, match="duplicate stop id"):
        load_viewer_config(toml)


def test_local_media_resolves_under_the_assets_dir(tmp_path: Path) -> None:
    assets = tmp_path / "pics"
    assets.mkdir()
    (assets / "wharf.jpg").write_bytes(b"jpeg")
    toml = _story_world(
        tmp_path,
        [_story(_stop(media=[{"src": "wharf.jpg", "alt": "the wharf", "credit": "LOC"}]))],
        block='[viewer.stories]\nfile = "s.json"\nassets = "pics"\n',
    )
    viewer = load_viewer_config(toml)
    media = site_dict("Port Vireo", viewer)["stories"][0]["stops"][0]["media"][0]
    # staged under one conventional name, so the deploy bundle needs no config
    assert media["src"] == "story-assets/wharf.jpg"
    staged = stage_story_assets(viewer.stories, tmp_path / "viewer")
    assert staged is not None and (staged / "wharf.jpg").is_file()


def test_a_story_block_with_no_file_is_refused(tmp_path: Path) -> None:
    """An empty block is a mistake, not an opt-out: a city that wants no
    stories writes no block, and silence here would ship no story UI."""
    toml = _story_world(tmp_path, [_story()], block="[viewer.stories]\n")
    with pytest.raises(ConfigError, match=r"viewer\.stories needs file"):
        load_viewer_config(toml)


@pytest.mark.parametrize("escape", ["../outside.json", "sub/../../outside.json"])
def test_an_asset_path_may_not_leave_the_assets_dir(tmp_path: Path, escape: str) -> None:
    """Resolved, not string-matched. `sub/../../x` starts with neither `..` nor
    `/`, and the file it names would otherwise be read and PUBLISHED."""
    (tmp_path / "outside.json").write_text('{"type": "FeatureCollection", "features": []}')
    assets = tmp_path / "story" / "pics"
    assets.mkdir(parents=True)
    (assets / "sub").mkdir()
    toml = _story_world(
        tmp_path / "story",
        [_story(_stop(overlay={"file": escape}))],
        block='[viewer.stories]\nfile = "s.json"\nassets = "pics"\n',
    )
    with pytest.raises(ConfigError, match=r"resolves outside|not absolute"):
        load_viewer_config(toml)


def test_a_media_link_must_be_a_real_scheme(tmp_path: Path) -> None:
    """`javascript:` in an href runs in the page the moment a reader clicks.
    `body_html` is trusted by design; a link target is not the same promise."""
    toml = _story_world(
        tmp_path,
        [_story(_stop(sources=[{"label": "x", "href": "javascript:alert(1)"}]))],
    )
    with pytest.raises(ConfigError, match="href must be"):
        load_viewer_config(toml)


def test_an_overlay_style_of_the_wrong_type_is_refused(tmp_path: Path) -> None:
    """The style reaches setPaintProperty verbatim, so a string width would
    surface as a MapLibre error in somebody's browser, not at build time."""
    toml = _story_world(
        tmp_path,
        [
            _story(
                _stop(
                    overlay={
                        "geojson": {"type": "FeatureCollection", "features": []},
                        "style": {"width": "thick"},
                    }
                )
            )
        ],
    )
    with pytest.raises(ConfigError, match="expected a number"):
        load_viewer_config(toml)


def test_staging_replaces_the_previous_images_and_survives_a_failed_copy(tmp_path: Path) -> None:
    """The swap is a rename, so a copy that dies partway leaves the previous
    images serving — they are referenced by a manifest that is still live."""
    assets = tmp_path / "pics"
    assets.mkdir()
    (assets / "new.jpg").write_bytes(b"new")
    viewer_dir = tmp_path / "viewer"
    (viewer_dir / "story-assets").mkdir(parents=True)
    (viewer_dir / "story-assets" / "old.jpg").write_bytes(b"old")
    toml = _story_world(
        tmp_path,
        [_story(_stop(media=[{"src": "new.jpg", "alt": "n"}]))],
        block='[viewer.stories]\nfile = "s.json"\nassets = "pics"\n',
    )
    stories = load_viewer_config(toml).stories

    with monkeypatched_copytree_failure(), pytest.raises(OSError, match="no space left"):
        stage_story_assets(stories, viewer_dir)
    assert (viewer_dir / "story-assets" / "old.jpg").is_file()

    stage_story_assets(stories, viewer_dir)
    assert (viewer_dir / "story-assets" / "new.jpg").is_file()
    assert not (viewer_dir / "story-assets" / "old.jpg").exists()
    # no staging residue left behind for the next publish to trip over
    assert not [p for p in viewer_dir.iterdir() if p.name.startswith(".story-assets")]


@contextmanager
def monkeypatched_copytree_failure() -> Iterator[None]:
    """`shutil.copytree` raising mid-tree, as a full disk would."""
    import autogeoref.viewer.stories as module

    original = module.shutil.copytree

    def boom(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
        original(src, dst, *args, **kwargs)
        raise OSError("no space left on device")

    module.shutil.copytree = boom
    try:
        yield
    finally:
        module.shutil.copytree = original


def test_staging_refuses_an_assets_dir_inside_the_staged_tree(tmp_path: Path) -> None:
    """Staging REPLACES the tree, so a source nested inside it would be deleted
    rather than copied — and the publish transaction cannot roll that back."""
    viewer_dir = tmp_path / "viewer"
    assets = viewer_dir / "story-assets" / "images"
    assets.mkdir(parents=True)
    (assets / "a.jpg").write_bytes(b"jpeg")
    toml = _story_world(
        viewer_dir,
        [_story(_stop())],
        block='[viewer.stories]\nfile = "s.json"\nassets = "story-assets/images"\n',
    )
    with pytest.raises(ConfigError, match="overlaps the staging directory"):
        stage_story_assets(load_viewer_config(toml).stories, viewer_dir)
    assert (assets / "a.jpg").is_file()  # and the source images are still there


def test_https_media_passes_through_but_other_schemes_do_not(tmp_path: Path) -> None:
    toml = _story_world(
        tmp_path, [_story(_stop(media=[{"src": "https://example.org/a.jpg", "alt": "a"}]))]
    )
    viewer = load_viewer_config(toml)
    stops = site_dict("Port Vireo", viewer)["stories"][0]["stops"]
    assert stops[0]["media"][0]["src"] == "https://example.org/a.jpg"

    bad = _story_world(
        tmp_path / "b", [_story(_stop(media=[{"src": "http://example.org/a.jpg", "alt": "a"}]))]
    )
    with pytest.raises(ConfigError, match="https:// URL"):
        load_viewer_config(bad)
