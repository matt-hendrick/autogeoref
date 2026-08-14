"""One page per city: where a manifest lives, and that publishing cannot cross.

The defect this exists over is silent. Every input to a viewer manifest is one
city's config — the site block, the era buckets, the geocoder, the basemap, the
credit line — so a shared output path let `publish --city B` retitle city A's
page and re-derive all of its era chips, with no error anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autogeoref.viewer.layout import (
    CITIES_NAME,
    MANIFEST_NAME,
    city_dir,
    city_manifest,
    city_tiles,
    refresh_cities,
)
from autogeoref.viewer.publish import PublicationError, publish_volume
from viewer_support import _archive, _fake_pmtiles_bounds, _publication_config


def test_a_manifest_is_named_for_its_city() -> None:
    """The slug every other per-city artifact already uses — the manifest was
    the one that did not carry it."""
    assert city_manifest("Crystal Lake, Ill.", Path("viewer")) == Path(
        "viewer/crystal-lake-ill/manifest.json"
    )
    assert city_dir("Chicago, Ill.", Path("out")) == Path("out/chicago-ill")


def test_a_city_publishes_into_the_first_directory_it_declares() -> None:
    """`serving_dirs` lists what a city serves; the FIRST is the one this
    pipeline writes. A later entry is a partner archive the city vouches for
    and must never be published into."""
    assert city_tiles("autogeoref", Path("deploy/tiles")) == Path("deploy/tiles/autogeoref")
    assert city_tiles("crystal-lake", Path("deploy/tiles")) == Path("deploy/tiles/crystal-lake")


def _manifest_at(root: Path, slug: str, name: str) -> Path:
    path = root / slug / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"volumes": [], "site": {"name": name}}), encoding="utf-8")
    return path


def test_the_city_index_is_derived_from_the_manifests_on_disk(tmp_path: Path) -> None:
    """Derived, never merged: a city whose directory is gone leaves the index
    on the next build, so a stale entry cannot offer a link to nothing."""
    viewer = tmp_path / "viewer"
    _manifest_at(viewer, "zed-city", "Zed City")
    gone = _manifest_at(viewer, "old-town", "Old Town")
    refresh_cities(viewer)
    listed = json.loads((viewer / CITIES_NAME).read_text())["cities"]
    # by NAME, so the list a visitor reads is alphabetical rather than slug-ordered
    assert listed == [
        {"slug": "old-town", "name": "Old Town", "manifest": "old-town/manifest.json"},
        {"slug": "zed-city", "name": "Zed City", "manifest": "zed-city/manifest.json"},
    ]

    gone.unlink()
    refresh_cities(viewer)
    assert [c["slug"] for c in json.loads((viewer / CITIES_NAME).read_text())["cities"]] == [
        "zed-city"
    ]


def test_one_unreadable_manifest_does_not_take_the_others_off_the_index(tmp_path: Path) -> None:
    """The index is navigation. A half-written file has to cost its own entry
    and nothing else's."""
    viewer = tmp_path / "viewer"
    _manifest_at(viewer, "good", "Good")
    (viewer / "broken").mkdir()
    (viewer / "broken" / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    refresh_cities(viewer)
    assert [c["slug"] for c in json.loads((viewer / CITIES_NAME).read_text())["cities"]] == ["good"]


def test_refresh_writes_nothing_where_there_is_no_page(tmp_path: Path) -> None:
    """An index beside no page files indexes nothing. Writing one anyway would
    leave a stray file wherever the caller happened to be."""
    refresh_cities(tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def _second_city(tmp_path: Path) -> Path:
    """A second city on the same host, serving from its own tiles directory.

    It has to DECLARE that directory: the tripwire that keeps every served
    layer under one credit line does not care that the name is the city's.
    """
    toml = tmp_path / "city-b.toml"
    toml.write_text(
        '[city]\nname = "Other Town"\ncenterlines = "streets.geojson"\naliases_dir = "aliases"\n'
        '[viewer]\nserving_dirs = ["other-town"]\n',
        encoding="utf-8",
    )
    return toml


def test_publishing_a_second_city_leaves_the_first_citys_page_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole claim of the rename, in one comparison.

    Byte-identical, not "still has 1 volume": the failure it replaces kept the
    volume list and rewrote the title, the era of every entry, and the site
    block — a page that still renders and is no longer this city's.
    """
    import autogeoref.viewer.publish as viewer_mod

    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)
    first = _publication_config(tmp_path)
    _archive(first, "vol_a")
    publish_volume("vol_a", first)
    before = first.manifest.read_bytes()

    # a second city, sharing the work tree and the deploy root exactly as two
    # cities on one host do, and serving from its OWN tiles directory
    other_toml = _second_city(tmp_path)
    second = type(first)(
        work=first.work,
        city_toml=other_toml,
        manifest=city_manifest("Other Town", first.viewer_root),
        viewer_root=first.viewer_root,
        tiles_root=first.tiles_root,
        serve_dir="other-town",
        exports_root=first.exports_root,
        loc_item=first.loc_item,
        loc_cache=first.loc_cache,
    )
    _archive(second, "vol_b")
    publish_volume("vol_b", second)

    assert first.manifest.read_bytes() == before, "city B's publish rewrote city A's page"
    assert json.loads(second.manifest.read_text())["volumes"][0]["id"] == "vol_b"
    # and the index now offers both, so neither is unreachable from the page
    listed = json.loads((first.viewer_root / CITIES_NAME).read_text())["cities"]
    assert {c["slug"] for c in listed} == {"testville", "other-town"}


def test_each_citys_archives_land_in_its_own_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half: one shared archive directory would put city B's volume,
    unlabelled and uncredited, straight onto city A's page.

    Asserted by PUBLISHING both, not by re-deriving the destination — the
    latter would only restate `city_tiles` back to itself.
    """
    import autogeoref.viewer.publish as viewer_mod

    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)
    first = _publication_config(tmp_path)
    _archive(first, "vol_a")
    assert publish_volume("vol_a", first) == first.tiles_root / "autogeoref" / "vol_a.pmtiles"

    second = type(first)(
        work=first.work,
        city_toml=_second_city(tmp_path),
        manifest=city_manifest("Other Town", first.viewer_root),
        viewer_root=first.viewer_root,
        tiles_root=first.tiles_root,
        serve_dir="other-town",
        exports_root=first.exports_root,
        loc_item=first.loc_item,
        loc_cache=first.loc_cache,
    )
    _archive(second, "vol_b")
    assert publish_volume("vol_b", second) == first.tiles_root / "other-town" / "vol_b.pmtiles"
    # the directory city A's manifest is built from holds only city A's volume
    assert {p.name for p in (first.tiles_root / "autogeoref").glob("*.pmtiles")} == {
        "vol_a.pmtiles"
    }
    assert [v["id"] for v in json.loads(first.manifest.read_text())["volumes"]] == ["vol_a"]


def test_a_city_cannot_publish_into_a_directory_it_has_not_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tripwire that keeps every served layer under one credit line does
    not care that the directory is named for the city, and it fires BEFORE the
    archive lands rather than rolling it back out again."""
    import autogeoref.viewer.publish as viewer_mod

    monkeypatch.setattr(viewer_mod, "pmtiles_bounds", _fake_pmtiles_bounds)
    config = _publication_config(tmp_path)
    undeclared = type(config)(
        work=config.work,
        city_toml=config.city_toml,  # declares nothing, so the default applies
        manifest=config.manifest,
        viewer_root=config.viewer_root,
        tiles_root=config.tiles_root,
        serve_dir="somewhere-else",
        exports_root=config.exports_root,
        loc_item=config.loc_item,
        loc_cache=config.loc_cache,
    )
    _archive(undeclared, "vol_a")
    with pytest.raises(PublicationError, match="serving directory not declared"):
        publish_volume("vol_a", undeclared)
    assert not (config.tiles_root / "somewhere-else").exists()
