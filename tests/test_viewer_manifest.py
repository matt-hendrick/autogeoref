"""Manifest assembly: which layers become volumes, and what each entry says.

A served archive is what makes a volume, so the tests build tiny worlds of
pmtiles files and a catalog and read the emitted entries. The serving-directory
tripwire is here too: every layer publishes under one credit line, so a
directory nobody declared refuses the build. Bounds are injected, so no GDAL is
needed. The CLI parse check sits at the end.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

from autogeoref.config.model import ConfigError
from autogeoref.viewer.config import ViewerConfig, load_viewer_config
from autogeoref.viewer.manifest import AreaSource, BoundsProbes, build_manifest, write_manifest
from viewer_support import (
    CHICAGO_TOML,
    VIREO_TOML,
    _fake_pm_bounds,
    _stop,
    _story,
    _story_world,
    _write_catalog,
)


@pytest.fixture
def vireo_site(tmp_path: Path) -> dict[str, Path]:
    """A tiny fictional-city world: one per-volume pmtiles, a catalog, areas,
    aliases."""
    pmtiles = tmp_path / "autogeoref"
    out_dir = tmp_path / "site"
    pmtiles.mkdir()
    out_dir.mkdir()
    (pmtiles / "vireo_002.pmtiles").write_bytes(b"vol")
    # a bake's leftover companion, which nothing serves
    (pmtiles / "vireo_002-overview.pmtiles").write_bytes(b"underlay")
    catalog = _write_catalog(
        tmp_path / "catalog.json",
        [
            {
                "id": "http://www.loc.gov/item/vireo_001/",
                "description": ["Vol. 1, 1905. 10 sheet(s)."],
                "date": "1905",
            },
            {
                "id": "http://www.loc.gov/item/vireo_002/",
                "description": ["Vol. 2, 1906. 12 sheet(s)."],
                "date": "1906",
            },
        ],
    )
    areas = tmp_path / "areas.geojson"
    areas.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "properties": {"community": "HARBOURSIDE"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [172.5, -43.6],
                                    [172.8, -43.6],
                                    [172.8, -43.4],
                                    [172.5, -43.4],
                                    [172.5, -43.6],
                                ]
                            ],
                        },
                    }
                ],
            }
        )
    )
    return {
        "pmtiles": pmtiles,
        "out_dir": out_dir,
        "catalog": catalog,
        "areas": areas,
    }


def test_build_manifest_never_emits_a_citywide_era_layer(tmp_path: Path) -> None:
    """Nothing bakes a citywide per-era archive and the page cannot draw one,
    so a filename in the retired `<city>-<era>.pmtiles` shape is just a volume
    identifier — never a second kind of layer."""
    pmtiles = tmp_path / "autogeoref"
    pmtiles.mkdir()
    (pmtiles / "chicago-1950.pmtiles").write_bytes(b"old bucket")

    manifest = build_manifest(
        "Chicago, Ill.",
        # stories dropped: this world serves one synthetic archive, so the
        # coverage gate would refuse the city's real stops first
        replace(load_viewer_config(CHICAGO_TOML), stories=None),
        out_path=tmp_path / "viewer" / "manifest.json",
        pmtiles_dirs=[pmtiles],
        probes=BoundsProbes(pmtiles=_fake_pm_bounds),
    )

    assert "era_pmtiles" not in manifest
    assert [v["id"] for v in manifest["volumes"]] == ["chicago-1950"]


def test_build_manifest_end_to_end(vireo_site: dict[str, Path]) -> None:
    viewer = load_viewer_config(VIREO_TOML)
    out_path = vireo_site["out_dir"] / "manifest.json"
    manifest = build_manifest(
        "Port Vireo",
        viewer,
        out_path=out_path,
        pmtiles_dirs=[vireo_site["pmtiles"]],
        loc_catalog=vireo_site["catalog"],
        areas=AreaSource(vireo_site["areas"]),
        probes=BoundsProbes(pmtiles=_fake_pm_bounds),
    )
    by_id = {v["id"]: v for v in manifest["volumes"]}
    # a served ARCHIVE is the only thing that makes a volume: the catalog's
    # vireo_001 has none, so it is not in the manifest
    assert set(by_id) == {"vireo_002"}

    # bounds from the archive header, relative path, catalog title and era
    v2 = by_id["vireo_002"]
    assert v2["title"] == "Port Vireo | 1906 | Vol. 2"
    assert v2["era"] == "1905"
    assert v2["pmtiles"] == "../autogeoref/vireo_002.pmtiles"
    assert v2["bounds"] == _fake_pm_bounds(Path())
    assert v2["areas"] == ["Harbourside"]
    assert "tiles" not in v2  # the TiTiler field does not exist here, ever
    # a leftover `-overview.pmtiles` is served nowhere and named nowhere: not a
    # volume of its own (set(by_id) above), and not a field on its parent
    assert "overview_pmtiles" not in v2

    assert "era_pmtiles" not in manifest  # citywide era layers are retired
    # historical street renames are pipeline-only: the viewer's search geocodes
    # the query as typed, so no rename table is served
    assert "street_aliases" not in manifest
    site = manifest["site"]
    assert site["name"] == "Port Vireo"
    assert site["home_point"] == [172.64, -43.53]
    assert "region_labels" not in site  # fictional city: generic fallback

    write_manifest(manifest, out_path)
    assert json.loads(out_path.read_text()) == manifest


def test_build_manifest_first_serving_directory_wins(
    vireo_site: dict[str, Path], tmp_path: Path
) -> None:
    """`pmtiles_dirs` is ordered: a volume served from an earlier directory
    shadows the same volume in a later one, so a rebuilt archive takes over the
    moment its tiles land. Both directories must be declared."""
    first = tmp_path / "tiles" / "autogeoref"
    second = tmp_path / "tiles" / "partner-archive"
    for d in (first, second):
        d.mkdir(parents=True)
    (second / "vireo_002.pmtiles").write_bytes(b"vol-second")
    (first / "vireo_002.pmtiles").write_bytes(b"vol-first")
    (second / "vireo_003.pmtiles").write_bytes(b"vol-second-only")

    # the key belongs INSIDE the existing [viewer] table: appending it would
    # write a second [viewer] header, which is a TOML error
    toml = tmp_path / "both-dirs.toml"
    toml.write_text(
        VIREO_TOML.read_text(encoding="utf-8").replace(
            "[viewer]\n", '[viewer]\nserving_dirs = ["autogeoref", "partner-archive"]\n', 1
        ),
        encoding="utf-8",
    )
    shutil.copy(VIREO_TOML.parent / "vireo-stories.json", tmp_path / "vireo-stories.json")
    manifest = build_manifest(
        "Port Vireo",
        load_viewer_config(toml),
        out_path=vireo_site["out_dir"] / "manifest.json",
        pmtiles_dirs=[first, second],
        loc_catalog=vireo_site["catalog"],
        probes=BoundsProbes(pmtiles=_fake_pm_bounds),
    )
    by_id = {v["id"]: v for v in manifest["volumes"]}
    assert by_id["vireo_002"]["pmtiles"].endswith("autogeoref/vireo_002.pmtiles")
    assert by_id["vireo_003"]["pmtiles"].endswith("partner-archive/vireo_003.pmtiles")
    # provenance is not a manifest field: one pipeline places everything a
    # declared directory can hold, and the site credit says so once
    assert all("provenance" not in v for v in manifest["volumes"])


def test_an_undeclared_serving_directory_is_refused(
    vireo_site: dict[str, Path], tmp_path: Path
) -> None:
    """The tripwire. Every layer is published under the city's ONE credit line,
    so a directory nobody declared would publish someone else's georeferencing
    as this project's — silently, because there is nothing left to compose that
    could notice. Refuse the build instead.

    `vireo_003` is deliberately absent from the catalog, so it has no era and
    would never have reached a per-era check: the directory is what is tested,
    not the layer.
    """
    autogeo = tmp_path / "tiles" / "autogeoref"
    partner = tmp_path / "tiles" / "partner-archive"
    for d in (autogeo, partner):
        d.mkdir(parents=True)
    (autogeo / "vireo_002.pmtiles").write_bytes(b"ours")
    (partner / "vireo_003.pmtiles").write_bytes(b"theirs")

    build = partial(
        build_manifest,
        "Port Vireo",
        load_viewer_config(VIREO_TOML),
        out_path=vireo_site["out_dir"] / "manifest.json",
        loc_catalog=vireo_site["catalog"],
        probes=BoundsProbes(pmtiles=_fake_pm_bounds),
    )
    with pytest.raises(ConfigError, match="partner-archive"):
        build(pmtiles_dirs=[autogeo, partner])

    # the declared directory alone still builds — the refusal is about the
    # undeclared one, not about serving more than one thing
    assert build(pmtiles_dirs=[autogeo])["volumes"]


def test_an_undeclared_directory_holding_only_an_underlay_is_refused(
    vireo_site: dict[str, Path], tmp_path: Path
) -> None:
    """The overview companion rides on its PARENT's manifest entry, so a check
    that looked at volumes would see a clean build while a foreign raster drew
    beneath a credited one. Checking the DIRECTORY catches it by construction —
    which is the reason to check directories and not layers."""
    autogeo = tmp_path / "tiles" / "autogeoref"
    partner = tmp_path / "tiles" / "partner-archive"
    for d in (autogeo, partner):
        d.mkdir(parents=True)
    (autogeo / "vireo_002.pmtiles").write_bytes(b"ours")
    (partner / "vireo_002-overview.pmtiles").write_bytes(b"their underlay")

    with pytest.raises(ConfigError, match="partner-archive"):
        build_manifest(
            "Port Vireo",
            load_viewer_config(VIREO_TOML),
            out_path=vireo_site["out_dir"] / "manifest.json",
            pmtiles_dirs=[autogeo, partner],
            loc_catalog=vireo_site["catalog"],
            probes=BoundsProbes(pmtiles=_fake_pm_bounds),
        )


def test_a_relative_serving_directory_is_named_before_it_is_checked(
    vireo_site: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Path(".").name` is the empty string, which matches no declared name and
    would read as "no directory here" rather than as the directory it is.
    Resolve before naming, or a relative path walks straight past the tripwire."""
    partner = tmp_path / "tiles" / "partner-archive"
    partner.mkdir(parents=True)
    (partner / "vireo_002.pmtiles").write_bytes(b"theirs")
    monkeypatch.chdir(partner)

    with pytest.raises(ConfigError, match="partner-archive"):
        build_manifest(
            "Port Vireo",
            load_viewer_config(VIREO_TOML),
            out_path=vireo_site["out_dir"] / "manifest.json",
            pmtiles_dirs=[Path()],  # Path() is "." — the empty-name case
            loc_catalog=vireo_site["catalog"],
            probes=BoundsProbes(pmtiles=_fake_pm_bounds),
        )


def test_a_city_that_declares_nothing_still_gets_the_default_tripwire(
    vireo_site: dict[str, Path], tmp_path: Path
) -> None:
    """`serving_dirs` defaults to the one directory this pipeline publishes to,
    so a city that declares nothing is protected rather than unprotected — the
    failure the [viewer.credits] block used to have, where the guard was armed
    only by an optional config key."""
    scratch = tmp_path / "tiles" / "scratch-demo"
    scratch.mkdir(parents=True)
    (scratch / "vireo_002.pmtiles").write_bytes(b"demo")
    assert load_viewer_config(VIREO_TOML).serving_dirs == ("autogeoref",)

    with pytest.raises(ConfigError, match="scratch-demo"):
        build_manifest(
            "Port Vireo",
            load_viewer_config(VIREO_TOML),
            out_path=vireo_site["out_dir"] / "manifest.json",
            pmtiles_dirs=[scratch],
            loc_catalog=vireo_site["catalog"],
            probes=BoundsProbes(pmtiles=_fake_pm_bounds),
        )


def _label_world(tmp_path: Path) -> tuple[Path, Path]:
    """Three served volumes: two numbered, one special (unnumbered subject)."""
    pmtiles = tmp_path / "autogeoref"
    pmtiles.mkdir()
    for ident in ("vireo_001", "vireo_002", "vireo_003"):
        (pmtiles / f"{ident}.pmtiles").write_bytes(b"vol")
    catalog = _write_catalog(
        tmp_path / "label-catalog.json",
        [
            {
                "id": "http://www.loc.gov/item/vireo_001/",
                "description": ["Vol. 1, 1905. 10 sheet(s)."],
                "date": "1905",
            },
            {
                "id": "http://www.loc.gov/item/vireo_002/",
                "description": ["Vol. 2, 1906. 12 sheet(s)."],
                "date": "1906",
            },
            {
                "id": "http://www.loc.gov/item/vireo_003/",
                "description": ["1906. 1 sheet(s). Exposition grounds."],
                "date": "1906",
            },
        ],
    )
    return pmtiles, catalog


def test_build_manifest_label_chain(tmp_path: Path) -> None:
    """The district-list name, strongest claim first: a declared
    [viewer.labels] entry, then the LOC subject (specials only). A numbered
    volume with neither gets no label and the viewer renders its community
    areas."""
    pmtiles, catalog = _label_world(tmp_path)
    toml = tmp_path / "city.toml"
    toml.write_text(
        '[city]\nname = "Port Vireo"\ncenterlines = "c"\naliases_dir = "a"\n'
        "[viewer.labels]\n"
        'vireo_002 = "Harbourside Docks"\n'
        'vireo_003 = "1906 Exposition"\n'
        'vireo_999 = "names no served volume"\n'
    )
    manifest = build_manifest(
        "Port Vireo",
        load_viewer_config(toml),
        out_path=tmp_path / "manifest.json",
        pmtiles_dirs=[pmtiles],
        loc_catalog=catalog,
        probes=BoundsProbes(pmtiles=_fake_pm_bounds),
    )
    by_id = {v["id"]: v for v in manifest["volumes"]}
    assert "label" not in by_id["vireo_001"]  # numbered, undeclared: areas render
    assert by_id["vireo_002"]["label"] == "Harbourside Docks"  # declared wins on any volume
    assert by_id["vireo_003"]["label"] == "1906 Exposition"  # declared beats the LOC subject
    assert "vireo_999" not in by_id  # unmatched declaration: warned, never emitted


def test_special_volumes_are_labelled_by_their_loc_subject(tmp_path: Path) -> None:
    """Undeclared specials fall back to the catalogued subject — the fix for
    the stockyards volume rendering as its top community area ("New City")."""
    pmtiles, catalog = _label_world(tmp_path)
    manifest = build_manifest(
        "Port Vireo",
        ViewerConfig(),
        out_path=tmp_path / "manifest.json",
        pmtiles_dirs=[pmtiles],
        loc_catalog=catalog,
        probes=BoundsProbes(pmtiles=_fake_pm_bounds),
    )
    by_id = {v["id"]: v for v in manifest["volumes"]}
    assert by_id["vireo_003"]["label"] == "Exposition grounds"
    assert "label" not in by_id["vireo_001"]
    assert "label" not in by_id["vireo_002"]


def test_label_survives_a_catalogless_rebuild(tmp_path: Path) -> None:
    """A repair publish without a local catalog retains `label` exactly like
    title/year/volume_number — the specials must not lose their names to a
    manifest rebuild."""
    from autogeoref.viewer.publish import _manifest_metadata

    pmtiles, catalog = _label_world(tmp_path)
    out = tmp_path / "viewer" / "manifest.json"
    out.parent.mkdir()
    first = build_manifest(
        "Port Vireo",
        ViewerConfig(),
        out_path=out,
        pmtiles_dirs=[pmtiles],
        loc_catalog=catalog,
        probes=BoundsProbes(pmtiles=_fake_pm_bounds),
    )
    write_manifest(first, out)
    rebuilt = build_manifest(
        "Port Vireo",
        ViewerConfig(),
        out_path=out,
        pmtiles_dirs=[pmtiles],
        loc_catalog=None,
        metadata_fallback=_manifest_metadata(out),
        probes=BoundsProbes(pmtiles=_fake_pm_bounds),
    )
    labels = {v["id"]: v.get("label") for v in rebuilt["volumes"]}
    assert labels == {"vireo_001": None, "vireo_002": None, "vireo_003": "Exposition grounds"}


def test_build_manifest_bare_city_still_valid(tmp_path: Path) -> None:
    """No pmtiles, no catalog: an empty but well-formed manifest
    (the viewer renders it with generic fallbacks and the home-point view)."""
    manifest = build_manifest(
        "Nowhere",
        ViewerConfig(),
        out_path=tmp_path / "manifest.json",
    )
    assert manifest["volumes"] == []
    assert "era_pmtiles" not in manifest
    assert manifest["site"] == {"name": "Nowhere"}


def test_a_stop_over_nothing_stops_the_build(vireo_site: dict[str, Path], tmp_path: Path) -> None:
    """An uncovered camera is a beautifully written caption over blank paper.

    Built against a world that DOES serve layers, so what fails is the camera
    and not the absence of anything to stand on.
    """
    elsewhere = _stop(camera={"center": [10.0, 10.0], "zoom": 16})
    away = _story_world(tmp_path / "away", [_story(elsewhere)])
    build = partial(
        build_manifest,
        "Port Vireo",
        out_path=vireo_site["out_dir"] / "manifest.json",
        pmtiles_dirs=[vireo_site["pmtiles"]],
        loc_catalog=vireo_site["catalog"],
        probes=BoundsProbes(pmtiles=_fake_pm_bounds),
    )
    with pytest.raises(ConfigError, match="which no served layer covers"):
        build(load_viewer_config(away))
    # and the same world accepts a camera the served layers do cover
    over = _story_world(tmp_path / "over", [_story(_stop())])
    assert build(load_viewer_config(over))["site"]["stories"]


def test_cli_viewer_manifest_and_deploy_bundle_parse() -> None:
    from autogeoref.cli.parser import build_parser

    args = build_parser().parse_args(
        [
            "viewer-manifest",
            "--city",
            "configs/chicago/chicago.toml",
            "--pmtiles",
            "pm1",
            "--pmtiles",
            "pm2",
            "--loc-catalog",
            "cat.json",
        ]
    )
    assert [str(p) for p in args.pmtiles] == ["pm1", "pm2"]
    # left to the city: a shared default is what let a second city overwrite
    # the first city's page
    assert args.out is None
    args2 = build_parser().parse_args(
        ["deploy-bundle", "https://tiles.example.com", "--city", "configs/chicago/chicago.toml"]
    )
    assert args2.tiles_base_url == "https://tiles.example.com"
    assert str(args2.viewer) == "viewer"
    assert args2.out is None
