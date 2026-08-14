"""Paths and builders shared by the viewer test modules.

The constants name the real page files and the two city configs the viewer
tests read: the production one and the fictional one that proves no city fact
is baked into the code. The builders write the throwaway worlds — a LOC
catalog, a story sidecar, a stop — that several of those modules need.
"""

from __future__ import annotations

import errno
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autogeoref.viewer.deploy import SOCIAL_MARKER, SOCIAL_PAGES
from autogeoref.viewer.publish import PublicationConfig

if TYPE_CHECKING:
    import pytest


def page_stub(name: str) -> str:
    """Stand-in contents for one file in a synthetic viewer directory.

    The two pages the deploy splices head metadata into have to carry what it
    splices: a title to replace and the marker to replace it at. A bare comment
    makes every bundle test fail on the missing marker instead of on whatever
    it was written to check.
    """
    if name in SOCIAL_PAGES:
        return f"<!DOCTYPE html>\n<title>Placeholder</title>\n{SOCIAL_MARKER}\n<!-- {name} -->\n"
    return f"/* {name} */\n"


ROOT = Path(__file__).resolve().parent.parent
CHICAGO_TOML = ROOT / "configs" / "chicago" / "chicago.toml"
VIREO_TOML = Path(__file__).resolve().parent / "data" / "city-vireo.toml"
VIEWER_DIR = ROOT / "viewer"
VIEWER_HTML = VIEWER_DIR / "index.html"
NOT_FOUND_HTML = VIEWER_DIR / "404.html"
VIEWER_CSS = VIEWER_DIR / "app.css"
VIEWER_JS = VIEWER_DIR / "app.js"
VIEWER_LIB = VIEWER_DIR / "lib.js"
VIEWER_CONFIG = VIEWER_DIR / "config.js"
WALK_HTML = VIEWER_DIR / "walkthrough.html"
WALK_CSS = VIEWER_DIR / "walkthrough.css"
WALK_JS = VIEWER_DIR / "walkthrough.js"
WALK_DIR = VIEWER_DIR / "walkthrough"
WALK_PANELS = WALK_DIR / "panels.json"


def _write_catalog(path: Path, items: list[dict[str, Any]]) -> Path:
    path.write_text(json.dumps(items))
    return path


def _fake_pm_bounds(_pm: Path) -> list[float]:
    return [172.62, -43.54, 172.66, -43.52]


def _story_world(tmp_path: Path, stories: list[dict[str, Any]], *, block: str = "") -> Path:
    """A fictional-city TOML whose story sidecar is ``stories``."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = VIREO_TOML.read_text(encoding="utf-8").split("[viewer.stories]")[0]
    toml = tmp_path / "city.toml"
    toml.write_text(base + (block or '[viewer.stories]\nfile = "s.json"\n'), encoding="utf-8")
    (tmp_path / "s.json").write_text(json.dumps({"stories": stories}), encoding="utf-8")
    return toml


def _stop(**overrides: Any) -> dict[str, Any]:
    stop: dict[str, Any] = {
        "id": "one",
        "title": "A stop",
        "camera": {"center": [172.64, -43.53], "zoom": 16},
        "eras": ["1905"],
    }
    stop.update(overrides)
    return stop


def _story(*stops: dict[str, Any]) -> dict[str, Any]:
    return {"id": "s1", "title": "A story", "stops": list(stops) or [_stop()]}


def _publication_config(tmp_path: Path) -> PublicationConfig:
    city = tmp_path / "city.toml"
    city.write_text(
        '[city]\nname = "Test City"\ncenterlines = "streets.geojson"\naliases_dir = "aliases"\n'
    )
    # one jp2 variant per sheet; any test volume's page 1 maps onto it
    item = tmp_path / "loc-item.json"
    item.write_text(
        json.dumps(
            {
                "resources": [
                    {
                        "files": [
                            [
                                {
                                    "mimetype": "image/jp2",
                                    "height": 80,
                                    "url": (
                                        "https://tile.loc.gov/storage-services/"
                                        "service/gmd/test/vol-0001.jp2"
                                    ),
                                }
                            ]
                        ]
                    }
                ]
            }
        )
    )
    return PublicationConfig(
        work=tmp_path / "work",
        city_toml=city,
        # the shipping layout: one manifest per city, under the page directory
        manifest=tmp_path / "viewer" / "testville" / "manifest.json",
        viewer_root=tmp_path / "viewer",
        tiles_root=tmp_path / "deploy" / "tiles",
        exports_root=tmp_path / "exports",
        loc_item=item,
        loc_cache=tmp_path / "loc-cache",
    )


def _fake_pmtiles_bounds(path: Path) -> list[float]:
    if path.read_bytes()[:2] != b"PM":
        raise ValueError("not a PMTiles header")
    return [0, 0, 1, 1]


def _gcp_feature(px: float, py: float, lng: float, lat: float) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {"image": [px, py]},
        "geometry": {"type": "Point", "coordinates": [lng, lat]},
    }


def _result_record(page: str, status: str = "OK") -> dict[str, Any]:
    return {
        "page": page,
        "status": status,
        "gcps_geojson": {
            "type": "FeatureCollection",
            "features": [
                _gcp_feature(0, 0, -87.7, 41.9),
                _gcp_feature(100, 0, -87.6, 41.9),
                _gcp_feature(0, 80, -87.7, 41.8),
            ],
        },
    }


def _work_tree(config: PublicationConfig, volume: str) -> Path:
    """The minimal placed volume a publish can export: manifest + one committed sheet."""
    root = config.work / volume
    (root / "sheets").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "sheets" / "manifest.json").write_text(json.dumps({"p1": {"full_size": [100, 80]}}))
    (root / "results" / "p1.json").write_text(json.dumps(_result_record("1"), indent=2))
    return root


def _refuse_directory_renames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail every directory rename with EXDEV, the way overlayfs fails one.

    A container image's export trees sit in a lower layer, which can be read and
    deleted but not renamed out of. Files are unaffected there, so this leaves
    file renames alone: patching those too would test a filesystem nobody runs.
    """
    real = Path.rename

    def guarded(self: Path, target: str | Path) -> Path:
        if self.is_dir():
            raise OSError(errno.EXDEV, "Invalid cross-device link", str(self))
        return real(self, target)

    monkeypatch.setattr(Path, "rename", guarded)


def _archive(config: PublicationConfig, volume: str, data: bytes = b"PM archive") -> Path:
    _work_tree(config, volume)
    source = config.work / volume / f"{volume}.pmtiles"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(data)
    return source


#: The fixture city's slug for a deploy bundle. One bundle is one city, so its
#: manifest lives under its own directory and the bundle puts it beside the page.
DEPLOY_CITY = "testville"


def deploy_viewer_dir(tmp_path: Path) -> Path:
    """A synthetic viewer directory a deploy bundle can be built from."""
    from autogeoref.viewer.deploy import ICON_FILES, PAGE_FILES, PLATFORM_FILES

    d = tmp_path / "viewer"
    (d / "vendor").mkdir(parents=True)
    for name in (*PAGE_FILES, *ICON_FILES, *PLATFORM_FILES):
        (d / name).write_text(page_stub(name))
    (d / "vendor" / "maplibre-gl.js").write_text("// vendored")
    manifest = {
        "volumes": [
            {"id": "v1", "era": "1950", "bounds": [0, 0, 1, 1], "pmtiles": "../work/pm/v1.pmtiles"},
            {"id": "v2", "era": "1910", "bounds": [0, 0, 1, 1]},
            {"id": "v3", "era": None, "bounds": [0, 0, 1, 1], "pmtiles": "../work/pm/v3.pmtiles"},
        ],
        # a manifest written before search-modern-only still carries renames on
        # disk; the bundle must not republish them
        "street_aliases": {"OLD": "NEW"},
        "site": {"name": "Testville"},
    }
    (d / DEPLOY_CITY).mkdir()
    (d / DEPLOY_CITY / "manifest.json").write_text(json.dumps(manifest))
    return d
