"""Builders shared by the console test modules.

The console derives everything from a state index over a work tree, so nearly
every test needs a tree with images, results and served archives in it, plus a
city config that declares as much or as little as the test is about. These build
those; `_status` is the index the console is then asked to read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autogeoref.config.model import CityConfig, VolumeConfig
from autogeoref.status import build_status


def _tree(root: Path) -> dict[str, Path]:
    roots = {
        "work": root / "work",
        "fixtures": root / "fixtures",
        "tiles": root / "deploy" / "tiles",
        "ground_truth": root / "fixtures" / "ground-truth",
    }
    for p in roots.values():
        p.mkdir(parents=True, exist_ok=True)
    return roots


def _images(work: Path, volume: str, pages: int) -> None:
    regions = work / volume / "regions"
    regions.mkdir(parents=True, exist_ok=True)
    for i in range(1, pages + 1):
        (regions / f"chicago_1900_vol_1_p{i}.jpg").write_bytes(b"")


def _results(work: Path, volume: str, accepted: int, flagged: int) -> None:
    results = work / volume / "results"
    results.mkdir(parents=True, exist_ok=True)
    page = 1
    for _ in range(accepted):
        (results / f"p{page}.json").write_text(
            json.dumps(
                {"page": str(page), "status": "OK", "gcps_geojson": {"type": "FeatureCollection"}}
            )
        )
        page += 1
    for _ in range(flagged):
        (results / f"p{page}.json").write_text(
            json.dumps({"page": str(page), "status": "REJECTED (no valid RANSAC model)"})
        )
        page += 1


def _tiles(tiles: Path, provenance: str, volume: str) -> None:
    d = tiles / provenance
    d.mkdir(parents=True, exist_ok=True)
    # NOT zero bytes: `viewer.classify_pmtiles` reads an empty archive as an
    # in-progress bake and skips it, so a volume tiled with one would still be
    # a candidate — which is correct, and would make this fixture lie
    (d / f"{volume}.pmtiles").write_bytes(b"PMTiles")


#: any valid bbox: the console only asks whether a bounds SOURCE exists, never
#: where it is (`run_inputs.resolve_bounds` owns the where)
_BBOX = (-88.0, 41.6, -87.5, 42.1)


def _city(*, renumbering: bool = True, declared: dict[str, bool] | None = None) -> CityConfig:
    # a "declared" volume is a RUNNABLE one, and the runner demands a bounds
    # source as well as an era (`cli_run._cmd_run`, era first, bounds second) — so
    # declaring here declares both. A volume missing exactly one is built by
    # hand in the tests that are about that one.
    volumes = {
        vid: VolumeConfig(
            identifier=vid,
            addresses_modern=modern,
            evidence_channels=("junction", "addresses"),
            bounds_bbox=_BBOX,
        )
        for vid, modern in (declared or {}).items()
    }
    return CityConfig(
        name="Chicago, Ill.",
        centerlines_path=Path("cl.geojson"),
        aliases_dir=Path("aliases"),
        evidence_channels=("junction", "addresses"),
        escalation_models=("claude-sonnet-5", "claude-opus-4-8"),
        renumbering_table_path=Path("renumbering.json") if renumbering else None,
        volumes=volumes,
    )


def _status(roots: dict[str, Path]) -> list[Any]:
    return build_status(
        work=roots["work"],
        fixtures=roots["fixtures"],
        tiles=roots["tiles"],
        ground_truth=roots["ground_truth"],
    )


def _city_toml(tmp_path: Path, *, loc_catalog: str | None) -> Path:
    body = '[city]\nname = "Chicago, Ill."\ncenterlines = "cl.geojson"\naliases_dir = "aliases"\n'
    if loc_catalog is not None:
        body += f'loc_catalog = "{loc_catalog}"\n'
    p = tmp_path / "city.toml"
    p.write_text(body)
    (tmp_path / "cl.geojson").write_text('{"type":"FeatureCollection","features":[]}')
    (tmp_path / "aliases").mkdir(exist_ok=True)
    return p


def _console_args(tmp_path: Path, cfg: Path, *extra: str) -> argparse.Namespace:
    from autogeoref.cli.parser import build_parser

    roots = _tree(tmp_path)
    return build_parser().parse_args(
        [
            "queue",
            "--serve",
            "--city",
            str(cfg),
            "--work",
            str(roots["work"]),
            "--fixtures",
            str(roots["fixtures"]),
            "--tiles",
            str(roots["tiles"]),
            *extra,
        ]
    )
