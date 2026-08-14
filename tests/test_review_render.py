"""Batch ghost composites (`review --render`): the review UI's scriptable fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from autogeoref.affine import TO_4326, apply_affine
from autogeoref.errors import ReviewError
from autogeoref.paths import VolumePaths
from autogeoref.review.render import render_ghost_composite

BASE = np.array(
    [
        [-9756000.0, 0.549, 0.012],
        [5139000.0, 0.012, -0.549],
    ]
)
PIXELS = [(500.0, 700.0), (5000.0, 800.0), (4800.0, 8000.0), (700.0, 7600.0)]


def gcps_fc(pixels: list[tuple[float, float]]) -> dict[str, Any]:
    features = []
    for px, py in pixels:
        lng, lat = TO_4326.transform(*apply_affine(BASE, px, py))
        features.append(
            {
                "type": "Feature",
                "properties": {"image": [round(px), round(py)], "note": "auto: A x B"},
                "geometry": {"type": "Point", "coordinates": [lng, lat]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def make_volume(root: Path) -> VolumePaths:
    paths = VolumePaths(root=root / "volX")
    paths.results.mkdir(parents=True)
    paths.sheets.mkdir(parents=True)
    manifest = {
        f"p{n}": {
            "full_size": [5860, 8505],
            "small_size": [1378, 2000],
            "scale": 0.235,
            "file": f"p{n}_small.jpg",
        }
        for n in (2, 6)
    }
    paths.manifest.write_text(json.dumps(manifest))
    (paths.results / "p2.json").write_text(
        json.dumps({"page": "2", "status": "OK", "gcps_geojson": gcps_fc(PIXELS)})
    )
    (paths.results / "p6.json").write_text(
        json.dumps({"page": "6", "status": "REJECTED (no valid RANSAC model)"})
    )
    for entry in manifest.values():
        Image.new("RGB", tuple(entry["small_size"]), (210, 200, 180)).save(
            paths.sheets / entry["file"]
        )
    return paths


def write_centerlines(root: Path) -> Path:
    # one segment through the placed sheet, one far away that must be clipped
    lng, lat = TO_4326.transform(*apply_affine(BASE, 2000.0, 4000.0))
    lng2, lat2 = TO_4326.transform(*apply_affine(BASE, 4000.0, 4000.0))
    path = root / "centerlines.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[lng, lat], [lng2, lat2]],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[0.0, 0.0], [0.001, 0.0]],
                        },
                    },
                ],
            }
        )
    )
    return path


def test_render_draws_centerlines_and_gcp_ties(tmp_path: Path) -> None:
    paths = make_volume(tmp_path)
    centerlines = write_centerlines(tmp_path)
    out = tmp_path / "qa"
    entry = render_ghost_composite(paths, "volX", "2", centerlines, out)
    assert entry["status"] == "OK"
    assert entry["n_gcps"] == len(PIXELS)
    assert entry["centerlines_drawn"] == 1  # the faraway segment is clipped
    composite = Path(entry["composite"])
    assert composite == out / "volX" / "p2_qa.jpg"
    img = Image.open(composite).convert("RGB")
    assert img.width == 1400
    colors = {c for _n, c in img.getcolors(maxcolors=1 << 20)}

    def present(target: tuple[int, int, int]) -> bool:
        return any(sum(abs(a - b) for a, b in zip(c, target, strict=True)) < 90 for c in colors)

    assert present((0, 200, 255))  # centerline ink
    assert present((255, 40, 40))  # GCP tie ink


def test_render_routes_rotated_sheets_through_the_quarter_turn(tmp_path: Path) -> None:
    """Tie ink must land where full_px_to_small puts it, not at naive px*scale."""
    from autogeoref.frames import full_px_to_small

    paths = VolumePaths(root=tmp_path / "volR")
    paths.results.mkdir(parents=True)
    paths.sheets.mkdir(parents=True)
    info = {
        "full_size": [5860, 8505],
        "small_size": [2000, 1378],  # upright on-disk small of a 90°-rotated scan
        "scale": 0.235,
        "file": "p2_small.jpg",
        "rotation_applied": 90,
    }
    paths.manifest.write_text(json.dumps({"p2": info}))
    (paths.results / "p2.json").write_text(
        json.dumps({"page": "2", "status": "OK", "gcps_geojson": gcps_fc(PIXELS)})
    )
    Image.new("RGB", (2000, 1378), (210, 200, 180)).save(paths.sheets / info["file"])
    centerlines = write_centerlines(tmp_path)

    entry = render_ghost_composite(paths, "volR", "2", centerlines, tmp_path / "qa")
    img = Image.open(entry["composite"]).convert("RGB")
    assert (img.width, img.height) == (1400, 964)  # landscape after the quarter-turn
    ratio = 1400 / 2000

    def red_near(x: float, y: float, radius: int = 10) -> bool:
        box = (int(x) - radius, int(y) - radius, int(x) + radius, int(y) + radius)
        region = np.asarray(img.crop(box))
        return bool(
            ((region[..., 0] > 180) & (region[..., 1] < 120) & (region[..., 2] < 120)).any()
        )

    scale = info["scale"]
    for px, py in PIXELS:
        ux, uy = full_px_to_small(px, py, info)
        assert red_near(ux * ratio, uy * ratio), f"no tie ink at rotated anchor for {px, py}"
        naive = (px * scale * ratio, py * scale * ratio)
        anchors = [full_px_to_small(qx, qy, info) for qx, qy in PIXELS]
        if all(
            (naive[0] - ax * ratio) ** 2 + (naive[1] - ay * ratio) ** 2 > 40**2
            for ax, ay in anchors
        ):
            assert not red_near(*naive), f"tie ink at the unrotated position for {px, py}"


def test_cli_render_renders_the_placed_pool_and_writes_a_summary(tmp_path: Path) -> None:
    from autogeoref.cli.entry import main

    make_volume(tmp_path)
    centerlines = write_centerlines(tmp_path)
    cfg = tmp_path / "city.toml"
    cfg.write_text(
        f'[city]\nname = "Testville"\ncenterlines = "{centerlines}"\naliases_dir = "{tmp_path}"\n'
    )
    out = tmp_path / "qa"
    rc = main(
        [
            "review",
            "--city",
            str(cfg),
            "--work",
            str(tmp_path),
            "--volume",
            "volX",
            "--all",
            "--render",
            str(out),
        ]
    )
    assert rc == 0
    summary = json.loads((out / "volX.json").read_text())
    # p6 has no placement, so only the placed sheet renders
    assert set(summary) == {"2"}
    assert (out / "volX" / "p2_qa.jpg").exists()


def test_cli_render_partial_failure_exits_nonzero(tmp_path: Path) -> None:
    """Unattended sweeps read the exit code: a page that cannot render must fail it."""
    from autogeoref.cli.entry import main

    make_volume(tmp_path)
    centerlines = write_centerlines(tmp_path)
    cfg = tmp_path / "city.toml"
    cfg.write_text(
        f'[city]\nname = "Testville"\ncenterlines = "{centerlines}"\naliases_dir = "{tmp_path}"\n'
    )
    out = tmp_path / "qa"
    rc = main(
        [
            "review",
            "--city",
            str(cfg),
            "--work",
            str(tmp_path),
            "--volume",
            "volX",
            "--render",
            str(out),
            "--pages",
            "2,6",
        ]
    )
    assert rc == 1  # p6 has no displayable placement
    summary = json.loads((out / "volX.json").read_text())
    assert set(summary) == {"2"}


def test_cli_render_requires_a_volume(tmp_path: Path) -> None:
    from autogeoref.cli.entry import main

    make_volume(tmp_path)
    centerlines = write_centerlines(tmp_path)
    cfg = tmp_path / "city.toml"
    cfg.write_text(
        f'[city]\nname = "Testville"\ncenterlines = "{centerlines}"\naliases_dir = "{tmp_path}"\n'
    )
    rc = main(
        ["review", "--city", str(cfg), "--work", str(tmp_path), "--render", str(tmp_path / "qa")]
    )
    assert rc == 2


def test_render_without_placement_raises(tmp_path: Path) -> None:
    paths = make_volume(tmp_path)
    centerlines = write_centerlines(tmp_path)
    with pytest.raises(ReviewError, match="no displayable placement"):
        render_ghost_composite(paths, "volX", "6", centerlines, tmp_path / "qa")


def test_render_missing_page_raises(tmp_path: Path) -> None:
    paths = make_volume(tmp_path)
    centerlines = write_centerlines(tmp_path)
    with pytest.raises(ReviewError, match="no result"):
        render_ghost_composite(paths, "volX", "99", centerlines, tmp_path / "qa")
