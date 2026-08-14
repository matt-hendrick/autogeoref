"""A synthetic review world: one volume on disk, its sidecars, and the app over it.

`make_volume` writes three sheets against a plausible affine — one committed,
one rescue-revoked with a placement, one plain rejected — which is enough shape
for the queue, the payload, the save and the apply. `make_app` puts a review app
over it with centerlines to clip. `look` opens a sheet the way the UI does,
fetching the ghost raster too, because a verdict on a sheet the server never
painted is refused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from autogeoref.affine import TO_4326, apply_affine
from autogeoref.paths import VolumePaths
from autogeoref.review.app import ReviewApp
from autogeoref.review.materialize import compose_ops
from autogeoref.review.sidecars import result_sha256 as sha_of
from autogeoref.review.sidecars import sidecar_path

BASE = np.array(
    [
        [-9756000.0, 0.549, 0.012],
        [5139000.0, 0.012, -0.549],
    ]
)
FULL_SIZE = (5860.0, 8505.0)


def ops_translate(dx: float, dy: float) -> list[dict[str, Any]]:
    return [{"type": "translate", "dx_m": dx, "dy_m": dy}]


def make_sidecar_dict(**over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "volume": "volX",
        "page": "10",
        "base_result_sha256": "abc123",
        "verdict": "adjusted",
        "ops": ops_translate(10, -5),
        "affine": [[float(v) for v in row] for row in BASE],
        "mask_px": [[10.0, 10.0], [100.0, 10.0], [100.0, 100.0]],
    }
    d.update(over)
    return d


def gcps_fc(m: Any, pixels: list[tuple[float, float]], note: str = "auto: A x B") -> dict[str, Any]:
    features = []
    for px, py in pixels:
        x, y = apply_affine(m, px, py)
        lng, lat = TO_4326.transform(x, y)
        features.append(
            {
                "type": "Feature",
                "properties": {"image": [round(px), round(py)], "username": "admin", "note": note},
                "geometry": {"type": "Point", "coordinates": [lng, lat]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


PIXELS = [(500.0, 700.0), (5000.0, 800.0), (4800.0, 8000.0), (700.0, 7600.0)]


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
        for n in (2, 4, 6)
    }
    paths.manifest.write_text(json.dumps(manifest))
    # p2: committed OK; p4: revoked rescue with a provisional placement;
    # p6: plain rejected, no placement
    (paths.results / "p2.json").write_text(
        json.dumps({"page": "2", "status": "OK", "gcps_geojson": gcps_fc(BASE, PIXELS)})
    )
    shifted = compose_ops(BASE, ops_translate(400.0, 0.0))
    (paths.results / "p4.json").write_text(
        json.dumps(
            {
                "page": "4",
                "status": "REJECTED (rescue revoked: anchors share one street)",
                "layer": None,
                "gcps_geojson": gcps_fc(shifted, PIXELS),
            }
        )
    )
    (paths.results / "p6.json").write_text(
        json.dumps({"page": "6", "status": "REJECTED (no valid RANSAC model)"})
    )
    return paths


def save_ui_sidecar(paths: VolumePaths, page: str, **over: Any) -> None:
    over.setdefault("mask_px", None)
    d = make_sidecar_dict(
        page=page,
        base_result_sha256=sha_of(paths.results / f"p{page}.json"),
        **over,
    )
    p = sidecar_path(paths, page)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))


def make_app(tmp_path: Path) -> tuple[ReviewApp, VolumePaths]:
    from autogeoref.config.model import CityConfig

    paths = make_volume(tmp_path)
    centerlines = tmp_path / "centerlines.geojson"
    lng, lat = TO_4326.transform(*apply_affine(BASE, 2000, 4000))
    centerlines.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"street_nam": "MADISON", "street_typ": "ST"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[lng, lat], [lng + 0.001, lat]],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {"street_nam": "FARAWAY", "street_typ": "ST"},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[0.0, 0.0], [0.001, 0.0]],
                        },
                    },
                ],
            }
        )
    )
    city = CityConfig(name="Testville", centerlines_path=centerlines, aliases_dir=tmp_path)
    app = ReviewApp(work=tmp_path, city=city, ui_dir=tmp_path, vendor_dir=tmp_path)
    return app, paths


def composed_affine(payload: dict[str, Any], ops: list[dict[str, Any]]) -> list[list[float]]:
    """What the UI sends: the op log composed onto the served base affine."""
    m = compose_ops(np.asarray(payload["affine"]), ops)
    return [[float(v) for v in row] for row in m]


def look(app: ReviewApp, volume: str, page: str) -> dict[str, Any]:
    """Open a sheet the way the UI does: fetch its payload AND its ghost raster.

    Both, because `save` refuses a verdict on a sheet whose overlay this server never
    painted (`ReviewApp.overlay_shown`), and the overlay IS the media file — the
    payload alone is what a client that never drew anything would have fetched. A
    test that skipped this would be asserting the behaviour of a reviewer who never
    looked, which is the thing the gate exists to refuse.
    """
    payload = app.sheet_payload(volume, page)
    app.media_path(volume, Path(payload["small_url"]).name)
    return payload


PENTAGON_PX = [
    [300.0, 200.0],
    [5500.0, 350.0],
    [5400.0, 8100.0],
    [2900.0, 8400.0],
    [400.0, 8000.0],
]


def sidecar_with_mask(paths: VolumePaths, page: str) -> None:
    save_ui_sidecar(
        paths,
        page,
        verdict="adjusted",
        ops=ops_translate(-400.0, 0.0),
        mask_px=PENTAGON_PX,
    )
