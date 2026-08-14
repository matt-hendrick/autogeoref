"""Batch ghost composites: the scriptable fallback to the review UI.

The review UI (``autogeoref review``) is the QA medium of record. This renderer
exists for the one case the UI cannot cover: unattended batch QA sweeps when
no browser is available. Each composite draws the city's modern centerlines
and every recorded GCP tie into the sheet's own pixel frame through the
recorded placement — if the placement is right, the drawn streets land on the
printed streets. Offline: results, smalls, and centerlines are all on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageDraw

from ..affine import TO_3857, TO_4326
from ..errors import ReviewError
from ..frames import full_px_to_small
from .materialize import affine_from_record, displayable_affine

if TYPE_CHECKING:
    from ..paths import VolumePaths

RENDER_WIDTH = 1400

_CENTERLINE = (0, 200, 255, 210)
_TIE = (255, 40, 40, 255)
_NODE = (255, 235, 0, 255)


def _clipped_lines(
    centerlines_path: Path, bbox: tuple[float, float, float, float]
) -> list[list[list[float]]]:
    fc = json.loads(centerlines_path.read_text())
    minx, miny, maxx, maxy = bbox
    out = []
    for f in fc["features"]:
        geom = f.get("geometry")
        if geom is None:
            continue
        if geom["type"] == "LineString":
            lines = [geom["coordinates"]]
        elif geom["type"] == "MultiLineString":
            lines = geom["coordinates"]
        else:
            continue
        for line in lines:
            xs = [p[0] for p in line]
            ys = [p[1] for p in line]
            if max(xs) < minx or min(xs) > maxx or max(ys) < miny or min(ys) > maxy:
                continue
            out.append(line)
    return out


def render_ghost_composite(
    paths: VolumePaths,
    volume: str,
    page: str,
    centerlines_path: Path,
    out_dir: Path,
    *,
    width: int = RENDER_WIDTH,
) -> dict[str, Any]:
    """Render ``p<page>_qa.jpg`` into ``out_dir/<volume>/`` and summarize it."""
    rp = paths.results / f"p{page}.json"
    if not rp.exists():
        raise ReviewError(f"no result for {volume} p{page}")
    record = json.loads(rp.read_text())
    manifest = json.loads(paths.manifest.read_text())
    info = manifest.get(f"p{page}")
    if info is None:
        raise ReviewError(f"p{page}: no sheets/manifest.json entry")
    m = displayable_affine(affine_from_record(record))
    if m is None:
        raise ReviewError(f"p{page}: no displayable placement to draw")

    small = Image.open(paths.sheets / info["file"]).convert("RGB")
    ratio = width / small.width
    disp = small.resize((width, int(small.height * ratio)), Image.Resampling.LANCZOS)

    # Result pixels live in the un-rotated source frame; full_px_to_small
    # applies the recorded quarter-turn so rotated sheets draw correctly
    # to keep rotated sheets in the source frame.
    def full_to_disp(px: float, py: float) -> tuple[float, float]:
        ux, uy = full_px_to_small(px, py, info)
        return ux * ratio, uy * ratio

    linear = np.array([[m[0][1], m[0][2]], [m[1][1], m[1][2]]], dtype=float)
    linear_inv = np.linalg.inv(linear)
    offset = np.array([m[0][0], m[1][0]], dtype=float)

    def world_to_disp(lon: float, lat: float) -> tuple[float, float]:
        x, y = TO_3857.transform(lon, lat)
        px, py = linear_inv @ (np.array([x, y]) - offset)
        return full_to_disp(px, py)

    full_w, full_h = float(info["full_size"][0]), float(info["full_size"][1])
    corners = [
        TO_4326.transform(
            m[0][0] + m[0][1] * px + m[0][2] * py,
            m[1][0] + m[1][1] * px + m[1][2] * py,
        )
        for px, py in ((0, 0), (full_w, 0), (0, full_h), (full_w, full_h))
    ]
    lngs, lats = [p[0] for p in corners], [p[1] for p in corners]
    pad = 0.002
    bbox = (min(lngs) - pad, min(lats) - pad, max(lngs) + pad, max(lats) + pad)

    draw = ImageDraw.Draw(disp, "RGBA")
    drawn = 0
    margin = 200
    for line in _clipped_lines(centerlines_path, bbox):
        pts = [world_to_disp(x, y) for x, y in line]
        inside = sum(
            1
            for p in pts
            if -margin <= p[0] <= disp.width + margin and -margin <= p[1] <= disp.height + margin
        )
        if inside < 2:
            continue
        draw.line(pts, fill=_CENTERLINE, width=3)
        drawn += 1

    gcps = (record.get("gcps_geojson") or {}).get("features") or []
    for ft in gcps:
        ix, iy = ft["properties"]["image"]
        lon, lat = ft["geometry"]["coordinates"]
        sx, sy = full_to_disp(ix, iy)
        wx, wy = world_to_disp(lon, lat)
        draw.line([(sx, sy), (wx, wy)], fill=_TIE, width=3)
        draw.ellipse([sx - 7, sy - 7, sx + 7, sy + 7], outline=_TIE, width=3)
        draw.ellipse([wx - 5, wy - 5, wx + 5, wy + 5], fill=_NODE)

    vout = out_dir / volume
    vout.mkdir(parents=True, exist_ok=True)
    composite = vout / f"p{page}_qa.jpg"
    disp.save(composite, quality=85)

    return {
        "status": record.get("status"),
        "n_gcps": len(gcps),
        "rotation_applied": int(info.get("rotation_applied", 0)),
        "composite": str(composite),
        "centerlines_drawn": drawn,
        "world_bbox_4326": [round(v, 5) for v in bbox],
    }
