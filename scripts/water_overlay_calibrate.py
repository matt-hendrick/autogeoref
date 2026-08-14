"""Calibrate reviewed waterway label axes against committed sheet transforms.

Zero model spend, zero network. It does not alter a placement: it projects a
named modern waterway crossing through an existing committed affine and
compares that point to the manually reviewed label-axis candidate on the scan.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

# water.py is a shared helper, not an instrument: it sits under experiments/
# because nothing in the package imports it, and the other consumer lives there.
sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))

from water import WaterIndex, normalize_water_name, water_crossing_candidates

from autogeoref.affine import TO_3857, fit_affine, gcps_from_geojson, invert_affine
from autogeoref.centerlines import CenterlineIndex
from autogeoref.frames import full_px_to_small
from autogeoref.names import load_aliases

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("AUTOGEOREF_DATA_ROOT", str(ROOT)))
WORK = Path(os.environ.get("AUTOGEOREF_WORK", str(DATA_ROOT / "work")))
TESTBEDS = Path(__file__).with_name("water_calibration_testbeds.json")
WATER = DATA_ROOT / "fixtures" / "reference" / "water-chicago-overpass.json"
STREETS = DATA_ROOT / "fixtures" / "reference" / "street_center_lines.geojson"


def _project_full(inv: np.ndarray, lng: float, lat: float) -> tuple[float, float]:
    x, y = TO_3857.transform(lng, lat)
    px, py = (float(v) for v in inv @ np.array([1.0, x, y]))
    return px, py


def _water_lines(water: WaterIndex, group: str) -> list[list[tuple[float, float]]]:
    geometry = water.groups[group]
    if geometry.geom_type == "LineString":
        return [list(geometry.coords)]
    if geometry.geom_type == "MultiLineString":
        return [list(line.coords) for line in geometry.geoms]
    raise ValueError(f"{group}: expected line geometry, got {geometry.geom_type}")


def measure_testbed(testbed: dict[str, Any], *, render_dir: Path | None = None) -> dict[str, Any]:
    """Measure one reviewed water-label x street-label candidate."""
    volume = str(testbed["volume"])
    page = str(testbed["page"])
    volume_root = WORK / volume
    annotation = json.loads((volume_root / "annotations" / f"p{page}.json").read_text())
    result = json.loads((volume_root / "results" / f"p{page}.json").read_text())
    manifest = json.loads((volume_root / "sheets" / "manifest.json").read_text())
    info = manifest[f"p{page}"]
    aliases = load_aliases(DATA_ROOT / "configs" / "chicago" / "aliases" / f"aliases-{volume}.json")
    streets = CenterlineIndex.from_geojson(STREETS, aliases=aliases)
    label = dict(testbed["water_label"])
    group = normalize_water_name(str(testbed["group"]))
    water = WaterIndex.from_json(WATER, {normalize_water_name(label["name"]): (group,)})
    annotation["water_labels"] = [label]
    candidates = water_crossing_candidates(
        annotation, water, streets, aliases, float(info["scale"])
    )
    candidates = [
        candidate for candidate in candidates if candidate.streets[1] == testbed["street"]
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"{volume}:p{page}: expected one reviewed candidate, got {len(candidates)}"
        )

    candidate = candidates[0]
    model = fit_affine(gcps_from_geojson(result["gcps_geojson"]))
    inv = invert_affine(model)
    projected_full = _project_full(inv, *candidate.world4326)
    candidate_full = candidate.pixel
    candidate_world = model @ np.array([1.0, *candidate_full])
    crossing_world = np.array(TO_3857.transform(*candidate.world4326))
    offset_m = float(np.linalg.norm(candidate_world - crossing_world))
    outcome: dict[str, Any] = {
        "volume": volume,
        "page": page,
        "status": result["status"],
        "review_basis": testbed["review_basis"],
        "ground_truth": bool(testbed["ground_truth"]),
        "water_label": label["name"],
        "group": group,
        "street": testbed["street"],
        "candidate_full_px": [round(v, 2) for v in candidate_full],
        "projected_full_px": [round(v, 2) for v in projected_full],
        "candidate_small_px": [round(v, 2) for v in full_px_to_small(*candidate_full, info)],
        "projected_small_px": [round(v, 2) for v in full_px_to_small(*projected_full, info)],
        "offset_m": round(offset_m, 2),
    }
    if render_dir is not None:
        image = Image.open(volume_root / "sheets" / info["file"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for line in _water_lines(water, group):
            pixels = [full_px_to_small(*_project_full(inv, lng, lat), info) for lng, lat in line]
            draw.line(pixels, fill=(0, 200, 255), width=3)
        for point, color in ((candidate_full, (255, 50, 50)), (projected_full, (255, 235, 0))):
            x, y = full_px_to_small(*point, info)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=color, width=3)
        render_dir.mkdir(parents=True, exist_ok=True)
        image.save(render_dir / f"{volume}-p{page}.jpg", quality=90)
    return outcome


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the runbook's pre-registered calibration bar to one reach."""
    offsets = [float(row["offset_m"]) for row in rows]
    return {
        "testbeds": len(rows),
        "ground_truth_testbeds": sum(bool(row["ground_truth"]) for row in rows),
        "median_offset_m": round(float(np.median(offsets)), 2),
        "max_offset_m": round(max(offsets), 2),
        "bindable": float(np.median(offsets)) <= 5.0 and max(offsets) <= 12.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testbeds", type=Path, default=TESTBEDS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path)
    args = parser.parse_args()
    testbeds = json.loads(args.testbeds.read_text())
    rows = [measure_testbed(testbed, render_dir=args.render_dir) for testbed in testbeds]
    payload = {"rows": rows, "summary": summary(rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
