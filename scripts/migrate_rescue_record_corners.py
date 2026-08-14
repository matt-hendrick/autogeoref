#!/usr/bin/env python
"""One-off migration: append the rescue-model corner GCPs to committed records.

``rescue.with_synthetic_corners`` writes the pinned model's corners on every NEW
rescue record, but ``stage_rescue`` only processes ``REJECTED`` pages, so a
re-run walks past the committed records that predate the fix. For each of those
carrying no synthetic corners, this reconstructs the pinned model from the
record's own anchors and the volume constants, appends the corners through the
production serializer, and rewrites the record.

Skips, all reported: reviewer-confirmed rescues, records whose pinned model
cannot be reconstructed within the anchor-residual guard, records whose
augmented refit does not land nearer the placing model, and every REJECTED
record including revoked-provisional rescues. Dry-run by default; ``--apply``
writes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from autogeoref.affine import (
    AffineMatrix,
    fit_affine,
    fit_affine_checked,
    gcps_from_geojson,
    grid_rmse_m,
)
from autogeoref.config.load import load_city_config
from autogeoref.matching import gcps_geojson_from
from autogeoref.paths import VolumeBusyError, VolumePaths, volume_lock, write_result
from autogeoref.rescue import TOL_M, is_synthetic_gcp, pinned_linear, with_synthetic_corners
from autogeoref.volume import is_reviewer_verified


def determinant(m: AffineMatrix) -> float:
    return float(m[0][1] * m[1][2] - m[0][2] * m[1][1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--configs", type=Path, default=Path("configs"))
    ap.add_argument("--volume", action="append", default=[])
    ap.add_argument("--apply", action="store_true", help="rewrite records (default: dry-run)")
    args = ap.parse_args()

    multiples: dict[str, dict[str, float]] = {}
    for cfg in sorted(args.configs.glob("*/*.toml")):
        for vid, vol in load_city_config(cfg).volumes.items():
            multiples[vid] = dict(vol.page_scale_multiples or {})

    volumes = args.volume or sorted(
        p.parent.name for p in args.work.glob("*/results") if p.is_dir()
    )
    migrated: list[dict[str, Any]] = []
    skipped_rv: list[str] = []
    skipped_unrecon: list[str] = []
    skipped_guard: list[str] = []
    no_constants: list[str] = []
    revoked_no_corners = 0

    busy: list[str] = []

    def process_volume(vid: str) -> None:
        nonlocal revoked_no_corners
        vdir = args.work / vid
        cpath = vdir / "volume-constants.json"
        mpath = vdir / "sheets" / "manifest.json"
        has_constants = cpath.exists() and mpath.exists()
        if has_constants:
            consts = json.loads(cpath.read_text())
            pin_scale, pin_rot = consts["scale_m_per_px"], consts["rotation_deg"]
            manifest = json.loads(mpath.read_text())
        for path in sorted((vdir / "results").glob("*.json")):
            r = json.loads(path.read_text())
            status = str(r.get("status", ""))
            fc = r.get("gcps_geojson")
            is_rescue = "rescued" in status or bool(r.get("rescue_anchors"))
            if not is_rescue or not fc:
                continue
            feats = fc["features"]
            has_corners = any(is_synthetic_gcp(f) for f in feats)
            if "REJECT" in status.upper():
                revoked_no_corners += not has_corners
                continue
            if has_corners:
                continue
            page = path.stem.lstrip("p")
            name = f"{vid} p{page}"
            if not has_constants:
                no_constants.append(name)
                continue
            info = manifest.get(f"p{page}")
            if info is None:
                no_constants.append(name + " (no manifest entry)")
                continue
            if is_reviewer_verified(status):
                skipped_rv.append(name)
                continue
            w, h = info["full_size"]
            gcps = gcps_from_geojson(fc)
            m_rec = fit_affine(gcps)
            page_scale = pin_scale * multiples.get(vid, {}).get(page, 1.0)

            fits: list[tuple[float, list[list[float]]]] = []
            for quad in (0.0, 90.0, 180.0, 270.0):
                lin = pinned_linear(page_scale, pin_rot + quad)
                tx = sum(x - (lin[0][0] * px + lin[0][1] * py) for px, py, x, _ in gcps) / len(gcps)
                ty = sum(y - (lin[1][0] * px + lin[1][1] * py) for px, py, _, y in gcps) / len(gcps)
                m_pin = [[tx, lin[0][0], lin[0][1]], [ty, lin[1][0], lin[1][1]]]
                spread = max(
                    math.hypot(
                        m_pin[0][0] + m_pin[0][1] * px + m_pin[0][2] * py - x,
                        m_pin[1][0] + m_pin[1][1] * px + m_pin[1][2] * py - y,
                    )
                    for px, py, x, y in gcps
                )
                fits.append((spread, m_pin))
            spread, m_pin = min(fits, key=lambda fit: fit[0])
            if spread > 2 * TOL_M:
                skipped_unrecon.append(f"{name} (residual {spread:.1f} m)")
                continue

            corner_feats = gcps_geojson_from(with_synthetic_corners([], m_pin, (w, h)))["features"]
            aug_fc = {"type": "FeatureCollection", "features": feats + corner_feats}
            m_aug = fit_affine_checked(gcps_from_geojson(aug_fc))
            m_pin_arr = np.array(m_pin)
            before = grid_rmse_m(m_rec, m_pin_arr, w, h)
            if m_aug is None or determinant(m_aug) >= 0:
                skipped_guard.append(f"{name} (augmented refit not upright)")
                continue
            after = grid_rmse_m(m_aug, m_pin_arr, w, h)
            if after > before:
                skipped_guard.append(f"{name} ({before:.2f} -> {after:.2f} m, worse)")
                continue
            migrated.append({"name": name, "before": before, "after": after})
            if args.apply:
                r["gcps_geojson"] = aug_fc
                write_result(path, r)

    for vid in volumes:
        if args.apply:
            # the same per-volume exclusion every mutating entry point takes;
            # a volume mid-bake is skipped and named, never raced
            try:
                with volume_lock(VolumePaths(root=args.work / vid), "migrate-rescue-corners"):
                    process_volume(vid)
            except VolumeBusyError as e:
                busy.append(f"{vid} (held by {e.holder or 'unknown'})")
        else:
            process_volume(vid)

    mode = "APPLIED" if args.apply else "dry-run"
    print(f"{mode}: {len(migrated)} records migrated")
    if migrated:
        before = sorted(d["before"] for d in migrated)
        after = sorted(d["after"] for d in migrated)
        mid = len(migrated) // 2
        print(
            f"  displacement from placing model: median {before[mid]:.2f} -> {after[mid]:.2f} m,"
            f" max {before[-1]:.2f} -> {after[-1]:.2f} m,"
            f" >5 m {sum(1 for v in before if v > 5)} -> {sum(1 for v in after if v > 5)}"
        )
    for label, items in (
        ("reviewer-confirmed, untouched", skipped_rv),
        ("not reconstructible, untouched", skipped_unrecon),
        ("guard-refused, untouched", skipped_guard),
        ("no volume constants/manifest", no_constants),
        ("VOLUME BUSY — re-run for these", busy),
    ):
        if items:
            print(f"  {label} ({len(items)}): {', '.join(items)}")
    print(
        f"  revoked-provisional rescue records without corners (out of scope, "
        f"rewritten by any future run that reinstates them): {revoked_no_corners}"
    )


if __name__ == "__main__":
    main()
