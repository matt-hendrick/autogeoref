"""Back half of the pipeline (warp -> mask -> mosaic -> tile) end to end.

Runs the four CLI-wired stages on a synthetic two-sheet volume — blank
white sheets, corner GCPs placing them side by side
(with real overlap) on a Chicago-ish EPSG:3857 box — and asserts the full
chain: COGs with fingerprint sidecars, dry-run-accepted masks trimmed at
the shared seam, an alpha-carrying composited mosaic over cutline parts,
and a readable PMTiles archive. Needs the GDAL binaries but no fixture tree (everything is
generated), so it runs on a fresh clone.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from autogeoref.affine import TO_4326
from autogeoref.bake.masks import stage_masks
from autogeoref.bake.mosaic import stage_mosaic
from autogeoref.bake.tiles import stage_tiles
from autogeoref.bake.warp import stage_warp
from autogeoref.errors import PipelineError
from autogeoref.paths import VolumePaths
from conftest import antedate

pytestmark = pytest.mark.gdal

VOL = "volX"
W, H = 600, 400
M_PER_PX = 2.0
# sheet 1 top-left in EPSG:3857 (Chicago-ish); sheet 2 offset 500 px east so
# the page-rectangle masks still overlap by ~190 m (beyond the sliver floor)
X0, Y0 = -9760000.0, 5141000.0
SHEET2_OFFSET_M = 500 * M_PER_PX


def _sheet_image(path: Path) -> None:
    # blank paper, no printed frame: real plates carry none, so
    # `mask.geometry.detect_page_bounds` returns the whole scanned page and the
    # Voronoi split is the only thing that trims these masks
    #
    Image.new("RGB", (W, H), "white").save(path, "JPEG", quality=90)


def _gcps_fc(x0: float, y0: float) -> dict[str, Any]:
    feats = []
    for px, py in [(0, 0), (W, 0), (W, H), (0, H)]:
        lng, lat = TO_4326.transform(x0 + px * M_PER_PX, y0 - py * M_PER_PX)
        feats.append(
            {
                "type": "Feature",
                "properties": {"image": [px, py]},
                "geometry": {"type": "Point", "coordinates": [lng, lat]},
            }
        )
    return {"type": "FeatureCollection", "features": feats}


def _result(page: str, x0: float, y0: float) -> dict[str, Any]:
    return {"page": page, "status": "OK", "gcps_geojson": _gcps_fc(x0, y0)}


@pytest.fixture(scope="module")
def volume(tmp_path_factory: pytest.TempPathFactory) -> VolumePaths:
    root = tmp_path_factory.mktemp("backhalf") / VOL
    paths = VolumePaths(root=root)
    paths.regions.mkdir(parents=True)
    paths.results.mkdir(parents=True)
    _sheet_image(paths.regions / f"{VOL}_p1.jpg")
    _sheet_image(paths.regions / f"{VOL}_p2.jpg")
    (paths.results / "p1.json").write_text(json.dumps(_result("1", X0, Y0)))
    (paths.results / "p2.json").write_text(json.dumps(_result("2", X0 + SHEET2_OFFSET_M, Y0)))
    # committed but no full-res image: must be skipped, not fatal
    (paths.results / "p3.json").write_text(json.dumps(_result("3", X0, Y0 - 2 * H * M_PER_PX)))
    # rejected: must never reach the back half
    (paths.results / "p4.json").write_text(
        json.dumps({"page": "4", "status": "REJECTED (no valid RANSAC model)"})
    )
    # every warp input predates the COG made from it; say so on disk rather
    # than trusting the wall clock to record the two writes in order
    antedate(*sorted(paths.regions.glob("*.jpg")))
    return paths


@pytest.fixture(scope="module")
def warped(volume: VolumePaths) -> dict[str, Any]:
    return stage_warp(volume, VOL)


def test_warp_covers_committed_sheets_and_skips_honestly(
    volume: VolumePaths, warped: dict[str, Any]
) -> None:
    assert sorted(warped["warped"]) == [f"{VOL}_p1", f"{VOL}_p2"]
    assert warped["skipped_no_image"] == ["3"]
    for slug in warped["warped"]:
        assert (volume.warped / f"{slug}.tif").is_file()
        assert (volume.warped / f"{slug}.tif.gcps.json").is_file()
    assert not (volume.warped / f"{VOL}_p4.tif").exists()


def test_warp_rerun_is_idempotent(volume: VolumePaths, warped: dict[str, Any]) -> None:
    cog = volume.warped / f"{VOL}_p1.tif"
    summary_file = volume.warped / "warp-summary.json"
    before = (cog.stat().st_mtime, summary_file.stat().st_mtime)
    again = stage_warp(volume, VOL)
    assert again == warped
    assert (cog.stat().st_mtime, summary_file.stat().st_mtime) == before


def test_masks_are_dryrun_accepted_and_seam_trimmed(
    volume: VolumePaths, warped: dict[str, Any]
) -> None:
    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform

    from autogeoref.affine import TO_3857

    masks_path = stage_masks(volume, VOL)
    fc = json.loads(masks_path.read_text())
    by_slug = {f["properties"]["slug"]: shape(f["geometry"]) for f in fc["features"]}
    assert sorted(by_slug) == [f"{VOL}_p1", f"{VOL}_p2"]
    assert fc["unmasked"] == []
    for slug in by_slug:
        assert (volume.masks / f"{slug}.geojson").is_file()
    # the Voronoi split removes the overlap: each mask must be smaller than
    # its full sheet footprint
    sheet_area = (W * M_PER_PX) * (H * M_PER_PX)
    a = shp_transform(TO_3857.transform, by_slug[f"{VOL}_p1"])
    b = shp_transform(TO_3857.transform, by_slug[f"{VOL}_p2"])
    assert a.area < 0.95 * sheet_area
    assert b.area < 0.95 * sheet_area
    # seam split resolved the overlap down to at most a hairline
    assert a.intersection(b).area < 1.0


def test_mosaic_and_tiles_roundtrip(volume: VolumePaths, warped: dict[str, Any]) -> None:
    stage_masks(volume, VOL)
    mosaic = stage_mosaic(volume)
    assert mosaic.is_file() and mosaic.name == "mosaic.tif"
    parts = sorted(p.name for p in (volume.root / "mosaic-parts").glob("*.vrt"))
    assert parts == [f"{VOL}_p1.vrt", f"{VOL}_p2.vrt"]
    # the mosaic's declared inputs all predate it; pin them so the skip cannot
    # turn on the wall clock recording the two writes in order
    antedate(*sorted((volume.root / "mosaic-parts").iterdir()), *sorted(volume.warped.iterdir()))
    mosaic_mtime = mosaic.stat().st_mtime
    assert stage_mosaic(volume).stat().st_mtime == mosaic_mtime  # fresh-skip

    # transparency survives the chain: the composited mosaic must carry an
    # alpha band (the first _041 bake dropped the COGs' internal masks and
    # rendered nodata as opaque black, hiding whole sheets)
    import subprocess

    info = subprocess.run(
        ["gdalinfo", str(mosaic)], capture_output=True, text=True, timeout=60, check=True
    ).stdout
    assert "ColorInterp=Alpha" in info

    pm = stage_tiles(volume, VOL, min_zoom=13, max_zoom=15, processes=2)
    assert pm.name == f"{VOL}.pmtiles"
    from pmtiles.reader import MmapSource, Reader

    with pm.open("rb") as f:
        header = Reader(MmapSource(f)).header()
    assert header["min_zoom"] == 13
    assert header["max_zoom"] == 15
    # bounds must sit on the synthetic sheets (west hemisphere, Chicago-ish)
    assert header["min_lon_e7"] < header["max_lon_e7"] < 0


def test_ladder_exhaustion_unmasks_sheet_and_drops_stale_cutline(
    volume: VolumePaths, warped: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The never-exercised fallback (14/14 healed on real data): an exhausted
    heal ladder must unlink the sheet's cutline from an earlier run — a stale
    ``masks/<slug>.geojson`` would silently re-crop the sheet at mosaic time —
    and the maskless part must still hand the mosaic uniform RGBA."""
    import subprocess

    import autogeoref.mask.geometry as mask_mod
    import autogeoref.warp as warp_mod

    stage_masks(volume, VOL)  # the earlier run: accepted cutlines on disk
    assert (volume.masks / f"{VOL}_p1.geojson").is_file()
    monkeypatch.setattr(warp_mod, "gdalwarp_cutline_dryrun", lambda *_a, **_k: False)
    monkeypatch.setattr(mask_mod, "heal", lambda _poly, _accepts: None)
    fc = json.loads(stage_masks(volume, VOL).read_text())
    assert sorted(fc["unmasked"]) == [f"{VOL}_p1", f"{VOL}_p2"]
    assert fc["features"] == []
    assert not (volume.masks / f"{VOL}_p1.geojson").exists()
    assert not (volume.masks / f"{VOL}_p2.geojson").exists()

    mosaic = stage_mosaic(volume)  # parts rebuilt WITHOUT cutlines
    info = subprocess.run(
        ["gdalinfo", str(mosaic)], capture_output=True, text=True, timeout=60, check=True
    ).stdout
    assert "ColorInterp=Alpha" in info


def test_mosaic_excludes_cog_of_page_revoked_after_warp(
    volume: VolumePaths, warped: dict[str, Any]
) -> None:
    """A page committed at warp time and revoked later (revoke-stale, reviewer
    review, escalation flip) leaves its COG under warped/ — the mosaic must
    follow the warp summary's committed set, not the directory, and the
    shrunken part set must invalidate the previous mosaic (removing a source
    bumps no mtime)."""
    # This test owns its initial mosaic rather than relying on an earlier test
    # sharing the module-scoped volume fixture.
    stage_masks(volume, VOL)
    initial_mosaic = stage_mosaic(volume)
    os.utime(initial_mosaic, (1000, 1000))
    rp = volume.results / "p2.json"
    r = json.loads(rp.read_text())
    r["status"] = "REJECTED (revoked after re-run)"
    rp.write_text(json.dumps(r))

    before = initial_mosaic.stat().st_mtime
    summary = stage_warp(volume, VOL)
    assert sorted(summary["warped"]) == [f"{VOL}_p1"]
    mosaic = stage_mosaic(volume)
    parts = json.loads((volume.root / "mosaic-parts" / "parts.json").read_text())
    assert parts == [f"{VOL}_p1.vrt"]
    assert (volume.warped / f"{VOL}_p2.tif").is_file()  # stale COG remains...
    assert mosaic.stat().st_mtime > before  # ...but the mosaic dropped it


def test_warp_without_any_image_fails_loudly(tmp_path: Path) -> None:
    paths = VolumePaths(root=tmp_path / VOL)
    paths.results.mkdir(parents=True)
    (paths.results / "p1.json").write_text(json.dumps(_result("1", X0, Y0)))
    with pytest.raises(PipelineError, match=r"no.*full-res image"):
        stage_warp(paths, VOL)


def test_cli_wires_the_back_half_behind_the_warp_flag() -> None:
    from autogeoref.cli.parser import build_parser

    args = build_parser().parse_args(
        ["run", "volX", "--city", "c.toml", "--warp", "--max-zoom", "16"]
    )
    assert args.warp and args.max_zoom == 16
    args2 = build_parser().parse_args(["run", "volX", "--city", "c.toml"])
    assert not args2.warp and args2.max_zoom is None
