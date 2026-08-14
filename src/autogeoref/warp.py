"""Plain-GDAL warp chain: GCP attach -> gdalwarp -> Cloud-Optimized GeoTIFF.

Shells out to the GDAL CLI binaries — one warp per OS process, no Python bindings:

1. ``gdal_translate -of VRT -a_srs EPSG:3857 -gcp px py X Y ...`` attaches the ground
   control points (world coordinates in EPSG:3857 metres).
2. ``gdalwarp -t_srs EPSG:3857 -order 1 -dstalpha -r near`` rectifies with a first-order
   polynomial (the least-squares affine, GDAL "poly1").
3. ``gdal_translate -of COG`` writes the final artifact, its ``TILING_SCHEME`` snapping
   resolution to the nearest WebMercatorQuad zoom and aligning to tile boundaries.

Also hosts :func:`gdalwarp_cutline_dryrun`, the mask-acceptance test shared with
:mod:`autogeoref.mask.geometry`: static geometry validity does not guarantee GDAL cutline
acceptance, so candidate masks run through ``gdalwarp`` to a throwaway VRT.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon, mapping

from autogeoref.affine import TO_3857, TO_4326
from autogeoref.paths import atomic_write_text

logger = logging.getLogger(__name__)

#: (px, py, lng, lat) — full-resolution image pixels + EPSG:4326 degrees.
Gcp4326 = tuple[float, float, float, float]

_NICE = ("nice", "-n", "10")


class WarpError(Exception):
    """Base error for the warp module."""


class GdalCommandError(WarpError):
    """A GDAL CLI tool exited nonzero."""

    def __init__(self, cmd: Sequence[str], returncode: int, stderr: str) -> None:
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"{cmd[0]} exited {returncode}: {stderr.strip()[:1000]}")


class GdalTimeoutError(WarpError):
    """A GDAL CLI tool exceeded its timeout."""


@dataclass(frozen=True)
class WarpResult:
    """Outcome of :func:`warp_sheet`."""

    slug: str
    cog_path: Path
    #: COG raster extent as (minx, miny, maxx, maxy) in EPSG:4326 degrees.
    extent_4326: tuple[float, float, float, float]
    #: True when an up-to-date COG already existed and the warp was skipped.
    from_cache: bool


def _run_gdal(cmd: Sequence[str], *, timeout_s: float, nice: bool = True) -> str:
    """Run a GDAL CLI command; return stdout, raise a typed error on failure."""
    full = [*(_NICE if nice else ()), *cmd]
    logger.debug("running: %s", " ".join(full))
    try:
        proc = subprocess.run(full, capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        raise GdalTimeoutError(f"{cmd[0]} timed out after {timeout_s}s") from exc
    except FileNotFoundError as exc:
        # exc.filename may be `nice`, not the GDAL tool — report the real culprit
        raise WarpError(f"binary not found: {exc.filename or full[0]}") from exc
    if proc.returncode != 0:
        raise GdalCommandError(cmd, proc.returncode, proc.stderr)
    return proc.stdout


def gcps_from_feature_collection(fc: dict[str, Any]) -> list[Gcp4326]:
    """``[(px, py, lng, lat), ...]`` from a layer-style GCP FeatureCollection.

    Accepts the fixture / production API format where each feature carries
    ``properties.image = [px, py]`` (full-resolution pixels) and a 4326 Point
    geometry.
    """
    out: list[Gcp4326] = []
    for f in fc["features"]:
        px, py = f["properties"]["image"]
        lng, lat = f["geometry"]["coordinates"]
        out.append((float(px), float(py), float(lng), float(lat)))
    return out


def attach_gcps_vrt(
    image_path: Path,
    gcps: Sequence[Gcp4326],
    vrt_path: Path,
    *,
    timeout_s: float = 120.0,
) -> Path:
    """Write a VRT of ``image_path`` with the GCPs attached (world in 3857).

    image_path: Source raster (unreferenced sheet scan). gcps: ``(px, py, lng, lat)`` control
    points; lng/lat are transformed to EPSG:3857 meters for the ``-gcp`` arguments. vrt_path:
    Output VRT path. timeout_s: Subprocess timeout. Returns ``vrt_path``. Raises WarpError: On
    fewer than 3 GCPs or a failed gdal_translate.
    """
    if len(gcps) < 3:
        raise WarpError(f"need >= 3 GCPs to georeference, got {len(gcps)}")
    cmd = ["gdal_translate", "-q", "-of", "VRT", "-a_srs", "EPSG:3857"]
    for px, py, lng, lat in gcps:
        x, y = TO_3857.transform(lng, lat)
        cmd += ["-gcp", repr(float(px)), repr(float(py)), repr(x), repr(y)]
    cmd += [str(image_path), str(vrt_path)]
    _run_gdal(cmd, timeout_s=timeout_s, nice=False)
    return vrt_path


def _gdalinfo_json(raster_path: Path, timeout_s: float) -> dict[str, Any]:
    out = _run_gdal(["gdalinfo", "-json", str(raster_path)], timeout_s=timeout_s, nice=False)
    info: dict[str, Any] = json.loads(out)
    return info


def extent_of(raster_path: Path, *, timeout_s: float = 60.0) -> tuple[float, float, float, float]:
    """Raster extent as (minx, miny, maxx, maxy) in EPSG:4326 degrees.

    Reads the geotransform + size from ``gdalinfo -json`` and maps all four
    corners (handles rotated geotransforms); EPSG:3857 rasters are converted
    to 4326 with pyproj.

    Raises:
        WarpError: If the raster CRS is neither EPSG:3857 nor EPSG:4326.
    """
    info = _gdalinfo_json(raster_path, timeout_s)
    gt = info["geoTransform"]
    w, h = info["size"]
    xs: list[float] = []
    ys: list[float] = []
    for cx, cy in ((0, 0), (w, 0), (0, h), (w, h)):
        xs.append(gt[0] + gt[1] * cx + gt[2] * cy)
        ys.append(gt[3] + gt[4] * cx + gt[5] * cy)
    wkt = info.get("coordinateSystem", {}).get("wkt", "")
    if '"EPSG",3857' in wkt or "Pseudo-Mercator" in wkt:
        pts = [TO_4326.transform(x, y) for x, y in zip(xs, ys, strict=True)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
    elif '"EPSG",4326' not in wkt:
        raise WarpError(f"unsupported raster CRS for extent_of: {raster_path}")
    return (min(xs), min(ys), max(xs), max(ys))


def _is_fresh(target: Path, *sources: Path | None) -> bool:
    """True if ``target`` exists and is at least as new as every source."""
    if not target.is_file():
        return False
    mtime = target.stat().st_mtime
    return all(mtime >= s.stat().st_mtime for s in sources if s is not None)


def _gcps_fingerprint(gcps: Sequence[Gcp4326]) -> str:
    """Stable content hash of a GCP set (repr floats, order-sensitive).

    The GCPs are a warp INPUT that arrives in memory, not as a file — mtime
    freshness alone cannot see them change, and the pipeline moves GCPs
    after warping by design (seam adjustment rewrites ``gcps_geojson``).
    """
    payload = json.dumps([[repr(float(v)) for v in gcp] for gcp in gcps])
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def warp_sheet(
    image_path: Path,
    gcps: Sequence[Gcp4326],
    out_dir: Path,
    *,
    slug: str,
    timeout_s: float,
    cutline_geojson: Path | None = None,
    force: bool = False,
) -> WarpResult:
    """Georeference one sheet to a Web-Mercator COG (the production chain).

    Idempotent: an existing ``<slug>.tif`` newer than its file inputs whose recorded GCP
    fingerprint sidecar matches the GCPs passed in is kept, unless ``force``. A COG without a
    sidecar is stale — the GCPs are an input the mtime check cannot see, and seam adjustment
    moves them by design. All intermediates live in a temporary work dir inside ``out_dir``, so
    the finished COG moves into place atomically. Returns the COG path and its 4326 extent;
    raises ``WarpError`` on any failed GDAL step.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cog_path = out_dir / f"{slug}.tif"
    sidecar_path = out_dir / f"{slug}.tif.gcps.json"
    fingerprint = _gcps_fingerprint(gcps)
    sidecar_matches = sidecar_path.is_file() and sidecar_path.read_text().strip() == fingerprint
    if not force and sidecar_matches and _is_fresh(cog_path, image_path, cutline_geojson):
        logger.info("%s: COG up to date, skipping warp", slug)
        return WarpResult(
            slug=slug,
            cog_path=cog_path,
            extent_4326=extent_of(cog_path, timeout_s=timeout_s),
            from_cache=True,
        )
    with tempfile.TemporaryDirectory(prefix=f".warp-{slug}-", dir=out_dir) as tmp:
        work = Path(tmp)
        gcp_vrt = attach_gcps_vrt(image_path, gcps, work / "gcps.vrt", timeout_s=timeout_s)
        warped_vrt = work / "warped.vrt"
        warp_cmd = [
            "gdalwarp",
            "-q",
            "-overwrite",
            "-of",
            "VRT",
            "-t_srs",
            "EPSG:3857",
            "-order",
            "1",
            "-dstalpha",
            "-r",
            "near",
        ]
        if cutline_geojson is not None:
            warp_cmd += ["-cutline", str(cutline_geojson), "-crop_to_cutline"]
        warp_cmd += [str(gcp_vrt), str(warped_vrt)]
        _run_gdal(warp_cmd, timeout_s=timeout_s)
        tmp_cog = work / "cog.tif"
        _run_gdal(
            [
                "gdal_translate",
                "-q",
                "-of",
                "COG",
                "-co",
                "COMPRESS=JPEG",
                "-co",
                "TILING_SCHEME=GoogleMapsCompatible",
                str(warped_vrt),
                str(tmp_cog),
            ],
            timeout_s=timeout_s,
        )
        tmp_cog.replace(cog_path)
    # sidecar AFTER the COG lands: a crash between the two leaves a missing/
    # stale fingerprint, which reads as "stale" and re-warps — never the reverse
    atomic_write_text(sidecar_path, fingerprint)
    logger.info("%s: warped to %s", slug, cog_path)
    return WarpResult(
        slug=slug,
        cog_path=cog_path,
        extent_4326=extent_of(cog_path, timeout_s=timeout_s),
        from_cache=False,
    )


def _cutline_feature_collection(polygon: Polygon, crs_epsg: int | None = None) -> dict[str, Any]:
    """GeoJSON FeatureCollection for a cutline, with an optional explicit CRS.

    With ``crs_epsg`` the polygon is tagged with that CRS; 4326 uses the
    GeoJSON default. Without it, coordinate magnitude chooses 3857 or 4326,
    so callers should pass the CRS explicitly when the heuristic is ambiguous.
    """
    fc: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": mapping(polygon)}],
    }
    if crs_epsg is None:
        crs_epsg = 3857 if max(abs(v) for v in polygon.bounds) > 360.0 else 4326
    if crs_epsg != 4326:
        fc["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs_epsg}"}}
    return fc


def gdalwarp_cutline_dryrun(
    src: Path,
    mask_polygon: Polygon,
    *,
    order: int | None = 1,
    timeout_s: float = 120.0,
    crs_epsg: int | None = None,
) -> bool:
    """THE definitive mask-acceptance test: does the real gdalwarp take it?

    Runs the same cutline warp the mosaicker will run, but to a throwaway VRT, so it is cheap.
    Static validity checks do not replicate GDAL's cutline transform and densify pipeline, so
    acceptance requires a clean exit code. The cutline CRS is inferred from coordinate magnitude
    unless ``crs_epsg`` says otherwise. Returns true iff gdalwarp exited 0 with no cutline error
    on stderr; raises ``WarpError`` if gdalwarp is missing or hangs past the timeout — an
    infrastructure failure, not a rejection verdict.
    """
    with tempfile.TemporaryDirectory(prefix="cutline-dryrun-") as tmp:
        work = Path(tmp)
        cutline_path = work / "cutline.geojson"
        cutline_path.write_text(json.dumps(_cutline_feature_collection(mask_polygon, crs_epsg)))
        out_vrt = work / "out.vrt"
        cmd = ["gdalwarp", "-overwrite", "-of", "VRT", "-t_srs", "EPSG:3857"]
        if order is not None:
            cmd += ["-order", str(order)]
        cmd += ["-cutline", str(cutline_path), "-crop_to_cutline", str(src), str(out_vrt)]
        try:
            proc = subprocess.run(
                [*_NICE, *cmd], capture_output=True, text=True, timeout=timeout_s, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise GdalTimeoutError(f"cutline dry-run timed out after {timeout_s}s") from exc
        except FileNotFoundError as exc:
            raise WarpError(f"binary not found: {exc.filename or 'gdalwarp'}") from exc
    if proc.returncode != 0:
        logger.debug("cutline dry-run rejected (exit %d): %s", proc.returncode, proc.stderr.strip())
        return False
    stderr_upper = proc.stderr.upper()
    if "ERROR" in stderr_upper and "CUTLINE" in stderr_upper:
        logger.debug("cutline dry-run rejected (stderr): %s", proc.stderr.strip())
        return False
    return True
