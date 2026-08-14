"""Tiler stage: synthetic raster -> mosaic -> XYZ tiles -> PMTiles round-trip."""

import subprocess
from pathlib import Path

import pytest

from autogeoref.tiles import TilingError, mosaic_gtiff, pack_pmtiles, render_xyz_tiles

# Only the round-trip needs the system GDAL binaries; the rest are pure Python
# and the fast suite deselects `gdal`, so a module-level mark would hide them.

# a small Chicago-ish extent in EPSG:3857 meters
ULX, ULY, LRX, LRY = -9760000.0, 5141000.0, -9759000.0, 5140000.0


@pytest.fixture(scope="module")
def synthetic_raster(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("tiles")
    src = tmp / "src.tif"
    # 256x256 gradient with georeferencing
    import numpy as np
    from PIL import Image

    g = (np.linspace(0, 255, 256 * 256).reshape(256, 256)).astype("uint8")
    arr = np.stack([g, g[::-1], g.T], axis=-1)
    png = tmp / "src.png"
    Image.fromarray(arr, mode="RGB").save(png)
    subprocess.run(
        [
            "gdal_translate",
            "-of",
            "GTiff",
            "-a_srs",
            "EPSG:3857",
            "-a_ullr",
            str(ULX),
            str(ULY),
            str(LRX),
            str(LRY),
            str(png),
            str(src),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return src


@pytest.mark.gdal
def test_mosaic_tiles_pmtiles_roundtrip(synthetic_raster: Path, tmp_path: Path) -> None:
    mosaic = mosaic_gtiff([synthetic_raster], tmp_path / "mosaic.tif")
    assert mosaic.exists()

    # BigTIFF is forced, so even this tiny mosaic carries the 8-byte-offset
    # version word. GDAL's default declines to switch under compression and the
    # mosaic then dies at 4 GB, after every part has been warped.
    with mosaic.open("rb") as f:
        magic = f.read(4)
    version = magic[2:4]
    assert version == (b"+\x00" if magic[:2] == b"II" else b"\x00+"), (
        f"mosaic is not a BigTIFF: version word {version!r}"
    )

    tiles = render_xyz_tiles(
        mosaic, tmp_path / "tiles", min_zoom=12, max_zoom=13, processes=2, webp=True
    )
    webps = list(tiles.rglob("*.webp"))
    assert webps, "no WEBP tiles rendered"

    pm = pack_pmtiles(tiles, tmp_path / "out.pmtiles")
    assert pm.stat().st_size > 0

    # read back: header sane, a written tile retrievable
    from pmtiles.reader import MmapSource, Reader
    from pmtiles.tile import tileid_to_zxy

    with pm.open("rb") as f:
        reader = Reader(MmapSource(f))
        header = reader.header()
        assert header["min_zoom"] == 12
        assert header["max_zoom"] == 13
        # bounds must cover the source extent (Chicago-ish, west hemisphere)
        assert header["min_lon_e7"] < header["max_lon_e7"] < 0
        # every rendered tile must be retrievable byte-identically
        sample = webps[0]
        z = int(sample.parent.parent.name)
        x = int(sample.parent.name)
        y = int(sample.stem)
        data = reader.get(z, x, y)
        assert data == sample.read_bytes()
        del tileid_to_zxy  # imported for API parity documentation


def test_empty_mosaic_rejected(tmp_path: Path) -> None:
    with pytest.raises(TilingError):
        mosaic_gtiff([], tmp_path / "m.tif")


def test_mosaic_creation_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each option is silent to drop and costly to lose, so pin the command.

    Unmarked on purpose: the fast suite deselects `gdal`, and this is the run
    that catches a removal before a whole-volume bake does.
    """
    import autogeoref.tiles as tiles

    seen: list[list[str]] = []
    monkeypatch.setattr(tiles, "_run", lambda cmd, **_kwargs: seen.append(cmd))

    tiles.mosaic_gtiff([tmp_path / "part.tif"], tmp_path / "m.tif")

    (cmd,) = seen
    for option in ("TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=YES"):
        assert option in cmd, option
        assert cmd[cmd.index(option) - 1] == "-co"
    # without it gdalwarp updates in place and drops every -co silently
    assert "-overwrite" in cmd


def test_pack_requires_tiles(tmp_path: Path) -> None:
    (tmp_path / "12").mkdir()
    with pytest.raises(TilingError):
        pack_pmtiles(tmp_path, tmp_path / "out.pmtiles")


def test_failed_pmtiles_replacement_preserves_previous_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiles = tmp_path / "tiles"
    tile = tiles / "0" / "0" / "0.webp"
    tile.parent.mkdir(parents=True)
    tile.write_bytes(b"new tile")
    archive = tmp_path / "out.pmtiles"
    old_archive = b"complete prior archive"
    archive.write_bytes(old_archive)

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replacement failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        pack_pmtiles(tiles, archive)

    assert archive.read_bytes() == old_archive
    assert not list(tmp_path.glob(".out.pmtiles.*.tmp"))


def test_failed_pmtiles_finalization_preserves_previous_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiles = tmp_path / "tiles"
    tile = tiles / "0" / "0" / "0.webp"
    tile.parent.mkdir(parents=True)
    tile.write_bytes(b"new tile")
    archive = tmp_path / "out.pmtiles"
    old_archive = b"complete prior archive"
    archive.write_bytes(old_archive)

    class FailingWriter:
        def __init__(self, _file: object) -> None:
            pass

        def write_tile(self, _tileid: int, _data: bytes) -> None:
            pass

        def finalize(self, _header: object, _metadata: object) -> None:
            raise RuntimeError("finalization failed")

    import pmtiles.writer

    monkeypatch.setattr(pmtiles.writer, "Writer", FailingWriter)
    with pytest.raises(RuntimeError, match="finalization failed"):
        pack_pmtiles(tiles, archive)

    assert archive.read_bytes() == old_archive
    assert not list(tmp_path.glob(".out.pmtiles.*.tmp"))
