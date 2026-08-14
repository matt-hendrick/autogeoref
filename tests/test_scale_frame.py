"""Anchors the ``scale_m_per_px`` frame contract to physical ground truth.

The contract (``config.VolumeConfig``): pins are EPSG:3857 meters per
full-resolution pixel. A printed scale legend at scan DPI gives TRUE ground
m/px instead, ~26% under the 3857 value at Chicago's latitude — the mistake
that rendered every ``_190`` seeded ghost undersized by exactly 1/cos(lat)
.

Each pinned volume with a known printed legend is checked against
``legend / DPI / cos(lat)``. The 10% tolerance passes printing and
fit-derivation slop (measured pins sit within 3%) and fails a wrong-frame
pin (26%+) outright.
"""

import math
from pathlib import Path

import pytest

from autogeoref.config.load import load_city_config
from autogeoref.config.model import VolumeConfig
from autogeoref.review.materialize import seed_affine

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

#: For a volume pinned by counterpart footprint (no config bbox): the pipeline
#: derives its correction latitude from ground truth at runtime; this frame
#: check only needs city-scale precision against a 10% tolerance.
FALLBACK_LAT = 41.8

M_PER_FT = 0.3048
#: All three volumes' jp2 masters carry 300 DPI metadata, corroborated by the
#: LOC catalog's physical sizes (_188/_189: 9000 px / 300 DPI = 76.2 cm vs the
#: catalog's 112 x 77 cm sheet).
SCAN_DPI = 300.0

#: Printed scale legends, feet per inch: _024 is the standard Sanborn detail
#: scale; _189 is a 200 ft/in CBD overview; _190 p3 prints "Scale 150 ft. to
#: an inch (Approx.)".
PRINTED_FT_PER_IN = {
    "sanborn01790_024": 50.0,
    "sanborn01790_189": 200.0,
    "sanborn01790_190": 150.0,
}


def _cos_lat(vol: VolumeConfig) -> float:
    if vol.bounds_bbox is not None:
        lat = (vol.bounds_bbox[1] + vol.bounds_bbox[3]) / 2.0
    else:
        lat = FALLBACK_LAT
    return math.cos(math.radians(lat))


def _expected_3857_m_per_px(vol: VolumeConfig, ft_per_in: float) -> float:
    true_m_per_px = ft_per_in * M_PER_FT / SCAN_DPI
    return true_m_per_px / _cos_lat(vol)


@pytest.mark.parametrize("volume", sorted(PRINTED_FT_PER_IN))
def test_pin_is_3857_frame(volume: str) -> None:
    """A pin recorded in true ground meters (the printed-legend frame) fails."""
    vol = load_city_config(CONFIGS / "chicago" / "chicago.toml").volume(volume)
    assert vol.scale_m_per_px is not None, f"{volume}: expected a config pin"
    expected = _expected_3857_m_per_px(vol, PRINTED_FT_PER_IN[volume])
    assert vol.scale_m_per_px == pytest.approx(expected, rel=0.10), (
        f"{volume}: pin {vol.scale_m_per_px} vs printed-legend 3857 value "
        f"{expected:.4f} — a ~26% gap means the pin was recorded in true "
        f"ground meters; divide by cos(lat) (see config.VolumeConfig)"
    )


def test_seed_ghost_span_matches_printed_scale() -> None:
    """The check that would have caught the 26% undersize: a ghost seeded from
    ``_190``'s pin must draw the sheet's width at its physical ground span."""
    vol = load_city_config(CONFIGS / "chicago" / "chicago.toml").volume("sanborn01790_190")
    full_size = (6342.0, 7605.0)  # _190 region masters at 300 DPI
    lon = (vol.bounds_bbox[0] + vol.bounds_bbox[2]) / 2.0
    lat = (vol.bounds_bbox[1] + vol.bounds_bbox[3]) / 2.0
    radius = 6378137.0
    center = (
        math.radians(lon) * radius,
        radius * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)),
    )
    m = seed_affine(vol.scale_m_per_px, vol.rotation_deg, center, full_size)

    def apply(px: float, py: float) -> tuple[float, float]:
        return (
            m[0][0] + m[0][1] * px + m[0][2] * py,
            m[1][0] + m[1][1] * px + m[1][2] * py,
        )

    ax, ay = apply(0.0, 0.0)
    bx, by = apply(full_size[0], 0.0)
    drawn_3857 = math.hypot(bx - ax, by - ay)
    expected_3857 = full_size[0] * _expected_3857_m_per_px(
        vol, PRINTED_FT_PER_IN["sanborn01790_190"]
    )
    assert drawn_3857 == pytest.approx(expected_3857, rel=0.10)
