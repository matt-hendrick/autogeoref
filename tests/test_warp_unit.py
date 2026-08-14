"""GDAL-free unit tests for warp helpers (the gdal-marked test_warp_extent suite covers
the real chain; these run everywhere)."""

from autogeoref.warp import _gcps_fingerprint


def test_gcps_fingerprint_stable_and_content_sensitive() -> None:
    a = [(1.0, 2.0, -87.6, 41.8), (3.0, 4.0, -87.7, 41.9)]
    assert _gcps_fingerprint(a) == _gcps_fingerprint(list(a))
    # order matters (GCP order is part of what gdal_translate receives)
    assert _gcps_fingerprint(a) != _gcps_fingerprint(a[::-1])
    # a sub-centimeter world move must invalidate: repr() keeps full precision
    b = [(1.0, 2.0, -87.6, 41.8), (3.0, 4.0, -87.7, 41.9000000001)]
    assert _gcps_fingerprint(a) != _gcps_fingerprint(b)
