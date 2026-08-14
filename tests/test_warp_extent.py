"""Warp extents must match frozen COG fixture extents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autogeoref.warp import (
    WarpResult,
    extent_of,
    gcps_from_feature_collection,
    warp_sheet,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
REGIONS_DIR = FIXTURES / "ref-volume" / "regions"
SHEET_SLUGS = sorted(p.stem for p in REGIONS_DIR.glob("*.jpg")) if REGIONS_DIR.is_dir() else []

#: COG extents reproduce production exactly; see module docstring.
EXTENT_REL_TOL = 1e-9

pytestmark = [
    pytest.mark.golden,
    pytest.mark.gdal,
    pytest.mark.skipif(not SHEET_SLUGS, reason="ref-volume region fixtures not present"),
]


@pytest.fixture(scope="module")
def layers_by_slug(ground_truth_path: Path) -> dict[str, dict[str, Any]]:
    with ground_truth_path.open() as f:
        layers = json.load(f)
    return {layer["slug"]: layer for layer in layers}


@pytest.fixture(scope="module")
def warped(
    tmp_path_factory: pytest.TempPathFactory, layers_by_slug: dict[str, dict[str, Any]]
) -> dict[str, WarpResult]:
    """Warp every ref sheet once, sequentially, into a module-scoped cache."""
    out_dir = tmp_path_factory.mktemp("t5_cogs")
    results: dict[str, WarpResult] = {}
    for slug in SHEET_SLUGS:
        gcps = gcps_from_feature_collection(layers_by_slug[slug]["gcps_geojson"])
        results[slug] = warp_sheet(
            REGIONS_DIR / f"{slug}.jpg", gcps, out_dir, slug=slug, timeout_s=600.0
        )
    return results


@pytest.mark.parametrize("slug", SHEET_SLUGS)
def test_cog_extent_matches_production(
    slug: str, warped: dict[str, WarpResult], layers_by_slug: dict[str, dict[str, Any]]
) -> None:
    result = warped[slug]
    assert result.cog_path.is_file()
    expected = layers_by_slug[slug]["extent"]
    assert list(result.extent_4326) == pytest.approx(expected, rel=EXTENT_REL_TOL)
    # sanity: a mismatch beyond ~1e-3 deg (~100 m) would mean the warp itself
    # is wrong (axis order / CRS confusion), not a gridding artifact
    for got, want in zip(result.extent_4326, expected, strict=True):
        assert abs(got - want) < 1e-3


def test_extent_of_agrees_with_warp_result(warped: dict[str, WarpResult]) -> None:
    result = warped[SHEET_SLUGS[0]]
    assert extent_of(result.cog_path) == result.extent_4326


def test_warp_is_idempotent(
    warped: dict[str, WarpResult], layers_by_slug: dict[str, dict[str, Any]]
) -> None:
    """A second call with an up-to-date COG must skip the warp entirely."""
    slug = SHEET_SLUGS[0]
    first = warped[slug]
    gcps = gcps_from_feature_collection(layers_by_slug[slug]["gcps_geojson"])
    again = warp_sheet(
        REGIONS_DIR / f"{slug}.jpg",
        gcps,
        first.cog_path.parent,
        slug=slug,
        timeout_s=600.0,
    )
    assert again.from_cache
    assert again.cog_path == first.cog_path
    assert again.extent_4326 == first.extent_4326


def test_changed_gcps_invalidate_the_cached_cog(
    warped: dict[str, WarpResult], layers_by_slug: dict[str, dict[str, Any]]
) -> None:
    """GCPs are a warp input that mtime freshness cannot see (seam adjustment
    moves them by design): a changed GCP set must re-warp, never silently
    reuse the stale COG. Runs LAST in this module — it overwrites the
    module-scoped COG for its slug."""
    slug = SHEET_SLUGS[0]
    first = warped[slug]
    gcps = gcps_from_feature_collection(layers_by_slug[slug]["gcps_geojson"])
    # ~85 m eastward shift on every GCP — comfortably beyond the COG's
    # tile-grid snap, so the extent must visibly move too
    shifted = [(px, py, lng + 0.001, lat) for px, py, lng, lat in gcps]
    again = warp_sheet(
        REGIONS_DIR / f"{slug}.jpg",
        shifted,
        first.cog_path.parent,
        slug=slug,
        timeout_s=600.0,
    )
    assert not again.from_cache, "stale COG reused despite changed GCPs"
    assert again.extent_4326 != first.extent_4326
