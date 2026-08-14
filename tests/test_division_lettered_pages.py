"""A volume whose printed sheet numbers all end in an uppercase letter.

The skeleton class is derived, not declared, and the id form alone cannot carry
it: some volumes print the division letter as part of the sheet number, so every
map page is ``NS`` and none of them duplicates another. Misread as skeletons the
whole volume takes page rectangles, skips the overlap split, and paints each
sheet's blank margin over its neighbours' blocks — and the volume-level mask QA
that would say so is computed over an empty set.

The mixed case the rule was written for (numeric sheets plus outline twins of
some of them) is pinned in ``test_mosaic_stratification.py``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

from autogeoref.affine import TO_3857
from autogeoref.bake.masks import stage_masks
from autogeoref.bake.warp import stage_warp
from autogeoref.paths import VolumePaths
from test_mosaic_stratification import (
    SHEET2_OFFSET_M,
    X0,
    Y0,
    _one_block_sheet,
    _result,
)

if TYPE_CHECKING:
    from pathlib import Path

VOL = "volS"


def _mask_areas(fc: dict[str, Any]) -> dict[str, float]:
    return {
        f["properties"]["slug"]: shp_transform(TO_3857.transform, shape(f["geometry"])).area
        for f in fc["features"]
    }


@pytest.fixture(scope="module")
def volume(tmp_path_factory: pytest.TempPathFactory) -> VolumePaths:
    """p1S and p2S: neighbouring one-block sheets, neither a skeleton twin."""
    root: Path = tmp_path_factory.mktemp("division") / VOL
    paths = VolumePaths(root=root)
    paths.regions.mkdir(parents=True)
    paths.results.mkdir(parents=True)
    _one_block_sheet(paths.regions / f"{VOL}_p1S.png")
    _one_block_sheet(paths.regions / f"{VOL}_p2S.png")
    (paths.results / "p1S.json").write_text(json.dumps(_result("1S", X0, Y0)))
    (paths.results / "p2S.json").write_text(json.dumps(_result("2S", X0 + SHEET2_OFFSET_M, Y0)))
    stage_warp(paths, VOL)
    return paths


@pytest.mark.gdal
def test_sheets_with_no_numeric_twin_are_regular_sheets(volume: VolumePaths) -> None:
    """Each takes its colour box, and the split resolves the seam between them."""
    masks = _mask_areas(json.loads(stage_masks(volume, VOL).read_text()))
    # the exempt bake is the page-rectangle yardstick: same sheets, no colour bound
    rects = _mask_areas(
        json.loads(
            stage_masks(
                volume, VOL, content_masks=True, content_mask_exempt=("1S", "2S")
            ).read_text()
        )
    )
    for slug, area in masks.items():
        assert area < 0.6 * rects[slug], slug
    fc = json.loads(stage_masks(volume, VOL).read_text())
    polys = {
        f["properties"]["slug"]: shp_transform(TO_3857.transform, shape(f["geometry"]))
        for f in fc["features"]
    }
    assert polys[f"{VOL}_p1S"].intersection(polys[f"{VOL}_p2S"]).area < 1.0


#: A mixed volume where the twin is on disk but its placement was REJECTED.
MVOL = "volM"


@pytest.fixture(scope="module")
def uncommitted_twin(tmp_path_factory: pytest.TempPathFactory) -> VolumePaths:
    """p1, p1S and p2 on disk; only p1S and p2 have a committed record."""
    root: Path = tmp_path_factory.mktemp("uncommitted") / MVOL
    paths = VolumePaths(root=root)
    paths.regions.mkdir(parents=True)
    paths.results.mkdir(parents=True)
    for page in ("1", "1S", "2"):
        _one_block_sheet(paths.regions / f"{MVOL}_p{page}.png")
    (paths.results / "p1S.json").write_text(json.dumps(_result("1S", X0, Y0)))
    (paths.results / "p2.json").write_text(json.dumps(_result("2", X0 + SHEET2_OFFSET_M, Y0)))
    stage_warp(paths, MVOL)
    return paths


@pytest.mark.gdal
def test_the_twin_need_not_have_placed(uncommitted_twin: VolumePaths) -> None:
    """The class comes from the page INVENTORY, not from what committed.

    p1 is on disk and rejected, so nothing of it is served — but p1S still
    duplicates its ground and must stay out of the split, or a rejected sheet
    silently promotes its own twin into competition with its neighbours.
    """
    masks = _mask_areas(json.loads(stage_masks(uncommitted_twin, MVOL).read_text()))
    assert set(masks) == {f"{MVOL}_p1S", f"{MVOL}_p2"}
    rects = _mask_areas(
        json.loads(
            stage_masks(
                uncommitted_twin, MVOL, content_masks=True, content_mask_exempt=("2",)
            ).read_text()
        )
    )
    # the skeleton keeps its whole page rectangle; the regular neighbour does not
    assert masks[f"{MVOL}_p1S"] == pytest.approx(rects[f"{MVOL}_p1S"])
    assert masks[f"{MVOL}_p2"] < 0.6 * rects[f"{MVOL}_p2"]


@pytest.mark.gdal
def test_the_volume_is_measurable_by_mask_qa(volume: VolumePaths) -> None:
    """The volume-level statistics are computed over the REGULAR sheets.

    A volume classed entirely as duplicate coverage has an empty footprint, so
    coverage is omitted and the report reads exactly like a clean volume.
    """
    stage_masks(volume, VOL)
    doc = json.loads((volume.masks / "masks-qa.json").read_text())
    assert doc["coverage"]["footprint_m2"] > 0
    assert "volume_flags" in doc
