"""Allmaps annotation writer vs the recorded proof artifacts.

The writer, fed the same inputs (``_024`` p1 GCPs, LOC IIIF id + dims, border
mask), must reproduce ``tests/data/a1-annotation.json`` — the exact annotation
proven to render warped and correctly placed in viewer.allmaps.org. That file is
tracked, frozen evidence of a human check: regenerate it and the proof is gone.
The volume export (``autogeoref allmaps``) must reproduce the same proven item
from the fixture volume's recorded results alone.

The proof predates the data licence, so it carries no ``rights``. Rather than
regenerate it, the comparisons below take the licence off what the writer built
and assert it separately — strictly, so dropping the stamp fails here too.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from autogeoref.allmaps import (
    RIGHTS,
    AnnotationError,
    border_mask,
    export_volume,
    georef_annotation,
)
from autogeoref.paths import VolumePaths

DATA = Path(__file__).resolve().parent / "data"
WORK = Path(__file__).resolve().parent.parent / "work"

# Constants read off the proven annotation: LOC IIIF ImageService2, info.json
# 5882x7322 — identical frame to the recorded result GCP pixels.
IIIF_ID = (
    "https://tile.loc.gov/image-services/iiif/"
    "service:gmd:gmd410m:g4104m:g4104cm:g01790191707:01790_07_1917-0001"
)
WIDTH, HEIGHT = 5882, 7322
PROBE_ANNOTATION_ID = "http://localhost:8899/a1-annotation.json"


def without_rights(node: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``node`` with its licence removed, failing if it had none."""
    stripped = dict(node)
    assert stripped.pop("rights", None) == RIGHTS, f"no licence on {node.get('type')}"
    return stripped


@pytest.fixture(scope="module")
def probe_annotation() -> dict[str, Any]:
    annotation: dict[str, Any] = json.loads((DATA / "a1-annotation.json").read_text())
    return annotation


@pytest.fixture(scope="module")
def p1_gcps(fixtures_dir: Path) -> list[tuple[float, float, float, float]]:
    res = json.loads((fixtures_dir / "sanborn01790_024" / "results" / "p1.json").read_text())
    out = []
    for f in res["gcps_geojson"]["features"]:
        px, py = f["properties"]["image"]
        lng, lat = f["geometry"]["coordinates"]
        out.append((px, py, lng, lat))
    return out


def test_reproduces_probe_annotation(probe_annotation: dict[str, Any], p1_gcps: list[Any]) -> None:
    ann = georef_annotation(
        iiif_image_id=IIIF_ID,
        image_width=WIDTH,
        image_height=HEIGHT,
        gcps=p1_gcps,
        mask_pixels=border_mask(WIDTH, HEIGHT),
        transformation_order=1,
        annotation_id=PROBE_ANNOTATION_ID,
    )
    page = without_rights(ann)
    page["items"] = [without_rights(item) for item in page["items"]]
    assert page == probe_annotation


def _three_gcps() -> list[tuple[float, float, float, float]]:
    return [(0, 0, -87.7, 41.9), (100, 0, -87.6, 41.9), (0, 100, -87.7, 41.8)]


def test_mask_none_omits_selector_and_id() -> None:
    ann = georef_annotation(
        iiif_image_id=IIIF_ID,
        image_width=WIDTH,
        image_height=HEIGHT,
        gcps=_three_gcps(),
    )
    item = ann["items"][0]
    assert "selector" not in item["target"]
    assert "id" not in item
    assert item["target"]["source"]["type"] == "ImageService2"
    assert item["body"]["transformation"] == {"type": "polynomial", "options": {"order": 1}}


def test_border_mask_probe_values() -> None:
    # the exact corner points baked into the proven annotation
    assert border_mask(WIDTH, HEIGHT) == [(88, 110), (5794, 110), (5794, 7212), (88, 7212)]


def test_too_few_gcps_rejected() -> None:
    with pytest.raises(AnnotationError):
        georef_annotation(
            iiif_image_id=IIIF_ID,
            image_width=WIDTH,
            image_height=HEIGHT,
            gcps=_three_gcps()[:2],
        )
    # polynomial order 2 needs 6 points
    with pytest.raises(AnnotationError):
        georef_annotation(
            iiif_image_id=IIIF_ID,
            image_width=WIDTH,
            image_height=HEIGHT,
            gcps=_three_gcps(),
            transformation_order=2,
        )


def test_degenerate_mask_rejected() -> None:
    with pytest.raises(AnnotationError):
        georef_annotation(
            iiif_image_id=IIIF_ID,
            image_width=WIDTH,
            image_height=HEIGHT,
            gcps=_three_gcps(),
            mask_pixels=[(0, 0), (1, 1)],
        )


# ----------------------------------------------------------------------
# Volume export (the `autogeoref allmaps` path)
# ----------------------------------------------------------------------

VOL = "sanborn01790_024"


@pytest.fixture(scope="module")
def item_services() -> dict[str, str]:
    """Per-page IIIF service ids from the cached ``_024`` LOC item JSON."""
    from autogeoref.loc import sheet_iiif_services

    path = WORK / f"loc-item-{VOL}.json"
    if not path.exists():
        pytest.skip(f"work/loc-item-{VOL}.json not present")
    return sheet_iiif_services(json.loads(path.read_text()))


def test_export_volume_round_trips_probe_item(
    fixtures_dir: Path, probe_annotation: dict[str, Any], item_services: dict[str, str]
) -> None:
    from autogeoref.volume import is_committed

    paths = VolumePaths(root=fixtures_dir / VOL)
    page = export_volume(paths, page_services=item_services)
    assert page["type"] == "AnnotationPage"

    committed = sum(
        1
        for p in (fixtures_dir / VOL / "results").glob("p*.json")
        if is_committed(json.loads(p.read_text()))
    )
    assert len(page["items"]) == committed > 0

    # the proven p1 annotation falls out of the recorded data alone
    expected = dict(probe_annotation["items"][0])
    expected.pop("id")
    p1_item = next(i for i in page["items"] if i["target"]["source"]["id"].endswith("-0001"))
    assert without_rights(p1_item) == expected
    assert page["rights"] == RIGHTS

    # page-numeric order, not lexicographic
    tags = [i["target"]["source"]["id"].rsplit("-", 1)[1] for i in page["items"]]
    assert tags[:3] == ["0001", "0002", "0003"]


def test_export_volume_refuses_to_drop_a_committed_sheet(
    fixtures_dir: Path, item_services: dict[str, str]
) -> None:
    services = dict(item_services)
    del services["1"]
    with pytest.raises(AnnotationError, match="without a manifest entry or IIIF service"):
        export_volume(VolumePaths(root=fixtures_dir / VOL), page_services=services)


def test_export_volume_requires_a_manifest(tmp_path: Path) -> None:
    with pytest.raises(AnnotationError, match="no sheet manifest"):
        export_volume(VolumePaths(root=tmp_path / VOL), page_services={})


def test_cli_allmaps_exports_offline(
    fixtures_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from autogeoref.cli.entry import main

    item_path = WORK / f"loc-item-{VOL}.json"
    if not item_path.exists():
        pytest.skip(f"work/loc-item-{VOL}.json not present")
    out = tmp_path / "allmaps.json"
    rc = main(
        [
            "allmaps",
            VOL,
            "--work",
            str(fixtures_dir),
            "--item-json",
            str(item_path),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    page = json.loads(out.read_text())
    assert page["type"] == "AnnotationPage"
    assert len(page["items"]) > 0
    assert str(len(page["items"])) in capsys.readouterr().out


def test_cli_allmaps_errors_on_an_unplaced_volume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from autogeoref.cli.entry import main

    item_path = tmp_path / "item.json"
    item_path.write_text(json.dumps({"resources": []}))
    rc = main(["allmaps", VOL, "--work", str(tmp_path), "--item-json", str(item_path)])
    assert rc == 1
    assert "no sheet manifest" in capsys.readouterr().err
