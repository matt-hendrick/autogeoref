"""Researcher export staging vs the fixture volume.

The export tree a publish lands (``exports/<volume>/``) must hold exactly the
committed sheets — records byte-for-byte, the AnnotationPage matching the
proven ``autogeoref allmaps`` assembly — and must be deterministic: staging
the same volume twice yields byte-identical trees.
"""

import json
from pathlib import Path

import pytest

from autogeoref.exports import stage_export, volume_page_services
from autogeoref.paths import VolumePaths
from autogeoref.volume import is_committed

WORK = Path(__file__).resolve().parent.parent / "work"
VOL = "sanborn01790_024"


@pytest.fixture(scope="module")
def item_path() -> Path:
    path = WORK / f"loc-item-{VOL}.json"
    if not path.exists():
        pytest.skip(f"work/loc-item-{VOL}.json not present")
    return path


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_stage_export_fixture_volume(fixtures_dir: Path, item_path: Path, tmp_path: Path) -> None:
    paths = VolumePaths(root=fixtures_dir / VOL)
    services = volume_page_services(VOL, item_json=item_path, cache_dir=tmp_path)
    out = tmp_path / "staged"
    count = stage_export(paths, page_services=services, out_dir=out)

    committed = {
        p.name: p.read_bytes()
        for p in (fixtures_dir / VOL / "results").glob("p*.json")
        if is_committed(json.loads(p.read_text()))
    }
    assert committed, "fixture volume must have committed sheets"
    exported = {p.name: p.read_bytes() for p in (out / "gcps").iterdir()}
    # exactly the committed records, byte-for-byte — copied, never re-derived
    assert exported == committed

    page = json.loads((out / "allmaps.json").read_text())
    assert page["type"] == "AnnotationPage"
    assert len(page["items"]) == count == len(committed)


def test_stage_export_is_deterministic(fixtures_dir: Path, item_path: Path, tmp_path: Path) -> None:
    paths = VolumePaths(root=fixtures_dir / VOL)
    services = volume_page_services(VOL, item_json=item_path, cache_dir=tmp_path)
    first, second = tmp_path / "one", tmp_path / "two"
    stage_export(paths, page_services=services, out_dir=first)
    stage_export(paths, page_services=services, out_dir=second)
    assert _tree_bytes(first) == _tree_bytes(second)
