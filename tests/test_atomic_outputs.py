"""Crash-safety contracts for persistent pipeline publication boundaries."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from autogeoref.annotate_volume import _write_json
from autogeoref.bake.mosaic import stage_mosaic
from autogeoref.config.model import VolumeConfig
from autogeoref.paths import VolumePaths
from autogeoref.report import build_report, report_json, report_markdown
from autogeoref.review.sidecars import ReviewSidecar, save_sidecar
from autogeoref.stages import match as match_stage
from autogeoref.stages.match import stage_match
from autogeoref.stages.report import stage_report
from autogeoref.stages.seam import stage_seam
from autogeoref.viewer.deploy import (
    ICON_FILES,
    PAGE_FILES,
    PLATFORM_FILES,
    build_deploy_bundle,
)
from autogeoref.viewer.manifest import write_manifest
from conftest import antedate
from viewer_support import page_stub


def _fail_replace(self: Path, target: Path) -> Path:
    raise OSError("replacement failed")


def test_match_constants_replacement_preserves_previous_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = VolumePaths(root=tmp_path / "volume")
    old = json.dumps({"scale_m_per_px": 1.0, "rotation_deg": 2.0}, indent=2)
    paths.constants.parent.mkdir()
    paths.constants.write_text(old)
    monkeypatch.setattr(
        match_stage, "load_sheet_inputs", lambda _paths: [SimpleNamespace(page="1")]
    )
    monkeypatch.setattr(
        match_stage,
        "derive_constraints",
        lambda *_args, **_kwargs: SimpleNamespace(scale_median=0.5, rotation_median=3.0),
    )
    monkeypatch.setattr(match_stage, "match_sheet", lambda *_args, **_kwargs: {"status": "OK"})
    monkeypatch.setattr(Path, "replace", _fail_replace)

    with pytest.raises(OSError, match="replacement failed"):
        stage_match(paths, SimpleNamespace(aliases={}), VolumeConfig(identifier="volume"))  # type: ignore[arg-type]

    assert paths.constants.read_text() == old
    assert json.loads(paths.constants.read_text()) == json.loads(old)
    assert not list(paths.root.glob(".volume-constants.json.*.tmp"))


def test_seam_record_replacement_preserves_previous_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = VolumePaths(root=tmp_path / "volume")
    paths.results.mkdir(parents=True)
    old = json.dumps({"ties": 99, "gate": "PASSED"}, indent=2)
    paths.seam_deltas.write_text(old)
    monkeypatch.setattr(Path, "replace", _fail_replace)

    with pytest.raises(OSError, match="replacement failed"):
        stage_seam(paths)

    assert paths.seam_deltas.read_text() == old
    assert json.loads(paths.seam_deltas.read_text()) == json.loads(old)
    assert not list(paths.root.glob(".seam_deltas.json.*.tmp"))


def test_annotation_and_review_sidecar_replacement_preserve_previous_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    annotation = tmp_path / "annotations" / "p1.json"
    annotation.parent.mkdir()
    annotation.write_text('{"old": true}')
    monkeypatch.setattr(Path, "replace", _fail_replace)

    with pytest.raises(OSError, match="replacement failed"):
        _write_json(annotation, {"new": True})

    assert json.loads(annotation.read_text()) == {"old": True}
    assert not list(annotation.parent.glob(".p1.json.*.tmp"))

    first_write = annotation.parent / "p2.failed.json"
    with pytest.raises(OSError, match="replacement failed"):
        _write_json(first_write, {"error": "retry me"})

    assert not first_write.exists()
    assert not list(annotation.parent.glob(".p2.failed.json.*.tmp"))

    paths = VolumePaths(root=tmp_path / "volume")
    review = paths.root / "review" / "p1.json"
    review.parent.mkdir(parents=True)
    review.write_text('{"old": true}')
    sidecar = ReviewSidecar("volume", "1", "sha", "accept")

    with pytest.raises(OSError, match="replacement failed"):
        save_sidecar(paths, sidecar)

    assert json.loads(review.read_text()) == {"old": True}
    assert not list(review.parent.glob(".p1.json.*.tmp"))


def test_mosaic_failure_preserves_previous_output_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autogeoref.tiles as tiles

    paths = VolumePaths(root=tmp_path / "volume")
    paths.warped.mkdir(parents=True)
    slug = "volume_p1"
    cog = paths.warped / f"{slug}.tif"
    cog.write_text("cog")
    (paths.warped / "warp-summary.json").write_text(json.dumps({"warped": [slug]}))

    def cutline_vrt(_cog: Path, _cutline: Path | None, out: Path, **_kwargs: Any) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("<VRTDataset />")

    def successful_mosaic(_parts: list[Path], out: Path, **_kwargs: Any) -> None:
        out.write_text("complete mosaic")

    monkeypatch.setattr(tiles, "cutline_vrt", cutline_vrt)
    monkeypatch.setattr(tiles, "mosaic_gtiff", successful_mosaic)
    mosaic = stage_mosaic(paths)
    assert mosaic.read_text() == "complete mosaic"

    # antedate the composite rather than post-date the COG by a second: the
    # wall clock can step back further than that, leaving the mosaic fresh
    antedate(mosaic)

    def failing_mosaic(_parts: list[Path], out: Path, **_kwargs: Any) -> None:
        out.write_text("partial mosaic")
        raise RuntimeError("GDAL failed")

    monkeypatch.setattr(tiles, "mosaic_gtiff", failing_mosaic)
    with pytest.raises(RuntimeError, match="GDAL failed"):
        stage_mosaic(paths)

    assert mosaic.read_text() == "complete mosaic"
    assert not list(paths.root.glob(".mosaic.tif.*.tmp"))
    assert not list((paths.root / "mosaic-parts").glob(".*.tmp"))


def test_report_representations_preserve_bytes_and_prior_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = VolumePaths(root=tmp_path / "volume")
    paths.results.mkdir(parents=True)
    report = build_report("volume", {}, None)

    stage_report(paths, "volume")
    assert (paths.root / "report.json").read_text() == report_json(report)
    assert (paths.root / "report.md").read_text() == report_markdown(report)

    markdown = paths.root / "report.md"
    old_markdown = "# Previous complete report\n"
    markdown.write_text(old_markdown)
    original_replace = Path.replace

    def fail_markdown(self: Path, target: Path) -> Path:
        if target == markdown:
            raise OSError("replacement failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_markdown)
    with pytest.raises(OSError, match="replacement failed"):
        stage_report(paths, "volume")

    assert markdown.read_text() == old_markdown
    assert not list(paths.root.glob(".report.md.*.tmp"))


def test_annotation_sidecar_producers_preserve_previous_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Escalation is the LAST annotation-sidecar producer.

    It was one of two until the consensus-annotate stage was cut; the failure-
    marker half of this test went with it, because nothing else writes `p<N>.v2.*.failed.json`.
    """
    from autogeoref.escalate import stage_escalate

    paths = VolumePaths(root=tmp_path / "volume")
    paths.results.mkdir(parents=True)
    paths.annotations.mkdir()
    paths.sheets.mkdir()
    (paths.results / "p1.json").write_text(json.dumps({"page": "1", "status": "REJECTED"}))
    paths.manifest.write_text(json.dumps({"p1": {"file": "p1_small.jpg"}}))
    image = paths.sheets / "p1_small.jpg"
    image.write_bytes(b"image")
    escalated = paths.annotations / "p1.escalated.claude-sonnet-5.json"

    import autogeoref.junction_snap as junction_snap

    monkeypatch.setattr(
        junction_snap, "extract_junctions", lambda _image: SimpleNamespace(n_junctions=4)
    )
    monkeypatch.setattr(Path, "replace", _fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        stage_escalate(
            paths,
            SimpleNamespace(aliases={}),  # type: ignore[arg-type]
            SimpleNamespace(),  # type: ignore[arg-type]
            "claude-sonnet-5",
            annotate_fn=lambda _image, _model: {"streets": []},
        )

    assert not escalated.exists()
    assert not list(paths.annotations.glob(".p1.escalated.claude-sonnet-5.json.*.tmp"))
    assert not list(paths.annotations.glob("*.v2.*")), (
        "nothing writes a v2 sidecar since the consensus producer was cut"
    )


def test_warp_fingerprint_lands_only_after_the_cog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autogeoref.warp as warp

    image = tmp_path / "sheet.jpg"
    image.write_bytes(b"image")
    out_dir = tmp_path / "warped"
    slug = "sheet"

    def attach_gcps(_image: Path, _gcps: Any, out: Path, **_kwargs: Any) -> Path:
        out.write_text("vrt")
        return out

    def run_gdal(command: list[str], **_kwargs: Any) -> None:
        Path(command[-1]).write_text("gdal output")

    seen_cog: list[bool] = []
    from autogeoref.paths import atomic_write_text

    def record_fingerprint(path: Path, text: str) -> Path:
        seen_cog.append((out_dir / f"{slug}.tif").is_file())
        return atomic_write_text(path, text)

    monkeypatch.setattr(warp, "attach_gcps_vrt", attach_gcps)
    monkeypatch.setattr(warp, "_run_gdal", run_gdal)
    monkeypatch.setattr(warp, "extent_of", lambda *_args, **_kwargs: (0.0, 0.0, 1.0, 1.0))
    monkeypatch.setattr(warp, "atomic_write_text", record_fingerprint)

    result = warp.warp_sheet(
        image,
        [(0.0, 0.0, -87.6, 41.8)],
        out_dir,
        slug=slug,
        timeout_s=1,
    )

    assert result.cog_path.is_file()
    assert seen_cog == [True]
    assert (out_dir / f"{slug}.tif.gcps.json").read_text() == warp._gcps_fingerprint(
        [(0.0, 0.0, -87.6, 41.8)]
    )


def test_manifest_replacement_preserves_previous_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"old": true}')
    monkeypatch.setattr(Path, "replace", _fail_replace)

    with pytest.raises(OSError, match="replacement failed"):
        write_manifest({"new": True}, manifest)

    assert json.loads(manifest.read_text()) == {"old": True}
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


def test_deploy_manifest_replacement_preserves_previous_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer = tmp_path / "viewer"
    viewer.mkdir()
    for name in (*PAGE_FILES, *ICON_FILES, *PLATFORM_FILES):
        (viewer / name).write_text(page_stub(name))
    (viewer / "testville").mkdir()
    (viewer / "testville" / "manifest.json").write_text(
        json.dumps({"volumes": [{"id": "v1", "era": "1900", "pmtiles": "v1.pmtiles"}]})
    )
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    manifest = deploy / "manifest.json"
    manifest.write_text('{"old": true}')
    monkeypatch.setattr(Path, "replace", _fail_replace)

    with pytest.raises(OSError, match="replacement failed"):
        build_deploy_bundle(viewer, deploy, "https://tiles.example.com", city="testville")

    assert json.loads(manifest.read_text()) == {"old": True}
    assert not list(deploy.glob(".manifest.json.*.tmp"))
