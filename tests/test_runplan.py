"""Fixture-free contracts for the resolved placement plan."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from autogeoref.config.model import VolumeConfig
from autogeoref.dag import Runner
from autogeoref.paths import VolumePaths
from autogeoref.runplan.placement import build_stages
from autogeoref.runpolicy import RunPolicy


def _context(root: Path) -> SimpleNamespace:
    paths = VolumePaths(root=root)
    paths.regions.mkdir(parents=True)
    (paths.regions / "sheet_p1.jpg").touch()
    return SimpleNamespace(
        args=SimpleNamespace(
            volume="vol",
            dry_run=True,
            no_annotate=False,
            force=False,
            max_zoom=None,
            street_index=None,
            limit=None,
            annotate_jobs=1,
            allow_failed_reads=False,
            escalate=False,
            escalate_model=None,
            verify_junctions=False,
            verified_accept=False,
        ),
        city=SimpleNamespace(
            centerline_name_property="name",
            centerline_type_property="type",
            address_block_size=100,
        ),
        vol=VolumeConfig(identifier="vol"),
        paths=paths,
        bounds=(0.0, 0.0, 1.0, 1.0),
        index=SimpleNamespace(aliases={}),
        ground_truth=None,
        gt_layers=None,
        rail_index=None,
        index_windows=None,
    )


def test_build_stages_preserves_order_enablement_and_dry_run(tmp_path: Path) -> None:
    policy = RunPolicy(
        warp=False,
        escalation_models=("model",),
        run_escalation=True,
        run_junction=False,
        run_verified=False,
        allowed_channels=frozenset(),
    )
    ctx = _context(tmp_path)
    stages = build_stages(ctx, policy)

    assert [stage.name for stage in stages] == [
        "prep",
        "annotate",
        "match",
        "escalate",
        "revoke-stale",
        "street-index",
        "rescue",
        "seam",
        "corroborate",
        "junction-verify",
        "verified-accept",
        "report",
        "warp",
        "mask",
        "mosaic",
        "tile-params",
        "tile",
    ]
    assert {stage.name for stage in stages if stage.enabled()} == {
        "prep",
        "annotate",
        "match",
        "escalate",
        "revoke-stale",
        "rescue",
        "seam",
        "corroborate",
        "report",
    }
    assert not (ctx.paths.root / "tiles-params.json").exists()
    results = Runner(ctx.paths.markers).execute(stages, dry_run=True)
    assert [(result.name, result.status) for result in results] == [
        ("prep", "skipped"),
        ("annotate", "skipped"),
        ("match", "skipped"),
        ("escalate", "skipped"),
        ("revoke-stale", "skipped"),
        ("street-index", "disabled"),
        ("rescue", "skipped"),
        ("seam", "skipped"),
        ("corroborate", "skipped"),
        ("junction-verify", "disabled"),
        ("verified-accept", "disabled"),
        ("report", "skipped"),
        ("warp", "disabled"),
        ("mask", "disabled"),
        ("mosaic", "disabled"),
        ("tile-params", "disabled"),
        ("tile", "disabled"),
    ]


def test_warp_plan_declares_back_half_freshness_inputs(tmp_path: Path) -> None:
    policy = RunPolicy(
        warp=True,
        escalation_models=(),
        run_escalation=False,
        run_junction=False,
        run_verified=False,
        allowed_channels=frozenset(),
    )
    ctx = _context(tmp_path)
    ctx.args.dry_run = False
    ctx.args.max_zoom = 16

    stages = {stage.name: stage for stage in build_stages(ctx, policy)}

    assert all(stages[name].enabled() for name in ("warp", "mask", "mosaic", "tile-params", "tile"))
    params = ctx.paths.root / "tiles-params.json"
    assert not params.exists()
    assert stages["tile-params"].outputs == [params]
    assert stages["tile"].inputs == [
        ctx.paths.root / "mosaic.tif",
        ctx.paths.warped / "warp-summary.json",
        ctx.paths.masks / "masks.geojson",
        params,
    ]
    assert stages["tile"].outputs == [ctx.paths.root / "vol.pmtiles"]


def test_escalation_stage_forwards_shared_annotate_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autogeoref.escalate as escalate
    from autogeoref.runplan import placement as runplan

    policy = RunPolicy(
        warp=False,
        escalation_models=("model",),
        run_escalation=True,
        run_junction=False,
        run_verified=False,
        allowed_channels=frozenset(),
    )
    ctx = _context(tmp_path)
    ctx.args.annotate_jobs = 4
    seen: dict[str, object] = {}

    monkeypatch.setattr(runplan.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(runplan, "resolve_constants", lambda *_args: (0.1, 0.0))
    monkeypatch.setattr(escalate, "stage_escalate", lambda *_args, **kwargs: seen.update(kwargs))

    stages = {stage.name: stage for stage in build_stages(ctx, policy)}
    stages["escalate"].run()

    assert seen["jobs"] == 4
