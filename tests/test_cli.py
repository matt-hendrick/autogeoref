"""CLI-level regression tests.

``run`` takes no ground truth at all: pins grade a finished placement and never
influence one, and the only way to keep that true across the hand-typed run, the
queue, and every script was to delete the input. The empty-marker convention it
used to carry ("the volunteer corpus was checked: never pinned", a real recorded
state — see fixtures/prod/) now lives in the scoring pass.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest

from autogeoref.cli.entry import main

ROOT = Path(__file__).resolve().parent.parent
VOL = "sanborn01790_024"


@pytest.fixture
def tiny_volume(tmp_path: Path, fixtures_dir: Path) -> Path:
    """A two-sheet copy of _024 — enough to drive every stage, fast."""
    vol_dir = tmp_path / VOL
    (vol_dir / "sheets").mkdir(parents=True)
    (vol_dir / "annotations").mkdir()
    shutil.copy2(
        fixtures_dir / VOL / "sheets" / "manifest.json",
        vol_dir / "sheets" / "manifest.json",
    )
    for ann in sorted((fixtures_dir / VOL / "annotations").glob("p*.json"))[:2]:
        shutil.copy2(ann, vol_dir / "annotations" / ann.name)
    return vol_dir


def _escalating_city(tmp_path: Path) -> Path:
    """A city whose ladder makes the (default-ON) escalate stage run for VOL."""
    cfg = tmp_path / "esc.toml"
    cl = ROOT / "fixtures" / "reference" / "street_center_lines.geojson"
    cfg.write_text(
        "[city]\n"
        'name = "Chicago, Ill."\n'
        f'centerlines = "{cl}"\n'
        f'aliases_dir = "{ROOT / "configs" / "chicago" / "aliases"}"\n'
        'escalation_models = ["claude-sonnet-5"]\n'
        f"[volumes.{VOL}]\n"
        "bounds_bbox = [-87.66, 41.87, -87.60, 41.90]\n"
        "scale_m_per_px = 0.0669\n"
        "rotation_deg = 1.20\n"
    )
    return cfg


@pytest.mark.golden
def test_unlaunchable_backend_does_not_sink_a_default_run(
    tiny_volume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing `claude` must not cost a DEFAULT run its honest report.

    Escalation is additive — a page it cannot re-read stays honestly REJECTED —
    but the stage is now default-ON, so an OSError from spawning the CLI would
    otherwise fail the stage and (stop_on_failure) abort rescue, corroborate and
    report for every machine without the CLI installed.
    """
    import autogeoref.escalate as escalate

    def unlaunchable(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError(2, "No such file or directory: 'claude'")

    monkeypatch.setattr(escalate, "stage_escalate", unlaunchable)
    cfg = _escalating_city(tmp_path)
    with caplog.at_level(logging.WARNING):
        rc = main(["run", VOL, "--city", str(cfg), "--work", str(tmp_path)])
    assert rc == 0, "an unlaunchable escalation backend must not fail the run"
    assert "escalation skipped" in caplog.text
    assert (tiny_volume / "report.json").exists(), "the honest report still lands"


@pytest.mark.golden
def test_unlaunchable_backend_still_fails_an_explicit_escalate(
    tiny_volume: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--escalate asked for the spend: tell that caller loudly instead of skipping."""
    import autogeoref.escalate as escalate

    def unlaunchable(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError(2, "No such file or directory: 'claude'")

    monkeypatch.setattr(escalate, "stage_escalate", unlaunchable)
    cfg = _escalating_city(tmp_path)
    rc = main(["run", VOL, "--city", str(cfg), "--work", str(tmp_path), "--escalate"])
    assert rc == 1


@pytest.mark.golden
def test_dry_run_with_street_index_spends_nothing(
    tiny_volume: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G2 finding 5: reading the street index spends model budget, so it
    must live behind the runner's stage discipline — a --dry-run must never
    reach it.

    Test-local pinned config: a dry run cannot derive bounds (derivation
    spends), so the volume must declare a bbox — and the exact box is
    irrelevant here."""
    import autogeoref.street_index as street_index

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("read_index must not run under --dry-run")

    monkeypatch.setattr(street_index, "read_index", boom)
    fake_index_img = tmp_path / "keymap.jpg"
    fake_index_img.write_bytes(b"\xff\xd8fake")
    rc = main(
        [
            "run",
            VOL,
            "--city",
            str(_pinned_city(tmp_path)),
            "--work",
            str(tmp_path),
            "--street-index",
            str(fake_index_img),
            "--dry-run",
        ]
    )
    assert rc == 0


def test_dry_run_preamble_prints_the_unified_budget_render(
    tiny_volume: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The run preamble's estimate line carries `budget.render()`'s output
    verbatim — the CLI half of the one-estimator contract (`budget` module
    docstring; the console half is pinned in test_console_cli.py). Bounds come from
    the declared bounds_bbox, keeping this independent of the bounds-resolution
    failures above."""
    from autogeoref.budget import estimate_spend

    cfg = _escalating_city(tmp_path)
    rc = main(["run", VOL, "--city", str(cfg), "--work", str(tmp_path), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    manifest = json.loads((tiny_volume / "sheets" / "manifest.json").read_text())
    sheets = len([p for p in manifest if not p.startswith("_")])
    # the two copied p*.json annotations are legacy reads (reused, not planned)
    # and no page has a small on disk, so the plan reads nothing: the range is
    # 0 up to the escalation ceiling for the city's 1-tier ladder
    expected = estimate_spend(sheets=sheets, unread=0, escalation_tiers=1)
    assert expected.low == 0 and expected.ceiling > 0
    assert expected.render() in out


def test_run_flags_parse() -> None:
    from autogeoref.cli.parser import build_parser

    args = build_parser().parse_args(
        ["run", "volX", "--city", "c.toml", "--verified-accept", "--verify-junctions"]
    )
    assert args.verified_accept and args.verify_junctions
    args2 = build_parser().parse_args(["run", "volX", "--city", "c.toml"])
    assert not args2.verified_accept and not args2.verify_junctions
    # the addresses channel lost its flag with its producer
    #: it is config-only now
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "volX", "--city", "c.toml", "--consensus-annotate"])


def test_annotate_jobs_help_covers_annotation_and_escalation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from autogeoref.cli.parser import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--help"])
    assert "pages annotated and escalated concurrently" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "vol", "--city", "city.toml", "--limit=-1"],
        ["run", "vol", "--city", "city.toml", "--annotate-jobs", "0"],
        ["run", "vol", "--city", "city.toml", "--max-zoom", "31"],
        ["run", "..", "--city", "city.toml"],
        ["run", "../other", "--city", "city.toml"],
        ["queue", "--place-lanes", "0"],
        ["queue", "--add", "../other"],
        ["prep", "../other"],
        ["report", ".."],
        ["queue", "--serve-lanes", "-1"],
        ["queue", "--interval", "nan"],
        ["queue", "--port", "65536"],
        ["review", "--city", "city.toml", "--port", "0"],
    ],
)
def test_parser_rejects_invalid_operational_limits(argv: list[str]) -> None:
    from autogeoref.cli.parser import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_missing_street_index_fails_before_annotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit missing input cannot reach the annotation spender."""
    import autogeoref.annotate_volume as annotate_volume

    cfg = tmp_path / "city.toml"
    # fixture-free on purpose (runs in CI): the contract is validation order,
    # so an empty centerlines file is enough — it only has to exist
    centerlines = tmp_path / "cl.geojson"
    centerlines.write_text('{"type": "FeatureCollection", "features": []}')
    (tmp_path / "aliases").mkdir()
    cfg.write_text(
        "[city]\n"
        'name = "X"\n'
        f'aliases_dir = "{tmp_path / "aliases"}"\n'
        f'centerlines = "{centerlines}"\n'
        "[volumes.vol]\n"
        "bounds_bbox = [-87.7, 41.8, -87.6, 41.9]\n"
        "scale_m_per_px = 0.1\n"
        "rotation_deg = 0\n"
    )
    calls: list[Path] = []

    def annotation_tripwire(*args: object, **kwargs: object) -> object:
        calls.append(Path("called"))
        raise AssertionError("annotation must not run")

    monkeypatch.setattr(annotate_volume, "annotate_volume", annotation_tripwire)
    with pytest.raises(SystemExit, match="--street-index must be an existing file"):
        main(
            [
                "run",
                "vol",
                "--city",
                str(cfg),
                "--work",
                str(tmp_path / "work"),
                "--street-index",
                str(tmp_path / "missing-index.jpg"),
            ]
        )
    assert calls == []


@pytest.mark.parametrize(
    ("extra_config", "run_args", "error"),
    [
        ("", ["--escalate-model", "codx:gpt-5.6-terra"], "--escalate-model"),
        (
            'evidence_channels = ["addresses"]\nrenumbering_table = "bad-renumbering.json"\n',
            [],
            "configured renumbering_table is invalid",
        ),
    ],
)
def test_invalid_pre_spend_configuration_cannot_reach_annotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_config: str,
    run_args: list[str],
    error: str,
) -> None:
    """Model and conversion failures must not be discovered after a paid read."""
    import autogeoref.annotate_volume as annotate_volume

    cfg = tmp_path / "city.toml"
    # fixture-free on purpose (runs in CI): pre-spend refusal never reads the
    # centerlines, so an existing empty file is enough
    centerlines = tmp_path / "cl.geojson"
    centerlines.write_text('{"type": "FeatureCollection", "features": []}')
    (tmp_path / "aliases").mkdir()
    (tmp_path / "bad-renumbering.json").write_text("not json")
    cfg.write_text(
        "[city]\n"
        'name = "X"\n'
        f'aliases_dir = "{tmp_path / "aliases"}"\n'
        f'centerlines = "{centerlines}"\n'
        f"{extra_config}"
        "[volumes.vol]\n"
        "bounds_bbox = [-87.7, 41.8, -87.6, 41.9]\n"
        "scale_m_per_px = 0.1\n"
        "rotation_deg = 0\n"
        "addresses_modern = false\n"
    )

    def annotation_tripwire(*args: object, **kwargs: object) -> object:
        raise AssertionError("annotation must not run")

    monkeypatch.setattr(annotate_volume, "annotate_volume", annotation_tripwire)
    with pytest.raises(SystemExit, match=error):
        main(["run", "vol", "--city", str(cfg), "--work", str(tmp_path / "work"), *run_args])


@pytest.mark.golden
def test_run_osm_default_city_autofetches_once(
    tiny_volume: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A city TOML without `centerlines` runs on the OSM default: the
    per-city cache is fetched once for the volume's bbox (network
    stubbed here — no network in tests, ever) and a covered re-run touches the wire zero times."""
    import autogeoref.osm as osm

    sample = json.loads((Path(__file__).parent / "data" / "osm_overpass_sample.json").read_text())
    calls: list[str] = []

    def fake_fetch(query: str, timeout_s: float = 0.0) -> dict[str, object]:
        calls.append(query)
        return sample

    monkeypatch.setattr(osm, "fetch_overpass", fake_fetch)
    cfg = tmp_path / "vireo.toml"
    cfg.write_text(
        "[city]\n"
        'name = "Vireo City"\n'
        'aliases_dir = "aliases"\n'
        f"[volumes.{VOL}]\n"
        "bounds_bbox = [-87.66, 41.87, -87.60, 41.90]\n"
        # pinned constants: the sample-data bbox can't cover these sheets, so
        # pass-1 derivation would honestly find nothing and rescue would
        # refuse to run — the wiring under test is fetch/cache, not matching
        "scale_m_per_px = 0.0669\n"
        "rotation_deg = 1.20\n"
    )
    argv = ["run", VOL, "--city", str(cfg), "--work", str(tmp_path)]
    assert main(argv) == 0
    assert len(calls) == 1, "first run fetches the volume bbox exactly once"
    cache = tmp_path / "cache" / "osm-centerlines-vireo-city.geojson"
    fc = json.loads(cache.read_text())
    assert fc["fetched_bboxes"] and fc["features"]
    assert (tiny_volume / "report.json").exists()

    def boom(query: str, timeout_s: float = 0.0) -> dict[str, object]:
        raise AssertionError("covered re-run must not fetch")

    monkeypatch.setattr(osm, "fetch_overpass", boom)
    assert main(argv) == 0  # cache covers the bounds: zero network


@pytest.mark.golden
def test_dry_run_osm_default_city_fetches_nothing(
    tiny_volume: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run spends NOTHING — no model budget and no network fetch: an
    OSM-default city with a cold cache must plan without touching the wire."""
    import autogeoref.osm as osm

    def boom(query: str, timeout_s: float = 0.0) -> dict[str, object]:
        raise AssertionError("--dry-run must never fetch")

    monkeypatch.setattr(osm, "fetch_overpass", boom)
    cfg = tmp_path / "vireo.toml"
    cfg.write_text(
        "[city]\n"
        'name = "Vireo City"\n'
        'aliases_dir = "aliases"\n'
        f"[volumes.{VOL}]\n"
        "bounds_bbox = [-87.66, 41.87, -87.60, 41.90]\n"
        "scale_m_per_px = 0.0669\n"
        "rotation_deg = 1.20\n"
    )
    rc = main(["run", VOL, "--city", str(cfg), "--work", str(tmp_path), "--dry-run"])
    assert rc == 0
    assert not (tmp_path / "cache" / "osm-centerlines-vireo-city.geojson").exists()


@pytest.mark.golden
def test_run_takes_no_ground_truth_at_all(tiny_volume: Path, tmp_path: Path) -> None:
    """The flag is GONE from ``run``, and the run places from its declared bounds.

    Not "the flag is ignored": argparse must refuse it, so an operator or a script
    that still passes it is told rather than quietly placing against pins. The
    test-local pinned config supplies the bounds (the two-sheet fixture is
    deliberately too small for the bootstrap to derive a box instead).
    """
    city = str(_pinned_city(tmp_path))
    gt = tmp_path / "empty-gt.json"
    gt.write_text("")
    with pytest.raises(SystemExit):
        main(["run", VOL, "--city", city, "--work", str(tmp_path), "--ground-truth", str(gt)])

    rc = main(["run", VOL, "--city", city, "--work", str(tmp_path)])
    assert rc == 0
    seam = json.loads((tiny_volume / "seam_deltas.json").read_text())
    assert seam["gate"] == "N/A"  # no run gates a seam solve on human pins
    assert (tiny_volume / "report.json").exists()


# ---------------------------------------------------------------------------
# volume ownership: one mutating owner per work volume
# ---------------------------------------------------------------------------


def _pinned_city(tmp_path: Path) -> Path:
    """A minimal city TOML with pinned constants and no model-spending extras."""
    cfg = tmp_path / "pinned.toml"
    cl = ROOT / "fixtures" / "reference" / "street_center_lines.geojson"
    cfg.write_text(
        "[city]\n"
        'name = "Chicago, Ill."\n'
        f'centerlines = "{cl}"\n'
        f'aliases_dir = "{ROOT / "configs" / "chicago" / "aliases"}"\n'
        f"[volumes.{VOL}]\n"
        "bounds_bbox = [-87.66, 41.87, -87.60, 41.90]\n"
        "scale_m_per_px = 0.0669\n"
        "rotation_deg = 1.20\n"
    )
    return cfg


_LOCK_RACE_DRIVER = """
import os, sys, time, types
from pathlib import Path

sync = Path(sys.argv[1])
volume, city, work = sys.argv[2], sys.argv[3], sys.argv[4]

import autogeoref.annotate_volume as annotate_volume


def fake_annotate(paths, vol, **kwargs):
    # the mocked billable read: record that this process reached it, then hold
    # the run open so the sibling process must meet a HELD lock, not a released one
    (sync / f"entered-{os.getpid()}").touch()
    deadline = time.monotonic() + 120.0
    while not (sync / "release").exists():
        if time.monotonic() > deadline:
            raise RuntimeError("never released")
        time.sleep(0.05)
    return types.SimpleNamespace(unread=[])


annotate_volume.annotate_volume = fake_annotate

from autogeoref.cli.entry import main

(sync / f"ready-{os.getpid()}").touch()
deadline = time.monotonic() + 120.0
while not (sync / "go").exists():
    if time.monotonic() > deadline:
        raise RuntimeError("no go signal")
    time.sleep(0.01)

sys.exit(main(["run", volume, "--city", city, "--work", work]))
"""


def _await(condition, deadline_s: float = 120.0, what: str = "condition") -> None:
    import time

    deadline = time.monotonic() + deadline_s
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.05)


@pytest.mark.golden
def test_barrier_started_concurrent_runs_of_one_volume_annotate_once(
    tiny_volume: Path, tmp_path: Path
) -> None:
    """Two simultaneous `autogeoref run` processes on ONE volume: exactly one
    reaches the (mocked) annotation call; the other is refused, actionably,
    before any model work."""
    import subprocess
    import sys as _sys

    cfg = _pinned_city(tmp_path)
    sync = tmp_path / "sync"
    sync.mkdir()
    driver = tmp_path / "driver.py"
    driver.write_text(_LOCK_RACE_DRIVER)
    procs = [
        subprocess.Popen(
            [_sys.executable, str(driver), str(sync), VOL, str(cfg), str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    try:
        _await(lambda: len(list(sync.glob("ready-*"))) == 2, what="both children ready")
        (sync / "go").touch()  # the barrier drops: both race for the volume lock

        def one_lost() -> bool:
            return any(p.poll() is not None and p.returncode != 0 for p in procs)

        _await(one_lost, what="one child to be refused")
        (sync / "release").touch()  # let the winner finish its run
        outs = [p.communicate(timeout=120) for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    entered = list(sync.glob("entered-*"))
    assert len(entered) == 1, "exactly ONE process may reach the billable annotation call"
    codes = sorted(p.returncode for p in procs)
    assert codes[0] == 0 and codes[1] != 0, f"one winner, one refusal (got {codes})"
    loser_err = next(err for p, (_out, err) in zip(procs, outs, strict=True) if p.returncode != 0)
    assert "is busy" in loser_err and "volume.lock" in loser_err


@pytest.mark.golden
def test_run_is_refused_before_annotation_while_the_volume_is_owned(
    tiny_volume: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autogeoref.annotate_volume as annotate_volume
    from autogeoref.paths import VolumePaths, volume_lock

    def annotation_tripwire(*args: object, **kwargs: object) -> object:
        raise AssertionError("annotation must not run while another process owns the volume")

    monkeypatch.setattr(annotate_volume, "annotate_volume", annotation_tripwire)
    cfg = _pinned_city(tmp_path)
    with (
        volume_lock(VolumePaths(root=tmp_path / VOL), operation="run"),
        pytest.raises(SystemExit, match="is busy"),
    ):
        main(["run", VOL, "--city", str(cfg), "--work", str(tmp_path)])


@pytest.mark.golden
def test_warp_only_is_refused_while_the_volume_is_owned(tiny_volume: Path, tmp_path: Path) -> None:
    from autogeoref.paths import VolumePaths, volume_lock

    cfg = _pinned_city(tmp_path)
    with (
        volume_lock(VolumePaths(root=tmp_path / VOL), operation="run"),
        pytest.raises(SystemExit, match="is busy"),
    ):
        main(["run", VOL, "--city", str(cfg), "--work", str(tmp_path), "--warp-only"])


@pytest.mark.golden
def test_prep_is_refused_while_the_volume_is_owned(tiny_volume: Path, tmp_path: Path) -> None:
    from autogeoref.paths import VolumePaths, volume_lock

    (tiny_volume / "regions").mkdir()
    with volume_lock(VolumePaths(root=tmp_path / VOL), operation="run"):
        rc = main(["prep", VOL, "--work", str(tmp_path)])
    assert rc == 1


@pytest.mark.golden
def test_dry_run_does_not_take_or_respect_the_volume_lock(
    tiny_volume: Path, tmp_path: Path
) -> None:
    """A dry run is side-effect-free, so it must run EVEN while a real
    operation owns the volume — it is how you ask what a run would cost."""
    from autogeoref.paths import VolumePaths, volume_lock

    cfg = _pinned_city(tmp_path)
    vp = VolumePaths(root=tmp_path / VOL)
    with volume_lock(vp, operation="run"):
        rc = main(["run", VOL, "--city", str(cfg), "--work", str(tmp_path), "--dry-run"])
        assert rc == 0
        held = json.loads(vp.lock.read_text())
        assert held["operation"] == "run", "the dry run neither took nor disturbed the lock"


def test_a_stale_config_catalog_warns_and_is_omitted_for_display_commands(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """viewer-manifest and dashboard default --loc-catalog from the city TOML
    (the same fallback era and publish perform), but the config fallback is
    display context: a `loc_catalog` naming a missing or unparseable file warns
    and is omitted rather than taking a read-only command down. The explicit
    flag keeps failing loudly — the operator named that file, and silence
    would hide their typo (the console pins the same rule)."""
    import logging

    from autogeoref.cli_context import display_catalog as _display_catalog
    from autogeoref.config.load import load_city_config

    cfg = tmp_path / "city.toml"
    cfg.write_text(
        '[city]\nname = "Chicago, Ill."\ncenterlines = "cl.geojson"\n'
        'aliases_dir = "aliases"\nloc_catalog = "cat.json"\n'
    )
    (tmp_path / "cl.geojson").write_text('{"type":"FeatureCollection","features":[]}')
    (tmp_path / "aliases").mkdir()
    city = load_city_config(cfg)

    with caplog.at_level(logging.WARNING):
        assert _display_catalog(None, city) is None
    assert "unusable" in caplog.text

    # a catalog that exists but does not parse is the same story: warn, omit
    (tmp_path / "cat.json").write_text("{ not json")
    assert _display_catalog(None, city) is None

    # a usable declared catalog resolves (relative to the TOML, not the cwd)
    (tmp_path / "cat.json").write_text("[]")
    assert _display_catalog(None, city) == tmp_path / "cat.json"

    # the explicit flag passes through untouched; its failures stay loud downstream
    missing = tmp_path / "missing.json"
    assert _display_catalog(missing, city) == missing
