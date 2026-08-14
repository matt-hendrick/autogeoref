"""`evidence_channels`: the config key that makes the evidence channels a DEFAULT.

The advertised `autogeoref run` used to skip two of the three channels that turn a
flagged sheet into an honest placement — they hid behind `--verify-junctions` /
`--verified-accept`, and a half-configuration silently produced a volume with no
addresses channel and a clean-looking funnel. It went off for real on `_017`
These tests pin the replacement, which copies escalation's shape:
declared in the city TOML => on by default; `[]` on a volume cancels; `--no-verify`
skips one run; absent => the flags still work exactly as before.

The addresses channel has no producer STAGE, so declaring it changes what
verified-accept may HEAR, never what the DAG spends. That is why the stage list
below has two entries and the era demand below is keyed on the channel, not a stage.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import shutil
from pathlib import Path

import pytest

from autogeoref.cli.entry import main
from autogeoref.config.load import load_city_config
from autogeoref.config.model import ConfigError

ROOT = Path(__file__).resolve().parent.parent
VOL = "sanborn01790_024"
CHANNEL_STAGES = ("junction-verify", "verified-accept")


# --------------------------------------------------------------------------
# config resolution
# --------------------------------------------------------------------------


def test_chicago_declares_both_channels() -> None:
    """The shipped city config is the whole point of the item."""
    city = load_city_config(ROOT / "configs" / "chicago" / "chicago.toml")
    assert city.evidence_channels == ("junction", "addresses")
    # inheritance, not repetition: no volume section names the key
    assert city.volume(VOL).evidence_channels == ("junction", "addresses")
    assert city.volume("a-volume-with-no-section").evidence_channels == (
        "junction",
        "addresses",
    )


def test_volume_empty_list_cancels_the_city_channels(tmp_path: Path) -> None:
    """Presence-based, like `escalation_models = []`: an empty list must not fall
    through to the city list, or a volume could never turn the (now default-ON)
    channels off."""
    cfg = tmp_path / "city.toml"
    cfg.write_text(
        '[city]\nname = "X"\naliases_dir = "aliases"\n'
        'evidence_channels = ["junction", "addresses"]\n'
        "[volumes.volA]\nevidence_channels = []\n"
        '[volumes.volB]\nevidence_channels = ["addresses"]\n'
        "[volumes.volC]\n"
    )
    city = load_city_config(cfg)
    assert city.volume("volA").evidence_channels == ()
    assert city.volume("volB").evidence_channels == ("addresses",)
    assert city.volume("volC").evidence_channels == ("junction", "addresses")


def test_unknown_channel_name_is_an_error_not_an_off_switch(tmp_path: Path) -> None:
    """A typo must never read as "off": that is how a config slip becomes a
    quietly degraded run — the volume finishes and reports a tidy funnel."""
    cfg = tmp_path / "city.toml"
    cfg.write_text(
        '[city]\nname = "X"\naliases_dir = "aliases"\nevidence_channels = ["junctions"]\n'
    )
    with pytest.raises(ConfigError, match="unknown evidence_channels"):
        load_city_config(cfg)

    vol_cfg = tmp_path / "city2.toml"
    vol_cfg.write_text(
        '[city]\nname = "X"\naliases_dir = "aliases"\n'
        '[volumes.volA]\nevidence_channels = ["corroboration"]\n'
    )
    with pytest.raises(ConfigError, match="unknown evidence_channels"):
        load_city_config(vol_cfg)


def test_channels_must_be_a_list_of_strings(tmp_path: Path) -> None:
    cfg = tmp_path / "city.toml"
    cfg.write_text('[city]\nname = "X"\naliases_dir = "aliases"\nevidence_channels = "junction"\n')
    with pytest.raises(ConfigError, match="must be a list"):
        load_city_config(cfg)


def test_duplicate_channel_names_collapse(tmp_path: Path) -> None:
    cfg = tmp_path / "city.toml"
    cfg.write_text(
        '[city]\nname = "X"\naliases_dir = "aliases"\n'
        'evidence_channels = ["junction", "junction"]\n'
    )
    assert load_city_config(cfg).evidence_channels == ("junction",)


# --------------------------------------------------------------------------
# CLI wiring — these drive the real runner, so they need the fixture tree
# --------------------------------------------------------------------------


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


def _city(tmp_path: Path, channels: str | None, ladder: str | None = None) -> Path:
    """A minimal city; `channels` is the raw TOML value, or None to omit the key.

    No escalation ladder unless a test asks for one: this file is about the channels,
    and an unrelated default-ON stage would only add noise — except in the [cv] tests,
    where escalation's own cv2 dependency is precisely the point.
    """
    cfg = tmp_path / "city.toml"
    cl = ROOT / "fixtures" / "reference" / "street_center_lines.geojson"
    cfg.write_text(
        "[city]\n"
        'name = "Chicago, Ill."\n'
        f'centerlines = "{cl}"\n'
        f'aliases_dir = "{ROOT / "configs" / "chicago" / "aliases"}"\n'
        + (f"evidence_channels = {channels}\n" if channels is not None else "")
        + (f"escalation_models = {ladder}\n" if ladder is not None else "")
        + f"[volumes.{VOL}]\n"
        "bounds_bbox = [-87.66, 41.87, -87.60, 41.90]\n"
        "scale_m_per_px = 0.0669\n"
        "rotation_deg = 1.20\n"
    )
    return cfg


def _stage_status(vol_dir: Path, stage: str) -> str:
    marker = vol_dir / "markers" / f"{stage}.marker.json"
    if not marker.exists():
        return "missing"
    return str(json.loads(marker.read_text())["status"])


def _stage_names(volume: Path) -> set[str]:
    """Every stage the DAG recorded a marker for, under whatever name."""
    return {p.name.removesuffix(".marker.json") for p in (volume / "markers").glob("*.marker.json")}


@pytest.mark.golden
def test_declared_channels_run_without_any_flag(tiny_volume: Path, tmp_path: Path) -> None:
    """The advertised command — no flags — must run both channel stages."""
    cfg = _city(tmp_path, '["junction", "addresses"]')
    assert main(["run", VOL, "--city", str(cfg), "--work", str(tmp_path)]) == 0
    for stage in CHANNEL_STAGES:
        assert _stage_status(tiny_volume, stage) == "ok", f"{stage} did not run"


@pytest.mark.golden
def test_absent_key_leaves_the_channels_opt_in(tiny_volume: Path, tmp_path: Path) -> None:
    """A city TOML that never mentions channels behaves exactly as before —
    which is what keeps every synthetic test config and every other city inert."""
    cfg = _city(tmp_path, None)
    assert main(["run", VOL, "--city", str(cfg), "--work", str(tmp_path)]) == 0
    for stage in CHANNEL_STAGES:
        assert _stage_status(tiny_volume, stage) == "disabled", f"{stage} ran unasked"


@pytest.mark.golden
def test_one_declared_channel_enables_only_its_stage(tiny_volume: Path, tmp_path: Path) -> None:
    """Declaring "addresses" alone must not spend a junction verifier — but
    verified-accept still runs: it is the consumer of whatever channels exist
    (corroboration always votes).

    The addresses channel adds NO stage of its own. It is heard by verified-accept
    off caches already on disk, so "addresses" is the one channel whose declaration
    is purely a permission, never a spend.
    """
    cfg = _city(tmp_path, '["addresses"]')
    assert main(["run", VOL, "--city", str(cfg), "--work", str(tmp_path)]) == 0
    assert _stage_status(tiny_volume, "junction-verify") == "disabled"
    assert _stage_status(tiny_volume, "verified-accept") == "ok"
    assert _stage_names(tiny_volume).isdisjoint({"consensus-annotate"}), (
        "the consensus producer is cut; no stage may reappear under any name"
    )


@pytest.mark.golden
def test_volume_cancel_disables_the_channels(tiny_volume: Path, tmp_path: Path) -> None:
    cfg = tmp_path / "city.toml"
    cl = ROOT / "fixtures" / "reference" / "street_center_lines.geojson"
    cfg.write_text(
        "[city]\n"
        'name = "Chicago, Ill."\n'
        f'centerlines = "{cl}"\n'
        f'aliases_dir = "{ROOT / "configs" / "chicago" / "aliases"}"\n'
        'evidence_channels = ["junction", "addresses"]\n'
        f"[volumes.{VOL}]\n"
        "evidence_channels = []\n"
        "bounds_bbox = [-87.66, 41.87, -87.60, 41.90]\n"
        "scale_m_per_px = 0.0669\n"
        "rotation_deg = 1.20\n"
    )
    assert main(["run", VOL, "--city", str(cfg), "--work", str(tmp_path)]) == 0
    for stage in CHANNEL_STAGES:
        assert _stage_status(tiny_volume, stage) == "disabled", f"{stage} survived the cancel"


@pytest.mark.golden
def test_no_verify_skips_the_declared_channels(tiny_volume: Path, tmp_path: Path) -> None:
    cfg = _city(tmp_path, '["junction", "addresses"]')
    rc = main(["run", VOL, "--city", str(cfg), "--work", str(tmp_path), "--no-verify"])
    assert rc == 0
    for stage in CHANNEL_STAGES:
        assert _stage_status(tiny_volume, stage) == "disabled", f"{stage} ran under --no-verify"


@pytest.mark.golden
def test_no_verify_contradicting_an_explicit_flag_is_an_error(
    tiny_volume: Path, tmp_path: Path
) -> None:
    """Never guess which one the caller meant about a spend (same rule as
    --no-escalate vs --escalate)."""
    cfg = _city(tmp_path, None)
    for flag in ("--verify-junctions", "--verified-accept"):
        with pytest.raises(SystemExit, match="--no-verify contradicts"):
            main(["run", VOL, "--city", str(cfg), "--work", str(tmp_path), "--no-verify", flag])


def _renumbering_city(tmp_path: Path, *, era: str | None, channels: str | None) -> Path:
    """A city that RENUMBERED (it ships a table) — the shape that demands an era."""
    cfg = tmp_path / "renumbered.toml"
    cl = ROOT / "fixtures" / "reference" / "street_center_lines.geojson"
    cfg.write_text(
        "[city]\n"
        'name = "Chicago, Ill."\n'
        f'centerlines = "{cl}"\n'
        f'aliases_dir = "{ROOT / "configs" / "chicago" / "aliases"}"\n'
        f'renumbering_table = "{ROOT / "configs" / "chicago" / "renumbering-chicago-1909.json"}"\n'
        + (f"evidence_channels = {channels}\n" if channels is not None else "")
        + f"[volumes.{VOL}]\n"
        + (f"addresses_modern = {era}\n" if era is not None else "")
        + "bounds_bbox = [-87.66, 41.87, -87.60, 41.90]\n"
        "scale_m_per_px = 0.0669\n"
        "rotation_deg = 1.20\n"
    )
    return cfg


@pytest.mark.golden
def test_addresses_channel_refuses_to_run_on_an_undeclared_era(
    tiny_volume: Path, tmp_path: Path
) -> None:
    """A warning was enough while the channel was opt-in. It is not enough now.

    Undeclared era = MODERN, and the addresses channel is the ONLY one that may
    REFUTE — so on a pre-renumbering volume the default reads printed numerals
    against a grid they predate and VETOES correct sheets. Measured on the 1895
    `_006.5`: 17 abstentions become 9 refutes, 8 on sheets 2.4-7.8 m from the human
    GCPs. A plain `autogeoref run` now fires this channel on every volume of a city
    that declares it — including the ~30 with no TOML section, which is the queue.
    So the run REFUSES, before any stage, rather than guessing an era.
    """
    cfg = _renumbering_city(tmp_path, era=None, channels='["addresses"]')
    with pytest.raises(SystemExit, match="declares no address era"):
        main(["run", VOL, "--city", str(cfg), "--work", str(tmp_path)])


@pytest.mark.golden
@pytest.mark.parametrize("era", ["true", "false"])
def test_a_declared_era_runs(tiny_volume: Path, tmp_path: Path, era: str) -> None:
    """Either declaration is fine — the engine refuses to GUESS, not to work."""
    cfg = _renumbering_city(tmp_path, era=era, channels='["addresses"]')
    assert main(["run", VOL, "--city", str(cfg), "--work", str(tmp_path)]) == 0
    assert _stage_status(tiny_volume, "verified-accept") == "ok"


@pytest.mark.golden
def test_no_era_needed_when_the_addresses_channel_is_not_running(
    tiny_volume: Path, tmp_path: Path
) -> None:
    """The demand is scoped to the channel that reads house numbers, not to the run.

    A volume with no era can still be matched, rescued, seamed and corroborated —
    and its junction channel still votes. Only the numeral-reading channel needs to
    know which century's numbers it is looking at.
    """
    cfg = _renumbering_city(tmp_path, era=None, channels='["junction"]')
    assert main(["run", VOL, "--city", str(cfg), "--work", str(tmp_path)]) == 0
    assert _stage_status(tiny_volume, "verified-accept") == "ok"

    cfg_all = _renumbering_city(tmp_path, era=None, channels='["junction", "addresses"]')
    rc = main(["run", VOL, "--city", str(cfg_all), "--work", str(tmp_path), "--no-verify"])
    assert rc == 0, "--no-verify turns the channel off, so no era is owed"


def _hide_cv2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate an install WITHOUT the [cv] extra.

    Hide the module from `find_spec`, which is what the CLI asks — deliberately NOT
    by making a stage raise ImportError. A broken-but-installed cv2 (`libGL.so.1`, an
    ABI mismatch) raises ImportError too, and reporting THAT as "the extra is not
    installed" would send the operator to a `uv sync` that cannot fix it. Absent and
    broken are different facts and the code separates them; so must the test.
    """
    real = importlib.util.find_spec

    def fake(name: str, package: str | None = None) -> object | None:
        return None if name == "cv2" else real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


@pytest.mark.golden
def test_missing_cv_extra_warns_and_does_not_sink_a_default_run(
    tiny_volume: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An absent [cv] extra must not cost a DEFAULT run its honest report.

    Two default-ON stages need cv2, not one: junction-verify reads drawn junctions,
    and ESCALATION reads them as its evidence gate (which pages are worth re-reading).
    A guard on junction alone is unreachable on any Chicago config — escalate runs
    first and `stop_on_failure` would kill the run there, before the junction warning
    could ever fire. Both must degrade, and both must SAY so.
    """
    _hide_cv2(monkeypatch)
    cfg = _city(tmp_path, '["junction", "addresses"]', ladder='["claude-sonnet-5"]')
    with caplog.at_level(logging.WARNING):
        rc = main(["run", VOL, "--city", str(cfg), "--work", str(tmp_path)])
    assert rc == 0, "an absent [cv] extra must not fail a run that never asked for it"
    assert "escalation skipped" in caplog.text, "escalate needs cv2 for its evidence gate"
    assert "junction channel skipped" in caplog.text
    assert (tiny_volume / "report.json").exists(), "the honest report still lands"
    # the consumer still ran, with one fewer channel to hear from — and said so
    assert _stage_status(tiny_volume, "verified-accept") == "ok"


@pytest.mark.golden
@pytest.mark.parametrize(
    ("flag", "message"),
    [
        ("--verify-junctions", "the [cv] extra is not installed"),
        ("--verified-accept", "the [cv] extra is not installed"),
        ("--escalate", "the [cv] extra is not installed"),
    ],
)
def test_missing_cv_extra_still_errors_for_a_caller_who_asked(
    tiny_volume: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str, message: str
) -> None:
    """A caller who NAMED the channel (or the spend) gets an error, not a shrug."""
    _hide_cv2(monkeypatch)
    cfg = _city(tmp_path, None, ladder='["claude-sonnet-5"]')
    with pytest.raises(SystemExit, match=re.escape(message)):
        main(["run", VOL, "--city", str(cfg), "--work", str(tmp_path), flag])
