"""The run preamble's pre-spend refusals and warnings.

Everything here happens before `prep`, under `--dry-run` too: what a run would
cost, whether the machine can make the calls at all, and whether there is
anything on disk to read yet.

`shutil.which` is always faked, never delegated — which model CLIs a developer
happens to have installed must not decide whether these pass.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from autogeoref.cli.entry import main

if TYPE_CHECKING:
    import pytest


def _cityless_config(tmp_path: Path, extra_city: str = "") -> Path:
    """A minimal runnable city — fixture-free, so it runs cold in CI."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "city.toml"
    centerlines = tmp_path / "cl.geojson"
    centerlines.write_text('{"type": "FeatureCollection", "features": []}')
    (tmp_path / "aliases").mkdir(exist_ok=True)
    cfg.write_text(
        "[city]\n"
        'name = "X"\n'
        f'aliases_dir = "{tmp_path / "aliases"}"\n'
        f'centerlines = "{centerlines}"\n'
        f"{extra_city}"
        "[volumes.vol]\n"
        "bounds_bbox = [-87.7, 41.8, -87.6, 41.9]\n"
        "scale_m_per_px = 0.1\n"
        "rotation_deg = 0\n"
    )
    return cfg


def _which_without(*absent: str) -> object:
    """A `shutil.which` where exactly these are missing and everything else is not.

    Never delegates to the real one: which model CLIs a developer happens to have
    installed must not decide whether these pass.
    """

    def which(cmd: str, *_args: object, **_kwargs: object) -> str | None:
        return None if cmd in absent else f"/usr/bin/{cmd}"

    return which


def test_a_missing_model_cli_is_named_before_prep_rather_than_mid_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Every model is spawned as an executable, so an absent one is a stranger's
    first wall. Naming it in the preamble beats a FileNotFoundError after prep."""
    import autogeoref.runpolicy as runpolicy

    cfg = _cityless_config(
        tmp_path,
        'annotation_model = "codex:gpt-5.6-terra"\nescalation_models = ["claude-sonnet-5"]\n',
    )
    monkeypatch.setattr(runpolicy.shutil, "which", _which_without("codex", "claude"))
    with caplog.at_level(logging.WARNING):
        main(["run", "vol", "--city", str(cfg), "--work", str(tmp_path / "work"), "--dry-run"])

    assert "'codex' is not on PATH" in caplog.text
    assert "'claude' is not on PATH" in caplog.text
    # the annotation model and the escalation tier are attributed to their own
    # binaries, so a reader knows which install unblocks which stage
    assert "codex:gpt-5.6-terra" in caplog.text and "claude-sonnet-5" in caplog.text


def test_an_unfetched_volume_reads_as_the_first_state_not_a_broken_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The documented first command runs before anything is downloaded, so its
    preamble has to say "go fetch" rather than promise a prep that is about to
    print `disabled`. The nonzero exit stands: nothing can run yet, and a caller
    gating on `--dry-run` needs zero to keep meaning "this would run"."""
    cfg = _cityless_config(tmp_path)
    rc = main(["run", "vol", "--city", str(cfg), "--work", str(tmp_path / "work"), "--dry-run"])

    out = capsys.readouterr().out
    assert "no scans under" in out and "fetch_loc_volume.py vol" in out
    assert "prep runs first" not in out
    assert rc == 1


def test_a_present_model_cli_says_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Warning on every run would train the reader to ignore it."""
    import autogeoref.runpolicy as runpolicy

    cfg = _cityless_config(tmp_path, 'annotation_model = "codex:gpt-5.6-terra"\n')
    monkeypatch.setattr(runpolicy.shutil, "which", lambda cmd, *_a, **_k: f"/usr/bin/{cmd}")
    with caplog.at_level(logging.WARNING):
        main(["run", "vol", "--city", str(cfg), "--work", str(tmp_path / "work"), "--dry-run"])

    assert "not on PATH" not in caplog.text


def test_no_annotate_and_ollama_are_not_reported_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """--no-annotate buys no read, and an ollama tier is served over HTTP with no
    binary to find — flagging either would be a false wall."""
    import autogeoref.runpolicy as runpolicy

    cfg = _cityless_config(tmp_path, 'escalation_models = ["ollama:llava-next"]\n')
    monkeypatch.setattr(runpolicy.shutil, "which", _which_without("ollama"))
    with caplog.at_level(logging.WARNING):
        main(["run", "vol", "--city", str(cfg), "--work", str(tmp_path / "work"), "--dry-run"])
    assert "not on PATH" not in caplog.text

    cfg2 = _cityless_config(tmp_path / "b", 'annotation_model = "codex:gpt-5.6-terra"\n')
    caplog.clear()
    # codex genuinely absent this time: --no-annotate is what has to silence it.
    monkeypatch.setattr(runpolicy.shutil, "which", _which_without("codex"))
    with caplog.at_level(logging.WARNING):
        main(
            [
                "run",
                "vol",
                "--city",
                str(cfg2),
                "--work",
                str(tmp_path / "work"),
                "--dry-run",
                "--no-annotate",
            ]
        )
    assert "not on PATH" not in caplog.text
