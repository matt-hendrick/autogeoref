"""Contracts for the re-place spend probe.

What the probe must get right is the BILL: a tier already holding a reading or
a failure marker costs nothing, and a re-place's exposure is wider than the
pages that are eligible today. Both are pinned here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from autogeoref.annotate.providers import canonical_model, model_cache_key, prior_variant_cache_key
from autogeoref.escalate import MIN_JUNCTIONS_TO_ESCALATE
from autogeoref.paths import VolumePaths

if TYPE_CHECKING:
    import pytest

MODEL = "codex:gpt-5.6-terra"
VARIANT = "high"


def _module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "escalation_spend_probe.py"
    spec = importlib.util.spec_from_file_location("escalation_spend_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _volume(tmp_path: Path) -> VolumePaths:
    paths = VolumePaths(root=tmp_path / "vol")
    paths.annotations.mkdir(parents=True)
    paths.results.mkdir(parents=True)
    paths.sheets.mkdir(parents=True)
    paths.manifest.write_text("{}")
    return paths


def test_the_probe_reads_the_shipped_junction_floor() -> None:
    """Not a copy of the number: the stage's own floor, or the bill is fiction."""
    assert _module().MIN_JUNCTIONS_TO_ESCALATE is MIN_JUNCTIONS_TO_ESCALATE


def test_a_failure_marker_counts_as_a_cache_hit_not_a_call(tmp_path: Path) -> None:
    """A marked tier is skipped until someone deletes the marker, so it is free.

    A probe that priced a marker as a call would inflate every estimate on a
    volume that has already failed a tier, which is exactly the population a
    re-place is sized for.
    """
    probe = _module()
    paths = _volume(tmp_path)
    key = model_cache_key(canonical_model(MODEL), VARIANT)
    prior = prior_variant_cache_key(canonical_model(MODEL), VARIANT)
    assert prior is not None

    assert probe.tier_state(paths, "1", MODEL, VARIANT) == probe.SPENDS

    (paths.annotations / f"p1.escalated.{key}.failed.json").write_text("{}")
    assert probe.tier_state(paths, "1", MODEL, VARIANT) == "marker"

    # a reading outranks the marker for the same tier
    (paths.annotations / f"p1.escalated.{key}.json").write_text("{}")
    assert probe.tier_state(paths, "1", MODEL, VARIANT) == "reading"

    # the prior variant spelling is a hit too, on either kind of file
    (paths.annotations / f"p2.escalated.{prior}.failed.json").write_text("{}")
    assert probe.tier_state(paths, "2", MODEL, VARIANT) == "marker(prior-variant)"
    (paths.annotations / f"p3.escalated.{prior}.json").write_text("{}")
    assert probe.tier_state(paths, "3", MODEL, VARIANT) == "reading(prior-variant)"


def test_the_junction_floor_and_the_status_buckets_decide_who_is_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only pages over the drawn-junction floor reach the bill, tagged by status."""
    probe = _module()
    paths = _volume(tmp_path)
    pages = {
        "1": "REJECTED (weak fit)",
        "2": "OK",
        "3": "REJECTED (rescue revoked: disjoint anchors)",
        "4": "REJECTED (weak fit)",
    }
    manifest = {}
    for page, status in pages.items():
        (paths.results / f"p{page}.json").write_text(json.dumps({"page": page, "status": status}))
        (paths.sheets / f"p{page}_small.jpg").touch()
        manifest[f"p{page}"] = {"file": f"p{page}_small.jpg", "scale": 1.0}
    paths.manifest.write_text(json.dumps(manifest))

    class _Extraction:
        def __init__(self, n: int) -> None:
            self.n_junctions = n

    # p4 alone sits under the floor; everything else clears it
    monkeypatch.setattr(
        probe,
        "extract_junctions",
        lambda image: _Extraction(
            MIN_JUNCTIONS_TO_ESCALATE - 1
            if image.name == "p4_small.jpg"
            else MIN_JUNCTIONS_TO_ESCALATE
        ),
    )

    passed, skipped = probe.gate_pages(paths)
    assert dict(passed) == {"1": "rejected", "2": "ok", "3": "revoked"}
    assert sum(skipped.values()) == 1


def test_eligible_now_bills_rejected_pages_and_worst_case_bills_them_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The two figures are different questions, and the return value is the exposure.

    ELIGIBLE NOW prices the pages the stage would touch today. WORST CASE drops
    the status filter, because a re-place can demote an accepted page and an
    accepted page has no escalation cache to spare it.
    """
    probe = _module()
    paths = _volume(tmp_path)
    city = tmp_path / "city.toml"
    city.write_text(
        '[city]\nname = "X"\naliases_dir = "aliases"\n'
        f'escalation_models = ["{MODEL}"]\nescalation_variants = ["{VARIANT}"]\n'
        "[volumes.vol]\n"
    )
    # one rejected page, one accepted page, neither cached anywhere
    monkeypatch.setattr(probe, "gate_pages", lambda _p: ([("1", "rejected"), ("2", "ok")], {}))

    worst = probe.probe("vol", city, tmp_path)
    out = capsys.readouterr().out
    assert "-- ELIGIBLE NOW: 1 page(s) --" in out
    assert "-- WORST CASE (status-blind): 2 page(s) --" in out
    assert worst == 2

    # caching the rejected page's only tier takes it off the eligible bill but
    # leaves the accepted page's exposure standing
    key = model_cache_key(canonical_model(MODEL), VARIANT)
    (paths.annotations / f"p1.escalated.{key}.json").write_text("{}")
    assert probe.probe("vol", city, tmp_path) == 1
    assert "ELIGIBLE NOW: 1 page(s) --\n       codex:gpt-5.6-terra/high: would spend 0" in (
        capsys.readouterr().out
    )


def test_a_volume_with_no_ladder_cannot_spend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No configured tier means ``stage_escalate`` has nothing to call."""
    probe = _module()
    _volume(tmp_path)
    city = tmp_path / "city.toml"
    city.write_text('[city]\nname = "X"\naliases_dir = "aliases"\n[volumes.vol]\n')

    assert probe.probe("vol", city, tmp_path) == 0
    assert "cannot spend here at all" in capsys.readouterr().out


def test_a_volume_with_no_results_is_reported_not_priced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    city = tmp_path / "city.toml"
    city.write_text('[city]\nname = "X"\naliases_dir = "aliases"\n[volumes.vol]\n')
    assert _module().probe("vol", city, tmp_path) == 0
    assert "nothing to size" in capsys.readouterr().out
