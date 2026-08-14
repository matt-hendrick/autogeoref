"""Tests for the zero-spend waterway incidence-survey harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "water_incidence_survey.py"
    spec = importlib.util.spec_from_file_location("water_incidence_survey", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sheet(volume: str, page: str) -> dict[str, object]:
    return {"volume": volume, "page": page, "golden": False, "status": "REJECTED", "image": "x"}


def test_draw_sample_is_stratified_and_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    survey = _module()
    monkeypatch.setattr(
        survey,
        "STRATA",
        (
            ("stockyards", frozenset({"015"}), 2),
            ("river-cbd", frozenset({"017"}), 2),
            ("remainder", frozenset(), 2),
        ),
    )
    pool = [
        *[_sheet("015", f"p{i}") for i in range(3)],
        *[_sheet("017", f"p{i}") for i in range(3)],
        *[_sheet("other", f"p{i}") for i in range(3)],
    ]

    first = survey.draw_sample(pool)
    assert survey.draw_sample(pool) == first
    assert {row["stratum"] for row in first} == {"stockyards", "river-cbd", "remainder"}
    assert all(row["eligible"] is False for row in first)


def test_report_requires_completed_reviews_and_enforces_lake_exclusion() -> None:
    survey = _module()
    incomplete: list[dict[str, Any]] = [
        {
            "volume": "v",
            "page": "p1",
            "review_note": "",
            "waterway_label": None,
            "feature_class": None,
            "crossing_streets": [],
            "eligible": False,
            "stratum": "x",
        }
    ]
    with pytest.raises(ValueError, match="unreviewed"):
        survey.report(incomplete)

    rows = [
        {
            "volume": "v",
            "page": f"p{i}",
            "review_note": "reviewed",
            "waterway_label": "Chicago River",
            "feature_class": "river",
            "crossing_streets": ["Ashland"],
            "eligible": True,
            "stratum": "x",
        }
        for i in range(5)
    ]
    rows.append(
        {
            "volume": "v",
            "page": "p5",
            "review_note": "lake excluded",
            "waterway_label": "Lake Michigan",
            "feature_class": "lake",
            "crossing_streets": ["Lake Shore"],
            "eligible": False,
            "stratum": "x",
        }
    )
    result = survey.report(rows)
    assert result["direct_test_ready"] is True
    assert result["eligible_flagged_sheets"] == 5


def test_flagged_pool_skips_accepted_and_missing_images(tmp_path: Path) -> None:
    survey = _module()
    volume = tmp_path / "sanborn01790_001"
    (volume / "results").mkdir(parents=True)
    (volume / "sheets").mkdir()
    (volume / "results" / "p1.json").write_text(json.dumps({"status": "REJECTED"}))
    (volume / "results" / "p2.json").write_text(json.dumps({"status": "OK"}))
    (volume / "results" / "p3.json").write_text(json.dumps({"status": "REJECTED"}))
    (volume / "sheets" / "p1_small.jpg").touch()
    (volume / "sheets" / "p2_small.jpg").touch()

    assert [row["page"] for row in survey.flagged_pool(tmp_path)] == ["p1"]
