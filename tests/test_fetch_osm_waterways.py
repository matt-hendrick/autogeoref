"""Contract tests for the waterway reference query."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "fetch_osm_waterways.py"
    spec = importlib.util.spec_from_file_location("fetch_osm_waterways", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_waterway_query_requests_only_named_linear_features() -> None:
    query = _module().waterway_query((-87.95, 41.62, -87.5, 42.05), timeout_s=123)

    assert '["waterway"]["name"](41.62,-87.95,42.05,-87.5)' in query
    assert "out geom;" in query
    assert "natural" not in query
