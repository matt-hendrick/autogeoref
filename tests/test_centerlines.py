"""Centerline-index cache contracts."""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from autogeoref.centerlines import CenterlineIndex

FEATURES = [
    {
        "properties": {"street_nam": "A", "street_typ": "ST"},
        "geometry": {"type": "LineString", "coordinates": [[-1, 0], [1, 0]]},
    },
    {
        "properties": {"street_nam": "B", "street_typ": "ST"},
        "geometry": {"type": "LineString", "coordinates": [[0, -1], [0, 1]]},
    },
]


def test_concurrent_intersection_cache_populates_once_and_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parallel matching has one cache publisher and unchanged replay output."""
    import autogeoref.centerlines as centerlines

    index = CenterlineIndex(FEATURES)
    real_union = centerlines.unary_union
    first_union = threading.Event()
    release_first_union = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def blocking_union(geometries: Any) -> Any:
        nonlocal calls
        with calls_lock:
            calls += 1
            first = calls == 1
        if first:
            first_union.set()
            assert release_first_union.wait(5)
        return real_union(geometries)

    monkeypatch.setattr(centerlines, "unary_union", blocking_union)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(index.intersections, "A", "B")
        assert first_union.wait(5)
        second = pool.submit(index.intersections, "A", "B")
        release_first_union.set()
        parallel = [first.result(), second.result()]

    assert calls == 2  # one merged geometry per street, not per worker
    assert parallel == [[(0.0, 0.0)], [(0.0, 0.0)]]
    assert index.intersections("A", "B") == [(0.0, 0.0)]
