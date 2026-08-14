"""Unit contracts for bounded page-level escalation concurrency."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from autogeoref.annotate.failures import BudgetLimitError, EmptyResponseError
from autogeoref.escalate import stage_escalate

SONNET = "claude-sonnet-5"
OPUS = "claude-opus-4-8"


@dataclass
class _Paths:
    results: Path
    annotations: Path
    sheets: Path
    manifest: Path


def _paths(root: Path, pages: tuple[str, ...] = ("1", "2")) -> _Paths:
    sheets = root / "sheets"
    paths = _Paths(root / "results", root / "annotations", sheets, sheets / "manifest.json")
    paths.results.mkdir()
    paths.annotations.mkdir()
    paths.sheets.mkdir()
    paths.manifest.write_text(json.dumps({f"p{page}": {} for page in pages}))
    for page in pages:
        (paths.sheets / f"p{page}_small.jpg").touch()
        (paths.results / f"p{page}.json").write_text(
            json.dumps({"status": "REJECTED (no valid RANSAC model)"})
        )
    return paths


def _patch_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    import autogeoref.escalate as escalate
    import autogeoref.junction_snap as junction_snap

    monkeypatch.setattr(
        junction_snap, "extract_junctions", lambda _image: SimpleNamespace(n_junctions=4)
    )
    monkeypatch.setattr(escalate, "sheet_input_from", lambda *_args: SimpleNamespace())
    monkeypatch.setattr(escalate, "constraints_for_page", lambda *_args: SimpleNamespace())


def _stage(paths: _Paths, annotate_fn: Any, *, model: str | list[str], jobs: int = 2) -> list[str]:
    return stage_escalate(
        paths,
        SimpleNamespace(aliases={}),
        SimpleNamespace(),
        model=model,
        annotate_fn=annotate_fn,
        jobs=jobs,
    )


def test_parallel_pages_enter_together_and_return_plan_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autogeoref.escalate as escalate

    paths = _paths(tmp_path)
    _patch_preflight(monkeypatch)
    entered = threading.Barrier(2, timeout=5)
    calls: list[str] = []
    calls_lock = threading.Lock()

    def concurrent(image_path: Path, _model: str) -> dict[str, Any]:
        with calls_lock:
            calls.append(image_path.name)
        entered.wait()
        return {}

    monkeypatch.setattr(
        escalate, "match_sheet", lambda *_args, **_kwargs: {"status": "OK", "n_inliers": 8}
    )
    assert _stage(paths, concurrent, model=SONNET) == ["1", "2"]
    assert sorted(calls) == ["p1_small.jpg", "p2_small.jpg"]


def test_parallel_pages_keep_each_ladder_serial_and_stop_at_first_flip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autogeoref.escalate as escalate

    paths = _paths(tmp_path)
    _patch_preflight(monkeypatch)
    first_calls = threading.Barrier(2, timeout=5)
    active: dict[str, int] = {}
    calls: list[tuple[str, str]] = []
    lock = threading.Lock()

    def tiered(image_path: Path, model: str) -> dict[str, Any]:
        page = image_path.stem.removeprefix("p").removesuffix("_small")
        with lock:
            active[page] = active.get(page, 0) + 1
            assert active[page] == 1
            calls.append((page, model))
            first = sum(call_page == page for call_page, _ in calls) == 1
        try:
            if first:
                first_calls.wait()
            if page == "1":
                return {}
            raise EmptyResponseError("p2 does not flip")
        finally:
            with lock:
                active[page] -= 1

    monkeypatch.setattr(
        escalate, "match_sheet", lambda *_args, **_kwargs: {"status": "OK", "n_inliers": 8}
    )
    assert _stage(paths, tiered, model=[SONNET, OPUS]) == ["1"]
    assert [model for page, model in calls if page == "1"] == [SONNET]
    assert [model for page, model in calls if page == "2"] == [SONNET] * 2 + [OPUS] * 2


def test_parallel_rerun_reuses_cache_and_exhausted_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autogeoref.escalate as escalate

    paths = _paths(tmp_path)
    _patch_preflight(monkeypatch)
    calls: list[str] = []

    def cached_or_failed(image_path: Path, _model: str) -> dict[str, Any]:
        calls.append(image_path.name)
        if image_path.name == "p1_small.jpg":
            return {}
        raise EmptyResponseError("p2 cannot read")

    monkeypatch.setattr(
        escalate, "match_sheet", lambda *_args, **_kwargs: {"status": "REJECTED (test)"}
    )
    _stage(paths, cached_or_failed, model=SONNET)
    assert (paths.annotations / f"p1.escalated.{SONNET}.json").exists()
    assert (paths.annotations / f"p2.escalated.{SONNET}.failed.json").exists()
    assert sorted(calls) == ["p1_small.jpg", "p2_small.jpg", "p2_small.jpg"]

    _stage(paths, cached_or_failed, model=SONNET)
    assert sorted(calls) == ["p1_small.jpg", "p2_small.jpg", "p2_small.jpg"]


def test_parallel_budget_stop_prevents_queued_retry_and_frontier_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path, ("1", "2", "3"))
    _patch_preflight(monkeypatch)
    p1_started = threading.Event()
    budget_reported = threading.Event()
    calls: list[tuple[str, str]] = []
    calls_lock = threading.Lock()

    def coordinated_budget(image_path: Path, model: str) -> dict[str, Any]:
        page = image_path.stem.removeprefix("p").removesuffix("_small")
        with calls_lock:
            calls.append((page, model))
        if page == "1":
            p1_started.set()
            assert budget_reported.wait(5)
            raise EmptyResponseError("in-flight call completed after the stop")
        if page == "2":
            assert p1_started.wait(5)
            budget_reported.set()
            raise BudgetLimitError("usage limit reached")
        raise AssertionError("queued page must not start")

    assert _stage(paths, coordinated_budget, model=[SONNET, OPUS]) == []
    assert sorted(calls) == [("1", SONNET), ("2", SONNET)]
    for page in ("1", "2", "3"):
        assert not (paths.annotations / f"p{page}.escalated.{SONNET}.failed.json").exists()
    assert not (paths.annotations / f"p1.escalated.{OPUS}.json").exists()


def test_budget_stop_rejects_an_attempt_after_its_initial_cancel_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Call admission closes the check-then-call race around a terminal stop."""
    import autogeoref.escalate as escalate

    paths = _paths(tmp_path)
    _patch_preflight(monkeypatch)
    first_checked = threading.Event()
    budget_set = threading.Event()
    provider_calls: list[str] = []

    class ObservedEvent(threading.Event):
        def set(self) -> None:
            super().set()
            budget_set.set()

    def controlled_retry(
        call: Any,
        _attempts: Any,
        label: str,
        *,
        cancelled: Any = None,
        admit_attempt: Any = None,
    ) -> Any:
        assert cancelled is not None and admit_attempt is not None
        if label.startswith("p1"):
            assert not cancelled()
            first_checked.set()
            assert budget_set.wait(5)
            assert not admit_attempt()
            return None, ""
        assert first_checked.wait(5)
        assert admit_attempt()
        return call(), ""

    def budgeted(image_path: Path, _model: str) -> dict[str, Any]:
        provider_calls.append(image_path.name)
        if image_path.name == "p2_small.jpg":
            raise BudgetLimitError("usage limit reached")
        raise AssertionError("p1 must be rejected before its provider call")

    monkeypatch.setattr(
        escalate,
        "threading",
        SimpleNamespace(Event=ObservedEvent, Lock=threading.Lock),
    )
    monkeypatch.setattr(escalate, "annotate_with_retry", controlled_retry)
    assert _stage(paths, budgeted, model=SONNET) == []
    assert provider_calls == ["p2_small.jpg"]


def test_default_annotator_creates_one_shared_backend_per_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import autogeoref.escalate as escalate

    entered_factory = threading.Event()
    release_factory = threading.Event()
    factory_calls = 0
    calls_lock = threading.Lock()

    class Backend:
        def annotate_extended(self, _image_path: Path) -> SimpleNamespace:
            return SimpleNamespace(raw={})

    def factory(_model: str) -> Backend:
        nonlocal factory_calls
        with calls_lock:
            factory_calls += 1
            first = factory_calls == 1
        if first:
            entered_factory.set()
            assert release_factory.wait(5)
        return Backend()

    monkeypatch.setattr(escalate, "backend_for_model", factory)
    annotate = escalate._default_annotator()
    image = tmp_path / "sheet.jpg"
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(annotate, image, SONNET)
        assert entered_factory.wait(5)
        second = pool.submit(annotate, image, SONNET)
        release_factory.set()
        assert first.result() == second.result() == {}

    assert factory_calls == 1
