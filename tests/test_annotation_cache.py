"""Primary annotation cache identity across model reasoning variants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import autogeoref.annotate_volume as annotation_volume
from autogeoref.annotate.failures import BudgetLimitError
from autogeoref.annotate.providers import model_cache_key, prior_variant_cache_key
from autogeoref.annotate.schema import EXTENDED_PROMPT_TEMPLATE, extended_from_raw
from autogeoref.annotate_volume import ReadIdentity
from autogeoref.paths import VolumePaths


def fake_backend_factory(read: Any) -> Any:
    """Stand in for `backend_for_model`: hand the reference to a `read` callable.

    The stage builds one reader per batch, so the model and variant arrive at
    construction; `read(image, model, variant)` still runs once per page.
    """

    def factory(model: str, *, variant: str | None = None, **_kwargs: Any) -> Any:
        class Reader:
            def annotate_extended(self, image: Path) -> Any:
                return read(image, model, variant)

        return Reader()

    return factory


def test_the_batch_builds_one_reader_and_only_when_it_will_read(
    tmp_path: Path, monkeypatch: object
) -> None:
    """A direct-API provider holds an HTTP client, so a reader per page would
    churn connection pools; and a fully cached volume must need no provider at
    all, which is what keeps `--dry-run` and a replay free of credentials.
    """
    paths = VolumePaths(root=tmp_path)
    paths.sheets.mkdir(parents=True)
    paths.sheets.joinpath("manifest.json").write_text(json.dumps({"p1": {}, "p2": {}}))
    for page in ("p1", "p2"):
        paths.sheets.joinpath(f"{page}_small.jpg").write_bytes(b"image")
    built: list[str] = []
    prompts: list[str | None] = []

    def factory(model: str, *, variant: str | None = None, **kwargs: Any) -> Any:
        built.append(model)
        prompts.append(kwargs.get("prompt_template"))

        class Reader:
            def annotate_extended(self, _image: Path) -> Any:
                return extended_from_raw({"streets": []})

        return Reader()

    monkeypatch.setattr(annotation_volume, "backend_for_model", factory)
    identity = ReadIdentity("claude-sonnet-5")
    annotation_volume.annotate_volume(paths, "vol", identity=identity, attempts=1, jobs=2)
    assert built == ["claude-sonnet-5"], "one reader for two pages"
    assert prompts[0] == EXTENDED_PROMPT_TEMPLATE, "the resolved prompt, not its name"

    # every page is cached now, so the second pass builds nothing
    annotation_volume.annotate_volume(paths, "vol", identity=identity, attempts=1)
    assert built == ["claude-sonnet-5"]


def test_primary_cache_is_keyed_by_model_and_variant(tmp_path: Path, monkeypatch: object) -> None:
    paths = VolumePaths(root=tmp_path)
    paths.sheets.mkdir(parents=True)
    paths.sheets.joinpath("manifest.json").write_text(json.dumps({"p1": {}}))
    paths.sheets.joinpath("p1_small.jpg").write_bytes(b"image")
    calls: list[tuple[str, str | None]] = []

    def read(_image: Path, model: str, variant: str | None) -> object:
        calls.append((model, variant))
        return extended_from_raw({"streets": []})

    monkeypatch.setattr(annotation_volume, "backend_for_model", fake_backend_factory(read))
    annotation_volume.annotate_volume(
        paths, "vol", identity=ReadIdentity("codex:gpt-5.6-terra", "high"), attempts=1
    )
    high_cache = paths.annotations / (
        f"p1.annotation.{model_cache_key('codex:gpt-5.6-terra', 'high')}.json"
    )
    assert high_cache.exists()

    # The generic active annotation must not make a default-effort read appear cached.
    annotation_volume.annotate_volume(
        paths, "vol", identity=ReadIdentity("codex:gpt-5.6-terra"), attempts=1
    )
    assert calls == [("codex:gpt-5.6-terra", "high"), ("codex:gpt-5.6-terra", None)]

    # Switching back to high restores its exact cache and makes no model call.
    annotation_volume.annotate_volume(
        paths, "vol", identity=ReadIdentity("codex:gpt-5.6-terra", "high"), attempts=1
    )
    assert calls == [("codex:gpt-5.6-terra", "high"), ("codex:gpt-5.6-terra", None)]


def test_keyed_failure_does_not_block_the_legacy_default(
    tmp_path: Path, monkeypatch: object
) -> None:
    paths = VolumePaths(root=tmp_path)
    paths.sheets.mkdir(parents=True)
    paths.sheets.joinpath("manifest.json").write_text(json.dumps({"p1": {}}))
    paths.sheets.joinpath("p1_small.jpg").write_bytes(b"image")
    calls: list[str] = []

    def read(_image: Path, model: str, variant: str | None) -> object:
        calls.append(model)
        if variant == "high":
            from autogeoref.annotate.failures import AnnotationCallError

            raise AnnotationCallError("failed high read")
        return extended_from_raw({"streets": []})

    monkeypatch.setattr(annotation_volume, "backend_for_model", fake_backend_factory(read))
    annotation_volume.annotate_volume(
        paths, "vol", identity=ReadIdentity("codex:gpt-5.6-terra", "high"), attempts=1
    )
    annotation_volume.annotate_volume(
        paths, "vol", identity=ReadIdentity("claude-sonnet-5"), attempts=1
    )
    assert calls == ["codex:gpt-5.6-terra", "claude-sonnet-5"]


def _legacy_only_volume(tmp_path: Path) -> VolumePaths:
    """One page whose only annotation is a bare pre-identity p1.json."""
    paths = VolumePaths(root=tmp_path)
    paths.sheets.mkdir(parents=True)
    paths.annotations.mkdir()
    paths.sheets.joinpath("manifest.json").write_text(json.dumps({"p1": {}}))
    paths.sheets.joinpath("p1_small.jpg").write_bytes(b"image")
    paths.annotations.joinpath("p1.json").write_text(json.dumps({"streets": []}))
    return paths


def test_provenance_free_read_is_reused_without_spend(tmp_path: Path) -> None:
    """A cache-layout migration must never become unplanned spend: a bare
    p<N>.json with no cache records is reused as-is, whatever model is configured."""
    paths = _legacy_only_volume(tmp_path)

    batch = annotation_volume.plan(paths, "vol", identity=ReadIdentity("claude-sonnet-5"))
    assert batch.todo == []
    assert batch.legacy == ["p1"]
    assert "unattributed" in batch.summary()


def test_reread_unattributed_is_the_explicit_opt_in(tmp_path: Path) -> None:
    paths = _legacy_only_volume(tmp_path)

    batch = annotation_volume.plan(
        paths, "vol", identity=ReadIdentity("claude-sonnet-5"), reread_unattributed=True
    )
    assert batch.todo == ["p1"]
    assert batch.legacy == []


def test_a_page_with_cache_records_is_not_legacy(tmp_path: Path) -> None:
    """Per-model caching still governs post-migration pages: a read keyed to one
    model does not satisfy another, and the switch plans a re-read as before."""
    paths = _legacy_only_volume(tmp_path)
    key = model_cache_key("codex:gpt-5.6-terra", "high")
    paths.annotations.joinpath(f"p1.annotation.{key}.json").write_text(json.dumps({"streets": []}))

    batch = annotation_volume.plan(paths, "vol", identity=ReadIdentity("claude-sonnet-5"))
    assert batch.todo == ["p1"]
    assert batch.legacy == []


def test_failed_replacement_removes_the_active_annotation(
    tmp_path: Path, monkeypatch: object
) -> None:
    paths = VolumePaths(root=tmp_path)
    paths.sheets.mkdir(parents=True)
    paths.annotations.mkdir()
    paths.sheets.joinpath("manifest.json").write_text(json.dumps({"p1": {}}))
    paths.sheets.joinpath("p1_small.jpg").write_bytes(b"image")
    paths.annotations.joinpath("p1.json").write_text(json.dumps({"streets": [{"name": "OLD"}]}))

    def fail(_image: Path, model: str, variant: str | None) -> object:
        from autogeoref.annotate.failures import AnnotationCallError

        assert model == "codex:gpt-5.6-terra" and variant == "high"
        raise AnnotationCallError("replacement failed")

    monkeypatch.setattr(annotation_volume, "backend_for_model", fake_backend_factory(fail))
    # replacing a legacy read requires the explicit flag; without it no call happens
    annotation_volume.annotate_volume(
        paths,
        "vol",
        identity=ReadIdentity("codex:gpt-5.6-terra", "high"),
        attempts=1,
        reread_unattributed=True,
    )
    assert not paths.annotations.joinpath("p1.json").exists()


def test_primary_reuses_the_prior_variant_cache(tmp_path: Path, monkeypatch: object) -> None:
    paths = VolumePaths(root=tmp_path)
    paths.sheets.mkdir(parents=True)
    paths.annotations.mkdir()
    paths.sheets.joinpath("manifest.json").write_text(json.dumps({"p1": {}}))
    paths.sheets.joinpath("p1_small.jpg").write_bytes(b"image")
    key = prior_variant_cache_key("codex:gpt-5.6-terra", "high")
    assert key is not None
    payload: dict[str, list[Any]] = {"streets": []}
    paths.annotations.joinpath(f"p1.annotation.{key}.json").write_text(json.dumps(payload))

    def should_not_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("v1 cache should be reused")

    monkeypatch.setattr(
        annotation_volume, "backend_for_model", fake_backend_factory(should_not_run)
    )
    annotation_volume.annotate_volume(
        paths, "vol", identity=ReadIdentity("codex:gpt-5.6-terra", "high"), attempts=1
    )
    assert json.loads(paths.annotations.joinpath("p1.json").read_text()) == payload


def test_a_budget_limit_writes_no_failure_marker_and_no_failed_cache_entry(
    tmp_path: Path, monkeypatch: object
) -> None:
    """A budget stop is not a page failure: the sheet is unread, not unreadable.

    The stage catches the limit above the marker write and re-raises. Swap those
    two and a provider refusal becomes a sticky marker that skips the page on
    every later run, so the sheet is lost to a fault that has already cleared.
    """
    paths = VolumePaths(root=tmp_path)
    paths.sheets.mkdir(parents=True)
    paths.sheets.joinpath("manifest.json").write_text(json.dumps({"p1": {}, "p2": {}}))
    for page in ("p1", "p2"):
        paths.sheets.joinpath(f"{page}_small.jpg").write_bytes(b"image")
    calls: list[str] = []

    def budget_limited(image: Path, _model: str, _variant: str | None) -> object:
        calls.append(image.name)
        raise BudgetLimitError("usage limit reached")

    monkeypatch.setattr(
        annotation_volume, "backend_for_model", fake_backend_factory(budget_limited)
    )
    identity = ReadIdentity("codex:gpt-5.6-terra")

    with pytest.raises(BudgetLimitError):
        annotation_volume.annotate_volume(paths, "vol", identity=identity, attempts=2, jobs=1)

    assert calls == ["p1_small.jpg"], "terminal: no retry, and no second page"
    key = model_cache_key("codex:gpt-5.6-terra", None)
    for name in (
        "p1.failed.json",
        f"p1.annotation.{key}.failed.json",
        "p1.annotation.active.json",
    ):
        assert not paths.annotations.joinpath(name).exists(), name
