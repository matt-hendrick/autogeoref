"""The markdown page is generated, so it must still match the panels on disk.

Without this, a caption edit committed without regenerating leaves
``docs/HOW-IT-WORKS.md`` silently stale, which is the defect generating it was
meant to remove. Rendering is a pure function of the committed JSON, so this
needs no ``work/`` tree.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
WALKTHROUGH = ROOT / "scripts" / "walkthrough"


def load_script(name: str) -> ModuleType:
    """Import one of the generator's libraries by path; `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(name, WALKTHROUGH / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PANELS = ROOT / "viewer" / "walkthrough" / "panels.json"
DOC = ROOT / "docs" / "HOW-IT-WORKS.md"


@pytest.fixture(scope="module")
def markdown() -> Any:
    return load_script("walkthrough_markdown")


@pytest.fixture(scope="module")
def page() -> dict[str, Any]:
    return dict(json.loads(PANELS.read_text(encoding="utf-8")))


def _panels(payload: dict[str, Any]) -> list[Any]:
    page_module = load_script("walkthrough_page")
    built = []
    for raw in payload["panels"]:
        panel = page_module.Panel(
            number=raw["number"],
            slug=raw["slug"],
            act=raw["act"],
            title=raw["title"],
            dek=raw["dek"],
            caption=raw["caption"],
            stage=raw["stage"],
            note=raw["note"],
            figures=[(f["label"], f["value"]) for f in raw["figures"]],
            states=[
                page_module.State(key=s["key"], label=s["label"], file=s["file"], alt=s["alt"])
                for s in raw["states"]
            ],
        )
        built.append(panel)
    return built


def test_the_committed_page_is_what_the_generator_renders(
    markdown: Any, page: dict[str, Any]
) -> None:
    rendered = markdown.render(
        page["meta"],
        _panels(page),
        [(g["term"], g["gloss"]) for g in page["glossary"]],
        "../viewer/walkthrough",
    )
    assert rendered == DOC.read_text(encoding="utf-8"), (
        "docs/HOW-IT-WORKS.md is stale; regenerate it with "
        "scripts/walkthrough/make_walkthrough_assets.py"
    )


def test_every_glossary_term_is_used_somewhere_on_the_page(page: dict[str, Any]) -> None:
    prose = " ".join(
        (panel.get(key) or "")
        for panel in page["panels"]
        for key in ("title", "dek", "caption", "note")
    ).lower()
    unused = [g["term"] for g in page["glossary"] if g["term"].lower() not in prose]
    assert not unused, f"glossary defines terms the walkthrough never uses: {unused}"


def test_every_referenced_plate_exists(page: dict[str, Any]) -> None:
    missing = [
        state["file"]
        for panel in page["panels"]
        for state in panel["states"]
        if not (PANELS.parent / state["file"]).is_file()
    ]
    assert not missing, f"panels.json names plates that are not on disk: {missing}"


def test_a_panel_with_no_plate_does_not_break_the_page(markdown: Any) -> None:
    page_module = load_script("walkthrough_page")
    empty = page_module.Panel(
        number=1, slug="x", act="I. Act", title="T", dek="D", caption="C body."
    )
    assert "### 1. T" in "\n".join(markdown._panel(empty, "../plates"))


def test_a_pipe_in_a_label_cannot_split_a_table_row(markdown: Any) -> None:
    page_module = load_script("walkthrough_page")
    states = [
        page_module.State(key="a", label="Before | after", file="a.jpg", alt="A|B"),
        page_module.State(key="b", label="Second", file="b.jpg", alt="C"),
    ]
    rows = markdown._labelled(states, "../plates")
    delimiters = [row.replace("\\|", "").count("|") for row in rows]
    assert delimiters == [3, 3, 3], rows
