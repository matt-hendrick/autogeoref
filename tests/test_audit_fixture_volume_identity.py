"""Contracts for the fixture-tree identity audit.

The audit answers one question — does ``fixtures/<vid>/`` hold volume ``<vid>``
— and the ways it can be wrong are all here: calling a misfiled tree OK, calling
a matching tree misfiled, and quietly passing a tree it never compared.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


def _module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_fixture_volume_identity.py"
    spec = importlib.util.spec_from_file_location("audit_fixture_volume_identity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(root: Path, vid: str, smalls: dict[str, bytes], streets: dict[str, list[str]]) -> None:
    volume = root / vid
    (volume / "results").mkdir(parents=True)
    (volume / "results" / "p1.json").write_text("{}")
    if smalls:
        (volume / "sheets").mkdir()
        for page, blob in smalls.items():
            (volume / "sheets" / f"p{page}_small.jpg").write_bytes(blob)
    if streets:
        (volume / "annotations").mkdir()
        for page, names in streets.items():
            (volume / "annotations" / f"p{page}.json").write_text(
                json.dumps({"streets": [{"name": n} for n in names]})
            )


def _row(report: dict[str, Any], vid: str) -> dict[str, Any]:
    row: dict[str, Any] = next(r for r in report["volumes"] if r["volume"] == vid)
    return row


def test_a_matching_tree_passes_and_a_misfiled_one_is_named(tmp_path: Path) -> None:
    """Identical scans are OK; a tree holding another volume's scans is MISFILED.

    The audit must also say WHICH volume the misfiled tree really holds, or the
    finding cannot be acted on without a second manual sweep.
    """
    audit = _module().audit
    work, fixtures = tmp_path / "work", tmp_path / "fixtures"
    right = {p: f"scan-right-{p}".encode() for p in ("1", "2", "3", "4")}
    other = {p: f"scan-other-{p}".encode() for p in ("1", "2", "3", "4")}

    _tree(fixtures, "vol_a", right, {})
    _tree(work, "vol_a", right, {})
    _tree(fixtures, "vol_b", other, {})  # filed as _b, holds _c's pixels
    _tree(work, "vol_b", right, {})
    _tree(work, "vol_c", other, {})

    report = audit(work, fixtures)
    assert _row(report, "vol_a")["verdict"] == "OK"
    assert _row(report, "vol_a")["method"] == "sheet image md5"
    assert _row(report, "vol_b")["verdict"] == "MISFILED"
    assert _row(report, "vol_b")["actually_holds"] == "vol_c"
    assert report["misfiled"] == ["vol_b"]


def test_the_match_floor_is_the_bar_and_it_is_inclusive(tmp_path: Path) -> None:
    """Exactly at ``MATCH_FLOOR`` is OK; one page below it is MISFILED."""
    module = _module()
    work, fixtures = tmp_path / "work", tmp_path / "fixtures"
    shared = {"1": b"same-1", "2": b"same-2"}

    # 2 of 4 identical == the floor
    _tree(fixtures, "at_floor", {**shared, "3": b"f-3", "4": b"f-4"}, {})
    _tree(work, "at_floor", {**shared, "3": b"w-3", "4": b"w-4"}, {})
    # 1 of 4 identical, below it
    _tree(fixtures, "under_floor", {"1": b"same-1", "2": b"f-2", "3": b"f-3", "4": b"f-4"}, {})
    _tree(work, "under_floor", {"1": b"same-1", "2": b"w-2", "3": b"w-3", "4": b"w-4"}, {})

    report = module.audit(work, fixtures)
    assert _row(report, "at_floor")["match_rate"] == module.MATCH_FLOOR
    assert _row(report, "at_floor")["verdict"] == "OK"
    assert _row(report, "under_floor")["verdict"] == "MISFILED"


def test_missing_smalls_fall_back_to_the_street_set_signal(tmp_path: Path) -> None:
    """With no prepped scans to hash, the annotation street sets still decide."""
    audit = _module().audit
    work, fixtures = tmp_path / "work", tmp_path / "fixtures"
    same = {"1": ["MAIN", "OAK"], "2": ["ELM", "1ST"]}
    different = {"1": ["CANAL", "HALSTED"], "2": ["ASHLAND", "DAMEN"]}

    _tree(fixtures, "agrees", {}, same)
    _tree(work, "agrees", {}, same)
    _tree(fixtures, "disagrees", {}, same)
    _tree(work, "disagrees", {}, different)

    report = audit(work, fixtures)
    assert _row(report, "agrees")["method"] == "annotation street sets"
    assert _row(report, "agrees")["verdict"] == "OK"
    assert _row(report, "disagrees")["verdict"] == "MISFILED"


def test_an_uncomparable_tree_is_never_reported_as_a_pass(tmp_path: Path) -> None:
    """ "Not checked" must read differently from "checked and fine".

    A fixture tree with no ``work/`` counterpart, or with no page in common, is
    unjudged — reporting either as OK would launder an unverified tree.
    """
    audit = _module().audit
    work, fixtures = tmp_path / "work", tmp_path / "fixtures"
    work.mkdir()

    _tree(fixtures, "orphan", {"1": b"x"}, {})
    _tree(fixtures, "no_overlap", {"1": b"x"}, {})
    _tree(work, "no_overlap", {"9": b"x"}, {})

    report = audit(work, fixtures)
    assert _row(report, "orphan")["verdict"] == "no work counterpart — not checkable here"
    assert _row(report, "orphan")["work_tree"] is False
    assert _row(report, "no_overlap")["verdict"] == "no comparable pages"
    assert report["misfiled"] == []
