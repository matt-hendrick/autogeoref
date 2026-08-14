"""The review queue, the edit sidecar, and the GCPs an edit materializes into.

A sidecar is the persisted form of one reviewer's decision, so it is validated
on the way in and rejected by name when it is malformed. Materializing it back
out must reproduce the composed affine exactly: refitting the moved control
points gives the same model, and the image pixels never move.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autogeoref.affine import (
    fit_affine,
    gcps_from_geojson,
)
from autogeoref.errors import ReviewError
from autogeoref.review.app import review_queue
from autogeoref.review.materialize import (
    compose_ops,
    final_gcps_geojson,
    synthetic_gcps_geojson,
    transformed_gcps_geojson,
)
from autogeoref.review.sidecars import (
    sidecar_from_dict,
)
from review_support import (
    BASE,
    FULL_SIZE,
    make_sidecar_dict,
    make_volume,
    ops_translate,
)


def test_sidecar_round_trips() -> None:
    side = sidecar_from_dict(make_sidecar_dict())
    assert side.verdict == "adjusted"
    assert side.ops == ops_translate(10, -5)
    assert side.mask_px is not None and len(side.mask_px) == 3
    assert side.timestamp  # filled when absent


@pytest.mark.parametrize(
    "bad",
    [
        {"verdict": "maybe"},
        {"ops": [{"type": "shear", "k": 2}]},
        {"ops": [{"type": "scale", "factor": -1, "center_3857": [0, 0]}]},
        {"affine": [[1, 2], [3, 4]]},
        {"mask_px": [[0, 0], [1, 1]]},
        {"base_result_sha256": ""},
        {"page": "10_1"},  # crop-layer id: crop pixels are not page pixels
        {"page": "../4"},  # apply interpolates the page into paths
        {"page": "cbd3"},  # named sheets are a literal allowlist, not a pattern
    ],
)
def test_sidecar_rejects_malformed(bad: dict[str, Any]) -> None:
    with pytest.raises(ReviewError):
        sidecar_from_dict(make_sidecar_dict(**bad))


def test_sidecar_accepts_named_and_suffixed_pages() -> None:
    assert sidecar_from_dict(make_sidecar_dict(page="cbd1")).page == "cbd1"
    assert sidecar_from_dict(make_sidecar_dict(page="13S")).page == "13S"


def test_review_queue_flagged_pool(tmp_path: Path) -> None:
    paths = make_volume(tmp_path)
    q = review_queue(paths, "volX")
    assert [e["page"] for e in q] == ["4", "6"]
    assert q[0]["has_placement"] and not q[1]["has_placement"]
    q_all = review_queue(paths, "volX", include_ok=True)
    assert [e["page"] for e in q_all] == ["2", "4", "6"]
    assert q_all[0]["committed"]


def test_transformed_gcps_refit_reproduces_composed_affine(tmp_path: Path) -> None:
    paths = make_volume(tmp_path)
    r = json.loads((paths.results / "p4.json").read_text())
    ops = [
        {"type": "translate", "dx_m": -400.0, "dy_m": 25.0},
        {"type": "rotate", "deg": 1.5, "center_3857": [-9755000.0, 5140000.0]},
    ]
    base = fit_affine(gcps_from_geojson(r["gcps_geojson"]))
    moved = transformed_gcps_geojson(r, ops)
    refit = fit_affine(gcps_from_geojson(moved))
    assert np.allclose(refit, compose_ops(base, ops), atol=1e-4)
    # image pixels never move
    for before, after in zip(r["gcps_geojson"]["features"], moved["features"], strict=True):
        assert before["properties"]["image"] == after["properties"]["image"]


def test_synthetic_reviewer_gcps_are_marked_synthetic() -> None:
    fc = synthetic_gcps_geojson(BASE, FULL_SIZE)
    assert len(fc["features"]) == 4
    assert all("synthetic" in f["properties"]["note"] for f in fc["features"])
    refit = fit_affine(gcps_from_geojson(fc))
    assert np.allclose(refit, BASE, atol=1e-4)


def test_final_gcps_accept_without_placement_is_none() -> None:
    side = sidecar_from_dict(make_sidecar_dict(verdict="accept", ops=[], affine=None, mask_px=None))
    assert final_gcps_geojson({"page": "6"}, side, FULL_SIZE) is None
