"""Junction-snap verifier vs its recorded validation behavior.

Golden tests replay ``_034`` p92 (diagonal avenues + rail ROW — the sheet
that separates at every radius) and p114 (pure grid) against the recorded
truth affines. Measured reference numbers (this port, matching the recorded
validation): p92 truth ratio 2.06 at R=100 (recorded: 2.06); p114 truth ratio
2.12. A 300 m offset prior on p92 scores 0.504 vs 0.875 at truth.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("skimage")

from autogeoref.affine import fit_affine, gcps_from_geojson
from autogeoref.junction_snap import (
    JunctionExtraction,
    JunctionSnapError,
    extract_junctions,
    extraction_in_source_frame,
    verify_placement,
    world_from_centerlines,
)

VOLUME = "sanborn01790_034"
_WGS84_RADIUS_M = 6378137.0


# ---------------------------------------------------------------------------
# unit tests (synthetic)
# ---------------------------------------------------------------------------


def _synthetic_features() -> list[dict[str, Any]]:
    def feat(name: str, coords: list[list[float]], fnode: int, tnode: int) -> dict[str, Any]:
        return {
            "type": "Feature",
            "properties": {"street_nam": name, "fnode_id": fnode, "tnode_id": tnode},
            "geometry": {"type": "LineString", "coordinates": coords},
        }

    # two streets crossing at node 1 (a corner: degree 2, distinct names)
    return [
        feat("A", [[0.0, 0.0], [0.001, 0.0]], 1, 2),
        feat("B", [[0.0, 0.0], [0.0, 0.001]], 1, 3),
    ]


def test_world_from_centerlines_synthetic() -> None:
    world = world_from_centerlines(_synthetic_features())
    assert len(world.nodes_3857) == 1  # only the shared corner is an intersection
    assert world.node_degrees[0] == 2
    assert len(world.polylines_3857) == 2
    # node is at lng/lat (0, 0) -> 3857 origin
    assert abs(world.nodes_3857[0][0]) < 1e-6
    assert abs(world.nodes_3857[0][1]) < 1e-6


def test_world_clip_bounds() -> None:
    world = world_from_centerlines(_synthetic_features(), bounds_3857=(1e6, 1e6, 2e6, 2e6))
    assert len(world.nodes_3857) == 0
    assert len(world.polylines_3857) == 0


def test_verify_rejects_too_few_junctions() -> None:
    world = world_from_centerlines(_synthetic_features())
    extraction = JunctionExtraction(
        junctions_px=np.zeros((2, 2)),
        junction_types=("any", "any"),
        skeleton=np.zeros((10, 10), dtype=bool),
        diagnostics={},
    )
    m = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    with pytest.raises(JunctionSnapError):
        verify_placement(extraction, m, world, small_to_full=1.0)


# ---------------------------------------------------------------------------
# rotation composition: upright small (prep's rotation_applied) -> source frame
# ---------------------------------------------------------------------------


def _marked_extraction() -> JunctionExtraction:
    """Skeleton with three lit pixels; the junctions sit ON those pixels."""
    skel = np.zeros((7, 5), dtype=bool)  # h=7, w=5 (portrait)
    pts = [(1.0, 0.0), (4.0, 2.0), (0.0, 6.0)]  # (x, y)
    for x, y in pts:
        skel[int(y), int(x)] = True
    return JunctionExtraction(
        junctions_px=np.asarray(pts, dtype=np.float64),
        junction_types=("X", "any", "any"),
        skeleton=skel,
        diagnostics={"n_junctions": 3.0},
    )


def test_source_frame_identity_without_rotation() -> None:
    e = _marked_extraction()
    assert extraction_in_source_frame(e, 0) is e


def test_source_frame_rejects_non_quarter_turn() -> None:
    with pytest.raises(JunctionSnapError):
        extraction_in_source_frame(_marked_extraction(), 45)


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_junctions_stay_on_their_skeleton_pixels(rotation: int) -> None:
    """The whole point: junctions and skeleton must turn together.

    ``rotation`` is prep's clockwise correction, so the composition turns the
    extraction back by ``360 - rotation``; a junction that sat on a lit
    skeleton pixel in the upright frame must still sit on one in the source
    frame, and the source frame's shape must be the un-turned one.
    """
    e = _marked_extraction()
    back = extraction_in_source_frame(e, rotation)
    h, w = e.skeleton.shape
    assert back.skeleton.shape == ((w, h) if rotation in (90, 270) else (h, w))
    assert back.skeleton.sum() == e.skeleton.sum()
    assert back.junction_types == e.junction_types
    for x, y in back.junctions_px.astype(int):  # the turned coords are exact
        assert back.skeleton[y, x]


def test_source_frame_round_trips() -> None:
    """Turning back by 360-r and then by r returns the original extraction."""
    e = _marked_extraction()
    there = extraction_in_source_frame(e, 90)  # back-turn of 270
    again = extraction_in_source_frame(there, 270)  # back-turn of 90 = inverse
    assert np.array_equal(again.skeleton, e.skeleton)
    assert np.allclose(again.junctions_px, e.junctions_px)


# ---------------------------------------------------------------------------
# golden tests: _034 p92 / p114 vs recorded truth (slow: real segmentation)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def centerline_features(centerlines_path: Path) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = json.loads(centerlines_path.read_text())["features"]
    return features


def _sheet_setup(
    fixtures_dir: Path, centerline_features: list[dict[str, Any]], page: str
) -> tuple[Any, Any, Any, Any, float]:
    vol = fixtures_dir / VOLUME
    res = json.loads((vol / "results" / f"p{page}.json").read_text())
    manifest = json.loads((vol / "sheets" / "manifest.json").read_text())
    # recover the small->full ratio per sheet, never assume a fixed value
    small_to_full = 1.0 / manifest[f"p{page}"]["scale"]
    gcps = gcps_from_geojson(res["gcps_geojson"])
    truth = fit_affine(gcps)
    xs = [g[2] for g in gcps]
    ys = [g[3] for g in gcps]
    bounds = (min(xs) - 3000, min(ys) - 3000, max(xs) + 3000, max(ys) + 3000)
    world = world_from_centerlines(centerline_features, bounds)
    extraction = extract_junctions(vol / "sheets" / f"p{page}_small.jpg")
    return extraction, truth, world, small_to_full, float(np.mean(ys))


def _offset_east(m: np.ndarray, ground_m: float, y_3857: float) -> np.ndarray:
    """Shift a full-res->3857 affine east by a ground-meter distance."""
    lat = math.degrees(math.atan(math.sinh(y_3857 / _WGS84_RADIUS_M)))
    out = m.copy()
    out[0, 0] += ground_m / math.cos(math.radians(lat))
    return out


@pytest.mark.golden
def test_p92_truth_supported_and_wrong_prior_scores_lower(
    fixtures_dir: Path, centerline_features: list[dict[str, Any]]
) -> None:
    extraction, truth, world, small_to_full, y_mid = _sheet_setup(
        fixtures_dir, centerline_features, "92"
    )
    # (a) extraction finds >= 4 junctions (measured: 11, 3 typed X)
    assert extraction.n_junctions >= 4

    # (b) the TRUE placement is supported (ratio 2.06, matches the recorded run)
    v_true = verify_placement(extraction, truth, world, small_to_full=small_to_full)
    assert v_true.supports
    assert v_true.separation_ratio >= 1.5
    assert v_true.best_offset_m <= 50.0  # truth is the argmax of its own window

    # (c) a 300 m offset prior scores clearly lower (measured: 0.504 vs 0.875).
    # NOTE: the wrong window can still contain a grid-aliased local peak
    # (the one-block grid-aliasing failure), so the robust signal is the score
    # drop — not the verdict, which at best ABSTAINS here (the channel supports
    # or abstains and never refutes; `supports=False` is unreachable).
    v_wrong = verify_placement(
        extraction,
        _offset_east(truth, 300.0, y_mid),
        world,
        small_to_full=small_to_full,
    )
    assert v_wrong.score_at_prior < 0.75 * v_true.score_at_prior


@pytest.mark.golden
@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotated_small_composes_back_to_the_same_verdict(
    fixtures_dir: Path, centerline_features: list[dict[str, Any]], tmp_path: Path, rotation: int
) -> None:
    """A prep-normalized (upright) small verifies like the source-frame one.

    p92's recorded small IS the source frame. Synthesize what prep would have written had the
    scan needed a clockwise correction — the same pixels, turned — and score the SAME truth
    affine. Composed, truth is supported at every quarter turn with real separation.
    UNCOMPOSED, in the negative control below, the score collapses and at 90 deg it still
    reports supports=True while sitting tens of metres off truth. The invariant is the VERDICT,
    not the score, since extraction is not bit-equivariant under a quarter turn.
    """
    import cv2

    extraction, truth, world, small_to_full, _ = _sheet_setup(
        fixtures_dir, centerline_features, "92"
    )
    v_source = verify_placement(extraction, truth, world, small_to_full=small_to_full)

    gray = cv2.imread(str(fixtures_dir / VOLUME / "sheets" / "p92_small.jpg"), cv2.IMREAD_GRAYSCALE)
    assert gray is not None
    upright = np.rot90(gray, k=-(rotation // 90))  # prep's clockwise correction
    upright_path = tmp_path / "p92_small.png"  # PNG: no re-encode noise
    cv2.imwrite(str(upright_path), upright)
    raw = extract_junctions(upright_path)

    v_rot = verify_placement(
        extraction_in_source_frame(raw, rotation), truth, world, small_to_full=small_to_full
    )
    assert v_source.supports is True
    assert v_rot.supports is True
    assert v_rot.best_offset_m <= 15.0  # truth is the argmax of its own window
    assert v_rot.separation_ratio >= 1.5  # measured 1.9-2.2; the accept margin is 1.10

    # negative control: the composition is load-bearing, not decoration —
    # the same junctions left in the upright frame score barely half as well
    v_uncomposed = verify_placement(raw, truth, world, small_to_full=small_to_full)
    assert v_uncomposed.score_at_prior < 0.75 * v_rot.score_at_prior


@pytest.mark.golden
def test_p114_pure_grid_truth_supported(
    fixtures_dir: Path, centerline_features: list[dict[str, Any]]
) -> None:
    extraction, truth, world, small_to_full, _ = _sheet_setup(
        fixtures_dir, centerline_features, "114"
    )
    assert extraction.n_junctions >= 4  # measured: 7
    v_true = verify_placement(extraction, truth, world, small_to_full=small_to_full)
    assert v_true.supports
    assert v_true.separation_ratio >= 1.5  # measured: 2.12


@pytest.mark.golden
def test_the_channel_supports_or_abstains_and_never_refutes(
    fixtures_dir: Path, centerline_features: list[dict[str, Any]]
) -> None:
    """CONTRACT: ``supports`` is True or None. Never False.

    A refute is the sole blocker in ``verified_accept``, and the old binary rule emitted one for
    anything its support clause did not catch. Measured, that veto never changed an outcome
    while falsely vetoing a fifth of the correctly placed human sheets, so the channel abstains
    where it used to refute. Sweeping the truth affine east across the whole prior window
    exercises both branches: near truth the support clause fires, far from it there is no
    evidence — and "no evidence" must read as None, not as evidence against.
    """
    extraction, truth, world, small_to_full, y_mid = _sheet_setup(
        fixtures_dir, centerline_features, "92"
    )
    seen: set[bool | None] = set()
    for offset_m in (0.0, 60.0, 90.0, 150.0, 300.0, 600.0):
        v = verify_placement(
            extraction,
            _offset_east(truth, offset_m, y_mid),
            world,
            small_to_full=small_to_full,
        )
        assert v.supports is not False, (
            f"junction channel refuted at {offset_m:.0f} m east — this channel "
            f"may only support or abstain (verdict: {v})"
        )
        seen.add(v.supports)
    assert seen == {True, None}, (
        f"the sweep must exercise BOTH branches, else it proves nothing: got {seen}"
    )
