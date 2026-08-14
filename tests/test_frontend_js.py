"""The frontend's own JavaScript, executed rather than grepped.

1. ``review_ui/affine.js`` against the Python that owns the same computation —
   :func:`compose_ops`, :func:`apply_affine`, :func:`invert_affine` and
   pyproj's EPSG:3857. The JS is display only, which is why a divergence is
   dangerous: the ghost lands where the JS says and the server writes
   something else.
2. ``affine.js``'s pin fit, against constructed transforms rather than against
   itself: the answer is known before the code runs.
3. ``viewer/lib.js``, which holds the viewer's decisions apart from the DOM. A
   literal-source assertion fails on a refactor and passes on a bug.

All need ``node`` on PATH; without it these skip.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autogeoref.affine import TO_3857, TO_4326, apply_affine, invert_affine
from autogeoref.errors import ReviewError
from autogeoref.review.materialize import MIN_PLACEMENT_SCALE_M_PER_PX, compose_ops
from js_support import REVIEW_AFFINE, queue, review, run_js, viewer

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
PARITY_CASES = DATA / "affine_parity_cases.json"

#: Both sides are IEEE doubles running the same formula, so this is round-off
#: room and not a fudge: 4e6 m from the origin, one ulp is about 1e-9 m.
PARITY_TOLERANCE_M = 1e-6

#: The characters the page actually renders in an era label — a merged decade
#: run is joined with an en dash, separate runs with a middle dot. Written as
#: escapes so an ASCII stand-in cannot creep into the expected value.
EN_DASH = "\u2013"
MIDDLE_DOT = " \u00b7 "


def js(*values: Any) -> str:
    """``values`` as a comma-joined JS argument list."""
    return ", ".join(json.dumps(v) for v in values)


def _parity_cases() -> dict[str, Any]:
    return json.loads(PARITY_CASES.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# review_ui/affine.js against review/materialize.py
# ---------------------------------------------------------------------------


def test_parity_fixture_covers_every_op_type_and_a_composed_chain() -> None:
    """The fixture is this file's reach; a case list that quietly lost an op
    type would stay green with the divergence live."""
    cases = _parity_cases()["cases"]
    kinds = {op["type"] for case in cases for op in case["ops"]}
    assert kinds == {"translate", "scale", "rotate"}
    assert any(len({op["type"] for op in case["ops"]}) == 3 for case in cases)
    # a rotation about a centre far from the origin, where an order-of-
    # composition error is largest and a near-origin case would pass
    assert any(
        op["type"] == "rotate" and math.hypot(*op["center_3857"]) > 1e6
        for case in cases
        for op in case["ops"]
    )


@pytest.mark.parametrize("case_index", range(len(_parity_cases()["cases"])))
def test_review_ui_affine_matches_compose_ops(case_index: int) -> None:
    fixture = _parity_cases()
    case = fixture["cases"][case_index]
    base, points = fixture["base"], fixture["points"]

    expected = compose_ops(base, case["ops"])
    got = run_js(REVIEW_AFFINE, f"L.composeOps({json.dumps(base)}, {json.dumps(case['ops'])})")
    for row_at, (row_expected, row_got) in enumerate(zip(expected, got, strict=True)):
        for col_at, (want, have) in enumerate(zip(row_expected, row_got, strict=True)):
            assert have == pytest.approx(float(want), abs=1e-9, rel=1e-12), (
                f"{case['name']}: affine[{row_at}][{col_at}] diverged"
            )

    # the matrix agreeing is the mechanism; where a pixel lands is the promise
    placed = run_js(
        REVIEW_AFFINE,
        f"{json.dumps(points)}.map(p => L.applyAff({json.dumps(got)}, p[0], p[1]))",
    )
    for (px, py), (x_got, y_got) in zip(points, placed, strict=True):
        x_want = expected[0][0] + expected[0][1] * px + expected[0][2] * py
        y_want = expected[1][0] + expected[1][1] * px + expected[1][2] * py
        assert x_got == pytest.approx(float(x_want), abs=PARITY_TOLERANCE_M)
        assert y_got == pytest.approx(float(y_want), abs=PARITY_TOLERANCE_M)


def test_the_python_side_is_the_validator_and_the_js_is_not() -> None:
    """Recorded so the parity above is not read as `the two are
    interchangeable`: the JS composes anything handed to it, and what refuses a
    degenerate op is the server, on the op log, at save."""
    bad = [{"type": "scale", "factor": 0, "center_3857": [0, 0]}]
    with pytest.raises(ReviewError):
        compose_ops(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), bad)
    assert run_js(REVIEW_AFFINE, f"L.composeOps([[0,1,0],[0,0,1]], {json.dumps(bad)})") == [
        [0, 0, 0],
        [0, 0, 0],
    ]


# ---------------------------------------------------------------------------
# affine.js's other three mirrors of Python: applyAff, toMerc/toLngLat, invAff
# ---------------------------------------------------------------------------

#: A placement at Chicago's scale and distance from the 3857 origin: ~1 m/px,
#: rotated a few degrees, ~9.7e6 m out. Round-off is worst this far from zero,
#: and a formula error is easiest to see under a rotation.
PLACEMENT = [
    [-9757400.0, 0.9962, -0.0872],
    [5138900.0, 0.0872, 0.9962],
]

#: Corners and a middle, in the full-resolution pixel frame a result uses.
SHEET_PIXELS = [(0.0, 0.0), (7000.0, 0.0), (7000.0, 5600.0), (0.0, 5600.0), (3123.7, 2011.4)]

#: Spread across the datum where a spherical/ellipsoidal mismatch is largest:
#: at 52 N the two mercators differ by tens of kilometres, and at the equator
#: not at all — so a case list without a high latitude would pass on the bug.
LNGLAT_CASES = [
    (-87.6298, 41.8781),
    (0.0, 0.0),
    (13.4050, 52.5200),
    (151.2093, -33.8688),
    (-122.4194, 37.7749),
    (-87.9, 42.05),
]


def test_apply_aff_places_a_pixel_where_apply_affine_does() -> None:
    """The affine agreeing matrix-for-matrix is the mechanism; this is the
    promise, and until now it was only ever exercised through a composed
    chain."""
    m = np.array(PLACEMENT, dtype=float)
    for px, py in SHEET_PIXELS:
        want = apply_affine(m, px, py)
        got = review(f"L.applyAff({js(PLACEMENT, px, py)})")
        assert got[0] == pytest.approx(want[0], abs=PARITY_TOLERANCE_M)
        assert got[1] == pytest.approx(want[1], abs=PARITY_TOLERANCE_M)


def test_to_merc_is_the_same_projection_pyproj_uses() -> None:
    """Every ghost corner, every pin and every mask vertex passes through this
    pair, and the file's own comment claimed the match with nothing checking
    it. A spherical-vs-ellipsoidal mismatch would not look like a bug on
    screen — it is smooth and latitude-dependent, so it would look like a
    slightly worse placement."""
    for lng, lat in LNGLAT_CASES:
        want_x, want_y = TO_3857.transform(lng, lat)
        got_x, got_y = review(f"L.toMerc({js(lng, lat)})")
        assert got_x == pytest.approx(want_x, abs=PARITY_TOLERANCE_M)
        assert got_y == pytest.approx(want_y, abs=PARITY_TOLERANCE_M)


def test_to_lng_lat_is_the_same_inverse_pyproj_uses() -> None:
    for lng, lat in LNGLAT_CASES:
        x, y = TO_3857.transform(lng, lat)
        want_lng, want_lat = TO_4326.transform(x, y)
        got_lng, got_lat = review(f"L.toLngLat({js(x, y)})")
        # a degree of latitude is ~111 km, so 1e-11 deg is well under a micron
        assert got_lng == pytest.approx(want_lng, abs=1e-11)
        assert got_lat == pytest.approx(want_lat, abs=1e-11)


def test_the_two_projections_are_each_other_s_inverse() -> None:
    """The oracle that needs no second implementation: a wrong-but-consistent
    pair cannot satisfy it."""
    for lng, lat in LNGLAT_CASES:
        back = review(f"L.toLngLat(...L.toMerc({js(lng, lat)}))")
        assert back[0] == pytest.approx(lng, abs=1e-11)
        assert back[1] == pytest.approx(lat, abs=1e-11)


def test_inv_aff_inverts_the_placement_python_inverts() -> None:
    """``invAff`` is a coefficient bag rather than a 2x3, so compare it where
    both conventions mean the same thing: ``[px, py] = Minv @ [1, X, Y]``."""
    want = invert_affine(np.array(PLACEMENT, dtype=float))
    got = review(f"L.invAff({js(PLACEMENT)})")
    linear = {"ia": want[0][1], "ib": want[0][2], "ic": want[1][1], "id": want[1][2]}
    for name, expected in linear.items():
        assert got[name] == pytest.approx(float(expected), rel=1e-12)
    # the JS carries the world origin instead of a folded constant term
    assert -(got["ia"] * got["x0"] + got["ib"] * got["y0"]) == pytest.approx(
        float(want[0][0]), rel=1e-9
    )
    assert -(got["ic"] * got["x0"] + got["id"] * got["y0"]) == pytest.approx(
        float(want[1][0]), rel=1e-9
    )


def test_a_world_point_round_trips_back_to_its_own_pixel() -> None:
    """``pxOf(invAff(m), applyAff(m, px, py))`` is the identity to the 0.1 px
    the function rounds to — a mask vertex dragged and re-read must not creep.
    """
    for px, py in SHEET_PIXELS:
        back = review(f"L.pxOf(L.invAff({js(PLACEMENT)}), L.applyAff({js(PLACEMENT, px, py)}))")
        assert back == [round(px, 1), round(py, 1)]


def test_inv_aff_divides_by_a_determinant_the_server_is_what_checks() -> None:
    """``invAff`` never checks the determinant, and a collapsed placement is
    real: a rescue whose anchors share one street has spread pixels and
    coincident world points. What stands between the two is
    ``displayable_affine`` — an affine under ``MIN_PLACEMENT_SCALE_M_PER_PX``
    is never handed to the page, because here it produces pixels rather than
    an error."""
    collapsed = [[100.0, 1e-9, 0.0], [200.0, 0.0, 1e-9]]
    assert math.sqrt(abs(1e-9 * 1e-9)) < MIN_PLACEMENT_SCALE_M_PER_PX
    singular = [[100.0, 1.0, 2.0], [200.0, 2.0, 4.0]]
    with pytest.raises(np.linalg.LinAlgError):
        invert_affine(np.array(singular, dtype=float))
    # JSON has no Infinity: node prints it as null, which is the point — the JS
    # returns a number-shaped nothing where Python raises
    assert review(f"L.pxOf(L.invAff({js(singular)}), [0, 0])") == [None, None]
    huge = review(f"L.pxOf(L.invAff({js(collapsed)}), [1, 1])")
    assert abs(huge[0]) > 1e6


# ---------------------------------------------------------------------------
# affine.js's pin fit — constructed transforms, not pinned outputs
# ---------------------------------------------------------------------------

#: Pin sources at Chicago's distance from the 3857 origin, spread over roughly
#: a sheet: a rotation about a centroid this far out is where an
#: order-of-composition error is largest.
PIN_SOURCE = [
    [-9757400.0, 5138900.0],
    [-9756900.0, 5138950.0],
    [-9757100.0, 5139400.0],
    [-9756800.0, 5139300.0],
]


def similarity_matrix(scale: float, deg: float, tx: float, ty: float) -> list[list[float]]:
    """A 2x3 ``[X,Y] = M @ [1,x,y]`` for scale-then-rotate-then-translate."""
    r = math.radians(deg)
    a, b = scale * math.cos(r), scale * math.sin(r)
    return [[tx, a, -b], [ty, b, a]]


def through(m: list[list[float]], points: list[list[float]]) -> list[list[float]]:
    arr = np.array(m, dtype=float)
    return [list(apply_affine(arr, x, y)) for x, y in points]


def fit(source: list[list[float]], world: list[list[float]]) -> Any:
    return review(f"L.similarityOps({js(source, world)})")


def fitted_matrix(result: Any) -> list[list[float]]:
    """The ops the fit returned, composed back into one affine."""
    return review(f"L.composeOps(L.IDENTITY, {js(result['ops'])})")


def sum_squared_residual(m: list[list[float]], source: Any, world: Any) -> float:
    placed = through(m, source)
    return sum((p[0] - w[0]) ** 2 + (p[1] - w[1]) ** 2 for p, w in zip(placed, world, strict=True))


def perturbed(
    m: list[list[float]], source: Any, d_scale: float, d_deg: float, dx: float, dy: float
) -> list[list[float]]:
    """``m`` nudged inside the similarity family, ABOUT THE PINS.

    About the pins and not about the 3857 origin: the sheet sits ~1e7 m out, so
    a 0.05% scale change about the origin moves it kilometres and every
    perturbation would look worse whatever the fit did.
    """
    placed = through(m, source)
    cx = sum(p[0] for p in placed) / len(placed)
    cy = sum(p[1] for p in placed) / len(placed)
    r = math.radians(d_deg)
    a, b = d_scale * math.cos(r), d_scale * math.sin(r)
    nudge = np.array(
        [[cx - a * cx + b * cy + dx, a, -b], [cy - b * cx - a * cy + dy, b, a]], dtype=float
    )
    base = np.array(m, dtype=float)
    linear = nudge[:, 1:3] @ base[:, 1:3]
    constant = nudge[:, 1:3] @ base[:, 0] + nudge[:, 0]
    return [[float(constant[i]), float(linear[i][0]), float(linear[i][1])] for i in (0, 1)]


@pytest.mark.parametrize(
    ("scale", "deg", "tx", "ty"),
    [
        (1.0, 0.0, 250.0, -180.0),  # pure move
        (1.0, 12.5, 0.0, 0.0),  # pure turn, about a centre 1.1e7 m from the origin
        (0.978, -3.25, 40.0, 12.0),  # shrink, turn and move together
        (1.14, 175.0, -900.0, 400.0),  # near half a turn, where a sign error hides
    ],
)
def test_the_fit_recovers_a_transform_it_was_never_told(
    scale: float, deg: float, tx: float, ty: float
) -> None:
    """The answer is known before the code runs: map a pin set through a
    similarity, hand the fitter both ends, and demand the ops it returns
    compose to that same similarity."""
    want = similarity_matrix(scale, deg, tx, ty)
    world = through(want, PIN_SOURCE)
    result = fit(PIN_SOURCE, world)
    assert "error" not in result, result
    got = fitted_matrix(result)
    # Compared where the sheet is, not coefficient by coefficient: the ops are
    # centred on the pins, so the composed constant term absorbs a centroid
    # ~1e7 m out and carries that magnitude's round-off (~1e-5 m). A tenth of a
    # millimetre is still four orders under anything a placement can express.
    probes = [*PIN_SOURCE, [-9760000.0, 5142000.0], [-9750000.0, 5130000.0]]
    for expected, actual in zip(through(want, probes), through(got, probes), strict=True):
        assert actual[0] == pytest.approx(expected[0], abs=1e-4)
        assert actual[1] == pytest.approx(expected[1], abs=1e-4)
    assert result["maxResidualM"] == pytest.approx(0.0, abs=1e-6)


def test_the_ops_are_the_vocabulary_the_server_recomputes_from() -> None:
    """A fit the server cannot replay is a ghost the operator approves and the
    pipeline then places somewhere else."""
    world = through(similarity_matrix(1.03, 7.0, 60.0, -25.0), PIN_SOURCE)
    ops = fit(PIN_SOURCE, world)["ops"]
    assert [op["type"] for op in ops] == ["scale", "rotate", "translate"]
    # the composition the server performs, over the same ops
    replayed = compose_ops(np.array(PLACEMENT, dtype=float), ops)
    mirrored = review(f"L.composeOps({js(PLACEMENT, ops)})")
    for row_want, row_got in zip(replayed, mirrored, strict=True):
        for expected, actual in zip(row_want, row_got, strict=True):
            assert actual == pytest.approx(float(expected), abs=1e-9, rel=1e-12)


def test_the_fit_is_least_squares_and_not_merely_a_fit() -> None:
    """With noise on the targets, no perturbation of the answer inside the
    similarity family may lower the residual. That is what `least squares`
    means, and nothing checked it — a fit that merely landed near the pins
    would pass a construction test on clean data."""
    truth = similarity_matrix(1.02, 4.0, 120.0, -60.0)
    noise = [[3.1, -2.4], [-1.8, 4.2], [2.6, 1.1], [-3.9, -0.7]]
    world = [
        [p[0] + n[0], p[1] + n[1]] for p, n in zip(through(truth, PIN_SOURCE), noise, strict=True)
    ]
    result = fit(PIN_SOURCE, world)
    got = fitted_matrix(result)
    best = sum_squared_residual(got, PIN_SOURCE, world)
    assert best > 0  # noise was added; a zero here would mean the test is vacuous

    # each nudge is worth a few tenths of a metre over a pin set ~500 m wide,
    # the same order as the noise — small enough to be local, big enough that a
    # fit sitting off the optimum is beaten by the one that steps toward it
    for d_scale, d_deg, dx, dy in [
        (1.0005, 0.0, 0.0, 0.0),
        (0.9995, 0.0, 0.0, 0.0),
        (1.0, 0.02, 0.0, 0.0),
        (1.0, -0.02, 0.0, 0.0),
        (1.0, 0.0, 0.5, 0.0),
        (1.0, 0.0, -0.5, 0.0),
        (1.0, 0.0, 0.0, 0.5),
        (1.0, 0.0, 0.0, -0.5),
    ]:
        nudge = (d_scale, d_deg, dx, dy)
        worse = sum_squared_residual(perturbed(got, PIN_SOURCE, *nudge), PIN_SOURCE, world)
        assert worse >= best, f"perturbation {nudge} beat the fit"


def test_one_pair_moves_the_sheet_and_does_not_resize_it() -> None:
    """A single pin says where a point goes and nothing about scale or angle;
    fitting either from it would be inventing evidence."""
    result = fit([[-9757400.0, 5138900.0]], [[-9757350.0, 5138880.0]])
    assert result["ops"] == [{"type": "translate", "dx_m": 50.0, "dy_m": -20.0}]
    assert result["maxResidualM"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("name", "source", "world"),
    [
        (
            "every pin on one spot",
            [[-9757400.0, 5138900.0]] * 3,
            [[-9757300.0, 5138800.0], [-9757310.0, 5138790.0], [-9757290.0, 5138810.0]],
        ),
        (
            "pins a millimetre apart",
            [[-9757400.0, 5138900.0], [-9757400.0004, 5138900.0]],
            [[-9757300.0, 5138800.0], [-9757250.0, 5138850.0]],
        ),
        (
            "every target on one spot",
            PIN_SOURCE,
            [[-9757300.0, 5138800.0]] * 4,
        ),
    ],
)
def test_a_pin_set_with_nothing_to_fit_is_refused(
    name: str, source: list[list[float]], world: list[list[float]]
) -> None:
    """Refused, not fitted. The returned object carries no ``ops`` at all,
    which is what makes the caller's op log unchangeable on this path rather
    than unchanged by convention."""
    result = fit(source, world)
    assert "ops" not in result, name
    assert "error" in result, name


def test_a_distinct_pair_is_fitted_however_collinear_it_is() -> None:
    """Recorded because `collinear` reads as degenerate and here it is not: a
    similarity has four degrees of freedom and two points give four equations,
    so refusing one would reject a legal placement."""
    source = [[-9757400.0, 5138900.0], [-9757000.0, 5138900.0], [-9756600.0, 5138900.0]]
    world = through(similarity_matrix(1.05, 9.0, 30.0, -15.0), source)
    result = fit(source, world)
    assert result["maxResidualM"] == pytest.approx(0.0, abs=1e-6)


def test_a_mirrored_pin_set_is_not_fitted_with_a_reflection() -> None:
    """The comment promises no reflection and the maths does not obviously
    enforce it. A mirrored sheet warps back-to-front, its labels reading
    backwards, and ``gdalwarp`` fills the reflected frame with opaque black —
    so the fit has to land badly instead, which is visible."""
    centre = [
        sum(p[0] for p in PIN_SOURCE) / len(PIN_SOURCE),
        sum(p[1] for p in PIN_SOURCE) / len(PIN_SOURCE),
    ]
    world = [[2 * centre[0] - x, y] for x, y in PIN_SOURCE]
    result = fit(PIN_SOURCE, world)
    got = fitted_matrix(result)
    det = got[0][1] * got[1][2] - got[0][2] * got[1][1]
    assert det > 0, "the fit produced a reflection"
    for op in result["ops"]:
        if op["type"] == "scale":
            assert op["factor"] > 0
    # a reflection would land every pin exactly, so the honest bar is that this
    # one lands nowhere near: hundreds of metres out on a pin set 350 m wide
    spread = max(abs(p[0] - centre[0]) for p in PIN_SOURCE)
    assert result["maxResidualM"] > spread / 2, "a reflection was reproduced after all"


# ---------------------------------------------------------------------------
# affine.js's op log: coalescing, and the copy an undo restores from
# ---------------------------------------------------------------------------

CENTRE = [-9757400.0, 5138900.0]


def test_consecutive_nudges_of_the_same_kind_become_one_entry() -> None:
    """An arrow key is a 1 m translate. Sixty of them are one move, and a log
    with sixty entries in it is unreadable — and it is what the server replays.
    """
    log = [{"type": "translate", "dx_m": 1.0, "dy_m": 0.0}]
    merged = review(f"L.coalesce({js(log)}, {js({'type': 'translate', 'dx_m': 2, 'dy_m': -3})})")
    assert merged == [{"type": "translate", "dx_m": 3.0, "dy_m": -3.0}]

    turns = [{"type": "rotate", "deg": 0.2, "center_3857": CENTRE}]
    spun = review(
        f"L.coalesce({js(turns)}, {js({'type': 'rotate', 'deg': 1, 'center_3857': CENTRE})})"
    )
    assert spun == [{"type": "rotate", "deg": 1.2, "center_3857": CENTRE}]


def test_edits_that_are_not_the_same_edit_stay_apart() -> None:
    """The same angle about a different centre is a different placement, and a
    merge would silently double the first turn about the wrong point."""
    turns = [{"type": "rotate", "deg": 0.2, "center_3857": CENTRE}]
    elsewhere = {"type": "rotate", "deg": 0.2, "center_3857": [CENTRE[0] + 400, CENTRE[1]]}
    assert len(review(f"L.coalesce({js(turns)}, {js(elsewhere)})")) == 2
    # a different kind of op never merges either, nor does the first op of all
    move = {"type": "translate", "dx_m": 1.0, "dy_m": 0.0}
    assert len(review(f"L.coalesce({js(turns)}, {js(move)})")) == 2
    assert review(f"L.coalesce([], {js(move)})") == [move]
    scale = [{"type": "scale", "factor": 1.02, "center_3857": CENTRE}]
    assert len(review(f"L.coalesce({js(scale)}, {js(scale[0])})")) == 2


def test_coalescing_leaves_the_log_it_was_given_alone() -> None:
    """The undo snapshot is taken BEFORE the merge, so a merge that wrote into
    the previous op in place would rewrite the history it was meant to restore.
    """
    log = [{"type": "translate", "dx_m": 1.0, "dy_m": 0.0}]
    after = review(
        "(() => {"
        f"const log = {js(log)};"
        f"L.coalesce(log, {js({'type': 'translate', 'dx_m': 5, 'dy_m': 5})});"
        "return log; })()"
    )
    assert after == log


def test_an_undo_snapshot_shares_nothing_with_the_live_edits() -> None:
    """The classic defect: a shallow copy, and a later edit rewrites the
    history it was supposed to be able to return to. Every nested piece counts
    — a rotate's centre is an array inside an op."""
    state = {
        "ops": [{"type": "rotate", "deg": 0.2, "center_3857": CENTRE}],
        "maskPx": [[10.0, 20.0], [30.0, 40.0]],
        "maskDirty": False,
        "pins": [{"px": 1.0, "py": 2.0, "w": [3.0, 4.0]}],
    }
    kept = review(
        "(() => {"
        f"const live = {js(state)};"
        "const snap = L.cloneEdits(live);"
        "live.ops[0].deg = 99; live.ops[0].center_3857[0] = 0;"
        "live.maskPx[0][0] = 99; live.pins[0].w[0] = 99; live.pins[0].px = 99;"
        "live.ops.push({type: 'translate', dx_m: 1, dy_m: 1});"
        "return snap; })()"
    )
    assert kept == state


# ---------------------------------------------------------------------------
# viewer/lib.js — the decisions, run
# ---------------------------------------------------------------------------


def test_adjacent_decades_merge_into_one_range() -> None:
    """Five chips have to read as a span, not a list — and a gap has to keep
    its separator, or the label claims coverage the atlas does not have."""
    assert viewer("L.selectionLabel(['1890s', '1900s', '1910s'])") == f"1890s{EN_DASH}1910s"
    assert viewer("L.selectionLabel(['1890s', '1910s'])") == f"1890s{MIDDLE_DOT}1910s"
    assert viewer("L.selectionLabel(['1900s'])") == "1900s"
    assert viewer("L.selectionLabel([])") == ""
    # unsorted in, ascending out
    assert viewer("L.selectionLabel(['1910s', '1890s', '1900s'])") == f"1890s{EN_DASH}1910s"
    two_runs = viewer("L.selectionLabel(['1890s', '1900s', '1930s', '1940s'])")
    assert two_runs == f"1890s{EN_DASH}1900s{MIDDLE_DOT}1930s{EN_DASH}1940s"


def test_a_dated_era_label_never_merges_into_a_decade_run() -> None:
    """`1906` is one survey, not a decade: merging it would print a range no
    chip can turn off."""
    assert viewer("L.selectionLabel(['1900s', '1906'])") == f"1900s{MIDDLE_DOT}1906"


def test_era_chips_sort_newest_first_by_leading_year() -> None:
    """Era labels are display text, so the order comes from the leading year
    and not from numeric-string coercion."""
    assert viewer("['1890s', '1950', '1910s'].sort(L.compareErasNewestFirst)") == [
        "1950",
        "1910s",
        "1890s",
    ]


def test_region_bands_pick_the_first_match_per_axis_then_collapse() -> None:
    labels = json.dumps(
        {
            "lat": [{"above": 10, "label": "North"}, {"below": 10, "label": "South"}],
            "lng": [{"below": 20, "label": "west"}, {"above": 20, "label": "east"}],
            "combine": "{lat} {lng}",
            "collapse": [["North west", "Northwest"]],
        }
    )
    assert viewer(f"L.regionLabel([0, 12, 4, 16], {labels}, null)") == "Northwest"
    assert viewer(f"L.regionLabel([30, 0, 34, 4], {labels}, null)") == "South east"


def test_an_unbanded_district_falls_back_to_a_compass_name_then_to_a_generic() -> None:
    assert viewer("L.regionLabel([0, 12, 4, 16], null, [10, 10])") == "Northwest district"
    assert viewer("L.regionLabel([0, 12, 4, 16], null, null)") == "District"


def test_stacking_is_chronological_across_and_within_eras() -> None:
    """Newest survey on top, so the map reads as the latest selected snapshot
    of each block and turning an era off reveals the older sheets beneath. A
    volume with no year of its own sorts at its era's year."""
    added = (
        "new Map(["
        '["v-1912", {era: "1910s", year: 1912}],'
        '["v-1890", {era: "1890s", year: 1890}],'
        '["v-noyear", {era: "1910s", year: null}],'
        '["v-1918", {era: "1910s", year: 1918}],'
        "])"
    )
    assert viewer(f"L.stackOrder({added})") == ["v-1890", "v-noyear", "v-1912", "v-1918"]


def test_a_switched_off_district_stays_off_through_its_era() -> None:
    """Two independent switches decide one layer. Hiding is per district, so an
    era coming back on must not quietly restore a district the reader switched
    off — a control that forgets is worse than one that does not."""
    on = "new Set(['1910s'])"
    off = "new Set([])"
    none_hidden = "new Set([])"
    hidden = "new Set(['v-1912'])"
    assert viewer(f"L.layerVisibility('v-1912', '1910s', {on}, {none_hidden})") == "visible"
    assert viewer(f"L.layerVisibility('v-1912', '1910s', {on}, {hidden})") == "none"
    # its era switched off: still none, and the hidden set is untouched by that
    assert viewer(f"L.layerVisibility('v-1912', '1910s', {off}, {hidden})") == "none"
    assert viewer(f"L.layerVisibility('v-1912', '1910s', {off}, {none_hidden})") == "none"
    # an id the hidden set does not name is visible whenever its era is on
    assert viewer(f"L.layerVisibility('v-1918', '1910s', {on}, {hidden})") == "visible"


def test_a_link_cannot_hide_a_district_the_manifest_no_longer_lists() -> None:
    """`off=` names districts by id, and ids outlive links. One that no longer
    exists must be dropped rather than ride every forwarded copy forever."""
    known = "['a', 'b']"
    assert sorted(viewer(f"[...L.hiddenFromLink('#off=a,b', 'off', {known})]")) == ["a", "b"]
    assert viewer(f"[...L.hiddenFromLink('#off=a,gone', 'off', {known})]") == ["a"]
    assert viewer(f"[...L.hiddenFromLink('#off=gone', 'off', {known})]") == []
    # absent and empty both mean the link says nothing about hiding
    assert viewer(f"[...L.hiddenFromLink('#at=1,2,3', 'off', {known})]") == []
    assert viewer(f"[...L.hiddenFromLink('#off=', 'off', {known})]") == []


def test_a_configured_basemap_is_used_on_every_host() -> None:
    vector = {"type": "vector", "styles": {"atlas": "vendor/basemap/style-muted.json"}}
    assert viewer(f"L.chooseBasemap({js(vector)}, 'atlas.example.com', 'atlas')") == {
        "kind": "vector",
        "styleHref": "vendor/basemap/style-muted.json",
    }
    raster = {"type": "raster", "tiles": "https://tiles.example.com/{z}/{x}/{y}.png"}
    assert viewer(f"L.chooseBasemap({js(raster)}, 'atlas.example.com', 'now')") == {
        "kind": "raster",
        "tiles": "https://tiles.example.com/{z}/{x}/{y}.png",
        "dev": False,
    }


def test_a_public_host_with_no_basemap_config_renders_bare_rather_than_borrowing() -> None:
    """The failure this guards is not cosmetic: a third-party raster basemap
    forbids this traffic shape, and what a visitor gets is a referer block and
    a blank map. The convenience default has to be local-only, and the bare
    style has to say why."""
    public = viewer("L.chooseBasemap({}, 'atlas.example.com', 'atlas')")
    assert public["kind"] == "bare"
    assert "no site.basemap" in public["reason"]
    for host in ("localhost", "127.0.0.1", "[::1]", ""):
        local = viewer(f"L.chooseBasemap({{}}, {json.dumps(host)}, 'atlas')")
        assert local["kind"] == "raster" and local["dev"] is True, host
        assert "openstreetmap.org" in local["tiles"]


def test_a_vector_basemap_with_no_style_for_this_pane_is_bare_and_names_the_pane() -> None:
    """Both panes ask for their own flavour, and a config that ships one is a
    half-drawn comparison — which reads as a rendering bug rather than a
    missing file unless the message says which pane went without."""
    one_pane = {"type": "vector", "styles": {"atlas": "vendor/basemap/style-muted.json"}}
    got = viewer(f"L.chooseBasemap({js(one_pane)}, 'atlas.example.com', 'now')")
    assert got["kind"] == "bare"
    assert "now" in got["reason"]


#: Three districts: a citywide sheet, a district inside it, and one that does
#: not contain the home point at all.
DISTRICTS = [
    {"id": "citywide", "bounds": [-1.0, -1.0, 1.0, 1.0]},
    {"id": "inner", "bounds": [-0.2, -0.2, 0.2, 0.2]},
    {"id": "elsewhere", "bounds": [5.0, 5.0, 5.5, 5.6]},
]


def test_the_union_of_several_districts_encloses_all_of_them() -> None:
    assert viewer(f"L.unionOf({js(DISTRICTS)})") == [-1.0, -1.0, 5.5, 5.6]
    assert viewer(f"L.unionOf({js(DISTRICTS[1:2])})") == [-0.2, -0.2, 0.2, 0.2]


def test_the_union_of_nothing_is_not_a_box() -> None:
    """The reduce seeds with an INVERTED box, so an empty list would hand
    `fitBounds` a west edge east of its east edge — a camera nobody asked
    for, from a manifest that simply has no volumes yet."""
    assert viewer("L.unionOf([])") is None


def test_the_atlas_opens_on_the_smallest_district_covering_the_home_point() -> None:
    """Both the citywide sheet and the block sheet contain the point, and only
    the block sheet shows anything — so `smallest covering` is the rule, and
    `largest` is only the fallback when nothing covers it at all."""
    assert viewer(f"L.startVolume({js(DISTRICTS)}, [0, 0])")["id"] == "inner"
    # nothing covers this point: the largest district on screen wins
    assert viewer(f"L.startVolume({js(DISTRICTS)}, [90, 80])")["id"] == "citywide"
    # no home point configured at all
    assert viewer(f"L.startVolume({js(DISTRICTS)}, null)")["id"] == "citywide"
    assert viewer("L.startVolume([], [0, 0])") is None


def test_a_query_that_already_names_the_city_is_not_given_the_suffix_twice() -> None:
    """The suffix is configuration and so is the token matched against it —
    taken from the suffix rather than written down here, which is the whole
    generalization contract in one function."""
    suffix = ", Springfield, IL"
    assert viewer(f"L.withCitySuffix('12 Main St', {json.dumps(suffix)})") == (
        "12 Main St, Springfield, IL"
    )
    assert viewer(f"L.withCitySuffix('12 Main St, springfield', {json.dumps(suffix)})") == (
        "12 Main St, springfield"
    )
    # no suffix configured: the query is geocoded exactly as typed
    assert viewer("L.withCitySuffix('12 Main St', '')") == "12 Main St"


def test_a_link_that_says_nothing_reads_as_nothing_and_not_as_zero() -> None:
    """`Number(null)` is 0, and 0 is a legal swipe fraction — so a key the link
    does not carry, and one it carries empty, both have to come back null or a
    plain visit opens with the divider hard left. That shipped once."""
    assert viewer("L.linkNumbers('#at=-1.5,2.5,12.25', 'at', 3)") == [-1.5, 2.5, 12.25]
    assert viewer("L.linkNumbers('#swipe', 'swipe', 1)") is None  # present but empty
    assert viewer("L.linkNumbers('#panel=open', 'swipe', 1)") is None  # absent
    # the wrong shape is nothing too, rather than a partly-applied camera
    assert viewer("L.linkNumbers('#at=1,2', 'at', 3)") is None
    assert viewer("L.linkNumbers('#at=1,2,here', 'at', 3)") is None
    assert viewer("L.linkText('#story=fair', 'story')") == "fair"
    assert viewer("L.linkText('#story', 'story')") is None
    assert viewer("L.linkText('#panel=open', 'story')") is None


def test_a_forwarded_link_carries_a_metre_of_position_and_no_more() -> None:
    """The precision contract for every link this site produces: enough to say
    which corner, not enough to churn the URL on sub-pixel drift."""
    assert viewer("L.viewValue(-71.05891234, 42.36012345, 14.987)") == "-71.05891,42.36012,14.99"
    assert viewer("L.viewValue(0, 0, 2)") == "0.00000,0.00000,2.00"


def test_the_story_list_is_offered_only_to_a_link_that_asks_for_it() -> None:
    """Configuring stories no longer puts an entry list in front of everyone:
    the query string opts a visit in, and a bare key counts (`?stories` is what
    someone types). Three answers, not two — `no opinion` is what lets a story
    permalink offer the list, and an explicit off is what overrules that."""
    assert viewer("L.storiesAsked('?stories=1', 'stories')") is True
    assert viewer("L.storiesAsked('?stories', 'stories')") is True
    assert viewer("L.storiesAsked('?a=1&stories&b=2', 'stories')") is True
    assert viewer("L.storiesAsked('', 'stories')") is None
    assert viewer("L.storiesAsked('?other=1', 'stories')") is None
    # a key whose NAME merely contains the word is a different key
    assert viewer("L.storiesAsked('?nostories=1', 'stories')") is None
    for off in ("0", "false", "off", "no", "OFF", "False"):
        assert viewer(f"L.storiesAsked('?stories={off}', 'stories')") is False, off


def test_writing_the_query_flag_replaces_it_rather_than_repeating_it() -> None:
    """The story opt-in is written into the address bar every time a story is
    entered, so writing it must be idempotent — three visits to a story would
    otherwise hand out `?stories=1&stories=1&stories=1`. Other keys are the
    fragment's contract, byte for byte, and a trailing separator is not
    somebody else's key."""
    assert viewer("L.queryWrite('', 'stories', '1')") == "?stories=1"
    assert viewer("L.queryWrite('?stories=1', 'stories', '1')") == "?stories=1"
    assert viewer("L.queryWrite('?a=1&b=2', 'stories', '1')") == "?a=1&b=2&stories=1"
    assert viewer("L.queryWrite('?a=1&', 'stories', '1')") == "?a=1&stories=1"
    # in place, so a key never jumps to the end of somebody's link
    assert viewer("L.queryWrite('?a=1&stories=0&b=2', 'stories', '1')") == "?a=1&stories=1&b=2"
    assert viewer("L.queryWrite('?q=a%26b%3Dc', 'stories', '1')") == "?q=a%26b%3Dc&stories=1"


def test_a_story_opens_at_a_named_stop_and_an_unknown_one_opens_at_the_first() -> None:
    """Stop resolution, run rather than grepped: `findIndex` returns -1 for a
    stop id a link names and the story no longer has, and -1 would read as the
    last stop — dropping a reader at the end of a story they just opened."""
    stops = '[{id: "a"}, {id: "b"}, {id: "c"}]'
    assert viewer(f"L.stopIndex({stops}, 'b')") == 1
    assert viewer(f"L.stopIndex({stops}, 'gone')") == 0
    assert viewer(f"L.stopIndex({stops}, null)") == 0


def test_the_stop_index_is_held_inside_the_story() -> None:
    assert viewer("L.clampStopIndex(-1, 3)") == 0
    assert viewer("L.clampStopIndex(9, 3)") == 2
    assert viewer("L.clampStopIndex(1, 3)") == 1


def test_an_absent_hash_key_is_null_and_a_valueless_one_is_empty() -> None:
    """Both must read as `this link says nothing` to a caller expecting a
    number: `Number(null)` is 0, which silently means `divider hard left`."""
    assert viewer("L.hashRead('#panel=closed&at=1,2,3', 'panel')") == "closed"
    assert viewer("L.hashRead('#panel=closed', 'swipe')") is None
    assert viewer("L.hashRead('#panel', 'panel')") == ""
    assert viewer("L.hashRead('', 'panel')") is None


def test_writing_one_hash_key_leaves_every_other_key_byte_for_byte() -> None:
    """The fragment is a shared namespace. A `URLSearchParams` round trip
    re-encodes `/` and `,` and turns a bare `#anchor` into `#anchor=`, so one
    panel click would corrupt a key this code knows nothing about."""
    shared = "#anchor&at=-1.5,2.5,12.25&path=a/b&panel=open"
    assert viewer(f"L.hashWrite('{shared}', 'panel', 'closed')") == (
        "#anchor&at=-1.5,2.5,12.25&path=a/b&panel=closed"
    )
    # a new key appends; it does not reorder the ones already there
    assert viewer(f"L.hashWrite('{shared}', 'swipe', '0.500')") == (
        "#anchor&at=-1.5,2.5,12.25&path=a/b&panel=open&swipe=0.500"
    )
    # null removes, so a feature that stops applying leaves the URL clean
    assert viewer(f"L.hashWrite('{shared}', 'at', null)") == "#anchor&path=a/b&panel=open"
    # removing the last key empties the fragment rather than leaving a bare `#`
    assert viewer("L.hashWrite('#panel=open', 'panel', null)") == ""
    # a key keeps its place when rewritten, so the URL does not churn
    assert viewer("L.hashWrite('#a=1&b=2&c=3', 'b', '9')") == "#a=1&b=9&c=3"


# ---------------------------------------------------------------------------
# queue_ui/board.js — which table an entry lands in, and how it reads
# ---------------------------------------------------------------------------

NOW_S = 1_800_000_000


def test_an_age_reads_in_the_unit_a_glance_needs() -> None:
    """Seconds while a stage is turning over, minutes for the rest of an hour,
    then `h:mm` — and an em dash for a job that has no timestamp, because `0s`
    would read as `just now` for something that never started."""
    assert queue(f"L.ago(null, {NOW_S})") == "—"
    assert queue(f"L.ago({NOW_S - 12}, {NOW_S})") == "12s"
    assert queue(f"L.ago({NOW_S - 89}, {NOW_S})") == "89s"
    assert queue(f"L.ago({NOW_S - 90}, {NOW_S})") == "1m"
    assert queue(f"L.ago({NOW_S - 5399}, {NOW_S})") == "89m"
    assert queue(f"L.ago({NOW_S - 5400}, {NOW_S})") == "1h30"
    # the minutes are zero-padded, or 4h05 would print as 4h5
    assert queue(f"L.ago({NOW_S - (4 * 3600 + 5 * 60)}, {NOW_S})") == "4h05"
    # a clock that has drifted backwards is not a negative age
    assert queue(f"L.ago({NOW_S + 500}, {NOW_S})") == "0s"


def test_a_context_line_drops_the_fields_it_does_not_have() -> None:
    """The line is `city, year - areas`, and every field is optional. A
    missing one has to disappear rather than print as a gap, or a volume with
    no year reads as a broken row."""
    context = {
        "vol_full": {"city": "Springfield", "year": 1894, "neighborhoods": ["North", "Levee"]},
        "vol_place_only": {"city": "Springfield", "year": None},
        "vol_areas_only": {"city": "", "neighborhoods": ["Levee", None, ""]},
        "vol_empty": {"city": None, "year": None, "neighborhoods": []},
    }
    assert queue(f"L.contextText('vol_full', {js(context)})") == "Springfield, 1894 - North / Levee"
    assert queue(f"L.contextText('vol_place_only', {js(context)})") == "Springfield"
    assert queue(f"L.contextText('vol_areas_only', {js(context)})") == "Levee"
    assert queue(f"L.contextText('vol_empty', {js(context)})") == ""
    # no context at all for this volume is not an error
    assert queue(f"L.contextText('unknown', {js(context)})") == ""


BOARD_ENTRIES = [
    {"volume": "a", "track": "place", "status": "queued"},
    {"volume": "b", "track": "place", "status": "running"},
    {"volume": "c", "track": "serve", "status": "needs-review"},
    {"volume": "d", "track": "fetch", "status": "failed"},
    {"volume": "e", "track": "serve", "status": "done"},
    {"volume": "f", "track": "serve", "status": "running"},
]


def test_every_entry_lands_in_the_table_the_operator_looks_in() -> None:
    """One flat list becomes four columns and a table per track. An entry in
    the wrong one is a volume the operator cannot find — and `done` belongs in
    none of the columns, only in its track's table."""
    got = queue(f"L.board({js(BOARD_ENTRIES, ['fetch', 'place', 'serve'])})")
    # running FIRST, then what is queued behind it: the column says so
    assert [e["volume"] for e in got["live"]] == ["b", "f", "a"]
    assert [e["volume"] for e in got["needsReview"]] == ["c"]
    assert [e["volume"] for e in got["failed"]] == ["d"]
    assert {t: [e["volume"] for e in rows] for t, rows in got["byTrack"].items()} == {
        "fetch": ["d"],
        "place": ["a", "b"],
        "serve": ["c", "e", "f"],
    }


def test_a_track_the_server_names_gets_its_own_table_and_an_unknown_one_gets_none() -> None:
    """The SERVER sends the track list, so this page cannot disagree with the
    queue about what a track is — but it must not invent a table either."""
    got = queue(f"L.board({js(BOARD_ENTRIES, ['place', 'audit'])})")
    assert list(got["byTrack"]) == ["place", "audit"]
    assert got["byTrack"]["audit"] == []
    # entries on a track nobody asked about are simply not in any table
    assert "serve" not in got["byTrack"]


NOW_MS = 1_800_000_000_000
MINUTE = 60_000


def test_a_read_rate_needs_enough_history_to_mean_anything() -> None:
    """Reads land only when a model call completes, so two samples a few
    seconds apart say nothing at all — and printing a rate from them would
    make a stalled drain look busy or a busy one look stalled."""
    assert queue("L.readsPerMin([])") is None
    assert queue(f"L.readsPerMin([{js({'t': NOW_MS, 'reads': 4})}])") is None
    close = [{"t": NOW_MS, "reads": 4}, {"t": NOW_MS + 30_000, "reads": 6}]
    assert queue(f"L.readsPerMin({js(close)})") is None
    apart = [{"t": NOW_MS, "reads": 4}, {"t": NOW_MS + 4 * MINUTE, "reads": 16}]
    assert queue(f"L.readsPerMin({js(apart)})") == pytest.approx(3.0)


def test_a_drain_that_stopped_reporting_reads_has_no_rate_rather_than_zero() -> None:
    """A rate of 0.0/min printed beside a running volume reads as a
    measurement; the honest answer is that there is nothing to report."""
    flat = [{"t": NOW_MS, "reads": 9}, {"t": NOW_MS + 5 * MINUTE, "reads": 9}]
    assert queue(f"L.readsPerMin({js(flat)})") is None


def test_the_rate_window_forgets_what_is_no_longer_evidence() -> None:
    """A twelve-minute window: an hour-old sample would flatten a fan-out that
    started five minutes ago into nothing."""
    samples = [
        {"t": NOW_MS - 60 * MINUTE, "reads": 1},
        {"t": NOW_MS - 13 * MINUTE, "reads": 2},
        {"t": NOW_MS - 11 * MINUTE, "reads": 3},
        {"t": NOW_MS, "reads": 40},
    ]
    kept = queue(f"L.withinWindow({js(samples, NOW_MS)})")
    assert [s["reads"] for s in kept] == [3, 40]
    # exactly on the boundary is still evidence
    edge = [{"t": NOW_MS - 12 * MINUTE, "reads": 1}, {"t": NOW_MS, "reads": 2}]
    assert len(queue(f"L.withinWindow({js(edge, NOW_MS)})")) == 2
