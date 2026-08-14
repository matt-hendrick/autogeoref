"""Property-based invariants for the functions everything else leans on."""

import math

from hypothesis import given
from hypothesis import strategies as st

from autogeoref.matching import fold_quadrant_deg
from autogeoref.names import normalize

street_text = st.text(
    alphabet=st.characters(
        codec="ascii", categories=("Lu", "Ll", "Nd", "P", "Z"), exclude_characters="\x00"
    ),
    min_size=0,
    max_size=40,
)


@given(street_text)
def test_normalize_is_idempotent(name: str) -> None:
    once = normalize(name)
    assert normalize(once) == once


@given(street_text)
def test_normalize_output_charset_and_shape(name: str) -> None:
    out = normalize(name)
    # uppercase alphanumerics and single spaces only, no leading/trailing space
    assert out == out.strip()
    assert "  " not in out
    assert all(c.isupper() or c.isdigit() or c == " " for c in out)


@given(street_text, st.sampled_from(["", ".", " ST", " ST.", " AVE", " BLVD"]))
def test_normalize_insensitive_to_case_and_trailing_dot(name: str, suffix: str) -> None:
    assert normalize(name.lower() + suffix) == normalize(name.upper() + suffix)


@given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False))
def test_fold_quadrant_range(angle: float) -> None:
    # half-open [-45, 45): exactly-45 folds to -45 (equivalent deviation;
    # every consumer gates on abs(), so the boundary choice is immaterial)
    folded = fold_quadrant_deg(angle)
    assert -45.0 <= folded < 45.0


@given(st.floats(min_value=-720, max_value=720, allow_nan=False))
def test_fold_quadrant_is_modulo_90(angle: float) -> None:
    folded = fold_quadrant_deg(angle)
    # the fold differs from the input by an exact multiple of 90 degrees
    k = (angle - folded) / 90.0
    assert math.isclose(k, round(k), abs_tol=1e-9)
    # idempotent
    assert math.isclose(fold_quadrant_deg(folded), folded, abs_tol=1e-12)


@given(
    st.floats(min_value=-360, max_value=360, allow_nan=False),
    st.sampled_from([0.0, 90.0, 180.0, 270.0, -90.0, -180.0]),
)
def test_fold_quadrant_turn_invariant(angle: float, turn: float) -> None:
    assert math.isclose(fold_quadrant_deg(angle + turn), fold_quadrant_deg(angle), abs_tol=1e-9)
