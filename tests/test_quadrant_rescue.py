"""Opt-in quadrant rescue: a 90-deg-rotated scan clusters only at its quadrant.

The capability is an orientation SEARCH, not a gate change: translation_fit
applies every rescue gate unchanged at each tried orientation.
"""

from pathlib import Path

from autogeoref.affine import TO_4326
from autogeoref.config.load import load_city_config
from autogeoref.matching import Candidate
from autogeoref.rescue import pinned_linear, translation_fit

SCALE = 0.067
ROT_DEG = 1.2
T0 = (-9760000.0, 5140000.0)


def _cand_at(
    linear: list[list[float]], px: float, py: float, streets: tuple[str, str]
) -> Candidate:
    ax = linear[0][0] * px + linear[0][1] * py
    ay = linear[1][0] * px + linear[1][1] * py
    lng, lat = TO_4326.transform(T0[0] + ax, T0[1] + ay)
    return Candidate(pixel=(px, py), world4326=(lng, lat), streets=streets)


def test_rotated_scan_fails_base_and_clusters_at_quadrant() -> None:
    # candidates whose true orientation is the volume rotation + 90 deg
    rotated = pinned_linear(SCALE, ROT_DEG + 90.0)
    cands = [
        _cand_at(rotated, 1000, 1000, ("A", "B")),
        _cand_at(rotated, 4000, 5200, ("C", "D")),
        _cand_at(rotated, 900, 5100, ("E", "F")),
    ]
    base = pinned_linear(SCALE, ROT_DEG)
    m, anchors = translation_fit(cands, base)
    assert m is None, "rotated candidates must NOT cluster at the base orientation"

    m, anchors = translation_fit(cands, rotated)
    assert m is not None
    assert len(anchors) == 3
    # the recovered translation is the true one
    assert abs(m[0][0] - T0[0]) < 0.5 and abs(m[1][0] - T0[1]) < 0.5


def test_gates_still_apply_at_the_quadrant() -> None:
    rotated = pinned_linear(SCALE, ROT_DEG + 90.0)
    # all anchors ride one street: the disjoint-pair rule must still reject
    cands = [
        _cand_at(rotated, 1000, 1000, ("MAIN", "1ST")),
        _cand_at(rotated, 1000, 3000, ("MAIN", "2ND")),
        _cand_at(rotated, 1000, 5000, ("MAIN", "3RD")),
    ]
    m, _ = translation_fit(cands, rotated)
    assert m is None


def test_quadrant_rescue_is_opt_in() -> None:
    cfg = load_city_config(
        Path(__file__).resolve().parent.parent / "configs" / "chicago" / "chicago.toml"
    )
    for vid in cfg.volumes:
        assert cfg.volume(vid).quadrant_rescue is False
    assert cfg.volume("sanborn01790_999").quadrant_rescue is False
