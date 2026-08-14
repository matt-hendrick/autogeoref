"""Corroboration contracts: replay the recorded
revocation->reinstatement decisions from the fixtures.

The corroborator must reinstate exactly the pages recorded as
neighbor-corroborated (7 in _024, 10 in _021, 7 in _034) and must NOT
reinstate any page that stayed revoked.

Decision-frame note: corroboration for _021/_034 was decided BEFORE the seam
adjustment shifted the recorded GCPs, for _024 AFTER — replaying each volume
in its recorded frame (reversing the recorded ``seam_adjusted`` deltas where
needed) reproduces every decision exactly.
"""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from autogeoref.affine import TO_3857, TO_4326
from autogeoref.corroborate import (
    committed_nodes,
    corroborations,
    is_corroborated,
    is_corroborated_near,
)
from autogeoref.seam import SheetFit, sheet_fit_from_result
from autogeoref.volume import REVOKED_PREFIX, STATUS_CORROBORATED

pytestmark = pytest.mark.golden

#: volume -> (expected reinstatements, replay pre-seam?)
CASES = {
    "sanborn01790_024": (7, False),
    "sanborn01790_021": (10, True),
    "sanborn01790_034": (7, True),
}


def unseam(r: dict[str, Any]) -> dict[str, Any]:
    """Reverse a recorded seam shift to recover the pre-seam GCP positions."""
    sa = r.get("seam_adjusted")
    if not sa:
        return r
    r = copy.deepcopy(r)
    for ft in (r.get("gcps_geojson") or {}).get("features") or []:
        lng, lat = ft["geometry"]["coordinates"]
        x, y = TO_3857.transform(lng, lat)
        ft["geometry"]["coordinates"] = list(TO_4326.transform(x - sa["dx_m"], y - sa["dy_m"]))
    return r


@pytest.mark.parametrize("volume", sorted(CASES))
def test_reinstates_exactly_the_recorded_pages(volume: str, fixtures_dir: Path) -> None:
    expected_n, pre_seam = CASES[volume]
    committed: dict[str, SheetFit] = {}
    candidates: dict[str, tuple[SheetFit, bool]] = {}
    for f in sorted((fixtures_dir / volume / "results").glob("p*.json")):
        r = json.loads(f.read_text())
        st = str(r.get("status", ""))
        fit = sheet_fit_from_result(r["page"], unseam(r) if pre_seam else r)
        if fit is None:
            continue
        if st.startswith(REVOKED_PREFIX):
            candidates[r["page"]] = (fit, False)
        elif st == STATUS_CORROBORATED:
            candidates[r["page"]] = (fit, True)
        elif st.startswith("OK") and r.get("layer"):
            committed[r["page"]] = fit

    nodes = committed_nodes(committed)
    reinstated = []
    for page, (fit, expected) in sorted(candidates.items()):
        verdict = is_corroborated(corroborations(fit, nodes))
        assert verdict == expected, (
            f"{volume} p{page}: corroborator says {verdict}, recorded "
            f"{'reinstated' if expected else 'stays revoked'}"
        )
        if verdict:
            reinstated.append(page)
    assert len(reinstated) == expected_n


def test_zero_shared_nodes_never_reinstates() -> None:
    assert not is_corroborated([])
    # one agreeing node is not enough
    assert not is_corroborated([((0.0, 0.0), "5", 3.0)])
    # two distinct agreeing nodes reinstate
    assert is_corroborated([((0.0, 0.0), "5", 3.0), ((10.0, 10.0), "6", 6.0)])
    # two hits on the SAME node do not
    assert not is_corroborated([((0.0, 0.0), "5", 3.0), ((0.0, 0.0), "6", 6.0)])
    # agreement outside 8 m does not count
    assert not is_corroborated([((0.0, 0.0), "5", 9.0), ((10.0, 10.0), "6", 8.5)])


def test_near_shape_is_a_channel_vote_not_a_gate() -> None:
    """The verified-accept channel shape: one strong node plus near agreement.

    The strong node must hold at FULL tolerance and a single agreeing node is
    never enough — real half-block-shifted sheets keep one node on their
    aligned axis, and a single-node vote accepted 3 labeled-BAD sheets.
    """
    strong = ((0.0, 0.0), "5", 4.2)
    near = ((10.0, 10.0), "6", 9.6)
    far = ((10.0, 10.0), "6", 16.1)
    # the recovered fringe: one node at tolerance, second within 2x
    assert is_corroborated_near([strong, near])
    # a single strong node stays worthless
    assert not is_corroborated_near([strong])
    # two hits on the SAME node collapse to one
    assert not is_corroborated_near([strong, ((0.0, 0.0), "6", 9.6)])
    # near-without-strong: both nodes past tolerance never vote
    assert not is_corroborated_near([((0.0, 0.0), "5", 9.0), near])
    # beyond 2x tolerance is disagreement, not near agreement
    assert not is_corroborated_near([strong, far])
    # superset of the strict gate: two strong nodes also vote
    assert is_corroborated_near([strong, ((10.0, 10.0), "6", 6.0)])
    # per-node identity uses the BEST observation of each node
    assert is_corroborated_near([((0.0, 0.0), "5", 20.0), strong, near])
