"""Rescue contracts, from the recorded fixture data.

Every anchor set recorded as revoked-or-corroborated (revocation happened
first; corroboration reinstated a subset) is a set whose anchors all ride one
street — the disjoint-pair rule must classify ALL of them as non-disjoint
(19 in _024, 28 in _021, 14 in _034). Sets with a disjoint pair must pass.

Also unit-tests the translation clustering itself (12 m tolerance, >=2
distinct pairs / pixel points, pinned linear part) and the synthetic-corner
serialization rule.
"""

import importlib.util
import json
import math
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from autogeoref.affine import (
    TO_3857,
    TO_4326,
    AffineMatrix,
    fit_affine,
    gcps_from_geojson,
    grid_rmse_m,
    model_determinant,
    model_scales,
)
from autogeoref.matching import Candidate
from autogeoref.rescue import (
    has_disjoint_pair,
    pinned_linear,
    translation_fit,
    with_synthetic_corners,
)
from autogeoref.volume import STATUS_CORROBORATED, STATUS_RESCUE_REVOKED

# recorded per-volume counts of all-share-one-street sets (revoked+corroborated)
RECORDED_SHARED_SETS = {"sanborn01790_024": 19, "sanborn01790_021": 28, "sanborn01790_034": 14}


def _iter_results(fixtures_dir: Path, volume: str) -> Iterator[dict[str, Any]]:
    for f in sorted((fixtures_dir / volume / "results").glob("p*.json")):
        yield json.loads(f.read_text())


def _load_script(name: str) -> ModuleType:
    """Import a ``scripts/`` harness by name (they are scripts, not an installed package)."""
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_revoked_and_corroborated_sets_all_fail_disjoint_rule(
    fixtures_dir: Path, auto_volumes: list[str]
) -> None:
    counts: dict[str, int] = {}
    for vol in auto_volumes:
        n = 0
        for r in _iter_results(fixtures_dir, vol):
            anchors = r.get("rescue_anchors")
            if not anchors:
                continue
            status = r.get("status", "")
            if status in (STATUS_RESCUE_REVOKED, STATUS_CORROBORATED):
                assert not has_disjoint_pair([tuple(a) for a in anchors]), (
                    f"{vol} p{r['page']}: recorded as revoked/corroborated but the "
                    f"disjoint-pair rule passes it: {anchors}"
                )
                n += 1
        if vol in RECORDED_SHARED_SETS:
            counts[vol] = n
    assert counts == RECORDED_SHARED_SETS


def test_disjoint_sets_recorded_as_rescued_pass(
    fixtures_dir: Path, auto_volumes: list[str]
) -> None:
    """Plain 'OK (rescued)' sets WITH a disjoint pair must pass the rule.

    (A handful of stale pre-rule rescue records with shared-street anchors
    escaped the production revocation sweep — see the project report — so we
    assert only on the disjoint ones, which is the rule's pass direction.)
    """
    n_pass = 0
    for vol in auto_volumes:
        for r in _iter_results(fixtures_dir, vol):
            anchors = r.get("rescue_anchors")
            if not anchors or r.get("status") != "OK (rescued)":
                continue
            if has_disjoint_pair([tuple(a) for a in anchors]):
                n_pass += 1
    assert n_pass >= 40  # 52 at fixture-freeze time; keep headroom


#: Every frozen fixture record whose recorded GCPs refit to a POSITIVE determinant, pinned
#: so no NEW one can appear. PERMANENT, not transitional: ``fixtures/`` is read-only by
#: contract, so no migration will bring this to zero. Nearly all are RESCUE records whose
#: GCP set is degenerate on one side, where ``fit_affine``'s minimum-norm solution is
#: arbitrary and the determinant's SIGN carries no information. The two MATCH accepts are
#: NOT live evidence for the gate — today's port refuses both on other gates.
MIRRORED_FIXTURE_RECORDS = {
    "ref-volume": ["p75"],
    "sanborn01790_021": ["p125", "p41", "p45", "p56", "p69", "p79", "p80", "p94"],
    "sanborn01790_024": ["p109", "p27", "p30", "p45", "p50", "p61"],
    "sanborn01790_034": ["p124", "p128", "p41", "p53", "p67"],
    "sanborn01790_038": ["p64"],
    "sanborn01790_089": ["p15"],
}

#: The MATCH accepts among them, which the rescue story above does not cover.
MIRRORED_FIXTURE_MATCHES = {("ref-volume", "p75"), ("sanborn01790_089", "p15")}

#: The conditioning bar the retired ``with_synthetic_corners`` branch used — on
#: BOTH the pixel and the world side, which is the half a pixel-only test misses
#: (``ref-volume`` p75 passes on pixels and fails on world by four decades).
_RETIRED_CONDITIONING_BAR = 1e-2


def _conditioning(points: np.ndarray) -> float:
    """Scale-invariant SVD ratio of a centred 2-D point set; 0 when rank-1."""
    sv = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return 0.0 if sv[0] == 0 else float(sv[-1] / sv[0])


def test_mirrored_fixture_records_are_the_known_pre_fix_set(fixtures_dir: Path) -> None:
    """The corpus audit, run as a contract over the frozen fixture volumes.

    Imports ``scripts/audit_reflected_placements.py`` rather than re-spelling
    its rule, so the harness the corpus finding was measured with is the one the
    suite defends. See :data:`MIRRORED_FIXTURE_RECORDS` for what the baseline is
    and why it is not zero.
    """
    audit = _load_script("audit_reflected_placements").audit
    found: dict[str, list[str]] = {}
    records: dict[tuple[str, str], dict[str, Any]] = {}
    checked = 0
    # every tree with results, NOT just sanborn* — ref-volume is the golden
    # volume this change re-baselined, and it holds one of the 22
    for results in sorted(fixtures_dir.glob("*/results")):
        volume = results.parent.name
        committed, mirrored = audit(volume, fixtures_dir)
        checked += committed
        if mirrored:
            found[volume] = sorted(page for page, _, _ in mirrored)
            records.update(((volume, page), rec) for page, _, rec in mirrored)
    assert checked == 698, f"{checked} committed fixture records read, expected 698"
    assert found == {v: sorted(p) for v, p in MIRRORED_FIXTURE_RECORDS.items()}

    for (volume, page), record in records.items():
        gcps = gcps_from_geojson(record["gcps_geojson"])
        pixel_cond = _conditioning(np.array([[px, py] for px, py, _, _ in gcps]))
        world_cond = _conditioning(np.array([[x, y] for _, _, x, y in gcps]))
        model = fit_affine(gcps)
        sx, sy = model_scales(model)
        # |det| / (sx*sy) is the sine of the angle between the two model axes:
        # 1.0 for a square placement, ~0 when the linear part collapses to a
        # line and the determinant's sign is numerical noise.
        perpendicularity = abs(model_determinant(model)) / (sx * sy)
        if (volume, page) in MIRRORED_FIXTURE_MATCHES:
            assert record.get("rescue_anchors") is None, f"{volume} {page} is a rescue record"
            continue
        # The twenty rescue records: degenerate on at least one side, so under
        # the retired branch's own bar — written before that branch existed.
        assert min(pixel_cond, world_cond) < _RETIRED_CONDITIONING_BAR, (
            f"{volume} {page}: pixel {pixel_cond:.3e}, world {world_cond:.3e} — "
            "both sides healthy, so this is NOT the known pre-fix class"
        )
        assert record.get("rescue_anchors"), f"{volume} {page} is not a rescue record"
        # ...and the nonsense shows in the model: one axis scale is orders off
        # the ~0.067 m/px these volumes are printed at.
        assert max(sx, sy) > 1.0 or min(sx, sy) < 1e-6, f"{volume} {page}: {sx}, {sy}"
        assert perpendicularity > 0.0


def test_the_two_mirrored_fixture_match_accepts_are_already_refused(
    fixtures_dir: Path,
) -> None:
    """Neither MATCH accept in the baseline is live evidence for the gate.

    Read naively, a frozen `OK` record that refits mirrored looks like a sheet
    the determinant test would newly catch. Both are instead sheets the port
    ALREADY refuses through gates that shipped after the record was frozen —
    the trap `test_golden_new_volumes.LOO_GATE_DIFFS` exists to name. Their
    geometry is pinned here so the distinction between them survives: p75's
    linear part is singular (the sign means nothing), p15's is square (a real
    reflection, and the only one in the tree).
    """
    from test_golden_new_volumes import LOO_GATE_DIFFS
    from test_golden_replay import MATCH_DEPARTURES

    assert MATCH_DEPARTURES["75"] == frozenset(), "ref-volume p75 is no longer a departure"
    assert LOO_GATE_DIFFS["sanborn01790_089"]["15"].startswith("REJECTED")

    geometry = {}
    for volume, page in sorted(MIRRORED_FIXTURE_MATCHES):
        record = json.loads((fixtures_dir / volume / "results" / f"{page}.json").read_text())
        gcps = gcps_from_geojson(record["gcps_geojson"])
        model = fit_affine(gcps)
        sx, sy = model_scales(model)
        geometry[(volume, page)] = (
            _conditioning(np.array([[x, y] for _, _, x, y in gcps])),
            abs(model_determinant(model)) / (sx * sy),
        )
        assert model_determinant(model) > 0
        assert 0.02 < sx < 0.5 and 0.02 < sy < 0.5

    world_cond, perp = geometry[("ref-volume", "p75")]
    assert world_cond < 1e-4 and perp < 1e-3, "p75's linear part is supposed to be singular"
    world_cond, perp = geometry[("sanborn01790_089", "p15")]
    assert world_cond > 0.5 and perp > 0.99, "p15 is supposed to be a square reflection"


def test_has_disjoint_pair_semantics() -> None:
    assert has_disjoint_pair([("A", "B"), ("C", "D")])
    assert not has_disjoint_pair([("A", "B"), ("B", "C"), ("C", "A")])  # chain, no disjoint
    assert not has_disjoint_pair([("MAIN", "1ST"), ("MAIN", "2ND"), ("MAIN", "3RD")])
    assert has_disjoint_pair([("MAIN", "1ST"), ("MAIN", "2ND"), ("OAK", "3RD")])
    # NORMALIZED comparison (deliberate switch; measured flip
    # set in scripts/measure_disjoint_flipset.py): the same street under two
    # raw spellings IS shared — spelling variance can no longer fake
    # disjointness the way the original raw-label rule allowed
    assert not has_disjoint_pair([("WASHBURN AV.", "RACINE ST"), ("Washburn Av.", "13TH ST.")])
    # alias-aware: a historical rename and its modern name are one street
    assert not has_disjoint_pair(
        [("ROBEY ST", "MONTROSE"), ("DAMEN AVE", "WILSON")], {"ROBEY": "DAMEN"}
    )
    assert has_disjoint_pair([("ROBEY ST", "MONTROSE"), ("DAMEN AVE", "WILSON")], {})


# --- translation_fit unit contracts ---------------------------------------

SCALE = 0.067
ROT_DEG = 1.0
LINEAR = pinned_linear(SCALE, ROT_DEG)
T0 = (-9760000.0, 5140000.0)


def _cand(
    px: float, py: float, streets: tuple[str, str], dx: float = 0.0, dy: float = 0.0
) -> Candidate:
    ax = LINEAR[0][0] * px + LINEAR[0][1] * py
    ay = LINEAR[1][0] * px + LINEAR[1][1] * py
    lng, lat = TO_4326.transform(T0[0] + ax + dx, T0[1] + ay + dy)
    return Candidate(pixel=(px, py), world4326=(lng, lat), streets=streets)


def test_translation_fit_disjoint_pair_accepts() -> None:
    cands = [
        _cand(1000, 1000, ("A", "B")),
        _cand(4000, 5000, ("C", "D")),
    ]
    m, anchors = translation_fit(cands, LINEAR)
    assert m is not None
    assert len(anchors) == 2
    assert math.isclose(m[0][0], T0[0], abs_tol=0.5)
    assert math.isclose(m[1][0], T0[1], abs_tol=0.5)


def test_translation_fit_shared_street_rejects() -> None:
    # three agreeing anchors, every one riding street "MAIN"
    cands = [
        _cand(1000, 1000, ("MAIN", "1ST")),
        _cand(1000, 3000, ("MAIN", "2ND")),
        _cand(1000, 5000, ("MAIN", "3RD")),
    ]
    m, anchors = translation_fit(cands, LINEAR)
    assert m is None
    assert anchors == []


def test_translation_fit_single_pair_cannot_self_confirm() -> None:
    # two candidates from the SAME street pair agreeing: rejected
    cands = [
        _cand(1000, 1000, ("A", "B")),
        _cand(1000, 1000, ("A", "B")),
    ]
    m, _ = translation_fit(cands, LINEAR)
    assert m is None


def test_translation_fit_skips_a_larger_invalid_cluster() -> None:
    # Multiple world intersections from one rail/street image crossing must not
    # evict a smaller street cluster that meets every existing rescue contract.
    cands = [
        _cand(1000, 1000, ("RR LINE", "MAIN")),
        _cand(1000, 1000, ("RR LINE", "MAIN")),
        _cand(1000, 1000, ("RR LINE", "MAIN")),
        _cand(1000, 1000, ("RR LINE", "MAIN")),
        _cand(3000, 2000, ("OAK", "PINE"), dx=40.0),
        _cand(5000, 4000, ("OAK", "MAPLE"), dx=40.0),
    ]
    m, anchors = translation_fit(cands, LINEAR, require_disjoint=False)
    assert m is not None
    assert [anchor.streets for anchor in anchors] == [("OAK", "PINE"), ("OAK", "MAPLE")]


def test_translation_fit_rail_shared_street_cluster_cannot_evict() -> None:
    # A rail-bearing cluster whose anchors all share one street can only end
    # revoked (it has no disjoint pair), so even under require_disjoint=False
    # it must not evict a smaller cluster that passes the fit gates
    # The disjoint requirement prevents a shared-street false recovery.
    cands = [
        _cand(1000, 1000, ("RR GALENA DIV", "CLINTON")),
        _cand(1000, 1400, ("RR GALENA DIV 2", "CLINTON")),
        _cand(1000, 2000, ("KINZIE", "CLINTON")),
        _cand(3000, 2000, ("KINZIE", "DESPLAINES"), dx=40.0),
        _cand(5000, 4000, ("DESPLAINES", "WAYMAN"), dx=40.0),
    ]
    m, anchors = translation_fit(cands, LINEAR, require_disjoint=False)
    assert m is not None
    assert [anchor.streets for anchor in anchors] == [
        ("KINZIE", "DESPLAINES"),
        ("DESPLAINES", "WAYMAN"),
    ]
    # a rail-bearing cluster WITH a disjoint pair still wins on size
    cands = [
        _cand(1000, 1000, ("RR GALENA DIV", "CLINTON")),
        _cand(1000, 2000, ("KINZIE", "HALSTED")),
        _cand(2000, 3000, ("LAKE", "MORGAN")),
        _cand(3000, 2000, ("KINZIE", "DESPLAINES"), dx=40.0),
        _cand(5000, 4000, ("DESPLAINES", "WAYMAN"), dx=40.0),
    ]
    m, anchors = translation_fit(cands, LINEAR, require_disjoint=False)
    assert m is not None
    assert len(anchors) == 3
    assert anchors[0].streets == ("RR GALENA DIV", "CLINTON")


def test_translation_fit_rail_shared_street_cluster_wins_when_alone() -> None:
    # When a shared-street rail cluster displaces nothing, it keeps the
    # recorded provisional lifecycle: revoked, then corroboration may vouch it
    # The disjoint requirement still applies after candidate recovery.
    # Only under require_disjoint=False; the direct-accept path still refuses.
    cands = [
        _cand(1000, 1000, ("RR KENOSHA SUB", "GRACE")),
        _cand(1000, 3000, ("RR KENOSHA SUB", "ADDISON")),
        _cand(1000, 5000, ("RR KENOSHA SUB", "ADDISON")),
    ]
    m, anchors = translation_fit(cands, LINEAR, require_disjoint=False)
    assert m is not None
    assert len(anchors) == 3
    m, anchors = translation_fit(cands, LINEAR)
    assert m is None
    assert anchors == []


def test_translation_fit_fallback_ignores_degenerate_street_cluster() -> None:
    # a degenerate street cluster (one pair, one pixel) fails the fit gates,
    # so it displaces nothing: the shared-street rail fallback still applies
    cands = [
        _cand(1000, 1000, ("RR KENOSHA SUB", "GRACE")),
        _cand(1000, 2000, ("RR KENOSHA SUB", "ADDISON")),
        _cand(1000, 3000, ("RR KENOSHA SUB", "ADDISON")),
        _cand(1000, 4000, ("RR KENOSHA SUB", "GRACE")),
        _cand(3000, 2000, ("MAIN", "1ST"), dx=40.0),
        _cand(3000, 2000, ("MAIN", "1ST"), dx=40.0),
        _cand(3000, 2000, ("MAIN", "1ST"), dx=40.0),
    ]
    m, anchors = translation_fit(cands, LINEAR, require_disjoint=False)
    assert m is not None
    assert len(anchors) == 4


def test_translation_fit_rail_guard_is_alias_aware() -> None:
    # The guard's disjoint test must agree with the caller's post-fit
    # disjointness decision, which normalizes with the volume alias table:
    # raw-label disjointness must not let a doomed rail cluster win.
    aliases = {"ROBEY": "DAMEN"}
    cands = [
        _cand(1000, 1000, ("RR X", "ROBEY ST")),
        _cand(1000, 3000, ("RR Y", "DAMEN AVE")),
        _cand(1000, 5000, ("RR X", "DAMEN AVE")),
        _cand(3000, 2000, ("OAK", "PINE"), dx=40.0),
        _cand(5000, 4000, ("OAK", "MAPLE"), dx=40.0),
    ]
    m, anchors = translation_fit(cands, LINEAR, require_disjoint=False, aliases=aliases)
    assert m is not None
    assert [anchor.streets for anchor in anchors] == [("OAK", "PINE"), ("OAK", "MAPLE")]
    # alias-blind, the same rail cluster looks disjoint and wins on size
    m, anchors = translation_fit(cands, LINEAR, require_disjoint=False)
    assert m is not None
    assert len(anchors) == 3


def test_translation_fit_street_only_provisional_lifecycle_unchanged() -> None:
    # street-only shared-street clusters keep the recorded provisional path:
    # under require_disjoint=False they still win and are handed to the
    # revoked-then-corroboration lifecycle by the caller
    cands = [
        _cand(1000, 1000, ("MAIN", "1ST")),
        _cand(1000, 3000, ("MAIN", "2ND")),
        _cand(1000, 5000, ("MAIN", "3RD")),
    ]
    m, anchors = translation_fit(cands, LINEAR, require_disjoint=False)
    assert m is not None
    assert len(anchors) == 3


def test_translation_fit_cluster_tolerance() -> None:
    # a 13 m disagreement is outside the 12 m cluster tolerance
    cands = [
        _cand(1000, 1000, ("A", "B")),
        _cand(4000, 5000, ("C", "D"), dx=13.0),
    ]
    m, _ = translation_fit(cands, LINEAR)
    assert m is None
    # 8 m agrees
    cands = [
        _cand(1000, 1000, ("A", "B")),
        _cand(4000, 5000, ("C", "D"), dx=8.0),
    ]
    m, anchors = translation_fit(cands, LINEAR)
    assert m is not None
    assert len(anchors) == 2


def test_synthetic_corners_are_unconditional() -> None:
    cands = [
        _cand(1000, 1000, ("A", "B")),
        _cand(4000, 5000, ("C", "D")),
    ]
    m, anchors = translation_fit(cands, LINEAR)
    assert m is not None
    gcps = with_synthetic_corners(anchors, m, (5900.0, 7300.0))
    # 2 anchors -> 3 synthetic corners appended
    assert len(gcps) == 5
    assert sum(c.streets == ("synthetic", "rescue-model-corner") for c in gcps) == 3
    # many COLLINEAR anchors must also get synthetics (the recorded bug)
    col = [_cand(1000, 1000 + 800 * k, ("S" + str(k), "X" + str(k))) for k in range(6)]
    gcps = with_synthetic_corners(col, m, (5900.0, 7300.0))
    assert len(gcps) == 9
    # NEAR-collinear anchors (pass a rank check, break GDAL's solver —
    # origin _015 p14) must also get synthetics
    near = [
        _cand(1000, 1000, ("N0", "M0")),
        _cand(2000, 2001, ("N1", "M1")),
        _cand(3000, 3005, ("N2", "M2")),
        _cand(4000, 3990, ("N3", "M3")),
    ]
    gcps = with_synthetic_corners(near, m, (5900.0, 7300.0))
    assert len(gcps) == 7
    # A HEALTHY 4-anchor non-collinear set gets them TOO. The old conditioning
    # test skipped exactly this case, and the skipped population is where the
    # defect lived: a well-conditioned anchor set still lets the consumers'
    # unconstrained refit drift off the model that passed the gate, because no
    # result record carries the model
    # .
    spread = [
        _cand(1000, 1000, ("A", "B")),
        _cand(4000, 1200, ("C", "D")),
        _cand(1100, 5000, ("E", "F")),
        _cand(4100, 5200, ("G", "H")),
    ]
    gcps = with_synthetic_corners(spread, m, (5900.0, 7300.0))
    assert len(gcps) == 7
    assert sum(c.streets == ("synthetic", "rescue-model-corner") for c in gcps) == 3


def test_synthetic_corners_added_for_world_degenerate_set() -> None:
    # A healthy PIXEL rectangle whose anchors map to only TWO distinct WORLD
    # points (two junctions, each contributing two pixel variants): gdalwarp's
    # refit from the raw GCPs is singular, so the serializer must emit the
    # model corners. Synthetic corners preserve a warpable GCP set.
    m = [[T0[0], LINEAR[0][0], LINEAR[0][1]], [T0[1], LINEAR[1][0], LINEAR[1][1]]]
    wa = TO_4326.transform(T0[0] + 100.0, T0[1] - 50.0)
    wb = TO_4326.transform(T0[0] + 100.0, T0[1] - 160.0)
    anchors = [
        Candidate(pixel=(741.0, 622.0), world4326=wa, streets=("WILMOT", "HOYNE")),
        Candidate(pixel=(622.0, 622.0), world4326=wa, streets=("WILMOT", "HOYNE")),
        Candidate(pixel=(741.0, 2314.0), world4326=wb, streets=("WABANSIA", "HOYNE")),
        Candidate(pixel=(622.0, 2314.0), world4326=wb, streets=("WABANSIA", "HOYNE")),
    ]
    gcps = with_synthetic_corners(anchors, m, (5900.0, 7300.0))
    assert len(gcps) == 7
    assert sum(c.streets == ("synthetic", "rescue-model-corner") for c in gcps) == 3


def test_recorded_gcps_refit_to_the_pinned_model_for_near_collinear_anchors() -> None:
    """THE option-A contract, and the `_056` p59 regression.

    p59's six anchors — four along one street at a single pixel column plus two rail crossings
    at one repeated pixel — cleared every conditioning bar the old branch tested, so no corners
    were written, and the consumers' unconstrained refit landed on a REFLECTION hundreds of
    metres from the model that placed the sheet. What is recorded must reproduce what passed the
    gate. The whole perpendicular dimension of the fit rests on a 60 px lever, and the rail
    anchors sit on the WRONG side of it — inside ``TOL_M``, so the rescue gate still passes,
    which is a separate open defect. Over that lever the x-axis does not stretch, it REVERSES.
    """
    anchors = [
        _cand(1180, 1000, ("S. WOOD", "W. 98TH")),
        _cand(1180, 2600, ("S. WOOD", "W. 99TH")),
        _cand(1180, 4200, ("S. WOOD", "W. 100TH")),
        _cand(1180, 5800, ("S. WOOD", "W. 101ST")),
        _cand(1240, 3400, ("RR ROCK ISLAND", "W. 99TH"), dx=-6.6),
        _cand(1240, 3400, ("RR ROCK ISLAND", "W. 100TH"), dx=-6.6, dy=4.44),
    ]
    m, kept = translation_fit(anchors, LINEAR, require_disjoint=False)
    assert m is not None and len(kept) == 6
    full_size = (5900.0, 7300.0)

    def refit(cands: list[Candidate]) -> AffineMatrix:
        gcps = [(c.pixel[0], c.pixel[1], *TO_3857.transform(*c.world4326)) for c in cands]
        return fit_affine(gcps)

    # WITHOUT corners the refit leaves the pinned model entirely — and it is
    # not even upright. This is the recorded p59 failure.
    bare = refit(kept)
    placing = np.array(m, dtype=float)
    assert model_determinant(bare) > 0
    assert grid_rmse_m(bare, placing, *full_size) > 100.0
    # WITH them it reproduces the placement across the whole page.
    recorded = refit(with_synthetic_corners(kept, m, full_size))
    assert model_determinant(recorded) < 0
    assert grid_rmse_m(recorded, placing, *full_size) < 2.0
