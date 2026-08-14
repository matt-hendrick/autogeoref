"""Sheets placed at the VOLUME's rotation and scale rather than their own.

When a rescued sheet's real anchors are collinear, ``rescue.pinned_linear``
fixes the linear part from the volume constants and solves only a translation,
and ``rescue.with_synthetic_corners`` injects corners lying exactly on that
model so the warper can refit it. Both are correct by design. What these tests
protect is that the result stops being INDISTINGUISHABLE from an
evidence-fitted placement: it is counted in the report and in `status`, its
residual is never presented as a quality score, and every consumer decides
"synthetic" the same way.

Evidence and the corpus census:
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autogeoref.placement_records import pinned_orientation
from autogeoref.report import build_report, load_results_dir, report_markdown
from autogeoref.rescue import SYNTHETIC_STREETS, is_synthetic_gcp
from autogeoref.seam import sheet_fit_from_result
from autogeoref.status import build_status
from autogeoref.status_render import format_table
from autogeoref.volume import (
    STATUS_OK,
    STATUS_REJECTED,
    STATUS_RESCUED,
    STATUS_REVIEWER_VERIFIED,
    status_verified,
)

#: exactly what ``matching.gcps_geojson_from`` writes for a synthetic corner
RESCUE_CORNER_NOTE = f"auto: {SYNTHETIC_STREETS[0]} x {SYNTHETIC_STREETS[1]}"
#: and what ``review.materialize.synthetic_gcps_geojson`` writes for a human one
REVIEWER_CORNER_NOTE = "synthetic: reviewer placement corner"


def _gcp(px: float, py: float, note: str, lng: float = -87.6, lat: float = 41.9) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {"image": [px, py], "username": "admin", "note": note},
        "geometry": {"type": "Point", "coordinates": [lng, lat]},
    }


def _record(
    page: str,
    status: str,
    *,
    notes: tuple[str, ...] = ("auto: A ST. x B ST.", "auto: C ST. x D ST."),
    residuals: list[float] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "page": page,
        "status": status,
        # spread in both frames: a set that is collinear or coincident cannot be
        # fitted at all, and `seam.sheet_fit_from_result` rightly refuses it
        "gcps_geojson": {
            "type": "FeatureCollection",
            "features": [
                _gcp(
                    100 * (i + 1),
                    100 * (i + 1) ** 2,
                    n,
                    lng=-87.6 + 0.01 * i,
                    lat=41.9 + 0.004 * i**2,
                )
                for i, n in enumerate(notes)
            ],
        },
    }
    if residuals is not None:
        record["auto_residuals_m"] = residuals
    return record


# --- the grammar -----------------------------------------------------------


def test_synthetic_is_the_note_marker_and_never_a_boolean_key() -> None:
    """THE gotcha: there is no ``synthetic`` property on a GCP feature.

    A reader that looks for one sees zero synthetics on every sheet in the
    corpus and silently concludes nothing is pinned.
    """
    assert is_synthetic_gcp(_gcp(1, 1, RESCUE_CORNER_NOTE)) is True
    assert is_synthetic_gcp(_gcp(1, 1, REVIEWER_CORNER_NOTE)) is True
    assert is_synthetic_gcp(_gcp(1, 1, "auto: W. NORTH AV. x ORCHARD ST.")) is False
    # a feature that DECLARES itself synthetic the way nobody writes it
    liar = _gcp(1, 1, "auto: W. NORTH AV. x ORCHARD ST.")
    liar["properties"]["synthetic"] = True
    assert is_synthetic_gcp(liar) is False


def test_seam_and_the_report_agree_on_which_gcps_are_synthetic() -> None:
    """One grammar: the seam solve's per-GCP flag and the pinned-orientation
    predicate must never drift apart — ties excluded here are exactly the
    corners counted there."""
    record = _record(
        "70",
        STATUS_RESCUED,
        notes=("auto: A ST. x B ST.", "auto: C ST. x D ST.", RESCUE_CORNER_NOTE),
    )
    fit = sheet_fit_from_result("70", record)
    assert fit is not None
    assert [g[4] for g in fit.gcps] == [False, False, True]
    assert pinned_orientation(record) is True


# --- the predicate ---------------------------------------------------------


def test_a_sheet_with_synthetic_corners_is_pinned() -> None:
    assert pinned_orientation(_record("70", STATUS_RESCUED, notes=(RESCUE_CORNER_NOTE,))) is True


def test_an_evidence_fitted_sheet_is_not_pinned() -> None:
    assert pinned_orientation(_record("71", STATUS_OK)) is False


def test_every_rescue_is_now_pinned_including_a_well_spread_one() -> None:
    """The predicate's scope widened at the always-corners change, and that is a correction.

    ``with_synthetic_corners`` used to skip a well-conditioned anchor set, so a
    rescue with four well-spread anchors reported NOT pinned — while its linear
    part came from ``pinned_linear`` like every other rescue's. The corners are
    now unconditional, so the predicate finally agrees with the geometry. This
    goes through the real serializer rather than a hand-built record, because
    the two must not drift.
    """
    from autogeoref.affine import TO_4326
    from autogeoref.matching import Candidate, gcps_geojson_from
    from autogeoref.rescue import pinned_linear, with_synthetic_corners

    linear = pinned_linear(0.067, 1.0)
    t0 = (-9760000.0, 5140000.0)
    m = [[t0[0], linear[0][0], linear[0][1]], [t0[1], linear[1][0], linear[1][1]]]

    def anchor(px: float, py: float, streets: tuple[str, str]) -> Candidate:
        ax = linear[0][0] * px + linear[0][1] * py
        ay = linear[1][0] * px + linear[1][1] * py
        return Candidate(
            pixel=(px, py), world4326=TO_4326.transform(t0[0] + ax, t0[1] + ay), streets=streets
        )

    well_spread = [
        anchor(1000, 1000, ("A", "B")),
        anchor(4000, 1200, ("C", "D")),
        anchor(1100, 5000, ("E", "F")),
        anchor(4100, 5200, ("G", "H")),
    ]
    gcps = with_synthetic_corners(well_spread, m, (5900.0, 7300.0))
    record = {
        "page": "70",
        "status": STATUS_RESCUED,
        "gcps_geojson": gcps_geojson_from(gcps),
    }
    assert pinned_orientation(record) is True
    # ...and the anchors alone, which is what the old branch would have
    # recorded for this set, report the opposite
    bare = {"page": "70", "status": STATUS_RESCUED, "gcps_geojson": gcps_geojson_from(well_spread)}
    assert pinned_orientation(bare) is False


def test_a_record_without_gcps_is_not_pinned() -> None:
    assert pinned_orientation({"page": "9", "status": STATUS_REJECTED}) is False


def test_a_reviewer_placement_is_not_pinned_orientation() -> None:
    """Reviewer corners are synthetic too, but a human set that orientation —
    counting them as volume-pinned would attribute a hand placement to the
    volume constants."""
    record = _record("12", STATUS_REVIEWER_VERIFIED, notes=(REVIEWER_CORNER_NOTE,) * 3)
    assert pinned_orientation(record) is False


def test_a_reviewer_who_confirms_a_pinned_rescue_has_not_unpinned_it() -> None:
    """The case a status test gets wrong, and there are eight of them on disk.

    ``review.materialize.final_gcps_geojson`` returns the record's OWN GCPs for
    an accept with no ops, so the sheet keeps the rescue's corners and the
    volume's orientation. Only the status changed.
    """
    record = _record(
        "5",
        STATUS_REVIEWER_VERIFIED,
        notes=("auto: A ST. x B ST.", "auto: C ST. x D ST.", RESCUE_CORNER_NOTE),
    )
    assert pinned_orientation(record) is True


def test_a_verified_accept_is_classified_on_its_gcps_like_any_other() -> None:
    """`OK (verified: ...)` is a promotion of a recorded placement, not a new
    one — a promoted rescue is still pinned, a promoted strict fit is not."""
    verified = status_verified(["corroboration", "junction"])
    assert pinned_orientation(_record("8", verified, notes=(RESCUE_CORNER_NOTE,))) is True
    assert pinned_orientation(_record("9", verified)) is False


# --- the report ------------------------------------------------------------


def test_report_counts_pinned_accepts_and_keeps_them_out_of_the_residual() -> None:
    """The residual statistics describe evidence-fitted accepts only.

    A residual measured through corners the placement model generated is the
    model measured against itself; averaging it into the volume's median would
    flatter the number with the sheets that have no evidence at all.
    """
    results = {
        "1": _record("1", STATUS_OK, residuals=[4.0, 6.0]),
        "2": _record("2", STATUS_RESCUED, notes=(RESCUE_CORNER_NOTE,), residuals=[0.0, 0.0]),
        "3": _record("3", STATUS_REJECTED),
    }
    report = build_report("vol", results)
    assert report.accepted_total == 2
    assert report.pinned_orientation == 1
    assert report.median_auto_residual_m == 5.0  # 0.0, 0.0 excluded
    assert report.p90_auto_residual_m == 6.0


def test_report_markdown_names_the_population_each_number_describes() -> None:
    results = {
        "1": _record("1", STATUS_OK, residuals=[4.0]),
        "2": _record("2", STATUS_RESCUED, notes=(RESCUE_CORNER_NOTE,)),
    }
    md = report_markdown(build_report("vol", results))
    assert "EVIDENCE-FITTED accepts only" in md
    assert "pinned-orientation accepts" in md and "1 of 2" in md


def test_a_pinned_sheet_a_reviewer_confirmed_stays_out_of_the_funnel() -> None:
    """It IS pinned (the predicate says so) but it is a human placement, and no
    funnel counter may absorb one — that is how a hand placement inflates an
    auto-acceptance number."""
    results = {
        "1": _record("1", STATUS_RESCUED, notes=(RESCUE_CORNER_NOTE,)),
        "2": _record("2", STATUS_REVIEWER_VERIFIED, notes=(RESCUE_CORNER_NOTE,)),
    }
    assert pinned_orientation(results["2"]) is True
    report = build_report("vol", results)
    assert (report.accepted_total, report.reviewer_verified) == (1, 1)
    assert report.pinned_orientation == 1


def test_a_volume_with_no_pinned_sheet_says_nothing_about_them() -> None:
    md = report_markdown(build_report("vol", {"1": _record("1", STATUS_OK, residuals=[4.0])}))
    assert "pinned-orientation" not in md


def test_frozen_024_funnel_carries_eight_pinned_accepts(fixtures_dir: Path) -> None:
    """Anchored on the recorded fixture, and a regression guard on the
    exclusion: those eight sheets carry no ``auto_residuals_m``, so the
    published median is exactly what it was before they were counted."""
    results = load_results_dir(fixtures_dir / "sanborn01790_024" / "results")
    report = build_report("sanborn01790_024", results)
    assert report.accepted_total == 97
    assert report.pinned_orientation == 8
    assert report.median_auto_residual_m == 3.21
    pinned = [r for r in results.values() if pinned_orientation(r)]
    assert pinned and not any(r.get("auto_residuals_m") for r in pinned)


# --- status ----------------------------------------------------------------


def test_status_prints_pinned_inside_the_ok_count(tmp_path: Path) -> None:
    """A subset of the accepts, not a fourth category — it must not read as if
    the volume placed more sheets than it did."""
    work = tmp_path / "work"
    results = work / "vol_019" / "results"
    results.mkdir(parents=True)
    for page, record in (
        ("1", _record("1", STATUS_OK)),
        ("2", _record("2", STATUS_RESCUED, notes=(RESCUE_CORNER_NOTE,))),
        ("3", _record("3", STATUS_REJECTED)),
    ):
        (results / f"p{page}.json").write_text(json.dumps(record))

    rows = build_status(work=work, fixtures=tmp_path / "fixtures", tiles=tmp_path / "tiles")
    row = next(r for r in rows if r.volume == "vol_019")
    assert (row.accepted, row.flagged, row.pinned_orientation) == (2, 1, 1)
    assert "2 ok (1 pinned) / 1 flagged" in format_table(rows)


def test_status_stays_quiet_when_nothing_is_pinned(tmp_path: Path) -> None:
    work = tmp_path / "work"
    results = work / "vol_034" / "results"
    results.mkdir(parents=True)
    (results / "p1.json").write_text(json.dumps(_record("1", STATUS_OK)))

    rows = build_status(work=work, fixtures=tmp_path / "fixtures", tiles=tmp_path / "tiles")
    row = next(r for r in rows if r.volume == "vol_034")
    # 0 is a count: this volume was scanned and none of its accepts are pinned.
    # None is reserved for a volume with no results to count at all.
    assert row.pinned_orientation == 0
    assert "1 ok / 0 flagged" in format_table(rows)
