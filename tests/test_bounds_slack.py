"""Which bounds sources get slack, and which do not.

`bounds` is a HARD membership filter, not a prior: `CenterlineIndex` drops every
feature that does not intersect the box, so a street outside it is a candidate
for no sheet in the volume. That is also why human pins are NOT a source: they
would be pins deciding which sheets can place. The remaining sources are padded
before they reach the index, and `bounds_bbox` — the only hand-authored one — is not.

Those are characterization tests, not a preference. The asymmetry was
undocumented and asserted nowhere until it was measured in
`` §14; pinning it here is
what makes a future change to it deliberate rather than silent. A change to
`resolve_bounds` that adds or removes slack SHOULD fail these, and that record is
where the evidence for changing them lives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autogeoref.bounds import BUFFER_DEG
from autogeoref.bounds_bootstrap import MARGIN_LAT_DEG, MARGIN_LON_DEG, localize_pages
from autogeoref.config.model import CityConfig, VolumeConfig
from autogeoref.run_inputs import NoBoundsSourceError, resolve_bounds

BBOX = (-87.70, 41.80, -87.60, 41.90)


def _city(tmp_path: Path, **kwargs: Any) -> CityConfig:
    return CityConfig(
        name="testville",
        centerlines_path=tmp_path / "centerlines.geojson",
        aliases_dir=tmp_path / "aliases",
        **kwargs,
    )


def test_bbox_is_returned_verbatim(tmp_path: Path) -> None:
    """The hand-authored source is the ONE that gets no slack."""
    city = _city(tmp_path)
    vol = VolumeConfig(identifier="v1", bounds_bbox=BBOX)
    assert resolve_bounds(city, vol, None) == BBOX


def test_a_declared_bbox_is_the_top_source(tmp_path: Path) -> None:
    """The whole signature: nothing outranks a declared box.

    Ground truth used to, which silently discarded the operator's curated box on
    every pinned volume — and, worse, let hand-placed pins decide which sheets could
    match at all. ``resolve_bounds`` cannot be handed pins any more, so this asserts
    the SIGNATURE as much as the value: a third positional argument is the viewer
    manifest and nothing else.
    """
    city = _city(tmp_path)
    vol = VolumeConfig(identifier="v1", bounds_bbox=BBOX)
    assert resolve_bounds(city, vol) == BBOX
    assert resolve_bounds(city, vol, None) == BBOX


def test_counterpart_takes_the_buffer(tmp_path: Path) -> None:
    manifest = tmp_path / "viewer.json"
    manifest.write_text(json.dumps({"volumes": [{"id": "other_001", "bounds": list(BBOX)}]}))
    city = _city(tmp_path)
    vol = VolumeConfig(identifier="v1", bounds_from_counterpart="other_001")
    got = resolve_bounds(city, vol, manifest)
    assert got == pytest.approx(
        (
            BBOX[0] - BUFFER_DEG,
            BBOX[1] - BUFFER_DEG,
            BBOX[2] + BUFFER_DEG,
            BBOX[3] + BUFFER_DEG,
        )
    )


def test_community_areas_take_the_buffer(tmp_path: Path) -> None:
    areas = tmp_path / "areas.geojson"
    areas.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "properties": {"community": "TESTLAND"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [BBOX[0], BBOX[1]],
                                    [BBOX[2], BBOX[1]],
                                    [BBOX[2], BBOX[3]],
                                    [BBOX[0], BBOX[3]],
                                    [BBOX[0], BBOX[1]],
                                ]
                            ],
                        },
                    }
                ]
            }
        )
    )
    city = _city(tmp_path, community_areas_path=areas)
    vol = VolumeConfig(identifier="v1", bounds_areas=("TESTLAND",))
    got = resolve_bounds(city, vol, None)
    assert got == pytest.approx(
        (
            BBOX[0] - BUFFER_DEG,
            BBOX[1] - BUFFER_DEG,
            BBOX[2] + BUFFER_DEG,
            BBOX[3] + BUFFER_DEG,
        )
    )


def test_no_source_raises_the_bootstrap_sentinel(tmp_path: Path) -> None:
    """`cli` answers this specific error with the derivation, so it must stay
    distinguishable from a declared source that failed to load."""
    with pytest.raises(NoBoundsSourceError):
        resolve_bounds(_city(tmp_path), VolumeConfig(identifier="v1"), None)


def _line(name: str, coords: list[list[float]]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {"street_nam": name, "street_typ": "AVE"},
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def test_the_derived_path_pads_more_than_the_declared_ones() -> None:
    """The fifth source, exercised rather than asserted from its constants.

    The derived box must be the union of the localized page boxes plus the
    bootstrap margins — read off the evidence the derivation itself reports, so
    this fails if the path stops padding. Comparing the two module constants to
    each other would pass either way.
    """
    lon, lat = -87.65, 41.85
    names = ["ALPHA", "BRAVO", "CHARLIE"]
    features = [
        _line("ALPHA", [[lon - 0.02, lat], [lon + 0.02, lat]]),
        _line("BRAVO", [[lon, lat - 0.02], [lon, lat + 0.02]]),
        _line("CHARLIE", [[lon - 0.02, lat + 1e-4], [lon + 0.02, lat + 1e-4]]),
    ]
    pages = {"p1": names, "p2": names, "p3": names}
    (w, s, e, n), evidence = localize_pages(pages, features, {})

    boxes = [p["bbox"] for p in evidence["pages"].values() if p["bbox"]]
    assert boxes, "no page localized — the fixture, not the margins, is wrong"
    assert w == pytest.approx(min(b[0] for b in boxes) - MARGIN_LON_DEG)
    assert s == pytest.approx(min(b[1] for b in boxes) - MARGIN_LAT_DEG)
    assert e == pytest.approx(max(b[2] for b in boxes) + MARGIN_LON_DEG)
    assert n == pytest.approx(max(b[3] for b in boxes) + MARGIN_LAT_DEG)
    # and strictly looser than what any DECLARED source would have got
    assert MARGIN_LON_DEG > BUFFER_DEG
    assert MARGIN_LAT_DEG > BUFFER_DEG
