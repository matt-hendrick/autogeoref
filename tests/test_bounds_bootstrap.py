"""The bounds bootstrap: derived search bounds must be evidence, not vibes.

What matters here is the shape of the guarantee, pinned without a model call:
a page localizes only where SEVERAL of its street names meet in one place; a
name combination that recurs across the city abstains rather than vote twice;
too few localized pages is a refusal with instructions, not a citywide guess;
and a persisted derivation replays without re-deriving (the
``volume-constants.json`` precedent).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autogeoref import bounds_bootstrap as bb
from autogeoref.bounds import BoundsError
from autogeoref.paths import VolumePaths


def _line(name: str, coords: list[list[float]]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {"street_nam": name, "street_typ": "AVE"},
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def _cluster(lon: float, lat: float, names: list[str]) -> list[dict[str, Any]]:
    """Several named streets meeting around (lon, lat): half horizontal, half vertical."""
    out = []
    for i, name in enumerate(names):
        if i % 2 == 0:
            out.append(_line(name, [[lon - 0.02, lat + i * 1e-3], [lon + 0.02, lat + i * 1e-3]]))
        else:
            out.append(_line(name, [[lon + i * 1e-3, lat - 0.02], [lon + i * 1e-3, lat + 0.02]]))
    return out


def test_pages_localize_where_their_names_co_occur() -> None:
    """Three sampled pages naming streets that meet near one corner must produce
    bounds around that corner — and a decoy segment reusing ONE of the names far
    away must not drag the box there: a single name is not a co-occurrence."""
    meet = _cluster(-87.65, 41.90, ["ALPHA", "BRAVO", "CHARLIE", "DELTA"])
    decoy = [_line("ALPHA", [[-87.55, 41.70], [-87.53, 41.70]])]
    pages = {
        "p1": ["ALPHA", "BRAVO", "CHARLIE"],
        "p5": ["BRAVO", "CHARLIE", "DELTA"],
        "p9": ["ALPHA", "CHARLIE", "DELTA"],
    }
    bounds, evidence = bb.localize_pages(pages, meet + decoy, {})
    w, s, e, n = bounds
    assert w < -87.65 < e and s < 41.90 < n
    assert evidence["localized"] == 3
    # the decoy corner stays outside: no page's names co-occur there
    assert not (w <= -87.55 <= e and s <= 41.70 <= n)


def test_a_name_set_that_recurs_across_the_city_abstains() -> None:
    """The same three names meeting in TWO distant places is exactly the numbered
    -streets trap; the page must abstain, and with every page abstaining the
    volume must refuse with instructions, never average the two corners."""
    names = ["ALPHA", "BRAVO", "CHARLIE"]
    features = _cluster(-87.70, 41.95, names) + _cluster(-87.55, 41.70, names)
    pages = {p: list(names) for p in ("p1", "p2", "p3", "p4")}
    with pytest.raises(BoundsError, match="key map"):
        bb.localize_pages(pages, features, {})


def test_too_few_matched_names_is_an_abstention_not_a_guess() -> None:
    """Two matched names co-occur somewhere in almost any grid city, so a page
    with fewer than MIN_NAMES_PER_PAGE matched names must not vote at all."""
    features = _cluster(-87.65, 41.90, ["ALPHA", "BRAVO", "CHARLIE", "DELTA"])
    localized, matched = bb.localize_page(
        ["ALPHA", "BRAVO"], bb._name_cells(features, {}, "street_nam", "street_typ"), {}
    )
    assert localized is None and matched == 2


def test_one_lying_page_is_dropped_when_enough_pages_agree() -> None:
    """A page whose (misread) names co-occur once in the WRONG part of the city
    must not inflate the union: with three or more agreeing pages the outlier is
    dropped and recorded, and the bounds stay the honest pages'."""
    names = ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]
    liar_names = ["ECHO", "FOXTROT", "GOLF"]
    features = _cluster(-87.70, 41.95, names) + _cluster(-87.53, 41.66, liar_names)
    pages = {
        "p1": names[:3],
        "p5": names[1:],
        "p9": [names[0], names[2], names[3]],
        "p12": liar_names,
    }
    bounds, evidence = bb.localize_pages(pages, features, {})
    w, s, e, n = bounds
    assert evidence["outliers"] == ["p12"]
    assert evidence["localized"] == 3
    assert evidence["pages"]["p12"]["outlier"] is True
    assert not (w <= -87.53 <= e and s <= 41.66 <= n), "liar corner stays outside"


def test_two_agreeing_pages_and_one_liar_refuse_rather_than_pick_sides() -> None:
    """At 2-vs-1 the quorum is not met once the outlier drops: refusing beats
    trusting a majority of two."""
    names = ["ALPHA", "BRAVO", "CHARLIE"]
    liar_names = ["ECHO", "FOXTROT", "GOLF"]
    features = _cluster(-87.70, 41.95, names) + _cluster(-87.53, 41.66, liar_names)
    pages = {"p1": names, "p5": names, "p9": liar_names}
    with pytest.raises(BoundsError, match="agreed"):
        bb.localize_pages(pages, features, {})


def test_persisted_bounds_round_trip_and_corruption(tmp_path: Path) -> None:
    """A persisted derivation replays free; a corrupt file re-derives instead of
    crashing the run that trusted it."""
    paths = VolumePaths(root=tmp_path / "vol_a")
    paths.root.mkdir(parents=True)
    assert bb.persisted_bounds(paths) is None
    bb.bounds_file(paths).write_text(json.dumps({"bounds": [-87.7, 41.8, -87.6, 41.9]}))
    assert bb.persisted_bounds(paths) == (-87.7, 41.8, -87.6, 41.9)
    bb.bounds_file(paths).write_text("{not json")
    assert bb.persisted_bounds(paths) is None
    # parseable garbage: an inverted box would sail into an empty centerline
    # index and die two stages later pointing at the wrong culprit
    bb.bounds_file(paths).write_text(json.dumps({"bounds": [-87.6, 41.9, -87.7, 41.8]}))
    assert bb.persisted_bounds(paths) is None


def test_sampling_spreads_through_the_volume() -> None:
    """Coverage drifts across a volume's page order, so the sample must include
    both ends and stay within budget — never the first k pages."""
    pages = [f"p{i}" for i in range(100)]
    sample = bb.sample_evenly(pages, 12)
    assert len(sample) <= 12
    assert "p0" in sample and "p99" in sample
    assert bb.sample_evenly(["p1", "p2"], 12) == ["p1", "p2"]


def test_sampling_uses_numeric_page_order_and_skips_map_less_plates(tmp_path: Path) -> None:
    """`manifest_pages` sorts lexicographically (p10 before p2) and still lists
    plates with no small on disk (ptitl). Sampling from that order is front-heavy
    and burns slots on pages that cannot name a single street."""
    paths = VolumePaths(root=tmp_path / "vol_a")
    paths.sheets.mkdir(parents=True)
    manifest: dict[str, dict[str, Any]] = {p: {} for p in ("p1", "p2", "p10", "p11", "ptitl")}
    paths.manifest.write_text(json.dumps(manifest))
    for p in ("p1", "p2", "p10", "p11"):  # ptitl never got a small
        (paths.sheets / f"{p}_small.jpg").write_bytes(b"jpg")

    assert bb.eligible_pages(paths) == ["p1", "p2", "p10", "p11"]


def test_spend_false_localizes_from_cache_and_never_annotates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run whose flags promised a capped/zero annotation spend (--limit,
    --no-annotate) must bootstrap from CACHED reads only: `annotate_volume` is
    not called at all, and the derivation still lands and persists when the
    cache suffices."""
    from autogeoref import annotate_volume as av_mod
    from autogeoref.config.model import CityConfig, VolumeConfig

    def _boom(*a: object, **k: object) -> None:
        raise AssertionError("spend=False must never reach annotate_volume")

    monkeypatch.setattr(av_mod, "annotate_volume", _boom)

    names = ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]
    centerlines = tmp_path / "cl.geojson"
    centerlines.write_text(
        json.dumps({"type": "FeatureCollection", "features": _cluster(-87.65, 41.90, names)})
    )
    paths = VolumePaths(root=tmp_path / "vol_a")
    paths.sheets.mkdir(parents=True)
    paths.annotations.mkdir()
    manifest: dict[str, dict[str, Any]] = {p: {} for p in ("p1", "p2", "p3")}
    paths.manifest.write_text(json.dumps(manifest))
    for p in manifest:
        (paths.sheets / f"{p}_small.jpg").write_bytes(b"jpg")
        (paths.annotations / f"{p}.json").write_text(
            json.dumps({"streets": [{"name": n} for n in names[:3]]})
        )
    city = CityConfig(name="Test", centerlines_path=centerlines, aliases_dir=tmp_path, volumes={})

    bounds = bb.derive_bounds(paths, "vol_a", city, VolumeConfig(identifier="vol_a"), spend=False)
    w, s, e, n = bounds
    assert w < -87.65 < e and s < 41.90 < n
    assert bb.persisted_bounds(paths) == bounds, "derivation persisted for replay"


def test_an_osm_default_city_cannot_bootstrap(tmp_path: Path) -> None:
    """OSM cities fetch centerlines BY bounds — the bootstrap must name the
    cycle and demand a declared bbox, not attempt a country-sized fetch."""
    from autogeoref.config.model import CityConfig, VolumeConfig

    city = CityConfig(
        name="Elsewhere",
        centerlines_path=tmp_path / "absent.geojson",
        aliases_dir=tmp_path,
        volumes={},
    )
    paths = VolumePaths(root=tmp_path / "vol_a")
    with pytest.raises(BoundsError, match="bounds_bbox"):
        bb.derive_bounds(paths, "vol_a", city, VolumeConfig(identifier="vol_a"))
