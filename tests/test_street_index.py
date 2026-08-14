"""Street-index prior channel: unit contracts + golden replays.

Unit tests exercise parse tolerance, tile dedupe (incl. the conflicting-
sheet drop rule), the tiled cached read with an injected annotate_fn, the
fuse_windows lens geometry against hand-computed cases, and the abstain
paths of index_priors.

Golden tests replay the REAL cached model reads (tests/data/index_038_
sonnet.json, tests/data/index_034_fable.json — byte copies of the raws
under work/g3/) through the production index_priors against the frozen
fixture tree: the 1913 post-renumbering volume must land its windows within
600 m of the recorded committed-sheet centroids, and the 1905 pre-1909
volume must produce ZERO windows under EMPTY_RENUMBERING (honest
abstention until the published 1909 conversion table is acquired).
"""

import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any

import pytest
from PIL import Image

from autogeoref.addresses import EMPTY_RENUMBERING
from autogeoref.affine import TO_3857
from autogeoref.annotate.failures import MalformedResponseError, ModelQualityError
from autogeoref.margins import PriorWindow
from autogeoref.names import load_aliases
from autogeoref.street_index import (
    HIT_RADIUS_M,
    IndexEntry,
    dedupe_entries,
    fuse_windows,
    index_priors,
    parse_index_entries,
    read_index,
)

DATA = Path(__file__).resolve().parent / "data"

# ~111319.49 m of 3857 easting per degree of longitude at the equator: keeps
# the synthetic-feature geometry hand-computable.
M_PER_DEG = 111319.490793


def _entry(street: str, lo: int, hi: int, sheet: str) -> IndexEntry:
    return IndexEntry(street=street, from_number=lo, to_number=hi, sheet=sheet)


def _feature(name: str, f_add: int, t_add: int, lon0: float, lon1: float) -> dict[str, Any]:
    """Synthetic E-W centerline segment on the equator with an l-side range."""
    return {
        "type": "Feature",
        "properties": {
            "street_nam": name,
            "l_f_add": str(f_add),
            "l_t_add": str(t_add),
            "r_f_add": "",
            "r_t_add": "",
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [[lon0, 0.0], [lon1, 0.0]],
        },
    }


# ----------------------------------------------------------------------
# parse_index_entries tolerance
# ----------------------------------------------------------------------


def test_parse_good_entries() -> None:
    raw = {
        "entries": [
            {"street": "Addison", "from": 2400, "to": 2518, "sheet": 92},
            {"street": "Byron ", "from": "100", "to": 198, "sheet": "12"},
        ]
    }
    entries = parse_index_entries(raw)
    assert entries == [
        _entry("Addison", 2400, 2518, "92"),
        _entry("Byron", 100, 198, "12"),
    ]


def test_parse_drops_malformed_entries_never_fatal() -> None:
    raw = {
        "entries": [
            {"street": "Addison", "from": 2400, "to": 2518, "sheet": 92},
            {"street": "", "from": 1, "to": 2, "sheet": 3},  # empty street
            {"street": "Byron", "from": "x", "to": 30, "sheet": 4},  # bad number
            {"street": "Cullom", "from": 600},  # missing keys
            {"street": None, "from": 1, "to": 2, "sheet": 5},  # non-str street
            {"street": "Dakin", "from": 1, "to": 2, "sheet": 1.5},  # non-int/str sheet
            {"street": "Eddy", "from": 1, "to": 2, "sheet": True},  # bool sheet
            "not even a dict",
            42,
        ]
    }
    assert parse_index_entries(raw) == [_entry("Addison", 2400, 2518, "92")]


def test_parse_tolerates_missing_or_bad_entries_key() -> None:
    assert parse_index_entries({}) == []
    assert parse_index_entries({"entries": "nope"}) == []
    assert parse_index_entries({"entries": None}) == []


# ----------------------------------------------------------------------
# dedupe_entries: overlap collapse + conflicting-sheet drop
# ----------------------------------------------------------------------


def test_dedupe_collapses_identical_duplicates() -> None:
    a = _entry("Addison", 2400, 2518, "92")
    b = _entry("addison ", 2400, 2518, "92")  # same key after normalization
    c = _entry("Byron", 100, 198, "12")
    assert dedupe_entries([a, b, c, a]) == [a, c]


def test_dedupe_drops_conflicting_sheet_readings_entirely() -> None:
    # The SAME (street, from, to) read as different sheets by two tiles:
    # at least one read is wrong — a misread must not become a prior.
    a = _entry("Addison", 2400, 2518, "92")
    b = _entry("ADDISON", 2400, 2518, "104")
    keep = _entry("Byron", 100, 198, "12")
    assert dedupe_entries([a, b, keep]) == [keep]


def test_dedupe_keeps_different_ranges_on_same_street() -> None:
    a = _entry("Addison", 2400, 2518, "92")
    b = _entry("Addison", 2520, 2644, "104")
    assert dedupe_entries([a, b]) == [a, b]


# ----------------------------------------------------------------------
# read_index: tiling, injected annotate_fn, caching
# ----------------------------------------------------------------------


def _index_page(tmp_path: Path) -> Path:
    path = tmp_path / "keymap_test.jpg"
    Image.new("RGB", (60, 300), "white").save(path)
    return path


def _namespace_dirs(cache: Path) -> list[Path]:
    return sorted(p for p in cache.iterdir() if p.is_dir())


def test_a_provider_refusal_never_becomes_a_cached_empty_index(tmp_path: Path) -> None:
    """A refusal is valid JSON, so it decodes and only a schema check rejects it.

    Read it out of the choke point unvalidated and `parse_index_entries` — which
    tolerates junk on purpose — turns it into zero entries with no exception, and
    the tile cache stores it. Every later run then serves the refusal from disk:
    an index that is permanently empty, never re-spends, and never errors.
    """
    page = _index_page(tmp_path)
    cache = tmp_path / "cache"
    refusal = {"type": "error", "code": "usage_limit_reached"}

    with pytest.raises(MalformedResponseError):
        read_index(page, annotate_fn=lambda _t: refusal, n_tiles=2, cache_dir=cache)
    assert not list(cache.rglob("*.json"))  # nothing poisoned the cache

    # a tile the model genuinely read and found nothing in is NOT a failure
    entries = read_index(page, annotate_fn=lambda _t: {"entries": []}, n_tiles=2, cache_dir=cache)
    assert entries == []
    assert len(list(cache.rglob("*.json"))) == 2


def test_read_index_tiles_calls_and_caches(tmp_path: Path) -> None:
    page = _index_page(tmp_path)
    cache = tmp_path / "cache"
    calls: list[Path] = []

    def fake_annotate(tile_path: Path) -> dict[str, Any]:
        calls.append(tile_path)
        i = len(calls)
        return {"entries": [{"street": f"Street{i}", "from": 100, "to": 198, "sheet": i}]}

    entries = read_index(page, annotate_fn=fake_annotate, n_tiles=5, cache_dir=cache)
    assert len(calls) == 5  # one model call per band
    assert [c.name for c in calls] == [f"tile{i}of5.jpg" for i in range(1, 6)]
    assert {e.street for e in entries} == {f"Street{i}" for i in range(1, 6)}
    # everything lives in ONE fingerprinted namespace directory: readable
    # stem prefix + hex fingerprint of the semantic inputs
    (namespace,) = _namespace_dirs(cache)
    assert re.fullmatch(r"keymap_test\.[0-9a-f]{16}", namespace.name)
    for i in range(1, 6):
        assert (namespace / f"tile{i}of5.jpg").exists()
        assert (namespace / f"tile{i}of5.json").exists()

    # unchanged repeat: every tile served from cache, zero new model calls
    again = read_index(page, annotate_fn=fake_annotate, n_tiles=5, cache_dir=cache)
    assert len(calls) == 5
    assert again == entries


def test_read_index_rereads_after_inplace_image_replacement(tmp_path: Path) -> None:
    page = _index_page(tmp_path)
    cache = tmp_path / "cache"
    calls: list[Path] = []

    def fake_annotate(tile_path: Path) -> dict[str, Any]:
        calls.append(tile_path)
        return {"entries": []}

    read_index(page, annotate_fn=fake_annotate, n_tiles=5, cache_dir=cache)
    assert len(calls) == 5
    # same path, different bytes: the stale crops/reads must NOT be reused
    Image.new("RGB", (60, 300), "gray").save(page)
    read_index(page, annotate_fn=fake_annotate, n_tiles=5, cache_dir=cache)
    assert len(calls) == 10
    assert len(_namespace_dirs(cache)) == 2


def test_read_index_rereads_on_tiling_change(tmp_path: Path) -> None:
    page = _index_page(tmp_path)
    cache = tmp_path / "cache"
    calls: list[Path] = []

    def fake_annotate(tile_path: Path) -> dict[str, Any]:
        calls.append(tile_path)
        return {"entries": []}

    read_index(page, annotate_fn=fake_annotate, n_tiles=5, overlap_frac=0.08, cache_dir=cache)
    assert len(calls) == 5
    # overlap changes the tile CONTENT even at the same tile names
    read_index(page, annotate_fn=fake_annotate, n_tiles=5, overlap_frac=0.12, cache_dir=cache)
    assert len(calls) == 10
    read_index(page, annotate_fn=fake_annotate, n_tiles=4, overlap_frac=0.08, cache_dir=cache)
    assert len(calls) == 14
    assert len(_namespace_dirs(cache)) == 3


def test_read_index_bare_and_qualified_model_share_namespace(tmp_path: Path) -> None:
    page = _index_page(tmp_path)
    cache = tmp_path / "cache"
    calls: list[Path] = []

    def fake_annotate(tile_path: Path) -> dict[str, Any]:
        calls.append(tile_path)
        return {"entries": []}

    read_index(page, model="claude-sonnet-5", annotate_fn=fake_annotate, n_tiles=5, cache_dir=cache)
    assert len(calls) == 5
    # the provider-qualified spelling of the SAME model reuses the cache
    read_index(
        page,
        model="anthropic:claude-sonnet-5",
        annotate_fn=fake_annotate,
        n_tiles=5,
        cache_dir=cache,
    )
    assert len(calls) == 5
    assert len(_namespace_dirs(cache)) == 1


def test_read_index_bands_cover_page_with_overlap(tmp_path: Path) -> None:
    page = _index_page(tmp_path)  # 60 x 300
    cache = tmp_path / "cache"

    def fake_annotate(tile_path: Path) -> dict[str, Any]:
        return {"entries": []}

    read_index(page, annotate_fn=fake_annotate, n_tiles=5, overlap_frac=0.08, cache_dir=cache)
    (namespace,) = _namespace_dirs(cache)
    heights = [Image.open(namespace / f"tile{i}of5.jpg").size for i in range(1, 6)]
    # full-width horizontal bands; interior bands carry overlap on both edges
    assert all(w == 60 for w, _h in heights)
    band = 300 / 5
    pad = band * 0.08
    assert heights[0][1] == math.ceil(band + pad) - 0  # edge band: one-sided overlap
    assert heights[2][1] == math.ceil(3 * band + pad) - math.floor(2 * band - pad)
    assert heights[4][1] == 300 - math.floor(4 * band - pad)


def test_read_index_dedupes_across_tile_overlap(tmp_path: Path) -> None:
    page = _index_page(tmp_path)
    cache = tmp_path / "cache"
    per_tile = [
        {"entries": [{"street": "Addison", "from": 2400, "to": 2518, "sheet": 92}]},
        # overlap re-read, identical -> collapses
        {"entries": [{"street": "Addison", "from": 2400, "to": 2518, "sheet": 92}]},
        # same key, DIFFERENT sheet -> the whole key is dropped
        {"entries": [{"street": "Byron", "from": 100, "to": 198, "sheet": 7}]},
        {"entries": [{"street": "Byron", "from": 100, "to": 198, "sheet": 8}]},
        {"entries": [{"street": "Cullom", "from": 600, "to": 719, "sheet": 114}]},
    ]

    def fake_annotate(tile_path: Path) -> dict[str, Any]:
        return per_tile.pop(0)

    entries = read_index(page, annotate_fn=fake_annotate, n_tiles=5, cache_dir=cache)
    assert entries == [
        _entry("Addison", 2400, 2518, "92"),
        _entry("Cullom", 600, 719, "114"),
    ]


def test_read_index_requires_cache_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cache_dir"):
        read_index(_index_page(tmp_path), annotate_fn=lambda _p: {"entries": []})


def test_read_index_enforces_model_quality_gate(tmp_path: Path) -> None:
    # conduct contract: the Sonnet-class minimum applies to this model path
    # exactly as it does to the annotator
    with pytest.raises(ModelQualityError):
        read_index(
            _index_page(tmp_path),
            model="claude-haiku-3",
            annotate_fn=lambda _p: {"entries": []},
            cache_dir=tmp_path / "cache",
        )


# ----------------------------------------------------------------------
# fuse_windows: lens geometry (hand-computed, on the equator where the
# mercator scale is exactly 1 and radii in ground meters == 3857 units)
# ----------------------------------------------------------------------


def test_fuse_containment_returns_smaller_window() -> None:
    big = PriorWindow(center_3857=(0.0, 0.0), radius_m=500.0)
    small = PriorWindow(center_3857=(10.0, 0.0), radius_m=100.0)
    assert fuse_windows(big, small) == small
    assert fuse_windows(small, big) == small


def test_fuse_identical_windows() -> None:
    w = PriorWindow(center_3857=(5.0, 0.0), radius_m=100.0)
    assert fuse_windows(w, w) == w


def test_fuse_partial_overlap_hand_computed() -> None:
    # a=(0,0) ra=100, b=(150,0) rb=100: d=150, x0=(150^2)/(2*150)=75,
    # h = sqrt(100^2 - 75^2) = sqrt(4375) ~ 66.144. The chord circle covers
    # both axial extremes (h=66.1 >= ra-x0=25 and >= x0-(d-rb)=25).
    a = PriorWindow(center_3857=(0.0, 0.0), radius_m=100.0)
    b = PriorWindow(center_3857=(150.0, 0.0), radius_m=100.0)
    fused = fuse_windows(a, b)
    assert fused is not None
    assert fused.center_3857 == pytest.approx((75.0, 0.0))
    assert fused.radius_m == pytest.approx(math.sqrt(4375.0))


def test_fuse_partial_overlap_smaller_disk_binds() -> None:
    # a=(0,0) ra=10, b=(95,0) rb=100: partial overlap (90 < d=95 < 110) but
    # x0 = (95^2 + 10^2 - 100^2)/(2*95) < 0, so A's cap pokes past the chord
    # circle and the minimal cover is disk A itself (corners + A's axial
    # extreme all lie on circle A).
    a = PriorWindow(center_3857=(0.0, 0.0), radius_m=10.0)
    b = PriorWindow(center_3857=(95.0, 0.0), radius_m=100.0)
    assert fuse_windows(a, b) == a
    assert fuse_windows(b, a) == a


def test_fuse_disjoint_returns_none() -> None:
    a = PriorWindow(center_3857=(0.0, 0.0), radius_m=100.0)
    b = PriorWindow(center_3857=(300.0, 0.0), radius_m=100.0)
    assert fuse_windows(a, b) is None
    # exact tangency: the lens is a single point — useless as a window
    c = PriorWindow(center_3857=(200.0, 0.0), radius_m=100.0)
    assert fuse_windows(a, c) is None


def test_fuse_symmetric_center_off_equator() -> None:
    # equal radii, partial overlap at a Chicago-like northing: the fused
    # center must sit at the midpoint regardless of the mercator scale
    y = 5_140_000.0
    a = PriorWindow(center_3857=(0.0, y), radius_m=100.0)
    b = PriorWindow(center_3857=(200.0, y), radius_m=100.0)
    fused = fuse_windows(a, b)
    assert fused is not None
    assert fused.center_3857 == pytest.approx((100.0, y))
    assert 0.0 < fused.radius_m < 100.0


# ----------------------------------------------------------------------
# index_priors: abstain paths + disambiguation (synthetic centerlines)
# ----------------------------------------------------------------------


def test_priors_abstain_on_no_range_match() -> None:
    features = [_feature("MAIN", 100, 198, 0.0, 0.001)]
    # range 900-998 matches no centerline field -> abstain (measured: such
    # entries produced zero wrong predictions; they just drop out)
    windows = index_priors([_entry("Main St", 900, 998, "5")], features, {})
    assert windows == {}


def test_priors_abstain_on_unknown_street() -> None:
    features = [_feature("MAIN", 100, 198, 0.0, 0.001)]
    windows = index_priors([_entry("Nowhere Av", 100, 198, "5")], features, {})
    assert windows == {}


def test_priors_empty_renumbering_abstains_everything() -> None:
    # the pre-1909 honest state: EMPTY_RENUMBERING converts nothing, so
    # every entry abstains rather than matching pre-renumbering numbers
    # against modern ranges
    features = [_feature("MAIN", 100, 198, 0.0, 0.001)]
    windows = index_priors(
        [_entry("Main St", 100, 198, "5")], features, {}, renumbering=EMPTY_RENUMBERING
    )
    assert windows == {}


def test_priors_modern_passthrough_produces_window() -> None:
    features = [_feature("MAIN", 100, 198, 0.0, 0.001)]
    windows = index_priors([_entry("Main St", 100, 198, "5")], features, {}, renumbering=None)
    assert set(windows) == {"5"}
    w = windows["5"]
    assert w.radius_m == HIT_RADIUS_M
    # midpoint 149 -> fraction (149-100)/98 = 0.5 -> lon 0.0005
    assert w.center_3857[0] == pytest.approx(0.0005 * M_PER_DEG, rel=1e-6)


def test_priors_inconsistent_page_abstains() -> None:
    # two rows for the same page resolving ~11 km apart: mutually
    # inconsistent index reads must not average into a fake window
    features = [
        _feature("MAIN", 100, 198, 0.0, 0.001),
        _feature("FAR", 100, 198, 0.1, 0.101),
    ]
    entries = [_entry("Main St", 100, 198, "5"), _entry("Far St", 100, 198, "5")]
    assert index_priors(entries, features, {}) == {}


def test_priors_disambiguate_repeated_range_toward_unambiguous_centroid() -> None:
    # ELM's range repeats on two segments (one near MAIN, one ~5.5 km east);
    # the single-hit MAIN entry anchors the page, so ELM resolves to its
    # near candidate and the page gets a tight window
    features = [
        _feature("MAIN", 100, 198, 0.0, 0.001),
        _feature("ELM", 100, 198, 0.001, 0.002),
        _feature("ELM", 100, 198, 0.05, 0.051),
    ]
    entries = [_entry("Main St", 100, 198, "5"), _entry("Elm St", 100, 198, "5")]
    windows = index_priors(entries, features, {})
    assert set(windows) == {"5"}
    # centroid of MAIN@0.0005 deg and near-ELM@0.0015 deg = 0.001 deg
    assert windows["5"].center_3857[0] == pytest.approx(0.001 * M_PER_DEG, rel=1e-6)


# ----------------------------------------------------------------------
# Golden replays from the REAL cached model reads
# ----------------------------------------------------------------------


def _committed_centroids(results_dir: Path) -> dict[str, tuple[float, float]]:
    """Recorded committed-sheet GCP centroids, exactly as the validation
    harness computes them."""
    centroids: dict[str, tuple[float, float]] = {}
    for f in sorted(results_dir.glob("p*.json")):
        r = json.loads(f.read_text())
        if not str(r.get("status", "")).startswith("OK"):
            continue
        pts = [
            TO_3857.transform(*ft["geometry"]["coordinates"])
            for ft in (r.get("gcps_geojson") or {}).get("features") or []
            if "synthetic" not in (ft["properties"].get("note") or "")
        ]
        if pts:
            centroids[str(r["page"])] = (
                sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts),
            )
    return centroids


@pytest.mark.golden
def test_golden_038_windows_hit_recorded_centroids(fixtures_dir: Path, aliases_dir: Path) -> None:
    """1913 (post-renumbering) replay: production index_priors on the real
    cached sonnet read must land >= 80% of validatable page windows within
    600 m of the recorded committed-sheet centroid."""
    raw = json.loads((DATA / "index_038_sonnet.json").read_text())
    entries = dedupe_entries(parse_index_entries(raw))
    assert len(entries) == 39  # the recorded read

    aliases = load_aliases(aliases_dir / "aliases-sanborn01790_038.json")
    features = json.loads((fixtures_dir / "reference" / "street_center_lines.geojson").read_text())[
        "features"
    ]
    windows = index_priors(entries, features, aliases, renumbering=None)

    centroids = _committed_centroids(fixtures_dir / "sanborn01790_038" / "results")
    dists = {
        page: math.hypot(
            w.center_3857[0] - centroids[page][0], w.center_3857[1] - centroids[page][1]
        )
        for page, w in windows.items()
        if page in centroids
    }
    assert dists, "no window landed on a committed page — replay is broken"
    within = sum(1 for d in dists.values() if d <= 600.0)
    assert within / len(dists) >= 0.8  # the coverage target

    # Pinned exact numbers from the fixture.
    # The experiment measured per-ENTRY: 15/15 within 600 m, median 229 m,
    # using nearest-hit-to-recorded-centroid disambiguation (truth-peeking,
    # unavailable in production). Production disambiguates toward the
    # page's own unambiguous-entry centroid and averages a page's entries,
    # so the per-PAGE numbers differ slightly:
    assert len(dists) == PINNED_N_PAGES
    assert within == PINNED_WITHIN_600
    assert median(dists.values()) == pytest.approx(PINNED_MEDIAN_M, abs=1.0)


# Pinned from recorded reads and frozen fixtures (see the
# golden test above for why per-PAGE numbers can differ from the per-entry
# experiment). This run: 22 pages got windows, 15 of them have recorded
# committed centroids, 15/15 within 600 m (100%), median 228.8 m, worst
# p95 at 400.4 m — the production per-page combination reproduces the
# experiment's per-entry quality (15/15, median 229 m) almost exactly.
PINNED_N_PAGES = 15
PINNED_WITHIN_600 = 15
PINNED_MEDIAN_M = 228.8


@pytest.mark.golden
def test_golden_034_pre1909_wall_abstains_everything(fixtures_dir: Path, aliases_dir: Path) -> None:
    """1905 (PRE-renumbering) replay: the real cached fable read parsed 16
    correct rows, but under EMPTY_RENUMBERING every entry must abstain —
    ZERO windows (the measured pre-1909 wall; honest abstention until the
    published 1909 conversion table is acquired)."""
    raw = json.loads((DATA / "index_034_fable.json").read_text())
    entries = dedupe_entries(parse_index_entries(raw))
    assert len(entries) == 16  # the rows WERE read — abstention is downstream

    aliases = load_aliases(aliases_dir / "aliases-sanborn01790_034.json")
    features = json.loads((fixtures_dir / "reference" / "street_center_lines.geojson").read_text())[
        "features"
    ]
    windows = index_priors(entries, features, aliases, renumbering=EMPTY_RENUMBERING)
    assert windows == {}
