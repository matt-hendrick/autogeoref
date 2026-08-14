"""The >=2-independent-verifiers acceptance path.

Unit tests drive every vote combination through the REAL stage on synthetic
records (identity-scale affine, hand-built centerline street, injectable
voucher nodes). Golden tests replay the real range-verified evidence: _024 p1
(47 consensus numerals) accepted at its recorded placement and refuted one
block off, and a whole-volume _034 sweep proving the anti-red-flag invariant
(no status may change except revoked -> verified; here, zero accepts).
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autogeoref.address_channel import (
    address_vote,
    address_vote_diagnostics,
)
from autogeoref.addresses import AddressNumeral
from autogeoref.affine import TO_4326
from autogeoref.annotate.providers import model_cache_key
from autogeoref.report import build_report
from autogeoref.seam import shift_gcps_geojson
from autogeoref.verified_accept import stage_verified_accept
from autogeoref.volume import (
    STATUS_VERIFIED_PREFIX,
    is_committed,
    status_verified,
)

DATA = Path(__file__).resolve().parent / "data"

# identity-scale affine: world = (X0 + px, Y0 - py); Chicago-ish 3857 origin
X0, Y0 = -9760000.0, 5140000.0


class _Paths:
    def __init__(self, root: Path) -> None:
        self.results = root / "results"
        self.annotations = root / "annotations"
        self.sheets = root / "sheets"
        self.manifest = self.sheets / "manifest.json"


def _gcp_feature(px: float, py: float) -> dict[str, Any]:
    lng, lat = TO_4326.transform(X0 + px, Y0 - py)
    return {
        "type": "Feature",
        "properties": {"image": [px, py], "note": "auto: A x B"},
        "geometry": {"type": "Point", "coordinates": [lng, lat]},
    }


def _revoked_record(page: str) -> dict[str, Any]:
    return {
        "page": page,
        "status": "REJECTED (rescue revoked: anchors share one street)",
        "layer": None,
        "gcps_geojson": {
            "type": "FeatureCollection",
            "features": [
                _gcp_feature(100, 100),
                _gcp_feature(800, 200),
                _gcp_feature(300, 900),
            ],
        },
    }


def _main_street_feature() -> dict[str, Any]:
    """E-W 'MAIN ST' along py=500, px 0..536: 100 house numbers per 134 m."""
    a = TO_4326.transform(X0 + 0.0, Y0 - 500.0)
    b = TO_4326.transform(X0 + 536.0, Y0 - 500.0)
    return {
        "type": "Feature",
        "properties": {
            "street_nam": "MAIN",
            "l_f_add": "101",
            "l_t_add": "499",
            "l_parity": "O",
            "r_f_add": "100",
            "r_t_add": "498",
            "r_parity": "E",
        },
        "geometry": {"type": "LineString", "coordinates": [list(a), list(b)]},
    }


def _numeral(value: int, px: float, py: float = 500.0) -> dict[str, Any]:
    return {"value": value, "bbox": [px - 10, py - 6, px + 10, py + 6], "street": "MAIN ST."}


#: three numerals whose implied positions sit exactly on their blocks
GOOD_NUMERALS = [_numeral(151, 67), _numeral(301, 268), _numeral(449, 469)]


@pytest.fixture
def vol(tmp_path: Path) -> _Paths:
    paths = _Paths(tmp_path)
    paths.results.mkdir()
    paths.annotations.mkdir()
    paths.sheets.mkdir()
    paths.manifest.write_text(
        json.dumps(
            {
                "p5": {
                    "full_size": [1000, 1000],
                    "small_size": [1000, 1000],
                    "scale": 1.0,
                    "file": "p5_small.jpg",
                }
            }
        )
    )
    return paths


def _write(paths: _Paths, record: dict[str, Any]) -> Path:
    rp = paths.results / f"p{record['page']}.json"
    rp.write_text(json.dumps(record))
    return rp


def _write_sidecars(paths: _Paths, page: str, numerals: list[dict[str, Any]]) -> None:
    for model in ("sonnet", "opus"):
        (paths.annotations / f"p{page}.v2.{model}.json").write_text(
            json.dumps({"streets": [], "page_number_seen": None, "address_numerals": numerals})
        )


def _vouch_nodes_for(record: dict[str, Any]) -> dict[Any, Any]:
    """Two committed neighbors holding the record's exact node positions."""
    nodes: dict[Any, Any] = {}
    from autogeoref.affine import TO_3857

    for ft in record["gcps_geojson"]["features"][:2]:
        x, y = TO_3857.transform(*ft["geometry"]["coordinates"])
        nodes[(round(x, 1), round(y, 1))] = [("77", (x, y))]
    return nodes


FEATURES = [_main_street_feature()]


def _run(
    paths: _Paths,
    era: str = "modern",
    vouch: dict[Any, Any] | None = None,
    channels: list[str] | None = None,
) -> dict[str, Any]:
    return stage_verified_accept(
        paths,
        FEATURES,
        aliases={},
        address_era=era,
        vouch_nodes=vouch if vouch is not None else {},
        # None (the default here and in the stage) = every channel may vote, which
        # is what every direct caller and every golden replay gets
        channels=channels,
    )


# ------------------------------------------------------------ vote combos


def test_zero_votes_stays_revoked(vol: _Paths) -> None:
    rp = _write(vol, _revoked_record("5"))
    scored = _run(vol)
    r = json.loads(rp.read_text())
    assert r["status"].startswith("REJECTED")
    assert scored["5"]["accepted"] is False
    assert r["verified_accept"]["votes"] == {
        "corroboration": None,
        "junction": None,
        "addresses": None,
    }


def test_single_channel_never_suffices(vol: _Paths) -> None:
    rec = _revoked_record("5")
    rec["junction_snap"] = {"supports": True}
    rp = _write(vol, rec)
    scored = _run(vol)
    assert scored["5"]["votes"]["junction"] is True
    assert scored["5"]["accepted"] is False
    assert json.loads(rp.read_text())["status"].startswith("REJECTED")


def test_addresses_alone_is_one_channel_not_two(vol: _Paths) -> None:
    """Many in-block numerals are still ONE channel — never a solo accept."""
    rec = _revoked_record("5")
    rp = _write(vol, rec)
    _write_sidecars(vol, "5", GOOD_NUMERALS)
    scored = _run(vol)
    assert scored["5"]["votes"]["addresses"] is True
    assert scored["5"]["accepted"] is False
    assert json.loads(rp.read_text())["status"].startswith("REJECTED")


def test_address_diagnostic_retains_vote_and_names_two_model_floor() -> None:
    per_model = {"sonnet": [AddressNumeral(151, (57, 494, 77, 506), "MAIN ST.")]}
    vote, detail = address_vote(
        per_model,
        [[X0, 1, 0], [Y0, 0, -1]],
        1.0,
        FEATURES,
        {},
        "modern",
    )
    diagnostic_vote, diagnostic_detail, reason = address_vote_diagnostics(
        per_model,
        [[X0, 1, 0], [Y0, 0, -1]],
        1.0,
        FEATURES,
        {},
        "modern",
        successful_models={"sonnet"},
    )
    assert (diagnostic_vote, diagnostic_detail["skipped"]) == (vote, detail["skipped"])
    assert reason == "fewer_than_2_successful_distinct_model_readings"


def _consensus_numeral(value: int, px: float, street: str | None = "MAIN ST.") -> AddressNumeral:
    return AddressNumeral(value, (px - 10, 494, px + 10, 506), street)


def _abstention_reason(
    numerals: list[AddressNumeral],
    era: str = "modern",
    opus_numerals: list[AddressNumeral] | None = None,
) -> tuple[bool | None, str | None]:
    vote, _, reason = address_vote_diagnostics(
        {"sonnet": numerals, "opus": numerals if opus_numerals is None else opus_numerals},
        [[X0, 1, 0], [Y0, 0, -1]],
        1.0,
        FEATURES,
        {},
        era,
        successful_models={"sonnet", "opus"},
    )
    return vote, reason


def test_diagnostics_pass_a_real_vote_through_unreasoned() -> None:
    numerals = [
        _consensus_numeral(151, 67),
        _consensus_numeral(301, 268),
        _consensus_numeral(449, 469),
    ]
    assert _abstention_reason(numerals) == (True, None)


def test_abstention_reason_era_unknown() -> None:
    vote, reason = _abstention_reason([_consensus_numeral(151, 67)], era="unknown")
    assert (vote, reason) == (None, "address_era_unknown")


def test_abstention_reason_no_cross_model_consensus() -> None:
    vote, reason = _abstention_reason(
        [_consensus_numeral(151, 67)], opus_numerals=[_consensus_numeral(999, 400)]
    )
    assert (vote, reason) == (None, "no_cross_model_consensus_numeral")


def test_abstention_reason_consensus_without_street_hint() -> None:
    vote, reason = _abstention_reason([_consensus_numeral(151, 67, street=None)])
    assert (vote, reason) == (None, "consensus_numerals_without_street_hint")


def test_abstention_reason_no_usable_era_conversion() -> None:
    """A renumbered-era volume with no table entry abstains, never guesses."""
    vote, reason = _abstention_reason([_consensus_numeral(151, 67)], era="renumbered")
    assert (vote, reason) == (None, "no_usable_era_conversion")


def test_abstention_reason_no_addressable_segment() -> None:
    vote, reason = _abstention_reason([_consensus_numeral(151, 67, street="ELM ST.")])
    assert (vote, reason) == (None, "no_matching_addressable_centerline_segment")


def test_abstention_reason_fewer_than_3_votable() -> None:
    vote, reason = _abstention_reason([_consensus_numeral(151, 67), _consensus_numeral(301, 268)])
    assert (vote, reason) == (None, "fewer_than_3_votable_numerals")


def test_abstention_reason_mixed_contradictory_evidence() -> None:
    """An in-block majority plus one numeral far off the band abstains."""
    numerals = [
        _consensus_numeral(151, 67),
        _consensus_numeral(301, 268),
        _consensus_numeral(449, 469),
        _consensus_numeral(429, 67),
    ]
    vote, reason = _abstention_reason(numerals)
    assert (vote, reason) == (None, "mixed_contradictory_evidence")


def test_abstention_reason_insufficient_in_block_support() -> None:
    """Votable but neither an in-block majority nor a clean zero: the fallback."""
    numerals = [
        _consensus_numeral(151, 67),
        _consensus_numeral(301, 67),
        _consensus_numeral(449, 67),
    ]
    vote, reason = _abstention_reason(numerals)
    assert (vote, reason) == (None, "enough_votable_insufficient_in_block_support")


def test_junction_plus_addresses_accepts(vol: _Paths) -> None:
    rec = _revoked_record("5")
    rec["junction_snap"] = {"supports": True, "separation_ratio": 1.4}
    rp = _write(vol, rec)
    _write_sidecars(vol, "5", GOOD_NUMERALS)
    scored = _run(vol)
    assert scored["5"]["accepted"] is True
    r = json.loads(rp.read_text())
    assert r["status"] == "OK (verified: junction+addresses)"
    assert r["verified_accept"]["previous_status"].startswith("REJECTED (rescue revoked")
    assert is_committed(r)  # OK prefix inherits committed semantics (intended)


def _vouch_with_offsets(rec: dict[str, Any], offsets_m: list[float]) -> dict[Any, Any]:
    """Vouch nodes for the record's first features, displaced by offsets_m."""
    from autogeoref.affine import TO_3857

    nodes: dict[Any, Any] = {}
    for ft, off in zip(rec["gcps_geojson"]["features"], offsets_m, strict=False):
        x, y = TO_3857.transform(*ft["geometry"]["coordinates"])
        nodes[(round(x, 1), round(y, 1))] = [("77", (x + off, y))]
    return nodes


def test_corroboration_fringe_plus_junction_accepts(vol: _Paths) -> None:
    """One node at tolerance + one within 2x is the corroboration channel's YES.

    The near-fringe contract: the strict
    standalone gate (>=2 nodes <=8 m) left this page revoked, but inside
    verified-accept the near-second-node shape is one channel, and with
    junction support the >=2-channel rule promotes.
    """
    rec = _revoked_record("5")
    rec["junction_snap"] = {"supports": True}
    rp = _write(vol, rec)
    scored = _run(vol, vouch=_vouch_with_offsets(rec, [0.0, 10.0]))
    assert scored["5"]["votes"]["corroboration"] is True
    assert scored["5"]["accepted"] is True
    assert json.loads(rp.read_text())["status"] == "OK (verified: corroboration+junction)"


def test_single_agreeing_node_never_votes_corroboration(vol: _Paths) -> None:
    """One agreeing node stays worthless — measured, not oversight
    (half-block-shifted bad sheets keep one
    node on their aligned axis plus a supporting junction verdict)."""
    rec = _revoked_record("5")
    rec["junction_snap"] = {"supports": True}
    rp = _write(vol, rec)
    scored = _run(vol, vouch=_vouch_with_offsets(rec, [0.0]))
    assert scored["5"]["votes"]["corroboration"] is None
    assert scored["5"]["accepted"] is False
    assert json.loads(rp.read_text())["status"].startswith("REJECTED (rescue revoked")


def test_second_node_beyond_near_band_never_votes(vol: _Paths) -> None:
    rec = _revoked_record("5")
    rec["junction_snap"] = {"supports": True}
    _write(vol, rec)
    scored = _run(vol, vouch=_vouch_with_offsets(rec, [0.0, 16.5]))
    assert scored["5"]["votes"]["corroboration"] is None
    assert scored["5"]["accepted"] is False


def test_addresses_mixed_evidence_abstains(vol: _Paths) -> None:
    """An in-block majority with a numeral hundreds of numbers off abstains.

    The _022 p60 lesson: 4-of-6 in-block coexisting with numerals 306-308
    numbers off voted YES and held that YES 200 m into the perpendicular
    blind flank. Contradictory evidence supports nothing either way.
    """
    rec = _revoked_record("5")
    rec["junction_snap"] = {"supports": True}
    rp = _write(vol, rec)
    # three clean in-block numerals plus one votable numeral ~278 numbers off
    _write_sidecars(vol, "5", [*GOOD_NUMERALS, _numeral(429, 67)])
    scored = _run(vol)
    assert scored["5"]["votes"]["addresses"] is None
    assert "abstained" in scored["5"]["addresses"]
    assert scored["5"]["accepted"] is False
    assert json.loads(rp.read_text())["status"].startswith("REJECTED (rescue revoked")


def test_unwarpable_gcp_set_never_promotes(vol: _Paths) -> None:
    """Two yes votes cannot promote a record serving would choke on.

    The p60 shape: a healthy PIXEL rectangle whose GCPs carry only TWO
    distinct world points. `fit_affine_checked` only ranks the pixel side, so
    the record scores normally — but gdalwarp's refit from the raw GCPs is
    singular and one such promotion blocked a whole volume's warp stage.
    """
    rec = _revoked_record("5")
    wa = TO_4326.transform(X0 + 100.0, Y0 - 100.0)
    wb = TO_4326.transform(X0 + 100.0, Y0 - 900.0)

    def ft(px: float, py: float, w: tuple[float, float]) -> dict[str, Any]:
        return {
            "type": "Feature",
            "properties": {"image": [px, py], "note": "auto: A x B"},
            "geometry": {"type": "Point", "coordinates": list(w)},
        }

    rec["gcps_geojson"]["features"] = [
        ft(100, 100, wa),
        ft(800, 100, wa),
        ft(100, 900, wb),
        ft(800, 900, wb),
    ]
    rec["junction_snap"] = {"supports": True}
    rp = _write(vol, rec)
    from autogeoref.affine import TO_3857

    vouch: dict[Any, Any] = {}
    for w in (wa, wb):
        x, y = TO_3857.transform(*w)
        vouch[(round(x, 1), round(y, 1))] = [("77", (x, y))]
    scored = _run(vol, vouch=vouch)
    assert scored["5"]["votes"]["corroboration"] is True
    assert scored["5"]["votes"]["junction"] is True
    assert scored["5"]["accepted"] is False
    assert scored["5"]["unwarpable_gcps"] is True
    assert json.loads(rp.read_text())["status"].startswith("REJECTED (rescue revoked")


# ------------------------------------------------- the channel allow-list


def test_an_undeclared_channel_does_not_vote_off_evidence_left_on_disk(vol: _Paths) -> None:
    """`channels` MUTES a channel — it does not merely skip a producer stage.

    The failure it prevents: a maintainer narrows `evidence_channels` to ["junction"] to silence
    a misbehaving addresses channel — and the sidecars and escalation tier caches are still in
    annotations/, so the addresses channel keeps voting, and keeps REFUTING (it is the only
    channel permitted to). The one thing an operator would reach for the key to do would be the
    one thing it could not do. This is the whole mechanism now that the addresses channel has NO
    producer stage: its evidence is always evidence left on disk, so muting it is the only way
    to stop it.
    """
    rec = _revoked_record("5")
    rec["junction_snap"] = {"supports": True, "separation_ratio": 1.4}
    rp = _write(vol, rec)
    _write_sidecars(vol, "5", GOOD_NUMERALS)  # the evidence is on disk either way

    heard = _run(vol, channels=["junction", "addresses"])
    assert heard["5"]["votes"]["addresses"] is True, "declared: the sidecars speak"
    assert heard["5"]["accepted"] is True

    _write(vol, rec)  # reset the record the first run promoted
    muted = _run(vol, channels=["junction"])
    assert muted["5"]["votes"]["addresses"] is None, "undeclared: the sidecars are mute"
    assert muted["5"]["votes"]["junction"] is True, "the declared channel still votes"
    assert muted["5"]["accepted"] is False, "one channel is not two"
    assert json.loads(rp.read_text())["status"].startswith("REJECTED")


def test_a_declared_channel_that_abstains_on_every_page_warns(
    vol: _Paths, caplog: pytest.LogCaptureFixture
) -> None:
    """The `_017` failure in its GENERAL form: a channel that is on, and silent.

    Forgetting a flag is impossible now, but the STATE is still reachable with no
    flag at all — missing smalls, an absent [cv] extra, a model that read no
    numerals — and it looks exactly like a clean funnel. The config key cannot
    prevent that; only saying so can.
    """
    _write(vol, _revoked_record("5"))  # no junction verdict, no sidecars: both silent
    with caplog.at_level(logging.WARNING):
        _run(vol, channels=["junction", "addresses"])
    assert (
        "the junction channel is DECLARED but abstained on ALL 1 provisional pages" in caplog.text
    )
    assert (
        "the addresses channel is DECLARED but abstained on ALL 1 provisional pages" in caplog.text
    )

    # ...and it does NOT cry wolf about a channel that actually spoke
    caplog.clear()
    rec = _revoked_record("5")
    rec["junction_snap"] = {"supports": True}
    _write(vol, rec)
    _write_sidecars(vol, "5", GOOD_NUMERALS)
    with caplog.at_level(logging.WARNING):
        _run(vol, channels=["junction", "addresses"])
    assert "abstained on ALL" not in caplog.text


def test_corroboration_plus_junction_accepts(vol: _Paths) -> None:
    rec = _revoked_record("5")
    rec["junction_snap"] = {"supports": True}
    rp = _write(vol, rec)
    scored = _run(vol, vouch=_vouch_nodes_for(rec))
    assert scored["5"]["votes"]["corroboration"] is True
    assert scored["5"]["accepted"] is True
    assert json.loads(rp.read_text())["status"] == "OK (verified: corroboration+junction)"


def test_junction_abstain_does_not_block(vol: _Paths) -> None:
    """A junction abstain is an ABSENCE of evidence, not evidence against."""
    rec = _revoked_record("5")
    rec["junction_snap"] = {"supports": None, "separation_ratio": 1.02}
    rp = _write(vol, rec)
    _write_sidecars(vol, "5", GOOD_NUMERALS)
    scored = _run(vol, vouch=_vouch_nodes_for(rec))
    assert scored["5"]["votes"]["junction"] is None
    # the other two channels still carry it
    assert scored["5"]["accepted"] is True
    assert json.loads(rp.read_text())["status"] == "OK (verified: corroboration+addresses)"


def test_legacy_recorded_junction_refute_is_read_as_abstain(vol: _Paths) -> None:
    """Legacy result records can carry a baked `supports: false`.

    The junction channel no longer refutes, and a stale record must not be able
    to cast a veto the channel is not allowed to cast — otherwise the contract
    would hold at the producer and silently leak at the consumer.
    """
    rec = _revoked_record("5")
    rec["junction_snap"] = {"supports": False, "separation_ratio": 0.98}
    rp = _write(vol, rec)
    _write_sidecars(vol, "5", GOOD_NUMERALS)
    scored = _run(vol, vouch=_vouch_nodes_for(rec))
    assert scored["5"]["votes"]["junction"] is None, "legacy False must read as abstain"
    assert scored["5"]["votes"]["corroboration"] is True
    assert scored["5"]["votes"]["addresses"] is True
    assert scored["5"]["accepted"] is True
    assert json.loads(rp.read_text())["status"] == "OK (verified: corroboration+addresses)"


def test_address_refute_blocks(vol: _Paths) -> None:
    """A placement 300 m off every numeral's street refutes and blocks."""
    rec = _revoked_record("5")
    rec["junction_snap"] = {"supports": True}
    shift_gcps_geojson(rec["gcps_geojson"], 0.0, 300.0)
    rp = _write(vol, rec)
    _write_sidecars(vol, "5", GOOD_NUMERALS)
    scored = _run(vol)
    assert scored["5"]["votes"]["addresses"] is False
    assert scored["5"]["accepted"] is False
    assert json.loads(rp.read_text())["status"].startswith("REJECTED")


def test_unknown_era_abstains_addresses(vol: _Paths) -> None:
    rec = _revoked_record("5")
    _write(vol, rec)
    _write_sidecars(vol, "5", GOOD_NUMERALS)
    scored = _run(vol, era="unknown")
    assert scored["5"]["votes"]["addresses"] is None
    assert "era unknown" in scored["5"]["addresses"]["skipped"]


def test_renumbered_era_without_table_abstains(vol: _Paths) -> None:
    """Pre-renumbering numbers never pass through unchanged."""
    rec = _revoked_record("5")
    _write(vol, rec)
    _write_sidecars(vol, "5", GOOD_NUMERALS)
    scored = _run(vol, era="renumbered")
    assert scored["5"]["votes"]["addresses"] is None
    assert scored["5"]["addresses"]["votable"] == 0


def test_single_model_numerals_abstain(vol: _Paths) -> None:
    """One model's readings never vote — consensus requires >=2 models."""
    rec = _revoked_record("5")
    _write(vol, rec)
    (vol.annotations / "p5.v2.sonnet.json").write_text(
        json.dumps({"streets": [], "page_number_seen": None, "address_numerals": GOOD_NUMERALS})
    )
    scored = _run(vol)
    assert scored["5"]["votes"]["addresses"] is None


def test_non_revoked_records_never_touched(vol: _Paths) -> None:
    ok = {
        "page": "7",
        "status": "OK",
        "gcps_geojson": {"type": "FeatureCollection", "features": []},
    }
    rp = _write(vol, ok)
    before = rp.read_text()
    scored = _run(vol)
    assert "7" not in scored
    assert rp.read_text() == before


#: NON-SQUARE source-scan small frame: a (w, h) transposition anywhere in the
#: composition changes the answer, which a square frame could never catch.
SRC_SMALL = (1000, 2000)


def _make_rotated(vol: _Paths, rotation: int) -> None:
    """Declare p5 orientation-normalized, exactly as prep records it.

    ``full_size`` and ``scale`` stay in the SOURCE frame while ``small_size``
    is the UPRIGHT (as-written) frame — the swap for 90/270 is the thing the
    composition has to get right.
    """
    upright = (SRC_SMALL[1], SRC_SMALL[0]) if rotation in (90, 270) else SRC_SMALL
    vol.manifest.write_text(
        json.dumps(
            {
                "p5": {
                    "full_size": list(SRC_SMALL),
                    "small_size": list(upright),
                    "scale": 1.0,
                    "file": "p5_small.jpg",
                    "rotation_applied": rotation,
                }
            }
        )
    )


def _pil_forward(x: float, y: float, rotation: int) -> tuple[float, float]:
    """Where a source-frame pixel lands on prep's upright small — per PIL.

    Deliberately NOT rotate_bbox: the production forward turn is
    ``Image.rotate(-rotation, expand=True)`` (prep.prep_sheet), so deriving the
    expected upright coordinates from PIL itself keeps the round trip from
    being "invert rotate_bbox with rotate_bbox", which any self-inverse but
    wrong-handed map would pass.
    """
    from PIL import Image

    im = Image.new("L", SRC_SMALL, 0)
    im.putpixel((int(x), int(y)), 255)
    arr = np.asarray(im.rotate(-rotation, expand=True))
    ys, xs = np.nonzero(arr)
    return float(xs[0]), float(ys[0])


def _upright(numerals: list[dict[str, Any]], rotation: int) -> list[dict[str, Any]]:
    """The same numerals as the annotator would read them on the upright small."""
    out = []
    for n in numerals:
        x0, y0, x1, y1 = n["bbox"]
        ax, ay = _pil_forward(x0, y0, rotation)
        bx, by = _pil_forward(x1, y1, rotation)
        bbox = [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)]
        out.append({**n, "bbox": bbox})
    return out


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotation_normalized_pages_compose_and_accept(vol: _Paths, rotation: int) -> None:
    """Numerals read on the upright small are turned back into the source frame.

    Same evidence as test_junction_plus_addresses_accepts, but the page is
    orientation-normalized and the sidecars carry the bboxes the annotator
    actually saw (placed by PIL, the production forward turn) on a NON-SQUARE
    sheet — the composition must recover the same vote. This is the coverage
    the frame guard used to cost rotated scans outright.
    """
    _make_rotated(vol, rotation)
    rec = _revoked_record("5")
    rec["junction_snap"] = {"supports": True}
    rp = _write(vol, rec)
    _write_sidecars(vol, "5", _upright(GOOD_NUMERALS, rotation))
    scored = _run(vol)
    assert scored["5"]["votes"]["addresses"] is True
    assert scored["5"]["accepted"] is True
    assert json.loads(rp.read_text())["status"] == "OK (verified: junction+addresses)"


def test_uncomposed_numerals_on_a_rotated_page_do_not_vote(vol: _Paths) -> None:
    """The frame trap itself: source-frame numerals on a page prep turned 90.

    Feeding the stage numerals that are NOT in the small's own frame must not
    produce a spurious yes — they land off their blocks and the channel
    abstains, leaving junction as the lone channel (never an accept).
    """
    _make_rotated(vol, 90)
    rec = _revoked_record("5")
    rec["junction_snap"] = {"supports": True}
    rp = _write(vol, rec)
    _write_sidecars(vol, "5", GOOD_NUMERALS)  # source frame, not the upright one
    scored = _run(vol)
    assert scored["5"]["votes"]["addresses"] is not True
    assert scored["5"]["accepted"] is False
    assert json.loads(rp.read_text())["status"].startswith("REJECTED")


def test_escalated_tier_caches_feed_the_addresses_channel(vol: _Paths) -> None:
    """The escalation ladder's per-tier v2 caches count as model readings:
    one sidecar + one escalated cache from a DIFFERENT model reach the
    2-model floor (G2 finding 1 producer gap, free half).

    Since the consensus producer was cut this is the channel's ONLY renewable
    input — a volume placed from here on reaches the floor through the ladder or
    not at all.
    """
    rec = _revoked_record("5")
    rec["junction_snap"] = {"supports": True}
    rp = _write(vol, rec)
    (vol.annotations / "p5.v2.sonnet.json").write_text(
        json.dumps({"streets": [], "page_number_seen": None, "address_numerals": GOOD_NUMERALS})
    )
    (vol.annotations / "p5.escalated.fable.json").write_text(
        json.dumps({"streets": [], "page_number_seen": None, "address_numerals": GOOD_NUMERALS})
    )
    scored = _run(vol)
    assert scored["5"]["votes"]["addresses"] is True
    assert json.loads(rp.read_text())["status"] == "OK (verified: junction+addresses)"


def test_same_model_never_self_confirms_across_producers(vol: _Paths) -> None:
    """A sidecar and an escalated cache from the SAME model are one reading."""
    rec = _revoked_record("5")
    _write(vol, rec)
    (vol.annotations / "p5.v2.fable.json").write_text(
        json.dumps({"streets": [], "page_number_seen": None, "address_numerals": GOOD_NUMERALS})
    )
    (vol.annotations / "p5.escalated.fable.json").write_text(
        json.dumps({"streets": [], "page_number_seen": None, "address_numerals": GOOD_NUMERALS})
    )
    scored = _run(vol)
    assert scored["5"]["votes"]["addresses"] is None  # one model, no consensus


def test_variant_escalation_cache_keeps_its_model_identity(vol: _Paths) -> None:
    rec = _revoked_record("5")
    _write(vol, rec)
    model = "codex:gpt-5.6-terra"
    (vol.annotations / f"p5.v2.{model}.json").write_text(
        json.dumps({"streets": [], "page_number_seen": None, "address_numerals": GOOD_NUMERALS})
    )
    cache_key = model_cache_key(model, "high")
    (vol.annotations / f"p5.escalated.{cache_key}.json").write_text(
        json.dumps({"streets": [], "page_number_seen": None, "address_numerals": GOOD_NUMERALS})
    )
    scored = _run(vol)
    assert scored["5"]["votes"]["addresses"] is None


def test_real_on_disk_sidecar_names_still_decode(vol: _Paths) -> None:
    """The 2000+ sidecars already on disk are bare Anthropic names, and the
       Codex ones carry BOTH a colon and dots.

       Nothing writes a `v2` sidecar any more — that producer was cut
    — so this frozen set of
       filenames IS the addresses channel's remaining input, and every shape in it
       must keep decoding to its model identity.
    """
    from autogeoref.address_channel import _sidecar_numerals

    ann = vol.annotations
    body = json.dumps({"streets": [], "page_number_seen": None, "address_numerals": GOOD_NUMERALS})
    for name in ("claude-sonnet-5", "claude-opus-4-8"):
        (ann / f"p5.v2.{name}.json").write_text(body)
    # a bare provider-qualified name: dots AND a colon, no variant encoding
    (ann / "p5.v2.codex:gpt-5.6-terra.json").write_text(body)
    # an encoded key, and a failure marker
    (ann / f"p5.v2.{model_cache_key('codex:gpt-5.6-sol', 'high')}.json").write_text(body)
    (ann / "p5.v2.claude-fable-5.failed.json").write_text("{}")
    # an escalation tier cache decodes by exactly the same rule
    (ann / f"p5.escalated.{model_cache_key('claude-fable-5')}.json").write_text(body)

    assert set(_sidecar_numerals(ann, "5")) == {
        "claude-sonnet-5",
        "claude-opus-4-8",
        "codex:gpt-5.6-terra",
        "codex:gpt-5.6-sol",
        "claude-fable-5",
    }, "a failure marker must not appear, and every real name must decode"


def test_two_spellings_of_one_model_are_still_one_voice(vol: _Paths) -> None:
    """ONE model must never satisfy the two-voice floor by agreeing with itself.

    This used to be guaranteed on the WRITE side: the consensus producer canonicalized a voice
    before naming its sidecar, so `claude-sonnet-5` and `anthropic:claude-sonnet-5` could not
    both appear. The producer is gone, so the reader enforces it — otherwise two spellings would
    be two keys and two "distinct models" in the only channel permitted to REFUTE a placement.
    """
    from autogeoref.address_channel import _sidecar_numerals

    ann = vol.annotations
    body = json.dumps({"streets": [], "page_number_seen": None, "address_numerals": GOOD_NUMERALS})
    (ann / "p5.v2.claude-sonnet-5.json").write_text(body)
    (ann / "p5.v2.anthropic:claude-sonnet-5.json").write_text(body)

    assert set(_sidecar_numerals(ann, "5")) == {"claude-sonnet-5"}


def test_an_unparseable_model_name_still_counts_as_a_voice(vol: _Paths) -> None:
    """Canonicalizing must not silently DROP a reading it cannot parse.

    The channel's own contract is that retired-model readings count toward the
    two-voice floor. A name whose provider no longer resolves must therefore be
    kept as read, not discarded — dropping it would demote accepts standing on
    sidecars written years of model churn ago, which is exactly what keeping the
    read path alive through the cut was meant to prevent.
    """
    from autogeoref.address_channel import _sidecar_numerals

    ann = vol.annotations
    body = json.dumps({"streets": [], "page_number_seen": None, "address_numerals": GOOD_NUMERALS})
    (ann / "p5.v2.claude-sonnet-5.json").write_text(body)
    (ann / "p5.v2.retired-vendor:some-old-model.json").write_text(body)

    assert set(_sidecar_numerals(ann, "5")) == {
        "claude-sonnet-5",
        "retired-vendor:some-old-model",
    }


def test_numbered_place_twin_numerals_are_votable(vol: _Paths) -> None:
    """street_nam 'W 37TH' + street_typ 'PL' must key as '37TH PL' — the
    numbered PLACE twin is a different parallel street (found by the 1919
    golden testbed: a whole page of 37th-Place numerals was invisible)."""
    rec = _revoked_record("5")
    rec["junction_snap"] = {"supports": True}
    rp = _write(vol, rec)
    place = _main_street_feature()
    place["properties"]["street_nam"] = "W 37TH"
    place["properties"]["street_typ"] = "PL"
    numerals = [{**n, "street": "W. 37TH PL."} for n in GOOD_NUMERALS]
    _write_sidecars(vol, "5", numerals)
    scored = stage_verified_accept(
        vol,
        [place],
        aliases={},
        address_era="modern",
        vouch_nodes={},
    )
    assert scored["5"]["addresses"]["votable"] == 3
    assert scored["5"]["votes"]["addresses"] is True
    assert json.loads(rp.read_text())["status"] == "OK (verified: junction+addresses)"


def test_era_from_config_mapping(caplog: pytest.LogCaptureFixture) -> None:
    """The one config->era mapping: undeclared is MODERN, and never renumbered.

    Undeclared meaning "abstain" silenced the addresses channel — and with it
    verified-accept, which cannot fire on one channel — for every volume nobody
    wrote a config line for. What undeclared must
    never mean is "renumbered": that is G2 finding 3, which converts MODERN
    numbers ~19 blocks away.
    """
    from autogeoref.era import era_from_config

    assert era_from_config(True) == "modern"
    assert era_from_config(False) == "renumbered"
    assert era_from_config(None) == "modern"

    # a city that ships a renumbering table HAS renumbered, so an undeclared
    # volume there is the one case the default can be wrong in: still modern
    # (we never guess an era from a date), but never silently
    with caplog.at_level(logging.WARNING):
        assert era_from_config(None, volume="v9", city_renumbered=True) == "modern"
    assert "v9" in caplog.text and "addresses_modern" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert era_from_config(True, volume="v9", city_renumbered=True) == "modern"
    assert not caplog.text  # declared: nothing to warn about


def test_clip_features_4326() -> None:
    from autogeoref.geometry import clip_features_4326

    inside = _main_street_feature()
    lng, lat = inside["geometry"]["coordinates"][0]
    far = json.loads(json.dumps(inside))
    far["geometry"]["coordinates"] = [[lng + 1.0, lat], [lng + 1.001, lat]]
    clipped = clip_features_4326([inside, far], (lng - 0.01, lat - 0.01, lng + 0.01, lat + 0.01))
    assert clipped == [inside]


# ------------------------------------------------------------ vocabulary


def test_status_vocabulary_and_report_counts() -> None:
    st = status_verified(["junction", "addresses"])
    assert st == "OK (verified: junction+addresses)"
    assert st.startswith(STATUS_VERIFIED_PREFIX)
    assert is_committed({"status": st})
    results = {
        "1": {"status": "OK"},
        "2": {"status": "OK (rescued)"},
        "3": {"status": "OK (rescued, neighbor-corroborated)"},
        "4": {"status": st},
        "5": {"status": "REJECTED (no valid RANSAC model)"},
    }
    report = build_report("volX", results)
    assert report.strict_accepted == 1
    assert report.rescued == 1
    assert report.corroborated == 1
    assert report.verified == 1
    assert report.accepted_total == 4
    assert report.flagged == 1


def test_verified_sheets_do_not_vouch(tmp_path: Path) -> None:
    """AR-5 extension: a verified sheet stays out of the corroboration pool."""
    from autogeoref.paths import VolumePaths
    from autogeoref.vouchers import committed_vouch_nodes

    paths = VolumePaths(root=tmp_path)
    paths.results.mkdir(parents=True)
    strict = {"page": "1", "status": "OK", "gcps_geojson": _revoked_record("1")["gcps_geojson"]}
    verified = {
        "page": "2",
        "status": status_verified(["junction", "addresses"]),
        "gcps_geojson": _revoked_record("2")["gcps_geojson"],
    }
    corrob = {
        "page": "3",
        "status": "OK (rescued, neighbor-corroborated)",
        "gcps_geojson": _revoked_record("3")["gcps_geojson"],
    }
    for r in (strict, verified, corrob):
        (paths.results / f"p{r['page']}.json").write_text(json.dumps(r))
    nodes = committed_vouch_nodes(paths)
    pages = {page for uses in nodes.values() for page, _pos in uses}
    assert pages == {"1"}


# ------------------------------------------------------------ golden


@pytest.mark.golden
def test_p1_real_numerals_accept_true_placement_refute_shifted(
    fixtures_dir: Path, aliases_dir: Path, tmp_path: Path
) -> None:
    """The range-verified _024 p1 evidence, end to end through the stage.

    p1's recorded strict placement is recast as a revoked provisional with a
    supporting junction verdict; the three real model sidecars must carry it
    to ``OK (verified: junction+addresses)``. The same record shifted one
    block north must be REFUTED by the addresses channel (measured: 0/47
    in-block at a perpendicular one-block shift).
    """
    vol_fix = fixtures_dir / "sanborn01790_024"
    rec = json.loads((vol_fix / "results" / "p1.json").read_text())
    manifest = json.loads((vol_fix / "sheets" / "manifest.json").read_text())
    from autogeoref.names import load_aliases

    aliases = load_aliases(aliases_dir / "aliases-sanborn01790_024.json")
    features = json.loads((fixtures_dir / "reference" / "street_center_lines.geojson").read_text())[
        "features"
    ]
    # cheap clip around p1 so segment building stays fast
    lng0, lat0 = rec["gcps_geojson"]["features"][0]["geometry"]["coordinates"]
    features = [
        f
        for f in features
        if f.get("geometry")
        and f["geometry"].get("coordinates")
        and abs(_first_coord(f["geometry"])[0] - lng0) < 0.03
        and abs(_first_coord(f["geometry"])[1] - lat0) < 0.03
    ]

    for shifted, expect_accept in ((False, True), (True, False)):
        paths = _Paths(tmp_path / ("shifted" if shifted else "true"))
        paths.results.mkdir(parents=True)
        paths.annotations.mkdir()
        paths.sheets.mkdir()
        paths.manifest.write_text(json.dumps({"p1": manifest["p1"]}))
        r = json.loads(json.dumps(rec))
        r["status"] = "REJECTED (rescue revoked: anchors share one street)"
        r["layer"] = None
        r["junction_snap"] = {"supports": True, "separation_ratio": 1.3}
        if shifted:
            shift_gcps_geojson(r["gcps_geojson"], 0.0, 134.0)
        (paths.results / "p1.json").write_text(json.dumps(r))
        for model in ("sonnet", "opus", "fable"):
            shutil.copy2(DATA / f"p1_v2_{model}.json", paths.annotations / f"p1.v2.{model}.json")
        scored = stage_verified_accept(
            paths, features, aliases, address_era="modern", vouch_nodes={}
        )
        out = json.loads((paths.results / "p1.json").read_text())
        detail = scored["1"]["addresses"]
        if expect_accept:
            assert scored["1"]["votes"]["addresses"] is True
            assert out["status"] == "OK (verified: junction+addresses)"
            # measured calibration: all 47 votable consensus numerals in-block
            assert detail["votable"] == 47
            assert detail["in_block"] == 47
        else:
            assert scored["1"]["votes"]["addresses"] is False
            assert out["status"].startswith("REJECTED")
            assert detail["in_block"] == 0


def _first_coord(geometry: dict[str, Any]) -> tuple[float, float]:
    c = geometry["coordinates"]
    while isinstance(c[0], list):
        c = c[0]
    return float(c[0]), float(c[1])


@pytest.mark.golden
def test_034_volume_sweep_changes_nothing_without_evidence(
    fixtures_dir: Path, aliases_dir: Path, tmp_path: Path
) -> None:
    """Anti-red-flag invariant on a real recorded volume.

    Without junction verdicts or v2 sidecars, the stage scores every revoked
    page of _034 and must accept ZERO and leave every status byte-identical;
    non-revoked records are untouched entirely. Even where the near-fringe
    corroboration shape could vote on a page the strict corroborate pass
    left revoked, one channel never suffices — with no junction or
    addresses evidence on disk there is no second vote to pair with.
    """
    vol_fix = fixtures_dir / "sanborn01790_034"
    root = tmp_path / "v034"
    shutil.copytree(vol_fix / "results", root / "results")
    (root / "sheets").mkdir()
    shutil.copy2(vol_fix / "sheets" / "manifest.json", root / "sheets" / "manifest.json")
    (root / "annotations").mkdir()
    paths = _Paths(root)
    features = json.loads((fixtures_dir / "reference" / "street_center_lines.geojson").read_text())[
        "features"
    ]
    from autogeoref.names import load_aliases

    aliases = load_aliases(aliases_dir / "aliases-sanborn01790_034.json")
    scored = stage_verified_accept(paths, features, aliases, address_era="renumbered")
    assert scored, "no revoked pages scored — sweep wiring broken"
    assert all(v["accepted"] is False for v in scored.values())
    for f in sorted((vol_fix / "results").glob("p*.json")):
        rec_status = json.loads(f.read_text())["status"]
        mine = json.loads((root / "results" / f.name).read_text())
        assert mine["status"] == rec_status
