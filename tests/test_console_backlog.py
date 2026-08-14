"""The backlog: which volumes the console offers, on which track, and why not.

A place candidate is images with no complete results; a serve candidate is a
placed volume this pipeline has not published. The console must never advertise
a volume the queue would refuse or a run that halts at stage one, so a volume
live on either queue is withheld and an undeclared address era blocks the card
and the paste line with it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autogeoref.config.model import CityConfig, VolumeConfig
from autogeoref.console import backlog as console_backlog
from autogeoref.console import text as console_text
from autogeoref.queue import store as qstore
from console_support import _BBOX, _city, _images, _results, _status, _tiles, _tree


def test_place_candidate_is_images_with_no_results(tmp_path: Path) -> None:
    """`images only` is the backlog — whatever the volume is SERVING.

    `vol_ours` is the control: images AND our own published archive, and it is
    still a place candidate, because serving is not processing. Without it the
    assertion would hold over a tree in which nothing serves anything.
    """
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 10)
    _images(roots["work"], "vol_ours", 8)
    _tiles(roots["tiles"], "autogeoref", "vol_ours")

    cands = console_backlog.candidates(
        _status(roots), work=roots["work"], city=_city(renumbering=False)
    )
    place = {c.volume: c for c in cands if c.track == "place"}
    assert set(place) == {"vol_a", "vol_ours"}


def test_a_volume_with_ground_truth_and_no_pixels_is_not_a_candidate(tmp_path: Path) -> None:
    """~35 volumes have human pins and no images here. They cannot be RUN, only re-fetched."""
    roots = _tree(tmp_path)
    (roots["ground_truth"] / "api-layers-vol_gt.json").write_text(
        json.dumps(
            [
                {
                    "slug": "chicago_1900_vol_1_p1",
                    "image_url": "http://x/p1.jpg",
                    "gcps_geojson": {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": {"type": "Point", "coordinates": [-87.6, 41.9]},
                                "properties": {"image": [10, 20]},
                            }
                        ],
                    },
                }
            ]
        )
    )
    rows = _status(roots)
    assert any(r.volume == "vol_gt" and r.gt for r in rows), "the fixture must produce a GT row"
    assert console_backlog.candidates(rows, work=roots["work"]) == []


def test_serve_candidate_is_placed_and_not_published_by_us(tmp_path: Path) -> None:
    roots = _tree(tmp_path)
    for name in ("vol_unserved", "vol_ours"):
        _images(roots["work"], name, 6)
        _results(roots["work"], name, accepted=4, flagged=2)
    _tiles(roots["tiles"], "autogeoref", "vol_ours")

    serve = {c.volume for c in console_backlog.candidates(_status(roots), work=roots["work"])}
    # OUR published archive is the only thing that takes a volume off the serve
    # track (which volume is which is the whole assertion — dropping the
    # `_tiles` line above puts vol_ours back in the set)
    assert serve == {"vol_unserved"}


def test_a_volume_live_on_the_queue_is_not_advertised_as_runnable(tmp_path: Path) -> None:
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 10)
    _images(roots["work"], "vol_b", 10)
    qstore.add(roots["work"], "vol_a", "place")

    cands = console_backlog.candidates(_status(roots), work=roots["work"])
    assert [c.volume for c in cands] == ["vol_b"]


def test_a_failed_run_is_a_candidate_again(tmp_path: Path) -> None:
    """`failed` is not `live`: a run that died is work you can retry, and the console
    is the only place that would tell you so."""
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 10)
    entry = qstore.add(roots["work"], "vol_a", "place")
    entry.status = "failed"
    qstore.save_queue(roots["work"], [entry])

    cands = console_backlog.candidates(_status(roots), work=roots["work"])
    assert [c.volume for c in cands] == ["vol_a"]


def test_a_half_placed_volume_goes_back_on_place_and_never_on_serve(tmp_path: Path) -> None:
    """The realistic kill: `stage_match` writes results/p<N>.json INSIDE its per-sheet
    loop, so an OOM kill leaves a partial results/ directory.

    Reading "placed" off the mere existence of that directory would strike the volume
    off the place backlog AND offer its 40-of-121 sheets to the serve track — which
    bakes them onto the public map. The test is `results >= sheets`.
    """
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_killed", 10)
    _results(roots["work"], "vol_killed", accepted=3, flagged=1)  # 4 of 10: killed
    entry = qstore.add(roots["work"], "vol_killed", "place")
    entry.status = "failed"
    qstore.save_queue(roots["work"], [entry])

    cands = console_backlog.candidates(_status(roots), work=roots["work"])
    assert [(c.volume, c.track) for c in cands] == [("vol_killed", "place")]
    assert any("PARTIAL" in note for note in cands[0].notes)


def test_a_placed_volume_held_for_review_can_still_be_offered_for_serving(
    tmp_path: Path,
) -> None:
    """`needs-review` is terminal for the runner (`queue.store._TERMINAL`) and nothing clears it.

    Since the queue end-to-end work
    only a volume that ASKED to be reviewed (`--review`) rests there
    — an ordinary entry serves itself. But those are exactly the volumes whose operator
    wanted a look, so a whole-volume queue exclusion would strand them: the console would
    have no route from "placed" to "served" for the one case a human is waiting on. The
    exclusion is per TRACK.
    """
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 4)
    _results(roots["work"], "vol_a", accepted=3, flagged=1)
    entry = qstore.add(roots["work"], "vol_a", "place", then_serve=False)
    entry.status = "needs-review"  # where `queue.run._execute` parks a review-first entry
    qstore.save_queue(roots["work"], [entry])

    cands = console_backlog.candidates(_status(roots), work=roots["work"])
    assert [(c.volume, c.track) for c in cands] == [("vol_a", "serve")]

    # ...and once it IS queued to serve, it stops being advertised
    qstore.add(roots["work"], "vol_a", "serve")
    assert console_backlog.candidates(_status(roots), work=roots["work"]) == []


def test_a_volume_queued_on_one_track_is_advertised_on_neither(tmp_path: Path) -> None:
    """`queue.store.add` refuses a second track while one is live — two `autogeoref run`
    processes on one work tree, and a serve run baking a funnel still moving under it.

    `_015` is exactly this today: complete results from an earlier run AND queued on
    place for a re-run. A per-track exclusion would advertise it for serving, and the
    queue would reject the very command the console printed.
    """
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 4)
    _results(roots["work"], "vol_a", accepted=3, flagged=1)  # complete: placeable AND servable
    qstore.add(roots["work"], "vol_a", "place", then_serve=False)  # ...but queued for a re-place

    assert console_backlog.candidates(_status(roots), work=roots["work"]) == []
    # and the command the console withheld is one the queue really does refuse
    with pytest.raises(qstore.QueueError, match="cannot be on serve at the same time"):
        qstore.add(roots["work"], "vol_a", "serve")


def test_a_volume_that_prep_would_halt_on_is_blocked(tmp_path: Path) -> None:
    """`prep` HALTS on a page it cannot name (`UnrecognizedSheetError`) — it stopped
    being a warning-and-skip when a nameless sheet turned out to mean every stage
    dropped it in silence. A backlog that offered such a volume would be offering a
    run that dies at stage one."""
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 4)
    # NOT `_pcbd1`: the Congested District sheets are page-addressable now, via the
    # explicit `slugs._NAMED_PAGES` allow-list. This is an id nothing can parse.
    (roots["work"] / "vol_a" / "regions" / "chicago_1900_vol_1_pzz9.jpg").write_bytes(b"")
    # `lost_sheets` only fires where the BASELINE RECORD names that page id — the
    # evidence it is a map sheet and not front matter (status._lost_sheets)
    frozen = roots["fixtures"] / "vol_a" / "results"
    frozen.mkdir(parents=True)
    (frozen / "pzz9.json").write_text(json.dumps({"page": "zz9", "status": "OK"}))

    rows = _status(roots)
    assert [r.lost_sheets for r in rows if r.volume == "vol_a"] == [["zz9"]]
    c = next(c for c in console_backlog.candidates(rows, work=roots["work"]) if c.volume == "vol_a")
    assert not c.runnable and c.blocked is not None and "prep HALTS" in c.blocked


def test_an_undeclared_address_era_blocks_the_candidate_and_the_paste_line(
    tmp_path: Path,
) -> None:
    """The console must never hand out an `--add` line for a run that will REFUSE.

    Chicago declares the addresses channel AND ships a renumbering table, so every
    volume with no `addresses_modern` is rejected by `cli_run._cmd_run` before a single
    stage runs. A candidate list that ignored that would be a list of failed runs —
    and it is most of the unprocessed queue, not an edge case.
    """
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_undeclared", 10)
    _images(roots["work"], "vol_declared", 10)
    city = _city(declared={"vol_declared": True})

    cands = console_backlog.candidates(_status(roots), work=roots["work"], city=city)
    by_vol = {c.volume: c for c in cands}
    assert by_vol["vol_undeclared"].blocked and not by_vol["vol_undeclared"].runnable
    assert by_vol["vol_declared"].blocked is None

    text = console_text.render_candidates(cands)
    add_lines = [ln for ln in text.splitlines() if "--add" in ln]
    assert add_lines and all("vol_undeclared" not in ln for ln in add_lines)
    # but it is still LISTED, with the fix — a blocked volume is real work
    assert "vol_undeclared" in text
    assert "addresses_modern" in text


def test_the_era_block_tracks_the_channel_and_the_table_together(tmp_path: Path) -> None:
    """The refusal needs BOTH: the addresses channel on, and a city that renumbered.
    A city with no table (most of them) never blocks; nor does a volume with the
    channel off. This is `config.era_undeclared` + the channel test, and it is the
    same pair `cli_run._cmd_run` applies. (Both volumes carry a bounds source, so the
    only rule under test is the era one.)"""
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_a", 10)
    rows = _status(roots)

    channel_on = VolumeConfig(
        identifier="vol_a", evidence_channels=("junction", "addresses"), bounds_bbox=_BBOX
    )
    no_table = CityConfig(**{**_city(renumbering=False).__dict__, "volumes": {"vol_a": channel_on}})
    assert console_backlog.candidates(rows, work=roots["work"], city=no_table)[0].runnable

    channel_off = VolumeConfig(identifier="vol_a", evidence_channels=(), bounds_bbox=_BBOX)
    no_channel = CityConfig(**{**_city().__dict__, "volumes": {"vol_a": channel_off}})
    assert console_backlog.candidates(rows, work=roots["work"], city=no_channel)[0].runnable


def test_a_missing_bounds_source_is_a_note_never_a_block(
    tmp_path: Path,
) -> None:
    """A volume with no declared bounds source used to be a refusal; since the
    bootstrap it DERIVES bounds on its first run (`bounds_bootstrap`), so the
    console files a note — the operator should know the first run samples
    sheets — and keeps the volume in the paste-ready line. Volunteer pins are a
    bounds source by themselves, so a volume with GT carries no note; and a
    volume missing the era too is still era-BLOCKED — that one the runner does
    refuse."""
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_unbounded", 10)
    _images(roots["work"], "vol_gt_bounded", 10)
    _images(roots["work"], "vol_neither", 10)
    _images(roots["work"], "vol_derived", 10)
    # a previous run's derivation replays free — its note must say so, not
    # promise fresh sampling
    (roots["work"] / "vol_derived" / "volume-bounds.json").write_text(
        json.dumps({"bounds": [-87.7, 41.8, -87.6, 41.9]})
    )
    (roots["ground_truth"] / "api-layers-vol_gt_bounded.json").write_text(
        json.dumps(
            [
                {
                    "slug": "chicago_1900_vol_1_p1",
                    "image_url": "http://x/p1.jpg",
                    "gcps_geojson": {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": {"type": "Point", "coordinates": [-87.6, 41.9]},
                                "properties": {"image": [10, 20]},
                            }
                        ],
                    },
                }
            ]
        )
    )
    era_only = {
        vid: VolumeConfig(
            identifier=vid, addresses_modern=True, evidence_channels=("junction", "addresses")
        )
        for vid in ("vol_unbounded", "vol_gt_bounded", "vol_derived")
    }
    city = CityConfig(**{**_city().__dict__, "volumes": era_only})

    cands = console_backlog.candidates(_status(roots), work=roots["work"], city=city)
    by_vol = {c.volume: c for c in cands}
    assert by_vol["vol_unbounded"].blocked is None
    assert any("derives" in n for n in by_vol["vol_unbounded"].notes)
    assert by_vol["vol_gt_bounded"].blocked is None
    assert not any("derives" in n for n in by_vol["vol_gt_bounded"].notes)
    assert by_vol["vol_derived"].blocked is None
    assert not any("derives" in n for n in by_vol["vol_derived"].notes)
    assert any("previous run" in n for n in by_vol["vol_derived"].notes)
    assert by_vol["vol_neither"].blocked and "addresses_modern" in by_vol["vol_neither"].blocked

    text = console_text.render_candidates(cands)
    add_lines = [ln for ln in text.splitlines() if "--add" in ln]
    assert any("vol_unbounded" in ln for ln in add_lines), "runnable: the run bootstraps"
    # the note names the skip route, so the operator can choose zero sample spend
    assert "bounds_bbox" in text


def test_the_board_puts_what_you_can_start_first(tmp_path: Path) -> None:
    """29 of Chicago's 33 candidates are blocked on an era declaration today. A list
    that leads with them buries the volumes an operator could actually begin."""
    roots = _tree(tmp_path)
    _images(roots["work"], "vol_blocked", 5)
    _images(roots["work"], "vol_ready", 5)
    city = _city(declared={"vol_ready": True})

    cands = console_backlog.candidates(_status(roots), work=roots["work"], city=city)
    assert [c.volume for c in cands] == ["vol_ready", "vol_blocked"]
