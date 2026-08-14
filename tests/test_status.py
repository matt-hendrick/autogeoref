"""`autogeoref status`: the generated state index.

The failure it exists to prevent is a volume that LOOKS done because tiles are
being served for it. Every test here is about that distinction — served tiles
are display artifacts and only `work/<volume>/results/` says "processed here".
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from autogeoref.cli.entry import main
from autogeoref.status import VolumeStatus, build_status
from autogeoref.status_render import format_table, status_json
from autogeoref.volume import (
    STATUS_CORROBORATED,
    STATUS_OK,
    STATUS_REJECTED,
    STATUS_RESCUE_REVOKED,
    STATUS_RESCUED,
    STATUS_REVIEWER_VERIFIED,
)


def _volume(work: Path, volume: str, *, sheets: int = 0, reads: int = 0) -> Path:
    root = work / volume
    for page in range(1, sheets + 1):
        (root / "regions").mkdir(parents=True, exist_ok=True)
        (root / "regions" / f"{volume}_p{page}.jpg").write_bytes(b"")
    for page in range(1, reads + 1):
        (root / "annotations").mkdir(parents=True, exist_ok=True)
        (root / "annotations" / f"p{page}.json").write_text("{}")
    return root


def _results(root: Path, statuses: dict[str, str]) -> None:
    (root / "results").mkdir(parents=True, exist_ok=True)
    for page, status in statuses.items():
        (root / "results" / f"p{page}.json").write_text(
            json.dumps({"page": page, "status": status})
        )


def _tiles(tiles: Path, provenance: str, *volumes: str) -> None:
    d = tiles / provenance
    d.mkdir(parents=True, exist_ok=True)
    for v in volumes:
        (d / f"{v}.pmtiles").write_bytes(b"x")


def _ground_truth(
    fixtures: Path,
    volume: str,
    *,
    pinned: int,
    unpinned: int = 0,
    split_pages: tuple[str, ...] = (),
) -> None:
    """A volunteer GCP export. Layers with no `gcps_geojson` were never pinned;
    `split_pages` are sheets georeferenced as two separate crops (`..._p90_1`)."""
    gt = fixtures / "ground-truth"
    gt.mkdir(parents=True, exist_ok=True)
    layers = [
        {"slug": f"chicago_ill_1950_vol_9_p{p}", "gcps_geojson": {"features": [{}]}}
        for p in range(1, pinned + 1)
    ] + [{"slug": f"chicago_ill_1950_vol_9_p{p}"} for p in range(pinned + 1, pinned + unpinned + 1)]
    layers += [
        {"slug": f"chicago_ill_1950_vol_9_p{page}_{part}", "gcps_geojson": {"features": [{}]}}
        for page in split_pages
        for part in (1, 2)
    ]
    (gt / f"api-layers-{volume}.json").write_text(json.dumps(layers))


def _row(rows: list[VolumeStatus], volume: str) -> VolumeStatus:
    return next(r for r in rows if r.volume == volume)


def test_a_baseline_funnel_does_not_read_as_processed(tmp_path: Path) -> None:
    """The whole point: images on disk + someone else's recorded funnel = NOT
    done here. Only ``work/<volume>/results/`` is this repo's work."""
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    _volume(work, "vol_017", sheets=9)
    _results(
        _volume(fixtures, "vol_017"),
        {"1": STATUS_OK, "2": STATUS_OK, "3": STATUS_REJECTED},
    )

    row = _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_017")
    assert row.processed_here is False
    assert row.results is None
    assert row.sheets == 9
    assert row.tiles is None
    assert row.note == "images only"
    # the archived funnel is shown as a baseline, and is never this repo's work
    assert (row.frozen_accepted, row.frozen_sheets) == (2, 3)
    assert row.frozen_source == "baseline"


def test_processed_here_reports_this_repos_funnel(tmp_path: Path) -> None:
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    root = _volume(work, "vol_024", sheets=4, reads=4)
    _results(
        root,
        {
            "1": STATUS_OK,
            "2": STATUS_RESCUED,
            "3": STATUS_CORROBORATED,
            "4": STATUS_REJECTED,
        },
    )
    _tiles(tiles, "autogeoref", "vol_024")

    row = _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_024")
    assert row.processed_here is True
    assert (row.results, row.accepted, row.flagged) == (4, 3, 1)
    assert row.reads == 4
    assert row.note == "processed here; tiled"


def test_processed_here_not_tiled(tmp_path: Path) -> None:
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    _results(_volume(work, "vol_050", sheets=2), {"1": STATUS_OK, "2": STATUS_OK})

    row = _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_050")
    assert row.note == "processed here; not tiled"
    assert row.tiles is None


def test_reviewer_verified_placements_stay_out_of_the_accepted_count(tmp_path: Path) -> None:
    """Human placements never inflate the auto-acceptance rate (report.py's rule,
    which status must not re-derive differently)."""
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    _results(
        _volume(work, "vol_041", sheets=3),
        {"1": STATUS_OK, "2": STATUS_REVIEWER_VERIFIED, "3": STATUS_REJECTED},
    )

    row = _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_041")
    assert (row.accepted, row.flagged, row.reviewer_verified) == (1, 1, 1)


def test_scratch_dirs_and_overview_companions_are_not_volumes(tmp_path: Path) -> None:
    """A work/ directory with no stage directories is somebody's scratch
    output, and an `-overview` companion is served by nothing — neither is a
    volume.

    An archive whose name merely ends in a year IS one. Nothing bakes a
    citywide era layer any more, so that filename shape carries no special
    meaning here; it is pinned because it used to, and because the rows it
    produces are exactly the phantoms this function's docstring warns about.
    """
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    _tiles(tiles, "autogeoref", "vol_041-overview", "chicago-1895", "chicago-1950s")
    (work / "bake").mkdir(parents=True)
    (work / "bake" / "queue.log").write_text("noise")

    rows = build_status(work=work, fixtures=fixtures, tiles=tiles)
    assert [r.volume for r in rows] == ["chicago-1895", "chicago-1950s"]


def test_empty_tree_degrades_honestly(tmp_path: Path) -> None:
    rows = build_status(
        work=tmp_path / "work", fixtures=tmp_path / "nope", tiles=tmp_path / "deploy/tiles"
    )
    assert rows == []
    assert "no volumes found" in format_table(rows)
    assert status_json(rows) == "[]"


def test_empty_result_says_where_it_looked(tmp_path: Path) -> None:
    """The defaults are relative: run from the wrong directory, a full repo looks
    empty. An empty answer must be debuggable, not just believable."""
    roots = {"work": tmp_path / "work", "tiles": tmp_path / "deploy/tiles"}
    out = format_table([], roots=roots)

    assert str(tmp_path / "work") in out


def test_a_damaged_result_file_does_not_take_down_the_index(tmp_path: Path) -> None:
    """Legacy or manually damaged records must not prevent status indexing.

    Status reads EVERY volume, so it must count the casualty and keep going:
    the volumes either side of it still have to be answerable.
    """
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    root = _volume(work, "vol_024", sheets=3)
    _results(root, {"1": STATUS_OK, "2": STATUS_OK})
    (root / "results" / "p3.json").write_text('{"page": "3", "status": "OK')  # truncated
    (root / "results" / "p4.json").write_text("[]")  # valid JSON, not a record
    _volume(work, "vol_025", sheets=1)

    rows = build_status(work=work, fixtures=fixtures, tiles=tiles)

    row = _row(rows, "vol_024")
    assert (row.results, row.accepted) == (2, 2)  # the readable records still count
    assert row.damaged_results == 2
    assert "2 unreadable result file(s)" in row.note
    assert _row(rows, "vol_025") is not None  # and the rest of the index survives


def test_tiles_this_repo_placed_are_never_reported_as_unprocessed(tmp_path: Path) -> None:
    """The inverse of the legacy trap: a pruned/absent work tree must not turn a
    layer THIS repo placed into 'nothing on disk'. The tiles say who placed it."""
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    _tiles(tiles, "autogeoref", "vol_024")

    row = _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_024")
    assert row.tiles == "autogeoref"
    assert row.note == "tiled by this repo; no work/ results on disk"


def test_a_foreign_serving_directory_is_not_a_tiles_provenance(tmp_path: Path) -> None:
    """`served_tiles` scans the KNOWN provenance directories only. Anything else
    under `deploy/tiles/` — a basemap archive, a restored foreign tier — must
    not mint a `tiles` value, or the column would report a volume as served that
    this repo never published."""
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    _results(_volume(work, "vol_124", sheets=2), {"1": STATUS_OK, "2": STATUS_OK})
    _tiles(tiles, "partner-archive", "vol_124")

    row = _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_124")
    assert row.tiles is None
    assert row.note == "processed here; not tiled"


def test_the_congested_district_sheets_are_addressable_now(tmp_path: Path) -> None:
    """`_017`/`_018` ship `..._pcbd1.jpg` / `..._pcbd2.jpg` — real map sheets that
    `page_from_slug` could not name, so every stage skipped them without a word.
    The named-page allow-list admits them (`slugs._NAMED_PAGES`)
    and the per-page scale override places them (they are 200 ft/in in a 50 ft/in
    book). They must now COUNT as sheets a run here can process, and no longer be
    reported as a loss. Title/index pages are still dropped by the numeric rule
    and are still not a loss — nobody wants those placed."""
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    root = _volume(work, "vol_017", sheets=2)
    (root / "regions" / "vol_017_pcbd1.jpg").write_bytes(b"")
    (root / "regions" / "vol_017_ptitl.jpg").write_bytes(b"")
    _results(
        _volume(fixtures, "vol_017"),
        {"1": STATUS_OK, "2": STATUS_OK, "cbd1": STATUS_REJECTED},
    )

    row = _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_017")
    assert row.sheets == 3  # the two numbered sheets AND the CBD sheet
    assert row.unaddressable == 1  # the title page only
    assert row.lost_sheets == []  # nothing is silently dropped any more
    assert "cannot address" not in (row.note or "")


def test_a_sheet_the_pipeline_cannot_name_is_reported_not_dropped_in_silence(
    tmp_path: Path,
) -> None:
    """The general property, kept alive after the CBD sheets were named.

    A sheet whose page id `page_from_slug` cannot parse is skipped by every stage
    in silence — so `status` must SAY so. The ORIGIN's recorded result is the
    evidence it is a map sheet and not front matter: title/index pages are dropped
    by the same rule and must NOT be reported, because nobody wants those placed.
    (The allow-list is literal ids, so an unrecognized name still fails closed.)"""
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    root = _volume(work, "vol_017", sheets=2)
    (root / "regions" / "vol_017_pannex1.jpg").write_bytes(b"")
    (root / "regions" / "vol_017_ptitl.jpg").write_bytes(b"")
    _results(
        _volume(fixtures, "vol_017"),
        {"1": STATUS_OK, "2": STATUS_OK, "annex1": STATUS_REJECTED},
    )

    row = _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_017")
    assert row.sheets == 2  # the addressable ones: what a run here can process
    assert row.unaddressable == 2  # both the annex sheet and the title page
    assert row.lost_sheets == ["annex1"]  # but only the SHEET is a loss
    assert "cannot address: annex1" in row.note


def test_an_image_we_simply_never_fetched_is_not_called_unaddressable(tmp_path: Path) -> None:
    """A recorded page whose image is missing from regions/ is ABSENT,
    not unnameable. Calling it unaddressable would be a confident false diagnosis
    on any partially downloaded volume — so the claim is keyed on the images on
    disk, never on 'the record has a page we don't'."""
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    _volume(work, "vol_020", sheets=2)  # p1, p2 on disk
    _results(
        _volume(fixtures, "vol_020"),
        {"1": STATUS_OK, "2": STATUS_OK, "3": STATUS_OK},  # the record also has a p3
    )

    row = _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_020")
    assert row.lost_sheets == []
    assert "cannot address" not in row.note


def test_a_damaged_origin_record_does_not_silently_shrink_the_baseline(tmp_path: Path) -> None:
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    _volume(work, "vol_038", sheets=3)
    fx = _volume(fixtures, "vol_038")
    _results(fx, {"1": STATUS_OK, "2": STATUS_OK})
    (fx / "results" / "p3.json").write_text('{"page": "3", "stat')  # truncated

    row = _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_038")
    assert (row.frozen_accepted, row.frozen_sheets) == (2, 2)  # what could be read
    assert row.damaged_frozen_results == 1
    assert "baseline understated" in row.note  # and it says so, rather than implying 2/2 is all


def test_front_matter_alone_is_not_reported_as_a_loss(tmp_path: Path) -> None:
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    root = _volume(work, "vol_002", sheets=2)
    (root / "regions" / "vol_002_ptitl.jpg").write_bytes(b"")

    row = _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_002")
    assert row.unaddressable == 1
    assert row.lost_sheets == []
    assert row.note == "images only"


def test_reads_counts_pages_not_files(tmp_path: Path) -> None:
    """Sidecars are extra reads OF A PAGE, not extra pages; a failed read is not
    a read at all."""
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    root = _volume(work, "vol_034", sheets=2, reads=2)  # p1.json, p2.json
    ann = root / "annotations"
    (ann / "p1.v2.claude-sonnet-5.json").write_text("{}")  # a legacy v2 sidecar for p1
    (ann / "p1.escalated.claude-opus-4-8.json").write_text("{}")  # escalated re-read of p1
    (ann / "p3.escalated.claude-opus-4-8.failed.json").write_text("{}")  # a read that did NOT land

    assert _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_034").reads == 2


def test_images_no_one_can_name_are_never_reported_as_nothing_on_disk(tmp_path: Path) -> None:
    """The first thing a new city gets wrong is the sheet naming (`<slug>_p<N>`).
    Its images are then all unaddressable — and reporting that volume as
    "nothing on disk" would send someone hunting for missing files that are
    sitting right there."""
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    (work / "vol_new" / "regions").mkdir(parents=True)
    (work / "vol_new" / "regions" / "scan_001.jpg").write_bytes(b"")

    row = _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_new")
    assert row.sheets is None  # nothing a run could process...
    assert row.unaddressable == 1  # ...but the image is RIGHT THERE
    assert "NONE page-addressable" in row.note
    assert "nothing on disk" not in row.note


def test_a_volume_whose_run_died_early_still_gets_a_row(tmp_path: Path) -> None:
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    (work / "vol_060" / "markers").mkdir(parents=True)

    row = _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_060")
    assert row.note == "nothing on disk"


def test_fixtures_only_volume_is_not_mistaken_for_our_work(tmp_path: Path) -> None:
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    _results(_volume(fixtures, "vol_089"), {"1": STATUS_OK, "2": STATUS_REJECTED})

    row = _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_089")
    assert row.processed_here is False
    assert row.results is None
    assert (row.frozen_accepted, row.frozen_sheets) == (1, 2)
    assert row.note == "baseline record only; no images here"


def test_our_own_golden_run_is_never_credited_to_the_baseline(tmp_path: Path) -> None:
    """`_041`/`_089`/`_130` sit in fixtures/ but were built end-to-end by THIS repo
    (LOC jp2 masters + a run here; FIXTURES.md records the funnels). Printing them
    in the baseline column would hand our own numbers to the
    thing we claim to be beating. `subset.json` — written by fetch_new_goldens.py —
    is the filesystem signature that tells them apart."""
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    golden = _volume(fixtures, "vol_041")
    _results(golden, {"1": STATUS_OK, "2": STATUS_REJECTED})
    (golden / "subset.json").write_text("[1, 2]")
    _results(_volume(fixtures, "vol_017"), {"1": STATUS_OK})  # a real baseline copy

    rows = build_status(work=work, fixtures=fixtures, tiles=tiles)

    ours = _row(rows, "vol_041")
    assert ours.frozen_source == "autogeoref"
    assert ours.note == "frozen golden run (this repo) only; no images here"
    assert "this repo" in format_table(rows)
    # ...and the archived copies still read as the baseline they are
    assert _row(rows, "vol_017").frozen_source == "baseline"


def test_ground_truth_without_images_is_visible_not_absent(tmp_path: Path) -> None:
    """The state that cost an agent a day: `_132` has 73 volunteer-pinned pages and
    zero images here. It has no work/ or fixtures/ directory at all, so an index
    keyed only on directories would not list it — and "can I validate against this
    volume?" would have no answer. GT alone earns a row."""
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    _ground_truth(fixtures, "vol_132", pinned=73)

    row = _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_132")
    assert (row.gt, row.sheets) == (73, None)
    assert row.note == "volunteer GT on 73 page(s); scoring it needs an LOC re-fetch"


def test_a_volume_with_both_gt_and_images_is_the_one_that_can_be_scored(tmp_path: Path) -> None:
    """The cell that matters. Only where human GCPs and pixels MEET can a run be
    scored — and they mostly do not meet, so the row has to say which is which."""
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    _volume(work, "vol_006.5", sheets=97)
    _ground_truth(fixtures, "vol_006.5", pinned=97)

    row = _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_006.5")
    assert (row.gt, row.sheets) == (97, 97)
    assert "scoreable here" in row.note


def test_an_unpinned_volume_is_not_carried_as_a_ground_truth_row(tmp_path: Path) -> None:
    """67 of the 102 exports are empty — the recorded fact "production was checked,
    nobody ever pinned this". They are not GT and must not pad the index."""
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    _ground_truth(fixtures, "vol_002", pinned=0, unpinned=40)  # layers, but no GCPs
    (fixtures / "ground-truth" / "api-layers-vol_003.json").write_text("")  # 0-byte marker

    assert build_status(work=work, fixtures=fixtures, tiles=tiles) == []


def test_region_split_ground_truth_is_counted_as_unusable_not_as_pages(tmp_path: Path) -> None:
    """Measured: 258 pinned layers (118 pages) corpus-wide are OHMG region splits —
    a sheet the volunteer pinned as two crops. Their GCP pixels are in the CROP's
    frame and the export carries no offset back to the page, so they can never touch
    the full-res scan; not one of those pages is also pinned whole. They must be
    counted as unusable and NOT folded into `gt`: the tempting "fix" (read `_p90_1`
    as page 90) would bind crop pixels to page pixels and fabricate a placement."""
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    _ground_truth(fixtures, "vol_130", pinned=78, split_pages=("90", "91"))

    rows = build_status(work=work, fixtures=fixtures, tiles=tiles)
    row = _row(rows, "vol_130")

    assert row.gt == 78  # the pages a run can actually score against
    assert row.gt_unscoreable == 4  # 2 pages x 2 crops — visible, never silent
    assert "78 +4 unusable" in format_table(rows)


def test_a_damaged_ground_truth_export_does_not_take_down_the_index(tmp_path: Path) -> None:
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    _ground_truth(fixtures, "vol_040", pinned=132)
    damaged = fixtures / "ground-truth" / "api-layers-vol_042.json"
    damaged.write_text('[{"slug": "p1"')  # truncated

    rows = build_status(work=work, fixtures=fixtures, tiles=tiles)

    assert _row(rows, "vol_040").gt == 132  # the readable export still counts
    assert [r.volume for r in rows] == ["vol_040"]


def test_an_in_progress_bake_is_not_a_served_layer(tmp_path: Path) -> None:
    """A zero-byte .pmtiles is a bake in flight; it must not read as serving."""
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    _volume(work, "vol_041", sheets=2)
    (tiles / "autogeoref").mkdir(parents=True)
    (tiles / "autogeoref" / "vol_041.pmtiles").write_bytes(b"")

    row = _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_041")
    assert row.tiles is None
    assert row.note == "images only"


def test_status_is_read_only(tmp_path: Path) -> None:
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    _results(_volume(work, "vol_024", sheets=2, reads=2), {"1": STATUS_OK, "2": STATUS_REJECTED})
    _tiles(tiles, "autogeoref", "vol_024")
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}

    build_status(work=work, fixtures=fixtures, tiles=tiles)

    assert {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")} == before


def test_cli_status_prints_the_table_and_json(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    _volume(work, "vol_017", sheets=3)
    _tiles(tiles, "autogeoref", "vol_050")
    argv = ["status", "--work", str(work), "--fixtures", str(fixtures), "--tiles", str(tiles)]

    assert main(argv) == 0
    table = capsys.readouterr().out
    # the ROWS, not the legend: every provenance word also appears in the
    # legend prose below the table, so `"autogeoref" in table` is satisfied by
    # an empty tiles root and proves nothing
    rows = {line.split()[0]: line for line in table.splitlines() if line.startswith("vol_")}
    assert "images only" in rows["vol_017"]
    assert "autogeoref" in rows["vol_050"]
    assert "autogeoref" not in rows["vol_017"]

    assert main([*argv, "--json"]) == 0
    payload = {row["volume"]: row for row in json.loads(capsys.readouterr().out)}
    assert payload["vol_017"]["results"] is None
    assert payload["vol_017"]["tiles"] is None
    assert payload["vol_050"]["tiles"] == "autogeoref"


# ---------------------------------------------------------------------------
# serve staleness: the served autogeoref archive vs the committed records.
# mtimes are set explicitly — write order is not a contract.
# ---------------------------------------------------------------------------


def _set_mtime(path: Path, when: float) -> None:
    os.utime(path, (when, when))


def test_stale_when_a_committed_record_outdates_the_bake(tmp_path: Path) -> None:
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    root = _volume(work, "vol_024", sheets=2)
    _results(root, {"1": STATUS_OK, "2": STATUS_REJECTED})
    _tiles(tiles, "autogeoref", "vol_024")
    record = root / "results" / "p1.json"
    _set_mtime(tiles / "autogeoref" / "vol_024.pmtiles", record.stat().st_mtime - 100)

    rows = build_status(work=work, fixtures=fixtures, tiles=tiles)
    row = _row(rows, "vol_024")

    assert row.serve_stale == "stale"
    assert row.stale_record == "p1.json"
    assert "committed record p1.json newer than the served bake — serve pass needed" in row.note
    assert "STALE" in format_table(rows)


def test_fresh_when_the_bake_postdates_every_committed_record(tmp_path: Path) -> None:
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    root = _volume(work, "vol_024", sheets=1)
    _results(root, {"1": STATUS_OK})
    _tiles(tiles, "autogeoref", "vol_024")
    record = root / "results" / "p1.json"
    _set_mtime(tiles / "autogeoref" / "vol_024.pmtiles", record.stat().st_mtime + 100)

    row = _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_024")

    assert row.serve_stale == "fresh"
    assert row.stale_record is None
    assert "serve pass needed" not in row.note


def test_no_bake_when_committed_results_have_no_autogeoref_archive(tmp_path: Path) -> None:
    """Judged against deploy/tiles/autogeoref/ ONLY: a foreign archive is not a
    rebake target, so committed results next to one still read `no bake`.

    `vol_024` is the control — same committed record, but OUR archive — so the
    verdict can differ and the assertion is not satisfied by every volume in
    the tree reading `no bake` for want of any archive at all.
    """
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    _results(_volume(work, "vol_050", sheets=1), {"1": STATUS_OK})
    _results(_volume(work, "vol_085", sheets=1), {"1": STATUS_OK})
    root = _volume(work, "vol_024", sheets=1)
    _results(root, {"1": STATUS_OK})
    _tiles(tiles, "partner-archive", "vol_085")
    _tiles(tiles, "autogeoref", "vol_024")
    _set_mtime(
        tiles / "autogeoref" / "vol_024.pmtiles",
        (root / "results" / "p1.json").stat().st_mtime + 100,
    )

    rows = build_status(work=work, fixtures=fixtures, tiles=tiles)

    assert _row(rows, "vol_024").serve_stale == "fresh"  # the control CAN differ
    assert _row(rows, "vol_050").serve_stale == "no bake"
    assert _row(rows, "vol_085").serve_stale == "no bake"
    assert "serve pass needed" not in _row(rows, "vol_085").note


def test_only_records_the_bake_would_serve_can_flag_staleness(tmp_path: Path) -> None:
    """Provisional churn never outdates a bake: the bake would not serve those
    records, so their mtimes are not evidence."""
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    root = _volume(work, "vol_024", sheets=3)
    _results(root, {"1": STATUS_OK, "2": STATUS_REJECTED, "3": STATUS_RESCUE_REVOKED})
    _tiles(tiles, "autogeoref", "vol_024")
    committed = (root / "results" / "p1.json").stat().st_mtime
    _set_mtime(tiles / "autogeoref" / "vol_024.pmtiles", committed + 100)
    # both non-serving records postdate the bake; the committed one predates it
    _set_mtime(root / "results" / "p2.json", committed + 200)
    _set_mtime(root / "results" / "p3.json", committed + 200)

    row = _row(build_status(work=work, fixtures=fixtures, tiles=tiles), "vol_024")

    assert row.serve_stale == "fresh"


def test_a_malformed_record_cannot_take_down_the_staleness_scan(tmp_path: Path) -> None:
    """A record that is not a JSON object at all is damaged, not evidence —
    skipped, like every other malformed input this command survives."""
    from autogeoref.status import newest_committed

    work = tmp_path / "work"
    root = _volume(work, "vol_024", sheets=2)
    _results(root, {"1": STATUS_OK})
    (root / "results" / "p2.json").write_text(json.dumps(["not", "a", "record"]))

    assert newest_committed(root / "results") == (
        (root / "results" / "p1.json").stat().st_mtime,
        "p1.json",
    )


def test_no_committed_results_means_no_staleness_verdict(tmp_path: Path) -> None:
    """All-flagged results and a pruned work tree alike: nothing a bake could
    serve, so there is nothing to be stale against — the column stays a dash."""
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    _results(_volume(work, "vol_024", sheets=1), {"1": STATUS_REJECTED})
    _tiles(tiles, "autogeoref", "vol_024", "vol_017")

    rows = build_status(work=work, fixtures=fixtures, tiles=tiles)

    assert _row(rows, "vol_024").serve_stale is None  # results, none committed
    assert _row(rows, "vol_017").serve_stale is None  # tiles only, no work tree


def test_cli_status_stale_lists_only_stale_volumes(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    work, fixtures, tiles = tmp_path / "work", tmp_path / "fixtures", tmp_path / "deploy/tiles"
    for volume, offset in (("vol_015", -100), ("vol_024", +100)):
        root = _volume(work, volume, sheets=1)
        _results(root, {"1": STATUS_OK})
        _tiles(tiles, "autogeoref", volume)
        record = root / "results" / "p1.json"
        _set_mtime(tiles / "autogeoref" / f"{volume}.pmtiles", record.stat().st_mtime + offset)
    roots = ["--work", str(work), "--fixtures", str(fixtures), "--tiles", str(tiles)]
    argv = ["status", "--stale", *roots]

    assert main(argv) == 0
    assert capsys.readouterr().out == "vol_015\n"

    assert main([*argv, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["volume"] for r in payload] == ["vol_015"]
    assert payload[0]["serve_stale"] == "stale"


def test_stage_report_notes_serve_staleness(tmp_path: Path) -> None:
    """The report generated after placement says plainly that the served
    archive predates a committed record — and says nothing when it does not."""
    from autogeoref.paths import VolumePaths
    from autogeoref.stages.report import stage_report

    work, tiles = tmp_path / "work", tmp_path / "deploy/tiles"
    root = _volume(work, "vol_024", sheets=1)
    _results(root, {"1": STATUS_OK})
    _tiles(tiles, "autogeoref", "vol_024")
    paths = VolumePaths(root=root)
    bake = tiles / "autogeoref" / "vol_024.pmtiles"
    record = root / "results" / "p1.json"

    _set_mtime(bake, record.stat().st_mtime - 100)
    stage_report(paths, "vol_024", tiles_root=tiles)
    assert (
        "NOTE: committed record p1.json newer than the served bake — serve pass needed"
        in (root / "report.md").read_text()
    )

    _set_mtime(bake, record.stat().st_mtime + 100)
    stage_report(paths, "vol_024", tiles_root=tiles)
    assert "serve pass needed" not in (root / "report.md").read_text()

    _set_mtime(bake, record.stat().st_mtime - 100)
    stage_report(paths, "vol_024")  # no tiles root offered: the note cannot fire
    assert "serve pass needed" not in (root / "report.md").read_text()
