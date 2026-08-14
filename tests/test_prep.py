"""Prep stage: downsample + manifest, the pixel-frame source of truth."""

import json
import threading
from pathlib import Path

import pytest
from PIL import Image

from autogeoref.prep import (
    DuplicatePageError,
    PrepError,
    UnrecognizedSheetError,
    page_of,
    prep_volume,
)
from conftest import antedate


@pytest.fixture
def regions(tmp_path: Path) -> Path:
    d = tmp_path / "regions"
    d.mkdir()
    # portrait sheet, long edge 4000 -> scale 0.5
    Image.new("RGB", (3000, 4000), (200, 180, 150)).save(d / "chicago_ill_1894_vol_9_p3.jpg")
    # landscape sheet, long edge 2500 -> scale 0.8
    Image.new("RGB", (2500, 1500), (200, 180, 150)).save(d / "chicago_ill_1894_vol_9_p10.jpg")
    # a scan is always older than the small prep makes from it; say so on disk
    # rather than trusting the wall clock to record the two writes in order
    antedate(*sorted(d.glob("*.jpg")))
    return d


def test_page_of() -> None:
    assert page_of(Path("chicago_ill_1894_vol_9_p3.jpg")) == "3"
    assert page_of(Path("p12.jpg")) == "12"
    assert page_of(Path("chicago_ill_1901_vol_x_p7a.jpg")) == "7a"
    assert page_of(Path("titlesheet.jpg")) is None


def test_prep_volume_manifest_and_scales(regions: Path, tmp_path: Path) -> None:
    sheets = tmp_path / "sheets"
    manifest = prep_volume(regions, sheets).manifest
    # page entries plus the volume's recorded orientation policy
    assert set(manifest) == {"p3", "p10", "_orientation_normalized"}
    assert manifest["_orientation_normalized"] is True
    p3 = manifest["p3"]
    assert p3["full_size"] == [3000, 4000]
    assert p3["small_size"] == [1500, 2000]
    assert p3["scale"] == pytest.approx(0.5)
    assert "rotation_applied" not in p3  # portrait, no compass -> upright
    with Image.open(sheets / "p3_small.jpg") as im:
        assert im.size == (1500, 2000)
    # orientation normalization is ON by default: a landscape scan without a
    # verified compass takes the documented 90-deg fallback (Sanborn sheets
    # are portrait-printed), so the written small and small_size are upright
    # while full_size stays the source frame
    p10 = manifest["p10"]
    assert p10["full_size"] == [2500, 1500]
    assert p10["rotation_applied"] == 90
    assert p10["small_size"] == [1200, 2000]
    assert p10["scale"] == pytest.approx(0.8)
    with Image.open(sheets / "p10_small.jpg") as im:
        assert im.size == (1200, 2000)
    # manifest round-trips as the pipeline's loader expects
    on_disk = json.loads((sheets / "manifest.json").read_text())
    assert on_disk == manifest


def test_prep_opt_out_keeps_source_orientation(regions: Path, tmp_path: Path) -> None:
    sheets = tmp_path / "sheets"
    manifest = prep_volume(regions, sheets, normalize_orientation=False).manifest
    p10 = manifest["p10"]
    assert "rotation_applied" not in p10
    assert p10["small_size"] == [2000, 1200]


def test_prep_orientation_policy_is_sticky_per_volume(regions: Path, tmp_path: Path) -> None:
    """Re-prep preserves the cached annotation frame."""
    sheets = tmp_path / "sheets"
    first = prep_volume(regions, sheets, normalize_orientation=False).manifest
    assert "rotation_applied" not in first["p10"]
    # re-prep with the (new) default True: policy stays source-frame
    again = prep_volume(regions, sheets).manifest
    assert "rotation_applied" not in again["p10"]
    assert again["p10"]["small_size"] == [2000, 1200]
    with Image.open(sheets / "p10_small.jpg") as im:
        assert im.size == (2000, 1200)  # small NOT rewritten upright

    # and the reverse: a normalized volume stays normalized on an
    # explicit-False re-prep
    sheets2 = tmp_path / "sheets2"
    prep_volume(regions, sheets2)  # default True, fresh volume
    again2 = prep_volume(regions, sheets2, normalize_orientation=False).manifest
    assert again2["p10"]["rotation_applied"] == 90

    # a legacy manifest WITHOUT the sentinel (pre-flip prep) reads as
    # source-frame from its entries
    legacy = json.loads((sheets / "manifest.json").read_text())
    legacy.pop("_orientation_normalized")
    (sheets / "manifest.json").write_text(json.dumps(legacy))
    third = prep_volume(regions, sheets).manifest
    assert "rotation_applied" not in third["p10"]


def test_prep_idempotent_and_resume_safe(regions: Path, tmp_path: Path) -> None:
    sheets = tmp_path / "sheets"
    prep_volume(regions, sheets)
    first_mtime = (sheets / "p3_small.jpg").stat().st_mtime
    # add a page manually to the manifest (simulating a removed source image)
    manifest_path = sheets / "manifest.json"
    m = json.loads(manifest_path.read_text())
    m["p99"] = {"full_size": [10, 10], "small_size": [10, 10], "scale": 1.0, "file": "x"}
    manifest_path.write_text(json.dumps(m))
    again = prep_volume(regions, sheets).manifest
    # mtime is the only witness: a re-encode of these pixels is byte-identical
    # and reuses the inode, so neither a hash nor st_ino can see it happen
    assert (sheets / "p3_small.jpg").stat().st_mtime == first_mtime  # not re-encoded
    assert "p99" in again  # foreign entries kept (resume-safe)


def test_prep_re_encodes_a_scan_replaced_in_place(regions: Path, tmp_path: Path) -> None:
    """The other side of that check: a newer scan must rewrite the small.

    Deleting the freshness comparison outright leaves every other test in this
    module green, because the manifest entry is recomputed from the source
    whether or not the small is rewritten. Only the written pixels say so.
    """
    sheets = tmp_path / "sheets"
    prep_volume(regions, sheets)
    small = sheets / "p3_small.jpg"
    before = small.read_bytes()
    antedate(sheets)  # the small predates the re-fetch, whatever the clock says
    Image.new("RGB", (3000, 4000), (10, 20, 30)).save(regions / "chicago_ill_1894_vol_9_p3.jpg")
    prep_volume(regions, sheets)
    assert small.read_bytes() != before, "a replaced scan must re-encode the small"


def test_failed_manifest_replacement_preserves_previous_complete_manifest(
    regions: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sheets = tmp_path / "sheets"
    prep_volume(regions, sheets)
    manifest_path = sheets / "manifest.json"
    old_text = manifest_path.read_text()

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replacement failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        prep_volume(regions, sheets)

    assert manifest_path.read_text() == old_text
    assert not list(sheets.glob(".manifest.json.*.tmp"))


def test_prep_empty_dir_raises(tmp_path: Path) -> None:
    empty = tmp_path / "none"
    empty.mkdir()
    with pytest.raises(PrepError):
        prep_volume(empty, tmp_path / "sheets")


def _sheet(d: Path, name: str) -> None:
    Image.new("RGB", (3000, 4000), (200, 180, 150)).save(d / name)


def test_prep_skips_known_map_less_sheets_and_reports_them(regions: Path, tmp_path: Path) -> None:
    """Map-less plates are skipped and included in reconciliation output."""
    _sheet(regions, "chicago_ill_1894_vol_9_pcovr.jpg")
    _sheet(regions, "chicago_ill_1894_vol_9_ptitl.jpg")
    _sheet(regions, "chicago_ill_1894_vol_9_pind1.jpg")
    _sheet(regions, "chicago_ill_1894_vol_9_pnote.jpg")
    result = prep_volume(regions, tmp_path / "sheets")
    assert result.images == 6
    assert sorted(result.pages) == ["10", "3"]
    assert result.skipped == {
        "chicago_ill_1894_vol_9_pcovr.jpg": "covr",
        "chicago_ill_1894_vol_9_ptitl.jpg": "titl",
        "chicago_ill_1894_vol_9_pind1.jpg": "ind1",
        "chicago_ill_1894_vol_9_pnote.jpg": "note",
    }
    # the map-less plates are not manifested, so nothing downstream reads them
    assert set(result.manifest) == {"p3", "p10", "_orientation_normalized"}
    assert "6 region images -> 2 addressable pages, 4 map-less" in result.summary()


def test_prep_halts_on_an_unrecognized_sheet(regions: Path, tmp_path: Path) -> None:
    """Unknown page names halt instead of silently dropping a sheet."""
    _sheet(regions, "chicago_ill_1894_vol_9_pcongested.jpg")
    with pytest.raises(UnrecognizedSheetError, match="pcongested"):
        prep_volume(regions, tmp_path / "sheets")


def test_prep_halts_before_writing_anything(regions: Path, tmp_path: Path) -> None:
    """Fail WHOLE, not half: a half-written manifest looks complete to every
    downstream stage, which is the failure this gate exists to prevent."""
    _sheet(regions, "chicago_ill_1894_vol_9_pcongested.jpg")
    sheets = tmp_path / "sheets"
    with pytest.raises(UnrecognizedSheetError):
        prep_volume(regions, sheets)
    assert not (sheets / "manifest.json").exists()


def test_prep_halts_when_two_images_claim_one_page(regions: Path, tmp_path: Path) -> None:
    """Duplicate page ids would mix image pixels with manifest dimensions."""
    Image.new("RGB", (2000, 1500), (1, 2, 3)).save(regions / "chicago_ill_1894_vol_9_rescan_p3.jpg")
    with pytest.raises(DuplicatePageError, match=r"p3"):
        prep_volume(regions, tmp_path / "sheets")


def test_prep_reruns_when_a_scan_is_replaced_in_place(regions: Path, tmp_path: Path) -> None:
    """Freshness must key on the IMAGE FILES, not the regions/ directory.

    Overwriting `..._p3.jpg` with a re-fetched scan does not change the parent
    directory's mtime. A directory-keyed check would call prep fresh and keep a
    manifest whose scale/full_size describe the OLD image — and that manifest is
    the only valid small<->full conversion, so every GCP would be scaled wrong.
    """
    from autogeoref.dag import Stage, is_fresh
    from autogeoref.paths import sheet_images

    sheets = tmp_path / "sheets"
    first = prep_volume(regions, sheets)
    assert first.manifest["p3"]["full_size"] == [3000, 4000]

    def prep_stage() -> Stage:
        # exactly how cli.py declares it
        return Stage(
            name="prep",
            run=lambda: None,
            inputs=sheet_images(regions),
            outputs=[sheets / "manifest.json"],
        )

    assert is_fresh(prep_stage())  # nothing changed

    # re-fetch: same filename, different pixels, directory mtime untouched.
    # The manifest is antedated first so the replacement is unambiguously newer.
    antedate(sheets / "manifest.json", when=2000.0)
    Image.new("RGB", (6000, 4500), (9, 9, 9)).save(regions / "chicago_ill_1894_vol_9_p3.jpg")
    assert not is_fresh(prep_stage()), "a replaced scan must re-prep"

    again = prep_volume(regions, sheets)
    assert again.manifest["p3"]["full_size"] == [6000, 4500]


def test_prep_serializes_pillow_global_pixel_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prep and page-bounds share one guard for Pillow's process-global cap.

    Several subsystems lift the cap around their own open, so the guard is only
    safe while it serializes: two threads inside it at once would leave the
    process uncapped after the first one restored the default.
    """
    from autogeoref.mask.geometry import detect_page_bounds
    from autogeoref.prep import prep_sheet

    source = tmp_path / "regions"
    source.mkdir()
    first = source / "p1.jpg"
    second = source / "p2.jpg"
    Image.new("RGB", (20, 30), (1, 2, 3)).save(first)
    Image.new("RGB", (20, 30), (4, 5, 6)).save(second)

    original_open = Image.open
    entered_first = threading.Event()
    entered_second = threading.Event()
    release_first = threading.Event()
    release_second = threading.Event()
    errors: list[BaseException] = []
    original_cap = Image.MAX_IMAGE_PIXELS

    def controlled_open(path: str | Path, *args: object, **kwargs: object) -> Image.Image:
        entered, release = (
            (entered_first, release_first)
            if Path(path).name == "p1.jpg"
            else (entered_second, release_second)
        )
        entered.set()
        assert Image.MAX_IMAGE_PIXELS is None
        assert release.wait(timeout=5)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", controlled_open)

    def prepare(image: Path, page: str) -> None:
        try:
            prep_sheet(image, tmp_path / "sheets", page, normalize_orientation=False)
        except BaseException as exc:  # communicate failures from the worker thread
            errors.append(exc)

    def bounds(image: Path) -> None:
        try:
            detect_page_bounds(image)
        except BaseException as exc:  # communicate failures from the worker thread
            errors.append(exc)

    first_thread = threading.Thread(target=prepare, args=(first, "1"))
    second_thread = threading.Thread(target=bounds, args=(second,))
    first_thread.start()
    assert entered_first.wait(timeout=5)
    second_thread.start()
    assert not entered_second.wait(timeout=0.1), "second decode bypassed the global-cap lock"
    release_first.set()
    assert entered_second.wait(timeout=5)
    assert Image.MAX_IMAGE_PIXELS is None
    release_second.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not errors
    assert original_cap == Image.MAX_IMAGE_PIXELS


def test_prep_named_congested_district_sheets_are_pages(regions: Path, tmp_path: Path) -> None:
    """`_pcbd1`/`_pcbd2` are real map sheets (slugs._NAMED_PAGES) — they must
    manifest as pages, never be skipped as map-less."""
    _sheet(regions, "chicago_ill_1894_vol_9_pcbd1.jpg")
    result = prep_volume(regions, tmp_path / "sheets")
    assert "cbd1" in result.pages
    assert result.skipped == {}
    assert result.manifest["pcbd1"]["file"] == "pcbd1_small.jpg"
