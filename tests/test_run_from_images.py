"""End-to-end runs starting from images without cached annotations."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import autogeoref.annotate_volume as annotate_volume
from autogeoref.annotate.schema import ExtendedAnnotation, extended_from_raw
from autogeoref.cli.entry import main

pytestmark = pytest.mark.golden  # needs the fixture tree for real scans + reads

ROOT = Path(__file__).resolve().parent.parent
VOL = "sanborn01790_024"
PAGES = ("p1", "p2")


def _reader_factory(read: Any) -> Any:
    """Stand in for `backend_for_model`; the stage builds one reader per batch."""

    def factory(model: str = "", **_kwargs: Any) -> Any:
        class Reader:
            def annotate_extended(self, image: Path) -> Any:
                return read(image, model)

        return Reader()

    return factory


@pytest.fixture
def scans_only(tmp_path: Path, fixtures_dir: Path) -> Path:
    """A volume with region images but no generated pipeline files."""
    vol = tmp_path / VOL
    (vol / "regions").mkdir(parents=True)
    for page in PAGES:
        shutil.copy2(
            fixtures_dir / VOL / "sheets" / f"{page}_small.jpg",
            vol / "regions" / f"{VOL}_{page}.jpg",
        )
    assert not (vol / "sheets").exists(), "the volume must start from scans alone"
    assert not (vol / "annotations").exists()
    return vol


@pytest.fixture
def recorded_reads(fixtures_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        page: json.loads((fixtures_dir / VOL / "annotations" / f"{page}.json").read_text())
        for page in PAGES
    }


@pytest.fixture
def backend(
    monkeypatch: pytest.MonkeyPatch, recorded_reads: dict[str, dict[str, Any]]
) -> list[str]:
    """Mock the annotate choke point; record every page it is asked to read."""
    calls: list[str] = []

    def fake_read(image_path: Path, model: str = "", **kwargs: Any) -> ExtendedAnnotation:
        page = image_path.stem.removesuffix("_small")
        calls.append(page)
        # built the way the real backend builds it (annotate.annotate_extended_cli),
        # so the stage stores exactly the shape a live read would have stored
        return extended_from_raw(recorded_reads[page])

    monkeypatch.setattr(annotate_volume, "backend_for_model", _reader_factory(fake_read))
    return calls


def _city(tmp_path: Path) -> Path:
    """A two-sheet city config with pinned scale and rotation."""
    cfg = tmp_path / "city.toml"
    cl = ROOT / "fixtures" / "reference" / "street_center_lines.geojson"
    cfg.write_text(
        "[city]\n"
        'name = "Chicago, Ill."\n'
        f'centerlines = "{cl}"\n'
        f'aliases_dir = "{ROOT / "configs" / "chicago" / "aliases"}"\n'
        f'[volumes."{VOL}"]\n'
        "bounds_bbox = [-87.66, 41.87, -87.60, 41.90]\n"
        "scale_m_per_px = 0.245\n"
        "rotation_deg = 1.20\n"
    )
    return cfg


def _run(tmp_path: Path, *extra: str) -> int:
    return main(["run", VOL, "--city", str(_city(tmp_path)), "--work", str(tmp_path), *extra])


def _primary_annotations(scans_only: Path) -> list[Path]:
    """The reads themselves, excluding the per-model cache records a bare glob
    would miscount."""
    return [p for p in (scans_only / "annotations").glob("p*.json") if ".annotation." not in p.name]


def test_a_volume_of_scans_becomes_a_volume_of_placements(
    scans_only: Path, tmp_path: Path, backend: list[str]
) -> None:
    """A run produces smalls, annotations, and results."""
    assert _run(tmp_path) == 0

    # prep produced the smalls and the manifest (the only valid pixel-frame conversion)
    assert (scans_only / "sheets" / "manifest.json").exists()
    # the ANNOTATE STAGE produced the annotations — nothing seeded them
    for page in PAGES:
        assert (scans_only / "annotations" / f"{page}.json").exists()
    assert sorted(backend) == sorted(PAGES), "one model call per uncached sheet"
    # and match actually MATCHED something, rather than reporting ok over zero sheets
    results = list((scans_only / "results").glob("p*.json"))
    assert len(results) == len(PAGES)


def test_match_halts_when_there_is_nothing_to_match(
    scans_only: Path, tmp_path: Path, backend: list[str]
) -> None:
    """Matching without annotations fails with actionable guidance."""
    rc = _run(tmp_path, "--no-annotate")
    assert rc == 1, "the run must fail, not sail on"
    assert backend == [], "--no-annotate spends nothing"

    marker = json.loads((scans_only / "markers" / "match.marker.json").read_text())
    assert marker["status"] == "failed"
    assert "no annotations to match" in marker["error"]
    assert "annotate" in marker["error"], "the error must name the stage that fixes it"


def test_dry_run_spends_nothing_and_still_says_what_it_would_spend(
    scans_only: Path, tmp_path: Path, backend: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """The budget gate on the pipeline's biggest spender.

    `--dry-run` is the only way to ask "what would this cost?" without paying it, so
    it must both PRINT the number and reach no backend. (The count here is the
    pre-prep upper bound: prep has not run, so there is no page list yet — and it
    says so rather than printing a flattering zero.)
    """
    assert _run(tmp_path, "--dry-run") == 0
    assert backend == [], "a dry run spends NOTHING"
    assert not (scans_only / "annotations").exists()

    printed = capsys.readouterr().out
    assert "2-4 model calls" in printed, printed


def test_limit_caps_the_spend_and_the_rest_resumes(
    scans_only: Path, tmp_path: Path, backend: list[str]
) -> None:
    """--limit caps the calls; the uncapped pages stay uncached for a later run.

    A capped run leaves the volume half-read, so `match` places only what was read —
    the honest outcome. What must NOT happen is the cap being ignored, or the skipped
    pages being marked as read.
    """
    _run(tmp_path, "--limit", "1")
    assert len(backend) == 1, "the cap is the cap"
    assert len(_primary_annotations(scans_only)) == 1

    # a second run reads the REST, and does not re-read the page it already has
    backend.clear()
    assert _run(tmp_path) == 0
    assert len(backend) == 1, "the cached page replays free; only the unread one is read"
    assert len(_primary_annotations(scans_only)) == 2


def test_a_failed_read_is_a_marker_not_an_annotation(
    scans_only: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call that did not land must never launder itself into "already annotated".

    The failure marker is what makes a retry possible and keeps the budget count
    honest (`status.annotation_reads` does not count a marker as a read).
    """
    from autogeoref.annotate.failures import AnnotationCallError

    calls: list[str] = []

    def always_fails(image_path: Path, model: str = "", **kwargs: Any) -> ExtendedAnnotation:
        calls.append(image_path.stem)
        raise AnnotationCallError("backend said no")

    monkeypatch.setattr(annotate_volume, "backend_for_model", _reader_factory(always_fails))
    assert _run(tmp_path) == 1, "a volume that could not be read must not place"

    for page in PAGES:
        assert (scans_only / "annotations" / f"{page}.failed.json").exists()
        assert not (scans_only / "annotations" / f"{page}.json").exists()
    # the retry is bounded: 2 pages x 2 attempts, and not one call more
    assert len(calls) == len(PAGES) * 2


def test_a_volume_one_sheet_short_does_not_place(
    scans_only: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed page read halts the run rather than shortening the volume."""
    from autogeoref.annotate.failures import AnnotationCallError

    recorded = json.loads((ROOT / "fixtures" / VOL / "annotations" / "p1.json").read_text())

    def one_page_fails(image_path: Path, model: str = "", **kwargs: Any) -> ExtendedAnnotation:
        if image_path.stem.startswith("p2"):
            raise AnnotationCallError("transient backend failure on ONE sheet")
        return extended_from_raw(recorded)

    monkeypatch.setattr(annotate_volume, "backend_for_model", _reader_factory(one_page_fails))

    rc = _run(tmp_path)

    assert rc == 1, "one unreadable sheet must fail the run, not shrink the volume"
    marker = json.loads((scans_only / "markers" / "annotate.marker.json").read_text())
    assert marker["status"] == "failed"
    assert "p2" in marker["error"], "the error must NAME the sheet the volume is missing"
    assert "--allow-failed-reads" in marker["error"], "and how to proceed deliberately"
    # the good page's read is kept: a retry re-reads only the page that failed
    assert (scans_only / "annotations" / "p1.json").exists()


def test_allow_failed_reads_places_the_short_volume_but_says_so(
    scans_only: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The escape hatch for a genuinely unreadable sheet — loud, never quiet."""
    import logging

    from autogeoref.annotate.failures import AnnotationCallError

    recorded = json.loads((ROOT / "fixtures" / VOL / "annotations" / "p1.json").read_text())

    def one_page_fails(image_path: Path, model: str = "", **kwargs: Any) -> ExtendedAnnotation:
        if image_path.stem.startswith("p2"):
            raise AnnotationCallError("this sheet is genuinely unreadable")
        return extended_from_raw(recorded)

    monkeypatch.setattr(annotate_volume, "backend_for_model", _reader_factory(one_page_fails))

    with caplog.at_level(logging.WARNING):
        rc = _run(tmp_path, "--allow-failed-reads")

    assert rc == 0
    assert "SHORT 1 SHEET(S)" in caplog.text
    assert "p2" in caplog.text
    assert len(list((scans_only / "results").glob("p*.json"))) == 1, "short, and honestly so"


def test_the_announced_number_is_not_less_than_the_number_spent(
    scans_only: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The budget contract: what it SAYS must bound what it SPENDS.

    A retry is a second billable call (`annotate.annotate_with_retry`), so a single
    figure would be a floor advertised as a total — `--limit 50` on a flaky backend
    could spend 100. The estimate names the ceiling, and the ceiling holds even when
    every page needs its retry.
    """
    recorded = {
        page: json.loads((ROOT / "fixtures" / VOL / "annotations" / f"{page}.json").read_text())
        for page in PAGES
    }
    calls: list[str] = []
    failed_once: set[str] = set()

    def flaky(image_path: Path, model: str = "", **kwargs: Any) -> ExtendedAnnotation:
        from autogeoref.annotate.failures import AnnotationCallError

        page = image_path.stem.removesuffix("_small")
        calls.append(page)
        if page not in failed_once:  # every page fails exactly once, then succeeds
            failed_once.add(page)
            raise AnnotationCallError("transient")
        return extended_from_raw(recorded[page])

    monkeypatch.setattr(annotate_volume, "backend_for_model", _reader_factory(flaky))

    assert _run(tmp_path) == 0
    printed = capsys.readouterr().out

    assert len(calls) == 4, "2 pages, each retried once"
    # the printed ceiling must be >= what was actually spent
    assert "2-4 model calls" in printed, printed
