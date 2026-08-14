"""Read-only filesystem-derived per-volume status index.

Served tiles are not evidence of local processing; result records are.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .bounds import load_ground_truth
from .mask.qa import load_masks_qa
from .paths import regions_by_page, sheet_images
from .report import VolumeReport, build_report
from .slugs import page_from_slug
from .viewer.config import SERVING_DIRS
from .viewer.sources import classify_pmtiles
from .volume import is_committed

logger = logging.getLogger(__name__)

#: A directory is a volume tree when it holds any of the stage directories.
#: Scratch/experiment directories under ``work/`` (bake logs, probe output)
#: hold none of them and are skipped without needing to be enumerated. A run
#: that died early may have written nothing but ``markers/`` — it is still a
#: volume, and a volume that exists must never be missing from the index.
VOLUME_MARKERS = ("results", "regions", "sheets", "annotations", "markers", "warped", "masks")


@dataclass(frozen=True)
class VolumeStatus:
    """One volume's true state. ``None`` means "nothing on disk", never zero."""

    volume: str
    #: Page-addressable full-resolution scans.
    sheets: int | None
    #: Volunteer-pinned pages usable for validation.
    gt: int | None
    #: Pinned crop layers that cannot be scored in the full-page pixel frame.
    gt_unscoreable: int | None
    #: images the pipeline cannot NAME (no parseable page id). Almost all are
    #: front matter it is right to skip (``_ptitl``, ``_pind1``, ``_pnote``), so
    #: this is recorded but not editorialized — ``lost_sheets`` is the one that
    #: means something went missing.
    unaddressable: int | None
    #: Recorded map pages whose on-disk image names are not addressable.
    lost_sheets: list[str]
    #: results files that could not be read (truncated / not a JSON object).
    #: A run killed mid-write leaves one; the row still prints.
    damaged_results: int | None
    #: the same, in the frozen results — reported because it silently SHRINKS
    #: the baseline denominator otherwise, which is the very failure (a number
    #: quietly getting smaller) this command exists to prevent
    damaged_frozen_results: int | None
    #: pages the model has read: the primary annotation (``p<N>.json``, written
    #: by whichever prompt was the backend default) plus the escalation tier
    #: caches and any ``p<N>.v2.<model>.json`` sidecar left by the retired
    #: consensus-annotate producer. Counting only the ``v2`` sidecars would print
    #: nothing for a normally processed volume — nothing writes them any more,
    #: and they were never the read every page gets.
    reads: int | None
    #: per-page records under ``work/<volume>/results/`` — THE processed-here test
    results: int | None
    accepted: int | None
    flagged: int | None
    reviewer_verified: int | None
    #: auto-accepts whose rotation and scale were PINNED rather than fitted
    #: (:func:`placement_records.pinned_orientation`) — a subset of ``accepted``, not an extra
    #: category. Correct by design and served like any other accept; printed because nothing
    #: else distinguishes them from evidence-fitted sheets. 0 = none are pinned; None = no
    #: results here to count.
    pinned_orientation: int | None
    #: sheets the bake's mask QA flagged (``masks/masks-qa.json``) — hull
    #: collapse, uncovered ink under a colour box, blank-margin overpaint, or a
    #: mirrored placement model. 0 = measured and clean; None = the
    #: volume was never baked with the QA step. Flags are advisory: the sheet
    #: still serves, and the remedy is an owner decision.
    mask_qa_flagged: int | None
    #: VOLUME-level mask-QA flags (``volume_flags`` in ``masks-qa.json``) —
    #: today only ``coverage_gaps``, the between-sheets uncovered-slot defect
    #: no per-sheet flag can see. () = measured and clean; None = never baked
    #: with the QA step, baked before the metric existed, or the metric was
    #: unmeasurable for this volume — either way NOT evidence of no holes.
    mask_qa_volume_flags: tuple[str, ...] | None
    #: the frozen recorded funnel under ``fixtures/<volume>/results/``, and WHOSE it is
    #: (:func:`_frozen_source`). Usually an archived baseline — a bar to beat, never this
    #: repo's work — but some are golden runs THIS repo produced end-to-end, and printing
    #: those as the baseline would credit our own numbers to what we measure against.
    frozen_source: str | None
    frozen_sheets: int | None
    frozen_accepted: int | None
    #: serving provenance = the ``deploy/tiles/`` subdirectory the archive sits in
    tiles: str | None
    #: is that a directory THIS pipeline publishes into — the reported city's
    #: first ``viewer.serving_dirs`` entry, or the default one every city that
    #: declares nothing writes to? A city may also vouch for a partner's
    #: archives, and those are never ours to rebake. This is the provenance
    #: test, never the directory's name, which differs per city.
    ours: bool
    #: does the served ``autogeoref`` archive reflect the committed records on disk? ``fresh``
    #: | ``stale`` | ``no bake``. None = no committed results, so nothing to be stale against.
    #: mtime-based, and wrong in both directions at the edges: a content-neutral rewrite reads
    #: stale once, and a record committed mid-bake reads fresh though the bake predates it —
    #: recheck after any mid-bake accept.
    serve_stale: str | None
    #: the newest committed record when it outdates the bake — stale rows only
    stale_record: str | None
    note: str

    @property
    def processed_here(self) -> bool:
        return bool(self.results)


def count_or_dash(value: int | None) -> str:
    """An optional count as digits, em-dash when unknown — 0 is a count, not a dash."""
    return "—" if value is None else str(value)


def _volume_dirs(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    return {
        p.name: p
        for p in sorted(root.iterdir())
        if p.is_dir() and any((p / m).is_dir() for m in VOLUME_MARKERS)
    }


def volume_funnel(volume: str, results_dir: Path) -> tuple[VolumeReport | None, int, set[str]]:
    """``(funnel, damaged_file_count, recorded_pages)`` for one results directory.

    Statuses are never re-derived here: :func:`report.build_report` owns what counts as
    accepted, so this command cannot disagree with `autogeoref report`. What it does NOT reuse
    is `report.load_results_dir`, which raises on a damaged record — deliberately, because a
    report's numbers are a contract artifact and must never quietly drop a sheet. `status` is
    the opposite case: it reads EVERY volume, and it is exactly what you reach for when the tree
    is in a bad state, so one truncated file (a run killed mid-write — results are not written
    atomically) must degrade to a counted skip, not take the whole index down with it.
    """
    if not results_dir.is_dir():
        return None, 0, set()
    results: dict[str, dict[str, Any]] = {}
    damaged = 0
    for f in sorted(results_dir.glob("p*.json")):
        try:
            record = json.loads(f.read_text())
            if not isinstance(record, dict):
                raise ValueError("record is not a JSON object")
        except (OSError, ValueError) as exc:  # JSONDecodeError is a ValueError
            logger.warning("%s: unreadable result record (%s); skipped", f, exc)
            damaged += 1
            continue
        results[str(record.get("page", f.stem.removeprefix("p")))] = record
    report = build_report(volume, results) if results else None
    return report, damaged, set(results)


def annotation_reads(annotations_dir: Path) -> int | None:
    """Distinct pages the model has read. A ``*.failed.json`` marker is a read
    that did NOT land, so it never counts as one."""
    if not annotations_dir.is_dir():
        return None
    pages = {
        f.name.split(".", 1)[0] for f in annotations_dir.glob("p*.json") if ".failed." not in f.name
    }
    return len(pages) or None


#: Marker identifying a fixture generated by this repository.
_GOLDEN_MARKER = "subset.json"


def _frozen_source(fixture_dir: Path) -> str:
    return "autogeoref" if (fixture_dir / _GOLDEN_MARKER).exists() else "baseline"


def _ground_truth_pages(ground_truth: Path) -> dict[str, tuple[int, int]]:
    """``{volume: (scoreable pages, unscoreable pinned layers)}`` over the exports.

    Volumes with NO pinned page are left out entirely, since carrying them as ``gt = 0`` rows
    would fill the index with volumes no one can do anything with. The scoreable count comes
    from :func:`bounds.load_ground_truth` — the same loader the PIPELINE scores with, so this
    column can never claim a page the run itself cannot use — and the second number is
    everything that loader drops, counted, because a silently shrinking denominator is the
    failure this command exists to prevent. Keyed on the LOC ITEM ID, never a rendered slug: two
    items can render to one slug, and pairing them mates one edition's GCPs to another's pixels.
    """
    if not ground_truth.is_dir():
        return {}
    out: dict[str, tuple[int, int]] = {}
    for f in sorted(ground_truth.glob("api-layers-*.json")):
        try:
            pages = load_ground_truth(f)
            layers = json.loads(f.read_text() or "[]")
        except (OSError, ValueError) as exc:  # JSONDecodeError is a ValueError
            logger.warning("%s: unreadable ground-truth export (%s); skipped", f, exc)
            continue
        pinned = sum(1 for lyr in layers if lyr.get("gcps_geojson"))
        if pages:
            out[f.stem.removeprefix("api-layers-")] = (len(pages), pinned - len(pages))
    return out


def _lost_sheets(images: list[Path], recorded_pages: set[str]) -> list[str]:
    """Sheets ON DISK that this pipeline cannot address but the baseline record names.

    Keyed on the images, deliberately, not on ``recorded_pages - our_pages``: a
    recorded page whose image we never fetched is simply ABSENT, and
    calling that "cannot be addressed" would be a confident false diagnosis on
    any partially downloaded volume. The claim made here is only ever about a
    file that is sitting in ``regions/`` right now with a page id nothing can
    parse (``..._pcbd1.jpg``), and the baseline record for that page id is
    the evidence it is a map sheet rather than front matter.
    """
    lost = []
    for image in images:
        if page_from_slug(image.stem) is not None:
            continue
        # the sheet naming convention is ``<slug>_p<id>``: recover the id the
        # record is filed under (``..._pcbd1`` -> ``cbd1``)
        _, _, ident = image.stem.rpartition("_p")
        if ident and ident in recorded_pages:
            lost.append(ident)
    return sorted(lost)


def served_tiles(tiles_root: Path, serving_dirs: Sequence[str] = SERVING_DIRS) -> dict[str, str]:
    """``{volume: serving directory}`` for every served per-volume archive.

    Scans the declared directories (a city's ``viewer.serving_dirs``), first
    wins. Scanning every sibling instead would let one serving something other
    than volumes (a basemap archive) mint phantom rows. Without the city's list
    only the default directory is scanned, so a city publishing under its own
    name reads as unbaked — pass the config in wherever there is one.
    """
    out: dict[str, str] = {}
    for label in serving_dirs:
        for ident in classify_pmtiles(tiles_root / label):
            out.setdefault(ident, label)
    return out


SERVE_FRESH = "fresh"
SERVE_STALE = "stale"
SERVE_NO_BAKE = "no bake"


def newest_committed(results_dir: Path) -> tuple[float, str] | None:
    """``(mtime, filename)`` of the newest record a rebake would serve.

    Committed means :func:`volume.is_committed` — the exact predicate
    ``bake.committed_layers`` selects with — so only records a bake would
    actually serve can flag a volume stale. Provisional churn, revocations,
    and over-commit-gate accepts (status OK but withheld from the mosaic)
    never do. Unreadable or malformed records are skipped;
    :func:`volume_funnel` counts the unreadable ones.
    """
    if not results_dir.is_dir():
        return None
    newest: tuple[float, str] | None = None
    for f in sorted(results_dir.glob("p*.json")):
        try:
            record = json.loads(f.read_text())
            # TypeError/ValueError also cover is_committed on a record whose
            # rmse field is not a number — malformed, so never evidence
            if not isinstance(record, dict) or not is_committed(record):
                continue
            mtime = f.stat().st_mtime
        except (OSError, ValueError, TypeError):
            continue
        if newest is None or mtime > newest[0]:
            newest = (mtime, f.name)
    return newest


def own_archives(tiles_root: Path, serve_dirs: Sequence[str] = SERVING_DIRS) -> dict[str, Path]:
    """``{volume: pmtiles path}`` under the directories this pipeline publishes
    into, first wins.

    The only archives a rebake replaces, and the only ones staleness is judged
    against — a directory a city merely vouches for is not ours to rebake."""
    out: dict[str, Path] = {}
    for label in serve_dirs:
        for ident, path in classify_pmtiles(tiles_root / label).items():
            out.setdefault(ident, path)
    return out


def serve_staleness(
    newest: tuple[float, str] | None, archive: Path | None
) -> tuple[str | None, str | None]:
    """``(verdict, newer-than-bake record name)`` for one volume."""
    if newest is None:
        return None, None
    if archive is None or not archive.exists():
        return SERVE_NO_BAKE, None
    if newest[0] > archive.stat().st_mtime:
        return SERVE_STALE, newest[1]
    return SERVE_FRESH, None


def stale_note(stale_record: str) -> str:
    """One wording for the stale verdict, shared with the volume report."""
    return f"committed record {stale_record} newer than the served bake — serve pass needed"


def _note(row: VolumeStatus, *, processed: bool, has_frozen: bool) -> str:
    """The one-line verdict for a row, read off the row itself.

    ``processed`` and ``has_frozen`` are the two things the row cannot say:
    whether this work tree HAS a funnel / a frozen record at all, as distinct
    from one whose counts are zero. The state it must never get backwards: an
    archive in this pipeline's own serving directory IS this repo's work even
    when the work tree it came from is gone.
    """
    sheets = row.sheets
    gt = row.gt
    unaddressable = row.unaddressable
    lost_sheets = row.lost_sheets
    damaged = row.damaged_results
    frozen_damaged = row.damaged_frozen_results
    frozen_source = row.frozen_source
    tiles = row.tiles
    mask_qa_flagged = row.mask_qa_flagged
    mask_qa_volume_flags = row.mask_qa_volume_flags
    serve_stale = row.serve_stale
    stale_record = row.stale_record
    parts: list[str] = []
    if processed:
        if row.ours:
            parts.append("processed here; tiled")
        else:
            parts.append("processed here; not tiled")
    elif row.ours:
        # placed by THIS repo (the tiles say so) but no results on disk — a
        # pruned or unshared work tree, NOT unprocessed work
        parts.append("tiled by this repo; no work/ results on disk")
    elif sheets:
        parts.append("images only" + ("; not processed" if tiles else ""))
    elif unaddressable:
        # images ARE on disk — the pipeline just cannot name a single one of
        # them. Saying "nothing on disk" here would be a flat lie, and it is
        # the first thing a new city gets wrong (sheets must be <slug>_p<N>).
        parts.append(
            f"{unaddressable} image(s) on disk, NONE page-addressable — "
            "nothing can be processed (sheets must be named <slug>_p<N>)"
        )
    elif tiles is not None:
        parts.append(f"{tiles} tiles only; no images here to process")
    elif has_frozen:
        # ours or the baseline's — never blur the two: some frozen runs are this
        # repo's own, and calling them baseline would hand our numbers to the
        # baseline we claim to be beating
        ours = frozen_source == "autogeoref"
        label = "frozen golden run (this repo)" if ours else "baseline record"
        parts.append(f"{label} only; no images here")
    elif not gt:
        parts.append("nothing on disk")
    if gt:
        # the cell that matters is (gt AND sheets): a volume can only be SCORED
        # where human GCPs and pixels meet, and they mostly do not — the census
        # fetched images for precisely the volumes nobody had pinned. Saying
        # "no images here" plainly is the point: the last agent spent a day
        # hunting a cache that was never missing.
        parts.append(
            f"volunteer GT on {gt} page(s)"
            + (" — scoreable here" if sheets else "; scoring it needs an LOC re-fetch")
        )
    if serve_stale == SERVE_STALE and stale_record:
        # the served archive predates a record it would serve — the one state
        # this column exists to catch, so the note names the evidence
        parts.append(stale_note(stale_record))
    if lost_sheets:
        # sheets the baseline record names and this pipeline cannot even name: a
        # run would drop them silently. Front matter (title/index pages) is
        # dropped too and is NOT reported — nobody wants it placed.
        parts.append("recorded page(s) this pipeline cannot address: " + ", ".join(lost_sheets))
    if damaged:
        parts.append(f"{damaged} unreadable result file(s)")
    if frozen_damaged:
        # otherwise the frozen column just quietly reads a smaller denominator
        parts.append(f"{frozen_damaged} unreadable frozen record(s) — baseline understated")
    if mask_qa_flagged:
        # advisory, not a serve blocker: the named sheets render, but the bake
        # measured their masks as defective (see mask.qa for the classes)
        parts.append(f"mask QA flags on {mask_qa_flagged} sheet(s) — see masks/masks-qa.json")
    if mask_qa_volume_flags:
        # the between-sheets defect class no per-sheet flag can see
        parts.append(
            "mask QA volume flag(s): "
            + ", ".join(mask_qa_volume_flags)
            + " — see masks/masks-qa.json"
        )
    return "; ".join(parts)


def build_status(
    *,
    work: Path,
    fixtures: Path,
    tiles: Path,
    ground_truth: Path | None = None,
    serving_dirs: Sequence[str] = SERVING_DIRS,
) -> list[VolumeStatus]:
    """Scan the tree and classify every volume it can see. Never writes.

    ``ground_truth`` defaults to ``<fixtures>/ground-truth/``. ``serving_dirs``
    is a city's ``viewer.serving_dirs``, and its entries are ADDED to the
    default set rather than swapped for it: this scan populates the whole
    report, so narrowing it to one city would drop every other city's served
    volumes from the table and report them as never scanned.
    """
    # directories this pipeline WRITES: the named city's own, plus the default
    # every city that declares nothing publishes into. A later `serving_dirs`
    # entry is a partner archive — served and vouched for, never ours.
    own = tuple(dict.fromkeys((serving_dirs[0], *SERVING_DIRS)))
    # the city's own first, so a volume served from two directories is
    # attributed to the one this run is about
    scan = tuple(dict.fromkeys((*serving_dirs, *SERVING_DIRS)))
    tile_provenance = served_tiles(tiles, scan)
    archives = own_archives(tiles, own)
    work_dirs = _volume_dirs(work)
    fixture_dirs = _volume_dirs(fixtures)
    # params-form twin of cli_context.ground_truth_root's args-form fallback;
    # cli_context imports build_status, so the default is restated here —
    # keep the two spellings identical
    gt_pages = _ground_truth_pages(ground_truth or fixtures / "ground-truth")
    # a pinned volume earns a row even with nothing else on disk: "GT 73,
    # sheets 0" IS the answer to "can I validate against this volume?", and it
    # is unreachable if the index only lists volumes that have a directory
    volumes = sorted(set(work_dirs) | set(fixture_dirs) | set(tile_provenance) | set(gt_pages))

    rows: list[VolumeStatus] = []
    for volume in volumes:
        wd = work_dirs.get(volume)
        fd = fixture_dirs.get(volume)
        funnel, damaged, _ = volume_funnel(volume, wd / "results") if wd else (None, 0, set())
        frozen, frozen_damaged, frozen_pages = (
            volume_funnel(volume, fd / "results") if fd else (None, 0, set())
        )
        images = sheet_images(wd / "regions") if wd else []
        pages = regions_by_page(wd / "regions") if wd else {}
        mask_qa = load_masks_qa(wd / "masks") if wd else None
        mask_qa_flagged = len(mask_qa.get("flagged") or {}) if mask_qa is not None else None
        # a doc from before the volume-level metric has no key at all: that is
        # "not measured" (None), never "measured clean" (())
        mask_qa_volume_flags = (
            tuple(mask_qa["volume_flags"])
            if mask_qa is not None and "volume_flags" in mask_qa
            else None
        )
        unaddressable = len(images) - len(pages)
        lost_sheets = _lost_sheets(images, frozen_pages)
        tile = tile_provenance.get(volume)
        gt, gt_unscoreable = gt_pages.get(volume, (0, 0))
        serve_stale, stale_record = serve_staleness(
            newest_committed(wd / "results") if wd else None, archives.get(volume)
        )
        row = VolumeStatus(
            volume=volume,
            sheets=len(pages) or None,
            gt=gt or None,
            gt_unscoreable=gt_unscoreable or None,
            unaddressable=unaddressable or None,
            lost_sheets=lost_sheets,
            damaged_results=damaged or None,
            damaged_frozen_results=frozen_damaged or None,
            reads=annotation_reads(wd / "annotations") if wd else None,
            results=funnel.n_sheets if funnel else None,
            accepted=funnel.accepted_total if funnel else None,
            flagged=funnel.flagged if funnel else None,
            reviewer_verified=(funnel.reviewer_verified or None) if funnel else None,
            pinned_orientation=funnel.pinned_orientation if funnel else None,
            mask_qa_flagged=mask_qa_flagged,
            mask_qa_volume_flags=mask_qa_volume_flags,
            frozen_source=_frozen_source(fd) if fd and frozen else None,
            frozen_sheets=frozen.n_sheets if frozen else None,
            frozen_accepted=frozen.accepted_total if frozen else None,
            tiles=tile,
            ours=tile in own,
            serve_stale=serve_stale,
            stale_record=stale_record,
            note="",
        )
        # the note is READ OFF the finished row, so the two can never disagree
        rows.append(
            replace(
                row,
                note=_note(row, processed=funnel is not None, has_frozen=frozen is not None),
            )
        )
    return rows
