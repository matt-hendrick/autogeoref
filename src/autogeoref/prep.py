"""Prep stage: full-resolution sheet scans -> downsampled JPEGs + manifest.

The vision annotator reads long-edge-2000 downsamples, and every downstream
pixel computation needs the per-page ``scale`` (small/full ratio) and
``full_size`` recorded at downsample time. The manifest is the single
source of truth for pixel-frame conversion — never assume a ratio.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .orient import detect_quarter_turn_image
from .paths import atomic_write_text, sheet_images
from .pillow import unlimited_image_pixels
from .slugs import non_addressable_kind, page_from_slug

logger = logging.getLogger(__name__)

TARGET_LONG_EDGE = 2000
JPEG_QUALITY = 90

#: Manifest key recording the volume's orientation policy (see prep_volume).
#: Not a page entry: consumers look pages up as ``manifest[f"p{page}"]``.
_ORIENTATION_SENTINEL = "_orientation_normalized"


class PrepError(RuntimeError):
    """A sheet image could not be prepared."""


class DuplicatePageError(PrepError):
    """Two region images claim the same page id.

    Fatal: merging duplicate page ids can associate annotations, dimensions,
    and images from different sheets.

    Usually a rescan left beside the original (``..._p12.jpg`` and
    ``..._rescan_p12.jpg``). Delete or move the one you do not want.
    """


class UnrecognizedSheetError(PrepError):
    """A region image's filename yields neither a page id nor a known map-less kind.

    Raised rather than warned so an unrecognized map sheet cannot disappear
    from every later stage. Add a map-less kind or a recognized page form.
    """


@dataclass(frozen=True)
class PrepResult:
    """What a prep pass actually did — the pre-spend reconciliation.

    ``images`` is every region file considered; ``pages`` the ids written to the
    manifest; ``skipped`` the map-less sheets, ``{filename: kind}``. The point of
    reporting all three is that the manifest count is legitimately LOWER than the
    region-file count (title and index plates carry no map), so the count alone
    can never tell you whether a sheet was dropped — only the reconciliation can.
    """

    manifest: dict[str, Any]
    images: int
    pages: tuple[str, ...] = field(default_factory=tuple)
    skipped: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """One line, safe to print before spending a model call per sheet."""
        kinds = ", ".join(sorted(set(self.skipped.values())))
        tail = f" ({kinds})" if kinds else ""
        return (
            f"{self.images} region images -> {len(self.pages)} addressable pages, "
            f"{len(self.skipped)} map-less{tail}"
        )


def page_of(image_path: Path) -> str | None:
    """Page id from a sheet filename (``<slug>_p12.jpg`` or ``p12.jpg``)."""
    return page_from_slug(image_path.stem)


def prep_sheet(
    image_path: Path,
    out_dir: Path,
    page: str,
    target_long_edge: int = TARGET_LONG_EDGE,
    normalize_orientation: bool = True,
) -> dict[str, Any]:
    """Downsample one sheet; returns its manifest entry.

    Idempotent: an existing, newer ``p<N>_small.jpg`` is kept and only the manifest entry is
    recomputed from the source dimensions. ``normalize_orientation`` runs the quarter-turn
    detector on the full-res image and writes the small UPRIGHT, recording ``rotation_applied``
    when nonzero. ``full_size`` stays the SOURCE scan frame while ``small_size`` is the written
    frame — compose ``scale`` with the recorded rotation to convert pixels. A sheet needing
    rotation is always re-encoded, since the mtime check cannot see whether an existing small
    was already rotated.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"p{page}_small.jpg"
    # Pillow's cap is process-global. Keep it raised through the lazy decode
    # (which occurs at convert()) and restore it before another decode can alter it.
    with unlimited_image_pixels(), Image.open(image_path) as im:
        # orientation detection and the downsample share one decode of
        # the full-res scan (the raster loads on first convert())
        rotation = detect_quarter_turn_image(im, image_path.name) if normalize_orientation else 0
        w, h = im.size
        scale = target_long_edge / max(w, h)
        small_size = (round(w * scale), round(h * scale))
        # Its own mtime comparison, not the stage runner's. A wall clock that
        # steps backwards can record this small as older than the scan it came
        # from, which re-encodes it: wasteful, never stale.
        fresh = (
            rotation == 0
            and out_path.exists()
            and out_path.stat().st_mtime >= image_path.stat().st_mtime
        )
        if not fresh:
            small = im.convert("RGB").resize(small_size, Image.Resampling.LANCZOS)
            if rotation:
                small = small.rotate(-rotation, expand=True)
            small.save(out_path, "JPEG", quality=JPEG_QUALITY)
            logger.info("p%s: %dx%d -> %dx%d", page, w, h, *small.size)
    if rotation in (90, 270):
        small_size = (small_size[1], small_size[0])
    entry = {
        "full_size": [w, h],
        "small_size": list(small_size),
        "scale": scale,
        "file": out_path.name,
    }
    if rotation:
        entry["rotation_applied"] = rotation
    return entry


def prep_volume(
    regions_dir: Path,
    sheets_dir: Path,
    target_long_edge: int = TARGET_LONG_EDGE,
    normalize_orientation: bool = True,
) -> PrepResult:
    """Downsample every full-res sheet in ``regions_dir``; write the manifest.

    Existing manifest entries for pages without a source image are kept, so this is resume-safe;
    entries for present images are recomputed. Raises :class:`UnrecognizedSheetError` if a
    region image's name yields neither a page id nor a known map-less kind. Returns the
    :class:`PrepResult` reconciliation — what you check BEFORE spending a call per sheet.
    """
    manifest_path = sheets_dir / "manifest.json"
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )
    # ORIENTATION POLICY IS STICKY PER VOLUME: a volume prepped in the source
    # frame has annotation caches MADE in that frame, and re-prepping it
    # normalized would rewrite the smalls upright and silently un-rotate those
    # caches. The written sentinel records the policy; a pre-sentinel manifest
    # is decided by whether any entry carries rotation_applied.
    if manifest:
        recorded = manifest.get(_ORIENTATION_SENTINEL)
        if recorded is None:
            recorded = any(
                isinstance(e, dict) and e.get("rotation_applied") for e in manifest.values()
            )
        if bool(recorded) != normalize_orientation:
            logger.warning(
                "%s: volume was prepped with normalize_orientation=%s; keeping that "
                "policy (cached annotations live in that frame). Re-prep into a "
                "fresh sheets dir to change it.",
                sheets_dir,
                recorded,
            )
        normalize_orientation = bool(recorded)
    # pipeline.sheet_images owns what counts as a scan (.jp2 is the LOC masters'
    # native format). Prep's gate and `regions_by_page` MUST agree on that, or
    # prep would fail to vet a file the pipeline goes on to address.
    images = sheet_images(regions_dir)
    if not images:
        raise PrepError(f"no sheet images under {regions_dir}")
    # CLASSIFY EVERY FILE BEFORE PREPPING ANY: a volume with an unrecognized
    # sheet must fail whole, not half-write a manifest that looks complete.
    pages: dict[Path, str] = {}
    skipped: dict[str, str] = {}
    unknown: list[str] = []
    for image_path in images:
        page = page_of(image_path)
        if page is not None:
            pages[image_path] = page
            continue
        kind = non_addressable_kind(image_path.stem)
        if kind is not None:
            skipped[image_path.name] = kind
            continue
        unknown.append(image_path.name)
    if unknown:
        raise UnrecognizedSheetError(
            f"{regions_dir}: {len(unknown)} region image(s) have no page id and are not a "
            f"known map-less sheet: {', '.join(sorted(unknown))}. Each is EITHER a new "
            "map-less kind (declare it in slugs._NON_ADDRESSABLE_RE) OR a map sheet whose "
            "naming slugs.page_from_slug does not recognize — in which case it would be "
            "dropped from the volume in silence. Look at the image; do not widen the page "
            "regex to make this go away (see slugs.__doc__)."
        )
    by_page: dict[str, list[str]] = {}
    for image_path, page in pages.items():
        by_page.setdefault(page, []).append(image_path.name)
    collisions = {p: names for p, names in by_page.items() if len(names) > 1}
    if collisions:
        detail = "; ".join(f"p{p}: {', '.join(sorted(n))}" for p, n in sorted(collisions.items()))
        raise DuplicatePageError(
            f"{regions_dir}: {len(collisions)} page id(s) claimed by more than one image "
            f"({detail}). One would silently win and the other would leave the volume — "
            "worse, the manifest could describe one image while the small on disk is a "
            "downsample of the other. Remove the sheet you do not want."
        )
    for image_path, page in pages.items():
        manifest[f"p{page}"] = prep_sheet(
            image_path, sheets_dir, page, target_long_edge, normalize_orientation
        )
    manifest[_ORIENTATION_SENTINEL] = normalize_orientation
    sheets_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
    result = PrepResult(
        manifest=manifest,
        images=len(images),
        pages=tuple(pages.values()),
        skipped=skipped,
    )
    logger.info("prep: %s -> %s", result.summary(), manifest_path)
    return result
