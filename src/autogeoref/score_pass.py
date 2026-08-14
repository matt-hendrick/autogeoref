"""Score a finished volume against human ground truth, after the fact.

The pass runs over result records a run already wrote and publishes its
verdicts to the volume's score sidecar. It never edits a record, and no
placement stage reads what it writes: scoring can take a sheet out of
committed evidence, never put one in.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .affine import TO_3857, apply_affine
from .bounds import load_ground_truth, mercator_correction_lat
from .paths import VolumePaths, iter_results
from .placement_records import preseam_record
from .scoring import (
    GT_COMMIT_RMSE_M,
    record_digest,
    score_record_vs_ground_truth,
    write_sidecar,
)
from .seam import SheetFit, rms, sheet_fit_from_result
from .slugs import page_from_slug
from .volume import is_reviewer_verified

logger = logging.getLogger(__name__)

#: Max allowed worsening of the human-GCP median before a seam solve is called
#: a regression. Diagnostic here: the solve has already been applied.
SEAM_GATE_M = 0.5

Deltas = Mapping[str, tuple[float, float]]


def resolve_pages(pages: Iterable[str], mapping: Mapping[str, Any]) -> dict[str, str]:
    """``{page: key}`` for the pages ``mapping`` holds, folding case ONE TO ONE.

    An export can spell a letter-suffixed page in lower case (`…_p12s`) where
    this repo names the same sheet from its master (`p12S`). Grading only:
    nothing here reaches a placement.

    The fold never chooses. An exact match wins and is never displaced, a key
    taken exactly is out of the running, and a key two pages both fold onto
    goes to neither — case is sometimes all that separates two real sheets.
    """
    wanted = list(pages)
    resolved = {page: page for page in wanted if page in mapping}
    taken = set(resolved.values())
    claims: dict[str, list[str]] = {}
    for page in wanted:
        if page in resolved:
            continue
        folded = page.casefold()
        hits = [k for k in mapping if k not in taken and k.casefold() == folded]
        if len(hits) == 1:
            claims.setdefault(hits[0], []).append(page)
    for key, claimants in claims.items():
        if len(claimants) == 1:
            resolved[claimants[0]] = key
    return resolved


def gt_gate(
    ground_truth_layers: Sequence[Mapping[str, Any]],
    sheets: Mapping[str, SheetFit],
    deltas: Deltas,
    gate_m: float = SEAM_GATE_M,
) -> tuple[float, float, int, bool] | None:
    """Median human-GCP RMSE before/after the deltas; ``None`` without GT.

    Returns ``(median_before, median_after, n_sheets, passed)``. Distances are
    3857 planar meters (matching the recorded seam_deltas fixtures).
    """
    before: list[float] = []
    after: list[float] = []
    pinned = [page_from_slug(lyr.get("slug") or "") for lyr in ground_truth_layers]
    resolved = resolve_pages([p for p in pinned if p is not None], sheets)
    # one sheet votes once however many layers reach it: two corpora spelling
    # the same page differently must not weight it twice in the median
    seen: set[str] = set()
    for lyr, page in zip(ground_truth_layers, pinned, strict=True):
        key = resolved.get(page) if page is not None else None
        if key is None or key in seen or key not in deltas:
            continue
        seen.add(key)
        s = sheets[key]
        errs_b: list[float] = []
        errs_a: list[float] = []
        for ft in (lyr.get("gcps_geojson") or {}).get("features") or []:
            px, py = ft["properties"]["image"]
            lng, lat = ft["geometry"]["coordinates"]
            x, y = TO_3857.transform(lng, lat)
            wx, wy = apply_affine(s.coef, px, py)
            errs_b.append(math.hypot(wx - x, wy - y))
            errs_a.append(math.hypot(wx + deltas[key][0] - x, wy + deltas[key][1] - y))
        if errs_b:
            before.append(rms(errs_b))
            after.append(rms(errs_a))
    if not before:
        return None
    med_b, med_a = float(np.median(before)), float(np.median(after))
    return med_b, med_a, len(before), med_a <= med_b + gate_m


@dataclass(frozen=True)
class GroundTruthSource:
    """One volunteer export considered for a volume, and what it held."""

    path: Path
    layers: list[dict[str, Any]]
    pages: dict[str, dict[str, Any]]
    #: The file exists but is EMPTY: the corpus was checked, this volume was
    #: never pinned. Different from no file at all, and the difference is the
    #: whole reason an operator can tell "unscoreable" from "unchecked".
    empty_marker: bool

    def as_record(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "pinned_pages": len(self.pages),
            "empty_marker": self.empty_marker,
        }


def export_path(directory: Path, volume: str) -> Path:
    """Where a volume's volunteer GCP export lives inside a ground-truth dir."""
    return directory / f"api-layers-{volume}.json"


def load_sources(volume: str, directories: Sequence[Path]) -> list[GroundTruthSource]:
    """Every readable export for ``volume``, in the order the caller gave.

    A volume's pins can sit in more than one corpus, so a single pass can grade
    against all of them; where two carry the same page, the FIRST directory
    named wins. A missing file is silent; an empty one is recorded as checked.
    One damaged export is skipped with a warning rather than sinking the pass,
    as ``status`` does over the same corpus.
    """
    out: list[GroundTruthSource] = []
    for directory in directories:
        path = export_path(directory, volume)
        if not path.is_file():
            continue
        try:
            text = path.read_text()
            if not text.strip():
                out.append(GroundTruthSource(path, [], {}, empty_marker=True))
                continue
            pages = load_ground_truth(path)
            layers: list[dict[str, Any]] = json.loads(text)
        except (OSError, ValueError) as exc:
            logger.warning("%s: unreadable ground-truth export (%s); skipped", path, exc)
            continue
        out.append(GroundTruthSource(path, layers, pages, empty_marker=False))
    return out


def merge_pages(sources: Sequence[GroundTruthSource]) -> dict[str, dict[str, Any]]:
    """``{page: layer}`` across the sources, earlier directories winning.

    Two corpora can spell one page differently in case, so precedence is
    decided case-insensitively: otherwise the later corpus's spelling survives
    as a second key and a placement joins to whichever spelling matches it,
    which is not the corpus the caller asked for.
    """
    merged: dict[str, dict[str, Any]] = {}
    claimed: set[str] = set()
    for source in sources:
        for page, layer in source.pages.items():
            if page.casefold() in claimed:
                continue
            claimed.add(page.casefold())
            merged[page] = layer
    return merged


def _seam_diagnostic(
    paths: VolumePaths, layers: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    """What the human GCPs say about the seam solve this tree already applied.

    Reconstructs the PRE-seam fits the solve worked from, so the verdict is the
    one the in-run gate used to give. Advisory: the shift is on disk either way.
    """
    if not paths.seam_deltas.exists():
        return None
    record: Any = json.loads(paths.seam_deltas.read_text())
    raw = (record or {}).get("deltas") if isinstance(record, dict) else None
    if not raw:
        return None
    deltas = {page: (float(d[0]), float(d[1])) for page, d in raw.items()}
    sheets: dict[str, SheetFit] = {}
    for page, r, _rp in iter_results(paths):
        if page not in deltas:
            continue
        fit = sheet_fit_from_result(page, preseam_record(r))
        if fit is not None:
            sheets[page] = fit
    gated = gt_gate(layers, sheets, deltas)
    if gated is None:
        return None
    med_b, med_a, n, passed = gated
    return {
        "gt_median_before_m": round(med_b, 3),
        "gt_median_after_m": round(med_a, 3),
        "n_sheets": n,
        "verdict": "PASSED" if passed else "FAILED",
    }


def score_volume(paths: VolumePaths, volume: str, directories: Sequence[Path]) -> dict[str, Any]:
    """Score every accepted placement in ``paths`` and write the sidecar.

    Reviewer-verified sheets are skipped: those placements are human work, and
    grading them against other human work measures nothing. Returns the payload,
    whose ``pages`` maps page -> ``rmse_vs_human_m``. Nothing is written when no
    export exists at all: a mistyped directory must not erase a real grading.
    """
    sources = load_sources(volume, directories)
    layers = merge_pages(sources)
    payload: dict[str, Any] = {
        "gate_m": GT_COMMIT_RMSE_M,
        "sources": [s.as_record() for s in sources],
        "pages": {},
    }
    if not sources:
        logger.warning(
            "%s: no ground-truth export in %s; nothing scored and nothing written",
            volume,
            ", ".join(str(d) for d in directories),
        )
        return payload
    if not layers:
        write_sidecar(paths, payload)
        logger.info(
            "%s: %d ground-truth source(s) carry no pinned page; nothing to score",
            volume,
            len(sources),
        )
        return payload
    correction_lat = mercator_correction_lat(layers)
    manifest = json.loads(paths.manifest.read_text())
    scored: dict[str, dict[str, Any]] = {}
    # resolved over EVERY page in the tree, not per page as it comes up: the
    # refusal to fold is about two pages competing for one pin, which a
    # one-page-at-a-time lookup cannot see
    resolved = resolve_pages((page for page, _r, _p in iter_results(paths)), layers)
    for page, record, result_path in iter_results(paths):
        status = str(record.get("status", ""))
        if not status.startswith("OK") or is_reviewer_verified(status):
            continue
        key = resolved.get(page)
        rmse = score_record_vs_ground_truth(
            record, manifest.get(f"p{page}"), layers.get(key) if key else None, correction_lat
        )
        if rmse is None:
            continue
        # stamped with the record it graded, so the score cannot outlive it
        scored[page] = {
            "rmse_vs_human_m": round(rmse, 2),
            "record_sha256": record_digest(result_path),
        }
    payload["pages"] = scored
    # the MERGED layers, not every source's raw list: a page two corpora both
    # pin would otherwise contribute twice and skew the median
    seam = _seam_diagnostic(paths, list(layers.values()))
    if seam is not None:
        payload["seam"] = seam
    write_sidecar(paths, payload)
    over = [p for p, e in scored.items() if e["rmse_vs_human_m"] > GT_COMMIT_RMSE_M]
    logger.info(
        "%s: scored %d accepted placement(s); %d beyond the %.0f m commit gate",
        volume,
        len(scored),
        len(over),
        GT_COMMIT_RMSE_M,
    )
    return payload


__all__ = [
    "SEAM_GATE_M",
    "GroundTruthSource",
    "export_path",
    "gt_gate",
    "load_sources",
    "merge_pages",
    "resolve_pages",
    "score_volume",
]
