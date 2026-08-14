"""Review sidecar schema, validation, and persistence."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ReviewError
from ..fixture_sums import sha256_of
from ..paths import VolumePaths, atomic_write_text
from ..slugs import valid_review_page
from .ops import op_linear_offset

VERDICTS = frozenset({"accept", "reject", "adjusted", "needs-manual-mask"})


@dataclass(frozen=True)
class ReviewSidecar:
    """One reviewed sheet: verdict + op log + derived placement/mask."""

    volume: str
    page: str
    base_result_sha256: str
    verdict: str
    ops: list[dict[str, Any]] = field(default_factory=list)
    affine: list[list[float]] | None = None
    mask_px: list[list[float]] | None = None
    timestamp: str = ""
    note: str = ""
    applied_result_sha256: str | None = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _validate_ring(mask_px: Sequence[Sequence[float]]) -> list[list[float]]:
    if len(mask_px) < 3:
        raise ReviewError(f"mask_px needs >= 3 vertices, got {len(mask_px)}")
    out = []
    for pt in mask_px:
        if len(pt) != 2:
            raise ReviewError(f"mask_px vertex must be [px, py], got {pt!r}")
        out.append([float(pt[0]), float(pt[1])])
    return out


def sidecar_from_dict(
    d: Mapping[str, Any], *, validate_op: Callable[[Mapping[str, Any]], object] | None = None
) -> ReviewSidecar:
    """Validate + build a sidecar from its JSON form (raises ReviewError)."""
    verdict = d.get("verdict")
    if verdict not in VERDICTS:
        raise ReviewError(f"verdict must be one of {sorted(VERDICTS)}, got {verdict!r}")
    if validate_op is None:
        validate_op = op_linear_offset
    ops = list(d.get("ops") or [])
    for op in ops:
        validate_op(op)
    affine = d.get("affine")
    if affine is not None:
        arr = np.asarray(affine, dtype=np.float64)
        if arr.shape != (2, 3) or not np.all(np.isfinite(arr)):
            raise ReviewError("affine must be a finite 2x3 matrix")
        affine = [[float(v) for v in row] for row in arr]
    mask_px = d.get("mask_px")
    if mask_px is not None:
        mask_px = _validate_ring(mask_px)
    for key in ("volume", "page", "base_result_sha256"):
        if not d.get(key):
            raise ReviewError(f"sidecar missing {key}")
    # apply interpolates the page into results/ and review/ paths, so an
    # unvalidated persisted id is a traversal vector, not just a typo
    page = str(d["page"])
    if not valid_review_page(page):
        raise ReviewError(f"sidecar page {page!r} is not a reviewable page id")
    return ReviewSidecar(
        volume=str(d["volume"]),
        page=str(d["page"]),
        base_result_sha256=str(d["base_result_sha256"]),
        verdict=str(verdict),
        ops=ops,
        affine=affine,
        mask_px=mask_px,
        timestamp=str(d.get("timestamp") or _now_iso()),
        note=str(d.get("note") or ""),
        applied_result_sha256=d.get("applied_result_sha256"),
    )


def sidecar_to_dict(s: ReviewSidecar) -> dict[str, Any]:
    return {
        "volume": s.volume,
        "page": s.page,
        "base_result_sha256": s.base_result_sha256,
        "verdict": s.verdict,
        "ops": s.ops,
        "affine": s.affine,
        "mask_px": s.mask_px,
        "timestamp": s.timestamp,
        "note": s.note,
        "applied_result_sha256": s.applied_result_sha256,
    }


def review_dir(paths: VolumePaths) -> Path:
    return paths.root / "review"


def sidecar_path(paths: VolumePaths, page: str) -> Path:
    return review_dir(paths) / f"p{page}.json"


def save_sidecar(paths: VolumePaths, s: ReviewSidecar) -> Path:
    path = sidecar_path(paths, s.page)
    return atomic_write_text(path, json.dumps(sidecar_to_dict(s), indent=2))


def load_sidecar(
    path: Path,
    *,
    volume: str | None = None,
    validate_op: Callable[[Mapping[str, Any]], object] | None = None,
) -> ReviewSidecar:
    """Load + validate one sidecar; ``volume`` pins whose tree it may touch.

    A sidecar carried over from another volume's review dir would otherwise
    materialize the WRONG volume's placement onto same-numbered pages here, so
    every caller that knows which volume it is reading passes it.
    """
    s = sidecar_from_dict(json.loads(path.read_text()), validate_op=validate_op)
    if volume is not None and s.volume != volume:
        raise ReviewError(f"sidecar {path.name} belongs to volume {s.volume!r}, not {volume!r}")
    return s


def result_sha256(path: Path) -> str:
    return sha256_of(path)
