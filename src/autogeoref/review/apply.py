"""Apply: materialize sidecars into reviewer-verified results + masks + warps.

A batch stage, not a server concern: the CLI's ``review --apply`` and the
console's apply route both call :func:`apply_reviews` directly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import ReviewError
from ..paths import (
    VolumePaths,
    regions_by_page,
    volume_lock,
    write_if_changed,
    write_result,
)
from ..scoring import drop_score
from ..slugs import page_sort_key, slug_for_page
from ..volume import (
    STATUS_REVIEWER_VERIFIED,
    is_committed,
    reviewer_result_key,
)
from .materialize import (
    affine_from_record,
    dryrun_against_region,
    final_gcps_geojson,
    mask_ring_4326,
)
from .sidecars import (
    ReviewSidecar,
    load_sidecar,
    result_sha256,
    review_dir,
    save_sidecar,
)

if TYPE_CHECKING:
    from ..affine import AffineMatrix


def apply_reviews(
    paths: VolumePaths,
    volume: str,
    *,
    timeout_s: float = 600.0,
    do_warp: bool = True,
) -> dict[str, Any]:
    """Materialize every sidecar for one volume, as the volume's sole owner.

    Takes :func:`paths.volume_lock` for the whole apply — result rewrites,
    mask upserts, and re-warps must not interleave with a concurrent run or
    prep of the same tree. Raises :class:`paths.VolumeBusyError` when another
    operation holds the volume; both CLI ``--apply`` and the console's apply
    route surface that as an actionable refusal.
    """
    with volume_lock(paths, operation="review --apply"):
        return _apply_reviews_locked(paths, volume, timeout_s=timeout_s, do_warp=do_warp)


def _apply_reviews_locked(
    paths: VolumePaths,
    volume: str,
    *,
    timeout_s: float,
    do_warp: bool,
) -> dict[str, Any]:
    """Materialize every sidecar for one volume.

    accept/adjusted -> the result becomes ``OK (reviewer-verified)`` with the op log as
    provenance; an edited mask re-runs the cutline dry-run and, when it passes, lands in
    ``masks/`` AND on the result as ``reviewer_mask_px``, which the back half uses instead of
    re-detecting page bounds so the next ``--warp`` cannot clobber the crop. reject /
    needs-manual-mask -> the verdict is recorded and the status untouched. A sidecar whose base
    sha no longer matches is SKIPPED with a warning, unless it matches the sha apply wrote.
    """
    summary = _new_summary(volume)
    sidecars = sorted(review_dir(paths).glob("p*.json"), key=lambda p: page_sort_key(p.stem))
    if not sidecars:
        summary["warnings"].append("no sidecars to apply")
        return summary
    manifest = json.loads(paths.manifest.read_text()) if paths.manifest.exists() else {}
    images = regions_by_page(paths.regions)
    needs_backhalf = False
    for sp in sidecars:
        cand = _load_candidate(paths, volume, manifest, sp, summary)
        if cand is None:
            continue
        if cand.side.verdict in ("reject", "needs-manual-mask"):
            _apply_dissent(paths, cand, summary)
            continue
        # accept / adjusted
        fc = _resolve_final_gcps(cand, summary)
        if fc is None:
            continue
        region = images.get(cand.page)
        m_final = affine_from_record({"gcps_geojson": fc})
        mask_valid = _validate_mask(cand, fc, m_final, region, summary, timeout_s=timeout_s)
        _materialize_accept(paths, cand, fc, mask_valid, summary)
        if mask_valid and cand.side.mask_px is not None and m_final is not None:
            _write_mask(paths, cand.slug, cand.side.mask_px, m_final)
            summary["masks_written"].append(cand.slug)
        needs_backhalf |= _rewarp_or_defer(
            paths, cand, fc, region, summary, do_warp=do_warp, timeout_s=timeout_s
        )
    if summary["warped"] or summary["masks_written"] or needs_backhalf:
        summary["rerun_hint"] = (
            f"re-run `autogeoref run {volume} --warp ...` to refresh mask splitting, "
            f"the mosaic, and {volume}.pmtiles"
        )
    return summary


def _new_summary(volume: str) -> dict[str, Any]:
    """The apply summary skeleton — its key set and value shapes are a contract:

    both operator surfaces render it (the CLI prints it, the console returns it
    as the apply route's JSON body), and tests pin the semantics of
    ``applied`` vs ``already_applied`` vs ``skipped``.
    """
    return {
        "volume": volume,
        "applied": [],
        "already_applied": [],
        "skipped": [],
        "warnings": [],
        "warped": [],
        "masks_written": [],
        "rerun_hint": None,
    }


@dataclass(frozen=True)
class _Candidate:
    """One loadable, sha-checked sidecar with the state its apply needs."""

    side: ReviewSidecar
    page: str
    rp: Path
    r: dict[str, Any]
    info: dict[str, Any]
    slug: str
    #: the sidecar was already applied to exactly this result (idempotent
    #: re-run): record nothing twice, but still late-materialize a mask
    already: bool


def _load_candidate(
    paths: VolumePaths,
    volume: str,
    manifest: Mapping[str, Any],
    sp: Path,
    summary: dict[str, Any],
) -> _Candidate | None:
    """Load + gate one sidecar file; ``None`` records the skip on the summary.

    A sidecar is UNTRUSTED input here: its page interpolates into result
    and mask paths, and its volume says whose tree it may touch — a
    forged or misplaced file is skipped loudly, never materialized.
    """
    try:
        side = load_sidecar(sp, volume=volume)
    except (ReviewError, json.JSONDecodeError) as exc:
        summary["skipped"].append(sp.stem.removeprefix("p"))
        summary["warnings"].append(f"{sp.name}: invalid sidecar not applied ({exc})")
        return None
    page = side.page
    if sp.name != f"p{page}.json":
        summary["skipped"].append(sp.stem.removeprefix("p"))
        summary["warnings"].append(
            f"{sp.name}: sidecar claims page {page!r} — mismatched file, not applied"
        )
        return None
    rp = paths.results / f"p{page}.json"
    if not rp.exists():
        summary["skipped"].append(page)
        summary["warnings"].append(f"p{page}: sidecar without a result record")
        return None
    current_sha = result_sha256(rp)
    already = side.applied_result_sha256 == current_sha
    if not already and side.base_result_sha256 != current_sha:
        summary["skipped"].append(page)
        summary["warnings"].append(f"p{page}: result changed since review (re-open it in the UI)")
        return None
    r = json.loads(rp.read_text())
    return _Candidate(
        side=side,
        page=page,
        rp=rp,
        r=r,
        info=manifest.get(f"p{page}") or {},
        slug=r.get("layer") or slug_for_page(volume, page),
        already=already,
    )


def _apply_dissent(paths: VolumePaths, cand: _Candidate, summary: dict[str, Any]) -> None:
    """reject / needs-manual-mask: record the verdict on the result, status untouched."""
    side, r, page = cand.side, cand.r, cand.page
    if cand.already:
        summary["already_applied"].append(page)
        return
    if is_committed(r) and side.verdict == "reject":
        # demoting a committed placement is destructive; record the
        # human dissent and surface it, never auto-revoke
        summary["warnings"].append(
            f"p{page}: reviewer REJECTED a committed placement — recorded as "
            f"dissent only; flag to the reviewer for manual action"
        )
    r.pop("owner_review", None)  # migrate the pre-rename spelling
    r["reviewer_review"] = {
        "verdict": side.verdict,
        "ops": side.ops,
        "timestamp": side.timestamp,
        "note": side.note,
    }
    write_result(cand.rp, r)
    save_sidecar(paths, replace(side, applied_result_sha256=result_sha256(cand.rp)))
    summary["applied"].append(page)


def _resolve_final_gcps(cand: _Candidate, summary: dict[str, Any]) -> dict[str, Any] | None:
    """The GCP collection this accept/adjusted materializes to.

    New applies derive it from the sidecar (op log over the record, or
    synthetic corners); an already-applied sidecar reuses the record's own —
    apply wrote those very GCPs, and re-deriving would need the pre-apply
    record that no longer exists. ``None`` records the skip on the summary.
    """
    side, page = cand.side, cand.page
    if cand.already:
        collection: dict[str, Any] = cand.r["gcps_geojson"]
        return collection
    if not cand.info:
        summary["skipped"].append(page)
        summary["warnings"].append(f"p{page}: no manifest entry, cannot apply")
        return None
    full_size = (float(cand.info["full_size"][0]), float(cand.info["full_size"][1]))
    fc = final_gcps_geojson(cand.r, side, full_size)
    if fc is None:
        summary["skipped"].append(page)
        summary["warnings"].append(f"p{page}: {side.verdict} without a placement")
        return None
    return fc


def _validate_mask(
    cand: _Candidate,
    fc: Mapping[str, Any],
    m_final: AffineMatrix | None,
    region: Path | None,
    summary: dict[str, Any],
    *,
    timeout_s: float,
) -> bool:
    """The edited mask must pass THE dry-run before anything records it.

    A validated mask lands on the result as ``reviewer_mask_px`` so the
    back half's mask stage uses it instead of re-detecting the page
    rectangle (which would silently clobber the reviewer's crop).
    """
    side, page = cand.side, cand.page
    if side.mask_px is None or m_final is None:
        return False
    if region is None:
        summary["warnings"].append(
            f"p{page}: edited mask NOT materialized — no full-res image to "
            f"dry-run against (add regions/{cand.slug} and re-run --apply)"
        )
        return False
    ok, detail = dryrun_against_region(region, fc, side.mask_px, m_final, timeout_s=timeout_s)
    if not ok:
        summary["warnings"].append(
            f"p{page}: gdalwarp rejected the edited mask — not written "
            f"({detail or 'no detail'}); sheet proceeds unmasked"
        )
    return ok


def _materialize_accept(
    paths: VolumePaths,
    cand: _Candidate,
    fc: dict[str, Any],
    mask_valid: bool,
    summary: dict[str, Any],
) -> None:
    """Land the verdict (and any validated mask) on the result record, once."""
    side, r = cand.side, cand.r
    changed = False
    if not cand.already:
        r["gcps_geojson"] = fc
        # the AUTO placement's score does not describe the HUMAN one that just
        # replaced it, so it is forgotten rather than recomputed: a reviewer-verified
        # placement is human truth and is never auto-scored. Leaving the stale score
        # in the sidecar would have a later demotion pass judge the human's work by
        # the machine's error.
        drop_score(paths, cand.page)
        r.pop("owner_review", None)  # migrate the pre-rename spelling
        r["reviewer_review"] = {
            "verdict": side.verdict,
            "ops": side.ops,
            "timestamp": side.timestamp,
            "note": side.note,
            "previous_status": str(r.get("status", "")),
        }
        r["status"] = STATUS_REVIEWER_VERIFIED
        changed = True
        summary["applied"].append(cand.page)
    else:
        summary["already_applied"].append(cand.page)
    if mask_valid and reviewer_result_key(r, "mask_px") != side.mask_px:
        # covers late materialization too: mask saved before the scan
        # arrived, validated on a later --apply run
        r.pop("owner_mask_px", None)  # migrate the pre-rename spelling
        r["reviewer_mask_px"] = side.mask_px
        changed = True
    if changed:
        write_result(cand.rp, r)
        save_sidecar(paths, replace(side, applied_result_sha256=result_sha256(cand.rp)))


def _rewarp_or_defer(
    paths: VolumePaths,
    cand: _Candidate,
    fc: dict[str, Any],
    region: Path | None,
    summary: dict[str, Any],
    *,
    do_warp: bool,
    timeout_s: float,
) -> bool:
    """Re-warp the sheet, or say why not; returns whether the back half is owed."""
    if not do_warp:
        # warp deferred by the caller (HTTP apply): the summary must still
        # say the back half is owed, or the hint only ever appears on CLI runs
        return not cand.already
    if region is None:
        summary["warnings"].append(f"p{cand.page}: no full-res image — not re-warped")
        return True
    from ..warp import gcps_from_feature_collection, warp_sheet

    warp_sheet(
        region,
        gcps_from_feature_collection(fc),
        paths.warped,
        slug=cand.slug,
        timeout_s=timeout_s,
    )
    summary["warped"].append(cand.slug)
    return False


def _write_mask(
    paths: VolumePaths, slug: str, mask_px: Sequence[Sequence[float]], m: AffineMatrix
) -> None:
    """Write the per-slug cutline + refresh the slug's entry in masks.geojson.

    The masks.geojson upsert is a read-modify-write of shared volume state, so
    the caller must hold :func:`paths.volume_lock`. Both writers take that same
    lock, under different operation strings — apply as "review --apply", the
    bake's writer inside `run`'s hold ("run" / "run --warp-only") — two
    unserialized upserts would each rewrite the collection from their own read
    and drop the other's slug.
    """
    ring = mask_ring_4326(mask_px, m)
    feature = {
        "type": "Feature",
        "properties": {"slug": slug},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }
    write_if_changed(
        paths.masks / f"{slug}.geojson",
        json.dumps({"type": "FeatureCollection", "features": [feature]}, indent=2),
    )
    coll_path = paths.masks / "masks.geojson"
    if coll_path.exists():
        coll = json.loads(coll_path.read_text())
    else:
        coll = {"type": "FeatureCollection", "features": [], "unmasked": []}
    coll["features"] = [
        f for f in coll.get("features", []) if f.get("properties", {}).get("slug") != slug
    ] + [feature]
    coll["features"].sort(key=lambda f: page_sort_key(str(f["properties"]["slug"])))
    coll["unmasked"] = [s for s in coll.get("unmasked", []) if s != slug]
    write_if_changed(coll_path, json.dumps(coll, indent=2))
