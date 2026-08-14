"""Review server state: the reviewable pool, sheet payloads, and the sidecar save.

The frame conventions, gate semantics, and mask-validity contract for the
whole surface are on the package docstring (mod:`autogeoref.review`).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ..affine import TO_3857, AffineMatrix, apply_affine
from ..errors import ReviewError
from ..frames import small_corners_full_px
from ..geometry import clip_features_4326
from ..paths import (
    VolumeBusyError,
    VolumePaths,
    iter_results,
    regions_by_page,
    volume_lock,
)
from ..placement_records import pinned_orientation
from ..scoring import load_scores
from ..slugs import page_sort_key, slug_for_page, valid_review_page
from ..validation import volume_id
from ..volume import is_committed, status_ok
from ..volume_constants import resolve_constants
from .materialize import (
    affine_from_record,
    compose_ops,
    corners_4326,
    displayable_affine,
    dryrun_against_region,
    final_gcps_geojson,
    mask_px_from_ring_4326,
    seed_affine,
)
from .sidecars import (
    _now_iso,
    load_sidecar,
    result_sha256,
    save_sidecar,
    sidecar_from_dict,
    sidecar_path,
    sidecar_to_dict,
)

if TYPE_CHECKING:
    from ..config.model import CityConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Queue + sheet payloads (what the UI consumes).
# --------------------------------------------------------------------------


def review_queue(
    paths: VolumePaths, volume: str, *, include_ok: bool = False
) -> list[dict[str, Any]]:
    """The reviewable pool, page-ordered: flagged sheets (default) or all.

    Committed sheets enter with ``include_ok`` for mask fixes; the UI tags
    them so a reviewer does not casually move a gate-passed placement.
    """
    entries: list[dict[str, Any]] = []
    for page, r, rp in iter_results(paths, sort_key=lambda p: page_sort_key(p.stem)):
        status = str(r.get("status", ""))
        flagged = not status_ok(status)
        if not flagged and not include_ok:
            continue
        side_p = sidecar_path(paths, page)
        verdict = None
        applied = False
        if side_p.exists():
            try:
                side = load_sidecar(side_p, volume=volume)
                verdict = side.verdict
                applied = side.applied_result_sha256 == result_sha256(rp)
            except (ReviewError, json.JSONDecodeError):
                verdict = "invalid-sidecar"
        entries.append(
            {
                "volume": volume,
                "page": page,
                "status": status,
                "flagged": flagged,
                "committed": is_committed(r),
                "has_placement": displayable_affine(affine_from_record(r)) is not None,
                "verdict": verdict,
                "applied": applied,
            }
        )
    return entries


# --------------------------------------------------------------------------
# The app: state shared by the request handler + the apply step's CLI entry.
# --------------------------------------------------------------------------

# Volume ids are validated by validation.volume_id — the one grammar, shared
# with the queue. Page ids are validated by slugs.valid_review_page — the one
# narrow grammar (digits + optional letter suffix, plus the literal cbd1/cbd2
# named sheets), shared with the sidecar schema so the UI and persisted files
# agree.


@dataclass
class ReviewApp:
    """Server state: work tree, city config, static dirs, lazy caches."""

    work: Path
    city: CityConfig
    ui_dir: Path
    vendor_dir: Path
    volumes: list[str] = field(default_factory=list)
    include_ok: bool = False
    dryrun_timeout_s: float = 120.0
    _manifests: dict[str, dict[str, Any]] = field(default_factory=dict)
    _centerline_features: list[dict[str, Any]] | None = None
    #: (volume, page) whose ghost overlay THIS SERVER has actually painted — i.e.
    #: whose sheet pixels it handed to the client. See :meth:`overlay_shown`.
    _shown: set[tuple[str, str]] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.volumes:
            self.volumes = sorted(
                p.parent.name
                for p in self.work.glob("*/results")
                if p.is_dir() and any(p.glob("p*.json"))
            )

    def paths(self, volume: str) -> VolumePaths:
        try:
            volume_id(volume)
        except ValueError:
            raise ReviewError(f"unknown volume {volume!r}") from None
        if volume not in self.volumes:
            raise ReviewError(f"unknown volume {volume!r}")
        return VolumePaths(root=self.work / volume)

    # -- "did you actually LOOK at it?" ---------------------------------------

    def resolve_media(self, volume: str, filename: str) -> Path:
        """Resolve a sheet image WITHOUT recording an overlay as shown.

        This is the reference route (``/refmedia``): neighbor sheets drawn as
        context around the sheet under review. It must not touch ``_shown`` —
        painting p11 as p12's neighbor is not evidence anyone reviewed p11's
        own placement, and letting it count would quietly reopen the unseen-
        verdict hole :meth:`overlay_shown` exists to close.
        """
        paths = self.paths(volume)  # validates the volume
        return paths.root / "sheets" / filename

    def media_path(self, volume: str, filename: str) -> Path:
        """Resolve a sheet image request, and RECORD that its overlay was painted.

        The ghost overlay is a raster of exactly this file (``sheet_payload``'s
        ``small_url``), so serving it is the closest thing the server has to proof
        that a human was shown where the sheet lands. :meth:`save` requires it.
        """
        path = self.resolve_media(volume, filename)
        for page, info in self.manifest(volume).items():
            if not page.startswith("_") and info.get("file") == filename:
                self._shown.add((volume, page[1:]))
                break
        return path

    def overlay_shown(self, volume: str, page: str) -> bool:
        """Has this server painted this sheet's ghost overlay for a client?

        **What this is, precisely**: evidence that the sheet's pixels were sent to a browser
        that asked for its placement — not proof that a retina received them, which a local
        admin tool with one trusted operator does not need. It refuses a verdict for a sheet
        whose overlay was never rendered: a bulk "accept all", an accept wired to a summary
        row, a client posting a verdict for the NEXT page. Server-side, because a client-side
        rule is one the next client forgets, and per-process. The media route sends ``no-store``
        so a cached re-open cannot leave this set empty.
        """
        return (volume, page) in self._shown

    def manifest(self, volume: str) -> dict[str, Any]:
        if volume not in self._manifests:
            mp = self.paths(volume).manifest
            self._manifests[volume] = json.loads(mp.read_text()) if mp.exists() else {}
        return self._manifests[volume]

    # -- placement seeds ---------------------------------------------------

    def _constants(self, volume: str) -> tuple[float, float] | None:
        # persisted first: the UI trusts what the pipeline actually derived
        return resolve_constants(
            self.paths(volume), self.city.volume(volume), prefer_persisted=True
        )

    def _committed_center_3857(self, volume: str) -> tuple[float, float] | None:
        xs: list[float] = []
        ys: list[float] = []
        for _page, r, _rp in iter_results(self.paths(volume)):
            if not is_committed(r):
                continue
            for ft in (r.get("gcps_geojson") or {}).get("features") or []:
                lng, lat = ft["geometry"]["coordinates"]
                x, y = TO_3857.transform(lng, lat)
                xs.append(x)
                ys.append(y)
        if not xs:
            return None
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def _seed(self, volume: str, page: str, full_size: tuple[float, float]) -> AffineMatrix | None:
        constants = self._constants(volume)
        center = self._committed_center_3857(volume)
        if center is None:
            # nothing committed yet: fall back to the volume's configured
            # bounds bbox so a zero-committed volume is still placeable by hand
            bbox = self.city.volume(volume).bounds_bbox
            if bbox is not None:
                west, south, east, north = bbox
                center = TO_3857.transform((west + east) / 2, (south + north) / 2)
        if constants is None or center is None:
            return None
        # a page printed at another scale seeds at ITS scale: seeding a 200 ft/in
        # sheet at the book's 50 ft/in draws the ghost overlay at a quarter size,
        # and the ghost review is the ONLY check these sheets get (no ground truth)
        multiple = self.city.volume(volume).page_scale_multiples.get(page, 1.0)
        return seed_affine(multiple * constants[0], constants[1], center, full_size)

    # -- payloads ------------------------------------------------------------

    #: Result-record fields the UI's evidence panel shows so the reviewer can
    #: see WHY the pipeline flagged the sheet instead of re-litigating blind.
    #: Pass-through of recorded pipeline facts only — the two DERIVED entries
    #: (the residual summary and the pinned-orientation flag) are added
    #: explicitly in :meth:`_evidence_summary` below.
    _EVIDENCE_KEYS: ClassVar[tuple[str, ...]] = (
        "n_streets",
        "n_candidates",
        "n_inliers",
        "rotation_deg",
        "inlier_streets",
        "rescue_anchors",
        "junction_snap",
        "verified_accept",
        "escalated_model",
        "seam_adjusted",
    )

    @classmethod
    def _evidence_summary(cls, r: Mapping[str, Any]) -> dict[str, Any]:
        evidence = {k: r[k] for k in cls._EVIDENCE_KEYS if r.get(k) is not None}
        residuals = r.get("auto_residuals_m")
        if isinstance(residuals, list) and residuals:
            values = [
                float(v)
                for v in residuals
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            if values:
                evidence["auto_residuals_m"] = {
                    "n": len(values),
                    "max": round(max(values), 1),
                    "mean": round(sum(values) / len(values), 1),
                }
        # the reviewer is looking at a placement whose orientation nothing
        # fitted, and the map shows its synthetic corners as perfect (zero
        # length) GCP ties. Say so, or the panel reads as strong evidence.
        if pinned_orientation(r):
            evidence["pinned_orientation"] = True
        return evidence

    def sheet_payload(self, volume: str, page: str) -> dict[str, Any]:
        if not valid_review_page(page):
            raise ReviewError(f"bad page {page!r}")
        paths = self.paths(volume)
        rp = paths.results / f"p{page}.json"
        if not rp.exists():
            raise ReviewError(f"no result for {volume} p{page}")
        r = json.loads(rp.read_text())
        info = self.manifest(volume).get(f"p{page}")
        if info is None:
            raise ReviewError(f"p{page}: no sheets/manifest.json entry")
        full_size = (float(info["full_size"][0]), float(info["full_size"][1]))
        m = displayable_affine(affine_from_record(r))
        seeded = m is None
        if m is None:
            m = self._seed(volume, page, full_size)
        slug = r.get("layer") or slug_for_page(volume, page)
        payload: dict[str, Any] = {
            "volume": volume,
            "page": page,
            "slug": slug,
            "status": str(r.get("status", "")),
            "committed": is_committed(r),
            "base_result_sha256": result_sha256(rp),
            "seeded": seeded,
            "affine": None if m is None else [[float(v) for v in row] for row in m],
            "small_url": f"/media/{volume}/{info['file']}",
            "small_size": info["small_size"],
            "full_size": info["full_size"],
            "scale": info["scale"],
            "rotation_applied": int(info.get("rotation_applied", 0)),
            "corners_px": [list(c) for c in small_corners_full_px(info)],
            # display only, and read from the scoring pass's sidecar: no
            # placement decision anywhere consults it
            "rmse_vs_human_m": load_scores(paths).get(page),
            "evidence": self._evidence_summary(r),
            "has_region": page in regions_by_page(paths.regions),
            "mask_px": None,
            "sidecar": None,
        }
        payload["gcps"] = [
            {
                "image": ft["properties"]["image"],
                "coordinates": ft["geometry"]["coordinates"],
                "note": ft["properties"].get("note", ""),
            }
            for ft in (r.get("gcps_geojson") or {}).get("features") or []
        ]
        side_p = sidecar_path(paths, page)
        if side_p.exists():
            try:
                payload["sidecar"] = sidecar_to_dict(load_sidecar(side_p, volume=volume))
                payload["mask_px"] = payload["sidecar"]["mask_px"]
            except (ReviewError, json.JSONDecodeError) as exc:
                # a corrupt sidecar must not brick the sheet view
                payload["sidecar_error"] = str(exc)
        if payload["mask_px"] is None and m is not None:
            mask_file = paths.masks / f"{slug}.geojson"
            if mask_file.exists():
                geom = json.loads(mask_file.read_text())["features"][0]["geometry"]
                ring = geom["coordinates"][0]
                payload["mask_px"] = mask_px_from_ring_4326(ring, m)
        return payload

    def placed_payload(self, volume: str) -> dict[str, Any]:
        """Every committed sheet's placement: neighbor context for the UI.

        The strongest alignment reference for a flagged sheet is usually its
        already-committed neighbor (Sanborn sheets adjoin), so the UI draws
        these as faint fixed overlays. Images are served via ``/refmedia``
        (no overlay-shown recording — see :meth:`resolve_media`).
        """
        paths = self.paths(volume)
        manifest = self.manifest(volume)
        placed: list[dict[str, Any]] = []
        for page, r, _rp in iter_results(paths, sort_key=lambda p: page_sort_key(p.stem)):
            if not is_committed(r):
                continue
            info = manifest.get(f"p{page}")
            m = displayable_affine(affine_from_record(r))
            if info is None or m is None:
                continue
            placed.append(
                {
                    "page": page,
                    "status": str(r.get("status", "")),
                    "small_url": f"/refmedia/{volume}/{info['file']}",
                    "corners": corners_4326(m, small_corners_full_px(info)),
                }
            )
        return {"volume": volume, "placed": placed}

    def centerlines_payload(self, volume: str) -> dict[str, Any]:
        """Modern centerlines clipped to the volume's evidence, slimmed."""
        bounds = self._volume_bounds_4326(volume)
        if bounds is None:
            return {"type": "FeatureCollection", "features": []}
        if self._centerline_features is None:
            self._centerline_features = json.loads(self.city.centerlines_path.read_text())[
                "features"
            ]
        out = []
        name_prop = self.city.centerline_name_property
        type_prop = self.city.centerline_type_property
        for f in clip_features_4326(self._centerline_features, bounds):
            props = f.get("properties") or {}
            out.append(
                {
                    "type": "Feature",
                    "properties": {
                        "name": " ".join(
                            str(props.get(k) or "") for k in (name_prop, type_prop)
                        ).strip()
                    },
                    "geometry": f.get("geometry") or {},
                }
            )
        return {"type": "FeatureCollection", "features": out}

    def _volume_bounds_4326(self, volume: str) -> tuple[float, float, float, float] | None:
        xs: list[float] = []
        ys: list[float] = []
        for _page, r, _rp in iter_results(self.paths(volume)):
            for ft in (r.get("gcps_geojson") or {}).get("features") or []:
                lng, lat = ft["geometry"]["coordinates"]
                xs.append(float(lng))
                ys.append(float(lat))
        if not xs:
            return None
        return min(xs), min(ys), max(xs), max(ys)

    # -- save ----------------------------------------------------------------

    def save(self, volume: str, page: str, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        """Validate + persist one sidecar; dry-run an edited mask when possible.

        Returns ``(http_status, payload)``. 409 = the result drifted under the edit, or another
        operation owns the volume right now (retry after it finishes); 422 = GDAL rejected the
        mask (the error text is returned, per the mask-validity contract); 428 = the overlay was
        never shown. **A verdict asserts that a human LOOKED**, so this refuses one for a sheet
        whose ghost overlay this server never painted — every verdict, not just ``accept``. A
        verdict that saw nothing is worth LESS than none, because ``apply`` records it as
        ``reviewer_verified`` and the auto-acceptance statistics then exclude the page.
        """
        if not valid_review_page(page):
            return 400, {"error": f"bad page {page!r}"}
        if not self.overlay_shown(volume, page):
            return 428, {
                "error": f"p{page}: this sheet's ghost overlay has not been rendered by this "
                "server, so a verdict on it would assert a look nobody took. Open the sheet "
                "in the review pane (which draws it over the map) and decide there."
            }
        paths = self.paths(volume)
        rp = paths.results / f"p{page}.json"
        if not rp.exists():
            return 404, {"error": f"no result for p{page}"}
        # One transaction under the volume lock: the result-hash check, the
        # mask dry-run, and the sidecar write. An apply (or run) sliding in
        # between the sha check and the write would persist this edit against
        # a result that no longer exists — the silent way reviewer work gets
        # stranded. The browser still holds the edit on a refusal.
        try:
            with volume_lock(paths, operation="review save"):
                return self._save_locked(paths, volume, page, body, rp)
        except VolumeBusyError as exc:
            return 409, {"error": str(exc)}

    def _save_locked(
        self, paths: VolumePaths, volume: str, page: str, body: Mapping[str, Any], rp: Path
    ) -> tuple[int, dict[str, Any]]:
        """The transactional tail of :meth:`save` (caller holds the volume lock)."""
        current_sha = result_sha256(rp)
        side = sidecar_from_dict(
            {
                **body,
                "volume": volume,
                "page": page,
                "timestamp": _now_iso(),
                "applied_result_sha256": None,
            }
        )
        if side.base_result_sha256 != current_sha:
            return 409, {
                "error": "the result record changed since this sheet was loaded — reload it"
            }
        info = self.manifest(volume).get(f"p{page}")
        if info is None:
            return 404, {"error": f"p{page}: no sheets/manifest.json entry"}
        r = json.loads(rp.read_text())
        full_size = (float(info["full_size"][0]), float(info["full_size"][1]))
        # the client's affine is DISPLAY math; apply materializes the op log —
        # so the two must agree, or what the reviewer saw is not what lands.
        # Only checkable when the base is the recorded GCP fit (seeded bases —
        # including degenerate records the payload seeded over — depend on
        # state the sha guard does not cover).
        base = displayable_affine(affine_from_record(r))
        if side.affine is not None and base is not None:
            expected = compose_ops(base, side.ops)
            got = np.asarray(side.affine)
            corners = ((0.0, 0.0), (full_size[0], 0.0), full_size, (0.0, full_size[1]))
            drift = max(
                math.hypot(
                    *(np.subtract(apply_affine(expected, px, py), apply_affine(got, px, py)))
                )
                for px, py in corners
            )
            if drift > 0.05:
                return 400, {
                    "error": f"affine does not match the op log (corner drift "
                    f"{drift:.3f} m) — client/server math disagree; reload the sheet"
                }
        dryrun = "not-run"
        if side.mask_px is not None and side.verdict in ("accept", "adjusted"):
            if side.affine is None:
                return 400, {"error": "mask edits need a placement affine"}
            fc = final_gcps_geojson(r, side, full_size)
            region = regions_by_page(paths.regions).get(page)
            if fc is not None and region is not None:
                try:
                    ok, detail = dryrun_against_region(
                        region,
                        fc,
                        side.mask_px,
                        np.asarray(side.affine),
                        timeout_s=self.dryrun_timeout_s,
                    )
                except Exception as exc:  # noqa: BLE001 - GDAL missing/hung: infrastructure
                    logger.warning("p%s: mask dry-run unavailable: %s", page, exc)
                    dryrun = f"unavailable ({exc})"
                else:
                    if not ok:
                        return 422, {"error": f"gdalwarp rejected the mask: {detail}"}
                    dryrun = "passed"
            else:
                dryrun = "skipped (no full-res image)"
        save_sidecar(paths, side)
        return 200, {"ok": True, "dryrun": dryrun, "sidecar": sidecar_to_dict(side)}
