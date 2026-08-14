"""Two implied sheet extents per scored page: the placement's and the pins'.

The measurement half of the extent audit beside it. Fitting an affine to each
side and walking it over the sheet gives the ground size each side thinks the
sheet is; where those disagree, a metre figure between the two affines is a
scale disagreement rather than a displacement. Read-only, no model call and no
network.
"""

from __future__ import annotations

import json
import logging
import math
import os
import statistics
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autogeoref.affine import (
    TO_4326,
    AffineMatrix,
    apply_affine,
    fit_affine_checked,
    gcps_from_geojson,
)
from autogeoref.bounds import mercator_correction_lat
from autogeoref.paths import VolumePaths, iter_results
from autogeoref.score_pass import load_sources, merge_pages, resolve_pages
from autogeoref.scoring import score_record_vs_ground_truth
from autogeoref.volume import is_reviewer_verified

logger = logging.getLogger("score_extent")

ROOT = Path(__file__).resolve().parents[1]
# read-only; point these at the primary checkout when running from a worktree
DATA_ROOT = Path(os.environ.get("AUTOGEOREF_DATA_ROOT", str(ROOT)))
WORK = Path(os.environ.get("AUTOGEOREF_WORK", str(DATA_ROOT / "work")))
CITY2 = Path(os.environ.get("AUTOGEOREF_CITY2_WORK", str(WORK / "city2-probe")))

#: Default extent band. A CHOICE, not a derivation: wide enough to keep an
#: ordinary manufacturing spread, tight enough to drop a whole-city key map and
#: a cropped placement. The audit beside this moves it and sweeps it.
BAND = (0.9, 1.1)

#: Bands the sweep prices, so a reader sees what the choice costs.
SWEEP_BANDS = ((0.5, 2.0), (0.85, 1.18), (0.9, 1.1), (0.95, 1.05))


#: Per city slug, the work root its operations use and the pin corpora scoring
#: reads, both of which are `--work` / `--ground-truth` arguments rather than
#: config keys and so have to be written down. The CITY LIST is not: it comes
#: from the configs on disk, and a city with no row here is a hard failure
#: rather than a silent omission — a corpus figure that quietly covers one city
#: is this project's characteristic defect.
CITY_ROOTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "chicago": ("work", ("fixtures/ground-truth", "fixtures/prod")),
    "cleveland": ("work/city2-probe", ("fixtures/ground-truth",)),
    "crystal-lake": ("work", ("fixtures/ground-truth",)),
    "staunton": ("work", ("fixtures/ground-truth",)),
}


@dataclass(frozen=True)
class CitySpec:
    """One city's work root and the pin corpora its scoring pass reads."""

    city: str
    config: Path
    work: Path
    ground_truth: tuple[Path, ...]


def city_specs(configs: Path | None = None) -> list[CitySpec]:
    """Every configured city, paired with the roots its operations use.

    The cities are enumerated from the config tree, so a city added there is
    covered the day it lands. Raises when one has no recorded roots. No glob or
    filesystem walk over the work tree substitutes: one city's volumes sit
    outside the other's root entirely.
    """
    root = configs or DATA_ROOT / "configs"
    out: list[CitySpec] = []
    for config in sorted(root.glob("*/*.toml")):
        city = config.parent.name
        if city not in CITY_ROOTS:
            raise SystemExit(
                f"{config}: no work root recorded for '{city}'; add one to CITY_ROOTS "
                "rather than letting a configured city drop out of a corpus figure"
            )
        work, pins = CITY_ROOTS[city]
        # keyed on the ROOT, not the city: cities sharing a work root have to
        # move together or an override repoints some of them and not the rest
        override = {"work": WORK, "work/city2-probe": CITY2}.get(work)
        out.append(
            CitySpec(
                city=city,
                config=config,
                work=override or DATA_ROOT / work,
                ground_truth=tuple(DATA_ROOT / p for p in pins),
            )
        )
    _refuse_shared_declarations(out)
    return out


def _refuse_shared_declarations(specs: Sequence[CitySpec]) -> None:
    """Fail when two configs declare one volume; it would be counted twice."""
    owners: dict[str, str] = {}
    for spec in specs:
        for volume in declared_volumes(spec):
            if volume in owners:
                raise SystemExit(
                    f"{volume} is declared by both '{owners[volume]}' and '{spec.city}'; "
                    "one volume belongs to one city or a corpus figure counts it twice"
                )
            owners[volume] = spec.city


def declared_volumes(spec: CitySpec) -> list[str]:
    """The volume ids ``spec``'s config declares, in id order."""
    config: dict[str, Any] = tomllib.loads(spec.config.read_text())
    return sorted(config.get("volumes") or {})


def is_scoreable(spec: CitySpec, volume: str) -> bool:
    """Does this volume have a finished tree under ``spec`` and a pinned page?"""
    tree = spec.work / volume
    if not (tree / "results").is_dir() or not (tree / "sheets/manifest.json").is_file():
        return False
    return any(source.pages for source in load_sources(volume, list(spec.ground_truth)))


def scoreable_volumes(spec: CitySpec) -> list[str]:
    """``spec``'s declared volumes that are scoreable end to end.

    Declaration is what attributes a volume to a city: three cities share one
    work root, so a directory scan would count the same tree once per city and
    treble a corpus figure. Filesystem shape alone cannot do it either — a work
    root holds hundreds of experiment arms indistinguishable from live volumes.
    """
    return [v for v in declared_volumes(spec) if is_scoreable(spec, v)]


def unclaimed_volumes(specs: Sequence[CitySpec]) -> list[str]:
    """Scoreable trees under a city's work root that NO config declares.

    Declaration decides membership, so a volume run but never written into a
    config would leave a corpus figure silently. This is the check that says so.
    """
    claimed = {v for spec in specs for v in declared_volumes(spec)}
    # keyed on (root, name), not name: two cities can share a root and read
    # different pin corpora, so a tree ruled out under one is not ruled out
    seen: set[tuple[Path, str]] = set()
    out: list[str] = []
    for spec in specs:
        if not spec.work.is_dir():
            continue
        for child in sorted(spec.work.iterdir()):
            key = (spec.work, child.name)
            if child.name in claimed or key in seen or not child.is_dir():
                continue
            seen.add(key)
            if is_scoreable(spec, child.name):
                out.append(f"{spec.work}/{child.name}")
    return sorted(set(out))


@dataclass
class PageExtent:
    """One page's two implied sheet extents and the ratio between them."""

    volume: str
    page: str
    status: str
    rmse_m: float
    pins: int
    width_px: float
    height_px: float
    pipeline_w_m: float
    pipeline_h_m: float
    human_w_m: float
    human_h_m: float
    #: Each side's implied metres per pixel over its own volume's median, set by
    #: :func:`attribute_sides` once the volume is complete. A sheet is a fixed
    #: object, so the side that departs from its own volume is the suspect one.
    pipeline_z: float = 1.0
    human_z: float = 1.0
    #: Which city's config declares the volume; empty when a caller named the
    #: volume directly rather than sweeping the corpus.
    city: str = ""

    @property
    def ratio_w(self) -> float:
        return self.human_w_m / self.pipeline_w_m

    @property
    def ratio_h(self) -> float:
        return self.human_h_m / self.pipeline_h_m

    @property
    def suspect(self) -> str:
        """Which side departs further from its own volume's typical scale.

        Reads `ambiguous` where neither leads by half, and where a volume
        genuinely mixes printed map scales it can read either way — check the
        sheet before quoting it.
        """
        pipeline, human = abs(math.log(self.pipeline_z)), abs(math.log(self.human_z))
        if human > pipeline * 1.5:
            return "human"
        if pipeline > human * 1.5:
            return "pipeline"
        return "ambiguous"

    @property
    def ratio(self) -> float:
        """The worse of the two axis ratios, by distance from 1 in log space."""
        return max((self.ratio_w, self.ratio_h), key=lambda r: abs(math.log(r)))

    def in_band(self, band: tuple[float, float]) -> bool:
        return band[0] <= self.ratio_w <= band[1] and band[0] <= self.ratio_h <= band[1]

    def as_record(self) -> dict[str, Any]:
        return {
            "volume": self.volume,
            "page": self.page,
            "status": self.status,
            "rmse_vs_human_m": round(self.rmse_m, 2),
            "pins": self.pins,
            "pipeline_w_m": round(self.pipeline_w_m, 1),
            "pipeline_h_m": round(self.pipeline_h_m, 1),
            "human_w_m": round(self.human_w_m, 1),
            "human_h_m": round(self.human_h_m, 1),
            "ratio_w": round(self.ratio_w, 4),
            "ratio_h": round(self.ratio_h, 4),
            "ratio": round(self.ratio, 4),
            "pipeline_scale_z": round(self.pipeline_z, 4),
            "human_scale_z": round(self.human_z, 4),
            "suspect": self.suspect,
        }


def attribute_sides(pages: list[PageExtent]) -> None:
    """Fill each page's per-side scale ratio against its own volume's median."""
    for volume in {p.volume for p in pages}:
        mine = [p for p in pages if p.volume == volume]
        pipeline_scale = [
            (p.pipeline_w_m + p.pipeline_h_m) / (p.width_px + p.height_px) for p in mine
        ]
        human_scale = [(p.human_w_m + p.human_h_m) / (p.width_px + p.height_px) for p in mine]
        typical_pipeline = statistics.median(pipeline_scale)
        typical_human = statistics.median(human_scale)
        for p, pipeline, human in zip(mine, pipeline_scale, human_scale, strict=True):
            p.pipeline_z = pipeline / typical_pipeline
            p.human_z = human / typical_human


def axis_extents_m(model: AffineMatrix, width: float, height: float) -> tuple[float, float]:
    """Ground metres the model puts along the sheet's pixel x and y axes.

    Walks the two edges from the sheet origin and corrects the projected metres
    to true ground metres at the sheet's own latitude — an uncorrected figure
    reads a third high at these latitudes and inverts nothing here, but any
    number quoted beside a metre gate has to be on the gate's scale.
    """
    ox, oy = apply_affine(model, 0.0, 0.0)
    wx, wy = apply_affine(model, width, 0.0)
    hx, hy = apply_affine(model, 0.0, height)
    _lng, lat = TO_4326.transform((ox + wx + hx) / 3.0, (oy + wy + hy) / 3.0)
    correction = math.cos(math.radians(lat))
    return math.hypot(wx - ox, wy - oy) * correction, math.hypot(hx - ox, hy - oy) * correction


def page_extent(
    volume: str,
    page: str,
    record: dict[str, Any],
    info: dict[str, Any] | None,
    layer: dict[str, Any] | None,
    correction_lat: float,
) -> PageExtent | None:
    """Both sides' implied extent for one page, or None where a side is missing."""
    rmse = score_record_vs_ground_truth(record, info, layer, correction_lat)
    if rmse is None or info is None or layer is None:
        return None
    human_gcps = gcps_from_geojson(dict(layer)["gcps_geojson"])
    pipeline = fit_affine_checked(gcps_from_geojson(dict(record)["gcps_geojson"]))
    human = fit_affine_checked(human_gcps)
    # the scorer fits these with the minimum-norm solver and grades them anyway,
    # so a page refused here is a page a published figure counted; say which
    if pipeline is None or human is None:
        logger.warning("%s p%s: degenerate GCP set, scored but not measurable here", volume, page)
        return None
    width, height = float(info["full_size"][0]), float(info["full_size"][1])
    pipeline_w, pipeline_h = axis_extents_m(pipeline, width, height)
    human_w, human_h = axis_extents_m(human, width, height)
    if min(pipeline_w, pipeline_h, human_w, human_h) <= 0.0:
        logger.warning("%s p%s: an implied extent is zero, scored but not measurable", volume, page)
        return None
    return PageExtent(
        volume=volume,
        page=page,
        status=str(record.get("status", "")),
        rmse_m=rmse,
        pins=len(human_gcps),
        width_px=width,
        height_px=height,
        pipeline_w_m=pipeline_w,
        pipeline_h_m=pipeline_h,
        human_w_m=human_w,
        human_h_m=human_h,
    )


def volume_extents(
    volume: str,
    work: Path,
    ground_truth: list[Path],
    unmeasurable: list[str] | None = None,
) -> list[PageExtent]:
    """Every auto-accepted, pinned page of one volume, measured on both sides.

    The population is the score pass's own: an `OK` status that is not a
    reviewer's, joined to a pin through the same case-folding resolver, so a
    page counted here is a page a published figure counted. A page the scorer
    grades and this cannot measure is appended to ``unmeasurable`` rather than
    dropped, because a denominator that moved without saying so is worse than
    the defect being measured.
    """
    paths = VolumePaths(work / volume)
    layers = merge_pages(load_sources(volume, ground_truth))
    if not layers:
        return []
    correction_lat = mercator_correction_lat(layers)
    manifest: dict[str, Any] = json.loads(paths.manifest.read_text())
    resolved = resolve_pages((page for page, _r, _p in iter_results(paths)), layers)
    out: list[PageExtent] = []
    for page, record, _result_path in iter_results(paths):
        status = str(record.get("status", ""))
        if not status.startswith("OK") or is_reviewer_verified(status):
            continue
        key = resolved.get(page)
        layer = layers.get(key) if key else None
        info = manifest.get(f"p{page}")
        measured = page_extent(volume, page, dict(record), info, layer, correction_lat)
        if measured is not None:
            out.append(measured)
        elif (
            unmeasurable is not None
            and score_record_vs_ground_truth(record, info, layer, correction_lat) is not None
        ):
            unmeasurable.append(f"{volume} p{page}")
    return out
