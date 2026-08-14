"""Load one placed volume and re-run the real matcher over it.

Everything a plate states comes from here: the pipeline's own functions called
on the pipeline's own output, never a second implementation. Reproduction is
checked rather than assumed — :func:`match_page` refuses a page whose recomputed
candidate and inlier counts disagree with the record on disk, because a figure
drawn from a drifted index would be confidently wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from autogeoref.config.load import load_city_config
from autogeoref.matching import (
    SPREAD_SPAN_FRAC,
    Candidate,
    candidate_gcps,
    ransac_affine,
    ransac_affine_diagnostics,
)
from autogeoref.paths import VolumePaths
from autogeoref.run_inputs import build_index, resolve_bounds
from autogeoref.sheet_inputs import load_sheet_inputs
from autogeoref.volume import (
    REVOKED_PREFIX,
    STATUS_CORROBORATED,
    STATUS_OK,
    STATUS_RESCUED,
    STATUS_VERIFIED_PREFIX,
    SheetInput,
    VolumeConstraints,
    constraints_for_page,
    constraints_from_constants,
)
from autogeoref.volume_constants import resolve_constants

if TYPE_CHECKING:
    from autogeoref.affine import AffineMatrix
    from autogeoref.centerlines import CenterlineIndex
    from autogeoref.config.model import CityConfig, VolumeConfig


class FidelityError(RuntimeError):
    """A figure would not describe the run that produced the record."""


@dataclass(frozen=True)
class Fit:
    """One page's matcher replay: what it proposed and what survived."""

    page: str
    sheet: SheetInput
    candidates: list[Candidate]
    model: AffineMatrix | None
    inliers: list[Candidate]
    constraints: VolumeConstraints
    record: dict[str, Any]

    @property
    def dropped(self) -> list[Candidate]:
        return [c for c in self.candidates if c not in self.inliers]

    def diagnostics(self) -> dict[str, Any]:
        return ransac_affine_diagnostics(
            self.candidates,
            self.sheet.full_size,
            scale_range=self.constraints.scale_range,
            rot_range_deg=self.constraints.rot_range_deg,
            rot_quadrant_fold=True,
            strict_result=(self.model, self.inliers),
        )


@dataclass(frozen=True)
class Escalation:
    """One page's two reads, as counts the escalation plate may state."""

    #: corner candidates the first read produced, and the re-read after it
    offered: int
    reoffered: int
    #: corners the re-read's fit kept
    kept: int
    #: the street every first-read candidate shares, if there is exactly one
    one_street: str
    #: whether those candidates span less of the sheet than the matcher allows
    narrow: bool


@dataclass
class Volume:
    """A placed volume on disk, with the inputs its run resolved."""

    city: CityConfig
    vol: VolumeConfig
    paths: VolumePaths
    bounds: tuple[float, float, float, float]
    index: CenterlineIndex
    features: list[dict[str, Any]]
    constraints: VolumeConstraints
    manifest: dict[str, Any]
    sheets: dict[str, SheetInput]
    results: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def identifier(self) -> str:
        return self.vol.identifier

    def small(self, page: str) -> Image.Image:
        entry = self.manifest[f"p{page}"]
        return Image.open(self.paths.sheets / entry["file"]).convert("RGB")

    def scale(self, page: str) -> float:
        """The small/full ratio the matcher used for this page."""
        return float(self.sheets[page].scale)

    def annotation(self, page: str) -> dict[str, Any]:
        return self.sheets[page].annotation

    def escalated_annotation(self, page: str) -> dict[str, Any]:
        """The re-read that flipped this page, taken from its own tier cache."""
        from autogeoref.annotate.providers import model_cache_key

        record = self.results[page]
        model = str(record.get("escalated_model", ""))
        key = model_cache_key(model, record.get("escalated_variant"))
        path = self.paths.annotations / f"p{page}.escalated.{key}.json"
        if not path.is_file():
            raise FidelityError(f"p{page}: no cached re-read at {path.name}")
        data: dict[str, Any] = json.loads(path.read_text())
        return data

    def unresolved(self, streets: list[dict[str, Any]]) -> set[str]:
        """Names that reach nothing on the modern map, renames applied."""
        from autogeoref.names import normalize

        return {
            s["name"]
            for s in streets
            if normalize(s["name"], self.index.aliases) not in self.index.by_name
        }

    def match(self, page: str, *, check: bool = True) -> Fit:
        """Replay the production match for one page."""
        return self._fit(page, self.sheets[page].annotation, check=check)

    def escalation(self, page: str) -> Escalation:
        """What the two reads of an escalated page offered the matcher.

        Both go back through the production matcher. It raises unless the first
        read is refused and the re-read reproduces the record, because a plate
        states these numbers as fact.
        """
        first = self._fit(page, self.sheets[page].annotation, check=False)
        if first.model is not None:
            raise FidelityError(f"p{page}: the first read places, so it illustrates nothing")
        second = self._fit(page, self.escalated_annotation(page), check=True)
        named = [set(c.streets) for c in first.candidates]
        shared = set.intersection(*named) if named else set()
        xs = [c.pixel[0] for c in first.candidates]
        width = first.sheet.full_size[0]
        return Escalation(
            offered=len(first.candidates),
            reoffered=len(second.candidates),
            kept=len(second.inliers),
            one_street=sorted(shared)[0] if len(shared) == 1 else "",
            # the matcher's own span gate, not a second copy of it
            narrow=bool(xs) and (max(xs) - min(xs)) < SPREAD_SPAN_FRAC * width,
        )

    def _fit(self, page: str, annotation: dict[str, Any], *, check: bool) -> Fit:
        sheet = self.sheets[page]
        constraints = constraints_for_page(page, self.constraints, self.vol.page_scale_multiples)
        cands = candidate_gcps(annotation, self.index, sheet.scale, self.index.aliases)
        model, inliers = ransac_affine(
            cands,
            sheet.full_size,
            scale_range=constraints.scale_range,
            rot_range_deg=constraints.rot_range_deg,
            rot_quadrant_fold=True,
        )
        record = self.results[page]
        if check:
            _assert_reproduced(page, record, len(cands), len(inliers))
        return Fit(page, sheet, cands, model, inliers, constraints, record)


def _assert_reproduced(page: str, record: dict[str, Any], cands: int, inliers: int) -> None:
    """The record on disk and this replay must agree, or nothing may be drawn."""
    if record.get("n_candidates") != cands:
        raise FidelityError(
            f"p{page}: replay found {cands} candidates, the record says "
            f"{record.get('n_candidates')} - the index has moved under it"
        )
    if record.get("status") == STATUS_OK and record.get("n_inliers") != inliers:
        raise FidelityError(
            f"p{page}: replay kept {inliers} inliers, the record says {record.get('n_inliers')}"
        )


def load(work: Path, city_toml: Path, identifier: str) -> Volume:
    """Open a placed volume and rebuild the index its run used."""
    city = load_city_config(city_toml)
    vol = city.volume(identifier)
    paths = VolumePaths(root=work / identifier)
    bounds = resolve_bounds(city, vol, None)
    features: list[dict[str, Any]] = json.loads(city.centerlines_path.read_text())["features"]
    index = build_index(city, vol, bounds, features=features)
    pins = resolve_constants(paths, vol)
    if pins is None:
        raise FidelityError(f"{identifier}: no pinned scale/rotation on disk or in config")
    manifest = json.loads(paths.manifest.read_text())
    sheets = {s.page: s for s in load_sheet_inputs(paths)}
    results = {
        path.stem.removeprefix("p"): json.loads(path.read_text())
        for path in sorted(paths.results.glob("p*.json"))
    }
    return Volume(
        city=city,
        vol=vol,
        paths=paths,
        bounds=bounds,
        index=index,
        features=features,
        constraints=constraints_from_constants(*pins),
        manifest={k: v for k, v in manifest.items() if isinstance(v, dict)},
        sheets=sheets,
        results=results,
    )


#: Stage order for the running counter, and the status each stage produces.
STAGES = ("match", "rescue", "corroborate", "verified-accept")


@dataclass(frozen=True)
class Funnel:
    """How a volume's sheets stand after each acceptance stage."""

    total: int
    #: stage -> (placed, provisional, refused)
    by_stage: dict[str, tuple[int, int, int]]
    counts: dict[str, int]


def funnel(volume: Volume) -> Funnel:
    """The running counter, derived from the final records alone."""
    counts = {"strict": 0, "rescued": 0, "corroborated": 0, "verified": 0, "revoked": 0, "none": 0}
    for record in volume.results.values():
        status = str(record.get("status", ""))
        if status == STATUS_OK:
            counts["strict"] += 1
        elif status == STATUS_RESCUED:
            counts["rescued"] += 1
        elif status == STATUS_CORROBORATED:
            counts["corroborated"] += 1
        elif status.startswith(STATUS_VERIFIED_PREFIX):
            counts["verified"] += 1
        elif status.startswith(REVOKED_PREFIX):
            counts["revoked"] += 1
        else:
            counts["none"] += 1
    total = sum(counts.values())
    # every corroborated, verified and still-revoked page was recorded
    # provisional by rescue; the reinstatements come later
    provisional = counts["corroborated"] + counts["verified"] + counts["revoked"]
    refused = counts["none"]
    stages = {
        "match": (counts["strict"], 0, total - counts["strict"]),
        "rescue": (counts["strict"] + counts["rescued"], provisional, refused),
        "corroborate": (
            counts["strict"] + counts["rescued"] + counts["corroborated"],
            provisional - counts["corroborated"],
            refused,
        ),
        "verified-accept": (
            total - counts["revoked"] - refused,
            0,
            refused + counts["revoked"],
        ),
    }
    return Funnel(total=total, by_stage=stages, counts=counts)


def exemplars(path: Path) -> dict[str, Any]:
    """The pinned page choices, kept beside the generator."""
    data: dict[str, Any] = json.loads(path.read_text())
    return data
