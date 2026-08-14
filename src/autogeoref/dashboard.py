"""Coverage & provenance dashboard: who placed what, where, and what is left.

It answers how much of the corpus is placed, WHO placed it, and where the
unplaced remainder is, broken down by era and by neighbourhood. That is the
coverage question, and it is the one the tree can answer honestly today.

It REFUSES to be an accuracy dashboard. Accuracy means residuals against human
GCPs, and a volume can only be scored where human pins and scanned pixels meet
— across the corpus that is a handful of volumes. A per-neighbourhood "median
error" panel built on that would be a number computed from a few volumes
wearing the clothes of a number computed from all of them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from importlib.resources import files as resource_files
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any

from .bounds import Bounds, BoundsError, load_ground_truth, volume_bounds
from .config.model import CityConfig, ConfigError
from .run_inputs import resolve_bounds
from .status import VolumeStatus, count_or_dash
from .viewer.era import era_label
from .viewer.sources import AreaIndex, loc_titles

if TYPE_CHECKING:
    from .viewer.config import ViewerConfig


@dataclass(frozen=True)
class CoverageState:
    """One bucket of the coverage burn-down. Ordered best-to-worst."""

    key: str
    label: str
    #: what a reader must not conclude from this bucket, where that is a live risk
    blurb: str


#: The states, in burn-down order. A volume is in EXACTLY ONE of them, decided by
#: :func:`coverage_state`. Deliberately not merged into "done / not done":
#: everything below the first state is open work of a different kind, and the
#: distinctions are what say which command would move it.
COVERAGE_STATES: tuple[CoverageState, ...] = (
    CoverageState("autogeoref", "Served — this pipeline", "placed here and published"),
    CoverageState(
        "processed-unserved",
        "Processed here, not published",
        "results on disk, no tiles: the warp/tile step is what is missing",
    ),
    CoverageState(
        "ready",
        "Scans ready, never processed",
        "full-res page-addressable scans on disk: this is the runnable queue",
    ),
    CoverageState(
        "no-scans",
        "No scans here",
        "nothing to run on — processing it needs a fresh LOC fetch",
    ),
)

_STATE_BY_KEY = {s.key: s for s in COVERAGE_STATES}

#: neighborhoods per volume. A volume footprint is a few km across and clips the
#: corner of everything nearby; the viewer has long used the three
#: largest-overlap areas as a volume's district names, and the dashboard reuses
#: that so the two never disagree about where a volume "is".
AREAS_PER_VOLUME = 3

#: the bucket a volume with no usable footprint lands in. Shown, never dropped:
#: an unlocatable volume is a gap in the coverage map, and quietly omitting it
#: would make the map look more complete than the tree is.
UNLOCATED = "(footprint unknown)"


@dataclass(frozen=True)
class VolumeCoverage:
    """One volume's coverage row: who placed it, where it is, what is left."""

    volume: str
    title: str | None
    year: int | None
    #: era-chip label from the city's viewer buckets, falling back to the year
    era: str | None
    state: str
    #: the ONE column that means "this repo processed it" — ``work/<v>/results/``.
    #: Independent of ``state`` on purpose: a volume can be placed here and not
    #: yet published.
    processed_here: bool
    sheets: int | None
    gt: int | None
    accepted: int | None
    flagged: int | None
    areas: list[str]
    #: which footprint the areas were derived from — "volunteer pins", "served
    #: layer", "city config", or None when the volume could not be located at
    #: all. Never hidden: an area rollup is only as honest as this column.
    bounds_source: str | None

    @property
    def scoreable(self) -> bool:
        """Human pins AND scanned pixels both present — the only volumes on which
        an auto-vs-human comparison can be made at all."""
        return bool(self.gt and self.sheets)


@dataclass(frozen=True)
class Bucket:
    """A rollup row: one era, or one neighborhood."""

    name: str
    volumes: int
    #: volumes per state key, in :data:`COVERAGE_STATES` order
    counts: dict[str, int]
    #: addressable scans in this bucket, where any are on disk. ``0`` is a real
    #: answer (nothing fetched), not a missing one.
    sheets: int


@dataclass(frozen=True)
class Coverage:
    rows: list[VolumeCoverage] = field(default_factory=list)
    by_era: list[Bucket] = field(default_factory=list)
    by_area: list[Bucket] = field(default_factory=list)

    @property
    def totals(self) -> Bucket:
        return _bucket("all volumes", self.rows)


def coverage_state(row: VolumeStatus) -> str:
    """Which bucket a volume is in. Serving provenance wins over processing:
    results on disk are not a published layer."""
    if row.ours:
        return "autogeoref"
    if row.results:
        return "processed-unserved"
    if row.sheets:
        return "ready"
    return "no-scans"


def _footprint(
    volume: str,
    city: CityConfig,
    *,
    ground_truth_dir: Path,
    viewer_manifest: Path | None,
    manifest_bounds: dict[str, Bounds],
) -> tuple[Bounds | None, str | None]:
    """``(bounds, source)`` for one volume — best available, source recorded.

    Priority is by TRUSTWORTHINESS of the footprint, not convenience: volunteer pins first
    (humans placed those sheets), then the served layer's published footprint, then the city
    config's curated bbox or area union. This repo's own accepted GCPs are deliberately NOT a
    source: every processed volume is already located by one of the three, so using them would
    buy nothing and would let a bad run relocate a volume on the coverage map.
    """
    gt_path = ground_truth_dir / f"api-layers-{volume}.json"
    if gt_path.exists():
        try:
            gt = load_ground_truth(gt_path)
            if gt:
                return volume_bounds(gt), "volunteer pins"
        except (OSError, ValueError, BoundsError):
            pass  # a damaged export is status's problem to report, not ours
    if volume in manifest_bounds:
        return manifest_bounds[volume], "served layer"
    try:
        return resolve_bounds(city, city.volume(volume), viewer_manifest), "city config"
    except (ConfigError, BoundsError, OSError, ValueError):
        return None, None


def _bucket(name: str, rows: list[VolumeCoverage]) -> Bucket:
    return Bucket(
        name=name,
        volumes=len(rows),
        counts={s.key: sum(1 for r in rows if r.state == s.key) for s in COVERAGE_STATES},
        sheets=sum(r.sheets or 0 for r in rows),
    )


def _rollup(rows: list[VolumeCoverage], keys: Any) -> list[Bucket]:
    grouped: dict[str, list[VolumeCoverage]] = {}
    for row in rows:
        for key in keys(row):
            grouped.setdefault(key, []).append(row)
    # the unlocated bucket sorts last: it is a gap in the map, not a place
    ordered = sorted(grouped.items(), key=lambda kv: (kv[0] == UNLOCATED, kv[0]))
    return [_bucket(name, members) for name, members in ordered]


def build_coverage(
    status_rows: list[VolumeStatus],
    city: CityConfig,
    viewer: ViewerConfig,
    *,
    ground_truth_dir: Path,
    loc_catalog: Path | None = None,
    viewer_manifest: Path | None = None,
) -> Coverage:
    """Join the filesystem state index to era and neighborhood.

    ``status_rows`` comes from :func:`status.build_status` — the dashboard never
    re-derives what is processed, so it cannot disagree with `autogeoref status`.
    """
    meta = loc_titles(loc_catalog, city.name) if loc_catalog else {}
    areas = (
        AreaIndex(city.community_areas_path)
        if city.community_areas_path is not None and city.community_areas_path.exists()
        else None
    )
    manifest_bounds: dict[str, Bounds] = {}
    if viewer_manifest is not None and viewer_manifest.exists():
        for entry in json.loads(viewer_manifest.read_text()).get("volumes") or []:
            if entry.get("bounds"):
                manifest_bounds[entry["id"]] = tuple(entry["bounds"])

    rows: list[VolumeCoverage] = []
    for status_row in status_rows:
        m = meta.get(status_row.volume) or {}
        bounds, source = _footprint(
            status_row.volume,
            city,
            ground_truth_dir=ground_truth_dir,
            viewer_manifest=viewer_manifest,
            manifest_bounds=manifest_bounds,
        )
        rows.append(
            VolumeCoverage(
                volume=status_row.volume,
                title=m.get("title"),
                year=m.get("year"),
                era=era_label(m.get("year"), viewer.era_buckets) if m.get("year") else None,
                state=coverage_state(status_row),
                processed_here=status_row.processed_here,
                sheets=status_row.sheets,
                gt=status_row.gt,
                accepted=status_row.accepted,
                flagged=status_row.flagged,
                areas=(
                    areas.names(bounds, top=AREAS_PER_VOLUME)
                    if areas is not None and bounds is not None
                    else []
                ),
                bounds_source=source,
            )
        )
    return Coverage(
        rows=rows,
        by_era=_rollup(rows, lambda r: [r.era or "(year unknown)"]),
        by_area=_rollup(rows, lambda r: r.areas or [UNLOCATED]),
    )


def coverage_json(coverage: Coverage) -> str:
    return json.dumps(
        {
            "totals": asdict(coverage.totals),
            "states": [asdict(s) for s in COVERAGE_STATES],
            "by_era": [asdict(b) for b in coverage.by_era],
            "by_area": [asdict(b) for b in coverage.by_area],
            "volumes": [asdict(r) for r in coverage.rows],
        },
        indent=2,
    )


def state_label(key: str) -> str:
    return _STATE_BY_KEY[key].label


# ---------------------------------------------------------------- rendering
#
# The page's markup, prose and palette are FILES (``dashboard_ui/``); the code
# below supplies data and nothing else. So the caveats can be edited without
# touching Python, and the state colours have exactly one home — the
# stylesheet's ``--s-<key>`` variables, which this module only references.


def _ui_dir() -> Path:
    return Path(str(resource_files("autogeoref") / "dashboard_ui"))


#: an em-dash for an absent number. The literal character, never the HTML entity:
#: these cells pass through :func:`_esc`, which would print "&mdash;" as text.
DASH = "—"


def _esc(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _swatch(state: str) -> str:
    return f'<i class="dot" style="background:var(--s-{state})"></i>{_esc(state_label(state))}'


def _bar(bucket: Bucket, total: int) -> str:
    """One stacked part-to-whole bar. Every segment carries its count in a
    tooltip, and shows it in place wherever it is wide enough to hold it: the
    palette's light-mode contrast obliges a channel other than colour."""
    segs = []
    for state in COVERAGE_STATES:
        n = bucket.counts[state.key]
        if not n:
            continue
        pct = 100 * n / total if total else 0
        label = str(n) if pct >= 7 else ""
        segs.append(
            f'<div data-state="{state.key}" '
            f'style="width:{pct:.4f}%;background:var(--s-{state.key})" '
            f'title="{_esc(state.label)}: {n} volume(s)">{label}</div>'
        )
    return f'<div class="bar">{"".join(segs)}</div>'


def _rollup_rows(buckets: list[Bucket]) -> str:
    cells = []
    for b in buckets:
        cells.append(f'<div class="lab">{_esc(b.name)}</div>')
        cells.append(_bar(b, b.volumes))
        cells.append(f'<div class="cnt">{b.volumes} vol</div>')
    return f'<div class="rows">{"".join(cells)}</div>'


def _kpi(value: object, key: str, hint: str = "") -> str:
    hint_html = f'<div class="h">{_esc(hint)}</div>' if hint else ""
    return (
        f'<div class="kpi"><div class="v">{_esc(value)}</div>'
        f'<div class="k">{_esc(key)}</div>{hint_html}</div>'
    )


def render_html(coverage: Coverage, *, generated: str | None = None) -> str:
    """Fill ``dashboard_ui/dashboard.html``, inlining the stylesheet so the page
    is a single self-contained file — no CDN, no fetch, openable from disk."""
    totals = coverage.totals
    c = totals.counts
    processed = [r for r in coverage.rows if r.processed_here]
    scoreable = [r for r in coverage.rows if r.scoreable]
    unlocated = sum(1 for r in coverage.rows if r.bounds_source is None)

    processed_rows = "".join(
        f"<tr><td><code>{_esc(r.volume)}</code></td><td>{_esc(r.era or '?')}</td>"
        f'<td class="n">{count_or_dash(r.accepted)}</td>'
        f'<td class="n">{count_or_dash(r.flagged)}</td>'
        f"<td>{_swatch(r.state)}</td></tr>"
        for r in sorted(processed, key=lambda r: r.volume)
    )
    volume_rows = "".join(
        f"<tr><td><code>{_esc(r.volume)}</code></td>"
        f'<td class="n">{count_or_dash(r.year)}</td>'
        f"<td>{_swatch(r.state)}</td>"
        f"<td>{'<span class=flag>yes</span>' if r.processed_here else DASH}</td>"
        f'<td class="n">{count_or_dash(r.sheets)}</td>'
        f'<td class="n">{count_or_dash(r.gt)}</td>'
        f'<td class="wrap">{_esc(", ".join(r.areas)) or DASH}</td>'
        f'<td class="wrap">{_esc(r.bounds_source or "unlocated")}</td></tr>'
        for r in sorted(coverage.rows, key=lambda r: (r.year or 0, r.volume))
    )
    names = ", ".join(r.volume for r in scoreable)
    scoreable_phrase = (
        f"{len(scoreable)} volume ({_esc(names)})"
        if len(scoreable) == 1
        else f"{len(scoreable)} volumes ({_esc(names)})"
        if scoreable
        else "no volume at all"
    )
    ui = _ui_dir()
    return Template((ui / "dashboard.html").read_text(encoding="utf-8")).substitute(
        css=(ui / "dashboard.css").read_text(encoding="utf-8"),
        # stamped, because a coverage page with no date is precisely the failure
        # this project keeps hitting: a stale state claim that reads as current
        stamp=(
            f'<p class="note">Generated {_esc(generated)} from the filesystem.</p>'
            if generated
            else ""
        ),
        kpis="".join(
            (
                _kpi(totals.volumes, "volumes in the corpus"),
                _kpi(
                    len(processed),
                    "processed by this pipeline",
                    f"{c['autogeoref']} of them served as ours",
                ),
                _kpi(
                    c["processed-unserved"],
                    "placed but not published",
                    "results on disk; the bake is what is missing",
                ),
                _kpi(c["ready"], "scans ready to run", f"{totals.sheets:,} sheets on disk"),
            )
        ),
        legend="".join(
            f'<span><i class="sw" style="background:var(--s-{s.key})"></i>{_esc(s.label)}</span>'
            for s in COVERAGE_STATES
        ),
        total_bar=_bar(totals, totals.volumes),
        era_rows=_rollup_rows(coverage.by_era),
        area_rows=_rollup_rows(coverage.by_area),
        processed_rows=processed_rows,
        volume_rows=volume_rows,
        n_volumes=totals.volumes,
        n_processed=len(processed),
        n_open=c["ready"] + c["no-scans"],
        n_unlocated=unlocated,
        pct_unlocated=round(100 * unlocated / totals.volumes) if totals.volumes else 0,
        areas_per_volume=AREAS_PER_VOLUME,
        unlocated_label=_esc(UNLOCATED),
        scoreable_phrase=scoreable_phrase,
    )
