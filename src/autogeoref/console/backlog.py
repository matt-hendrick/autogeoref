"""The backlog: which volumes could be started, on which track, today.

Everything here derives from :func:`status.build_status` rows and the queue
file — never from a re-derivation of its own (the package docstring owns that
rule). The output is :class:`Candidate` records; :mod:`.text` and
:mod:`.payload` render them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..bounds_bootstrap import BOUNDS_FILE
from ..budget import DEFAULT_GATED_FRACTION, SpendEstimate, estimate_spend
from ..config.model import CityConfig, VolumeConfig, era_undeclared
from ..loc import catalog_year
from ..queue.store import load_queue

if TYPE_CHECKING:
    from ..status import VolumeStatus

#: Queue statuses that make a volume un-addable, and therefore un-advertisable.
#: :func:`queue.store.add`'s own rule read back: a NON-TERMINAL entry blocks the
#: volume on BOTH tracks, and a terminal one blocks nothing. So `needs-review`
#: must NOT be here — it is terminal, and treating it as live would leave no
#: route from "placed" to "served" for exactly the volumes whose operator asked
#: to look. `failed` is absent for the same reason.
_LIVE = {"queued", "running"}


@dataclass(frozen=True)
class Candidate:
    """One volume that could be started on one track today."""

    volume: str
    track: str  # place | serve
    sheets: int
    #: LOC catalog edition year, when a catalog was supplied. Context for the
    #: operator's era declaration — NEVER an input to it. The engine does not
    #: infer an address era from an edition year (that bakes one city's calendar
    #: into every city, `era.era_from_config`), and neither does this.
    year: int | None
    #: pages the model has already read; they replay free
    cached_reads: int
    calls: SpendEstimate | None  # place track only — serve spends no model budget
    #: why a run would REFUSE to start, when it would. A blocked candidate is
    #: still listed (it is real work), but it is never in the paste-ready command:
    #: handing an operator an `--add` line for a volume whose run dies on the
    #: first line is handing them a failed run.
    blocked: str | None
    #: what the operator should know before starting it
    notes: list[str] = field(default_factory=list)

    @property
    def runnable(self) -> bool:
        return self.blocked is None


def _blocked_reason(city: CityConfig | None, row: VolumeStatus) -> str | None:
    """Why a run of this volume would REFUSE TO START, asked before you start it.

    A backlog that lists a volume the runner rejects on its first line is a
    backlog of failed runs, so every pre-flight refusal the tree can SEE is
    mirrored here. The rules themselves are never restated: ``prep.prep_volume``
    owns the unnameable-sheet halt and :func:`config.era_undeclared` the
    address-era one. A missing bounds source is a NOTE rather than a block,
    since the run derives bounds from the volume's own sampled sheets.
    """
    if row.lost_sheets:
        # prep HALTS on a page it cannot name — it stopped being a warning-and-skip
        # when a nameless sheet turned out to mean "every stage silently drops it"
        return "prep HALTS: recorded page(s) this pipeline cannot name — " + ", ".join(
            row.lost_sheets
        )
    if city is None:
        return None
    vol = city.volume(row.volume)
    if "addresses" in vol.evidence_channels and era_undeclared(city, vol):
        return "no addresses_modern declared, and this city renumbered its houses"
    return None


def _bounds_undeclared(vol: VolumeConfig, row: VolumeStatus) -> bool:
    """No declared bounds source: mirrors ``run_inputs.resolve_bounds``'s inputs.

    The ground-truth arm reads ``row.gt`` — the same frozen export
    ``resolve_bounds`` loads, already counted by ``status`` — so an empty ``[]``
    export reads as what it is: no bounds. True means the first run bootstraps
    (a note on the candidate), never that it refuses.
    """
    return (
        not row.gt
        and vol.bounds_bbox is None
        and not vol.bounds_from_counterpart
        and not vol.bounds_areas
    )


def _place_notes(vol: VolumeConfig | None, row: VolumeStatus, work: Path) -> list[str]:
    """What the operator should know before starting a place run of this volume."""
    notes: list[str] = []
    if row.results:
        notes.append(
            f"PARTIAL: {row.results} of {row.sheets} sheets have results — a killed "
            "run, not a placement. Re-run place; do not serve it"
        )
    if vol is not None and _bounds_undeclared(vol, row):
        # a persisted derivation replays free, so only a volume with
        # neither declaration nor derivation gets the sampling note
        if (work / row.volume / BOUNDS_FILE).exists():
            notes.append(
                "bounds derived on a previous run (volume-bounds.json); replayed "
                "free — declare bounds_bbox to override"
            )
        else:
            notes.append(
                "no declared bounds — the first run derives them from its own sampled "
                "sheets (bounds_bootstrap; the reads land in the annotation cache and "
                "replay free). Declare bounds_bbox from the volume's key map to skip"
            )
    return notes


def _place_candidate(
    row: VolumeStatus,
    *,
    work: Path,
    city: CityConfig | None,
    catalog: dict[str, dict[str, Any]],
) -> Candidate:
    assert row.sheets  # the driver's guard: a sheetless row is on neither track
    vol = city.volume(row.volume) if city else None
    return Candidate(
        volume=row.volume,
        track="place",
        sheets=row.sheets,
        year=catalog_year(catalog, row.volume),
        cached_reads=row.reads or 0,
        calls=estimate_spend(
            sheets=row.sheets,
            cached=row.reads or 0,
            escalation_tiers=len(vol.escalation_ladder()) if vol else 0,
            gated_fraction=(city.gated_fraction if city else DEFAULT_GATED_FRACTION),
        ),
        blocked=_blocked_reason(city, row),
        notes=_place_notes(vol, row, work),
    )


def _serve_candidate(row: VolumeStatus, catalog: dict[str, dict[str, Any]]) -> Candidate:
    assert row.sheets  # the driver's guard: a sheetless row is on neither track
    return Candidate(
        volume=row.volume,
        track="serve",
        sheets=row.sheets,
        year=catalog_year(catalog, row.volume),
        cached_reads=row.reads or 0,
        calls=None,
        blocked=None,
        notes=[f"{row.accepted} accepted, {row.flagged} flagged"],
    )


def candidates(
    rows: list[VolumeStatus],
    *,
    work: Path,
    city: CityConfig | None = None,
    catalog: dict[str, dict[str, Any]] | None = None,
) -> list[Candidate]:
    """The backlog on both tracks, derived from ``status`` and the queue.

    **place** = pixels on disk and no COMPLETE set of results from them, so the
    unprocessed volumes AND the half-placed ones. A killed run leaves a PARTIAL
    results directory, so the test is ``results >= sheets``, never that
    directory's mere existence. **serve** = completely placed here and not
    published by us. A volume with no sheets, or with a live queue entry on
    either track, is on neither (:data:`_LIVE`).
    """
    catalog = catalog or {}
    live = {e.volume for e in load_queue(work) if e.status in _LIVE}
    out: list[Candidate] = []
    for row in rows:
        if not row.sheets or row.volume in live:
            continue
        placed = (row.results or 0) >= row.sheets
        if not placed:
            out.append(_place_candidate(row, work=work, city=city, catalog=catalog))
        elif not row.ours:
            out.append(_serve_candidate(row, catalog))
    # what you can START comes first, in both views: the blocked pool is
    # usually the larger one, and leading with it buries the volumes an
    # operator could actually begin.
    return sorted(out, key=lambda c: (not c.runnable, c.volume))
