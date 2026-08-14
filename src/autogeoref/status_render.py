"""Render the status index: the terminal table, its legend, and the JSON form.

Reading is `status`; this only shapes what has already been derived, which is
why the two are separate — nothing here may go and look at the filesystem.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from .status import SERVE_STALE

if TYPE_CHECKING:
    from pathlib import Path

    from .status import VolumeStatus


LEGEND = (
    "sheets = full-res images in work/<volume>/regions/ that a run can ADDRESS. Images whose "
    "filename carries no page id are excluded (mostly title/index pages; --json counts them as "
    "`unaddressable`), so this can be lower than `ls regions/ | wc -l`.\n"
    "gt     = pages a volunteer pinned by hand (GCPs in fixtures/ground-truth/) — the only thing "
    "that makes a volume usable for VALIDATION. `gt` and `sheets` are near-disjoint by "
    "construction: the census fetched images for the volumes nobody had pinned, so a volume with "
    "GT and no sheets is not broken — its pixels were never downloaded here, and getting them "
    "means a fresh LOC fetch.\n"
    "         `+N unusable` = pinned layers that CANNOT be scored: OHMG region splits "
    "(`..._p10_1`, `..._p10_2`), where the volunteer georeferenced a sheet as separate crops. "
    "Their GCP pixels are in the CROP's frame and the export carries no offset back to the page, "
    "so they cannot touch the full-res scan. 258 layers / 118 pages corpus-wide, and not one of "
    "those pages is also pinned whole. Do NOT 'recover' them by teaching the page parser to read "
    "`_p10_1` as page 10 — that binds crop pixels to page pixels and fabricates a placement.\n"
    "         GT is filed by LOC ITEM ID and images must come from that same item. Never pair "
    "them by OHMG slug: two LOC items can share one slug (_066 and _132 are both "
    "`chicago_ill_1950_vol_49` — different scans of the same volume), and the mismatch produces "
    "a plausible-looking, entirely fabricated result.\n"
    "funnel = auto-accepted / flagged in work/<volume>/results/ — the ONLY evidence a volume was "
    "processed here.\n"
    "         `(N pinned)` = accepts, already counted in the ok number, whose ROTATION AND SCALE "
    "came from the volume constants instead of their own evidence: the rescue path for a sheet "
    "whose real anchors are collinear pins the linear part and solves only a translation, then "
    "injects synthetic corners so the warper can refit it. Correct by design and served like any "
    "other accept, but nothing has checked their orientation, and a fit residual over their "
    "synthetic corners is the placement model measured against itself — never read one as a "
    "quality score. Corpus-wide this was about a fifth of all committed sheets at the last "
    "census, and rises as records are rewritten under the always-corners rule; the pin was "
    "measured to cost well under the misregistration those sheets already carry, so it is "
    "deliberately left alone. Reviewer placements are excluded here as they are from every other "
    "counter, so this can read lower than the corpus census.\n"
    "frozen = a recorded funnel under fixtures/<volume>/results/: accepted / pages it RECORDED "
    "(not the volume's sheet count), tagged with WHOSE it is. `(baseline)` is archived — "
    "a bar to beat, never this repo's work. `(this repo)` is a golden run we produced "
    "end-to-end (_041/_089/_130) — ours, and not a baseline to credit anyone else with.\n"
    "tiles  = serving provenance, from the serving directory: autogeoref (placed here).\n"
    "serve  = the served autogeoref archive vs the committed records on disk: fresh | STALE "
    "(a committed record is newer than the bake — serve pass needed; the note names it) | "
    "no bake (committed results, no autogeoref archive). Committed = what "
    "`bake.committed_layers` would serve, so provisional churn never flags. mtime-based, in "
    "both directions: a content-neutral record rewrite can read STALE once (the next bake "
    "reconciles it byte-identically), and a record committed while a serve pass is mid-flight "
    "lands older than the publish timestamp and reads fresh — recheck after a mid-bake accept."
)

_COLUMNS = (
    "volume",
    "sheets",
    "gt",
    "reads",
    "results",
    "funnel (this repo)",
    "frozen record",
    "tiles",
    "serve",
    "note",
)


def _cells(row: VolumeStatus) -> tuple[str, ...]:
    def num(v: int | None) -> str:
        return "-" if v is None else str(v)

    # `pinned` rides inside the ok count because it IS part of it: these sheets
    # are accepted and served, they just were not oriented by their own evidence
    funnel = (
        f"{row.accepted} ok"
        + (f" ({row.pinned_orientation} pinned)" if row.pinned_orientation else "")
        + f" / {row.flagged} flagged"
        + (f" (+{row.reviewer_verified} reviewer)" if row.reviewer_verified else "")
        if row.results
        else "-"
    )
    # the denominator is what the frozen run RECORDED, which can be short of the
    # volume's sheet count — and the tag says whose numbers these are, so a
    # golden run of ours is never read as the baseline
    frozen = (
        f"{row.frozen_accepted}/{row.frozen_sheets} placed "
        f"({'this repo' if row.frozen_source == 'autogeoref' else 'baseline'})"
        if row.frozen_sheets is not None
        else "-"
    )
    # the unscoreable tail rides along in the gt cell: it is GT that exists and
    # cannot be used, and hiding it invites someone to "recover" it wrongly
    gt = num(row.gt) + (f" +{row.gt_unscoreable} unusable" if row.gt_unscoreable else "")
    # STALE is the row an operator must act on; it earns the caps
    serve = "STALE" if row.serve_stale == SERVE_STALE else (row.serve_stale or "-")
    return (
        row.volume,
        num(row.sheets),
        gt,
        num(row.reads),
        num(row.results),
        funnel,
        frozen,
        row.tiles or "-",
        serve,
        row.note,
    )


def format_table(rows: list[VolumeStatus], *, roots: dict[str, Path] | None = None) -> str:
    """Fixed-width table + legend.

    The scanned roots are always echoed: the defaults are RELATIVE, so the same
    command run from the wrong directory finds nothing — and a confident
    "nothing is done" is the worst answer this command could give. Say where it
    looked, so an empty result is debuggable instead of believable.
    """
    scanned = ["scanned: " + "  ".join(f"{k}={v}" for k, v in roots.items())] if roots else []
    if not rows:
        return "\n".join(["no volumes found under the scanned roots", *scanned, ""])
    body = [_cells(r) for r in rows]
    widths = [max(len(c[i]) for c in [_COLUMNS, *body]) for i in range(len(_COLUMNS))]

    def line(cells: tuple[str, ...]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)).rstrip()

    return "\n".join([line(_COLUMNS), *(line(c) for c in body), "", LEGEND, *scanned, ""])


def status_json(rows: list[VolumeStatus]) -> str:
    return json.dumps([asdict(r) for r in rows], indent=2)
