"""The queue as a text table — what `autogeoref queue` and `--watch` print."""

from __future__ import annotations

import time
from pathlib import Path

from ..status import count_or_dash
from .progress import iter_display
from .store import TRACKS


def _ago(when: float | None) -> str:
    if not when:
        return "—"
    secs = int(time.time() - when)
    if secs < 90:
        return f"{secs}s"
    if secs < 5400:
        return f"{secs // 60}m"
    return f"{secs // 3600}h{(secs % 3600) // 60:02d}"


_HEAD = (
    f"{'volume':<24} {'pages':>5} {'reads':>5} {'acc':>4} {'flag':>4}  "
    f"{'stage':<18} {'status':<12} {'age':>5}"
)


def render_text(work: Path) -> str:
    """The whole queue as a table — what `autogeoref queue` and `--watch` print.

    Every number except ``status`` is read off the work tree at render time, so
    this stays true even when no runner is alive to update the queue file.
    """
    rows = list(iter_display(work))
    out: list[str] = []
    # One heading per track, in TRACKS order. The prose is per-track and lives
    # here rather than being derived; a track with no entry gets its bare name, so
    # a new queue shows up on the board even before someone writes its line.
    titles = {
        "fetch": "FETCH — download jp2 masters from LOC (1 lane at >=5 s; promotes to place)",
        "place": "PLACE — annotate/match/verify (model budget, 1 at a time; promotes to serve)",
        "serve": "SERVE — warp/mask/mosaic/tile (no model budget, N lanes; runs alongside place)",
    }
    for track in TRACKS:
        title = titles.get(track, track.upper())
        out.append(title)
        track_rows = [(e, p) for e, p in rows if e.track == track]
        if not track_rows:
            out.append("  (empty)")
            out.append("")
            continue
        out.append("  " + _HEAD)
        for entry, prog in track_rows:
            stage = prog.stage or "—"
            if prog.stage_status == "failed":
                stage += " ✗"
            age = _ago(entry.started if entry.status == "running" else entry.finished)
            out.append(
                f"  {entry.volume:<24} {count_or_dash(prog.pages or None):>5} {prog.reads:>5} "
                f"{count_or_dash(prog.accepted):>4} {count_or_dash(prog.flagged):>4}  {stage:<18} "
                f"{entry.status:<12} {age:>5}"
            )
        out.append("")

    needs = [(e, p) for e, p in rows if e.status == "needs-review"]
    failed = [(e, p) for e, p in rows if e.status == "failed"]
    if needs:
        # Only an entry that ASKED for it (--review / then_serve=False) is ever here:
        # review is a diagnostic, not a gate (module docstring).
        out.append("NEEDS YOU — placed, and you asked to look before it bakes")
        for entry, prog in needs:
            out.append(
                f"  {entry.volume:<24} {count_or_dash(prog.accepted)} accepted, "
                f"{count_or_dash(prog.flagged)} flagged"
                f"   autogeoref review --volume {entry.volume}"
            )
            out.append(f"  {'':<24} then: autogeoref queue --track serve --add {entry.volume}")
        out.append("")
    if failed:
        out.append("FAILED")
        for entry, prog in failed:
            why = prog.error or entry.note or "see the log"
            out.append(f"  {entry.volume:<24} {why}")
            if entry.log:
                out.append(f"  {'':<24} {entry.log}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"
