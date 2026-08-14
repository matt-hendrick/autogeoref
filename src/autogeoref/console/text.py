"""The text view: the backlog as paste-ready terminal output."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.model import CityConfig
    from .backlog import Candidate

#: Where the two tools the console routes to actually listen. The console does
#: not absorb them: NEEDS YOU is a review job and SERVED is a map.
REVIEW_URL = "http://127.0.0.1:8765/"
#: `scripts/serve_viewer.py 8123` is what README tells you to run (PMTiles needs
#: a Range-capable server, so it is not `python -m http.server`). It serves the
#: repo root, so the page itself lives under `/viewer/`.
VIEWER_URL = "http://127.0.0.1:8123/viewer/"


def add_command(cands: list[Candidate], track: str) -> str | None:
    """The paste-ready ``--add`` line for one track — or ``None`` if nothing is ready.

    The line is complete: running it serves the volume. Review survives as
    ``--review``, which STOPS a volume at ``needs-review`` for a human to look
    at, and nothing here composes it — it is a thing you ask for, and the
    console has no way to know that you want to.
    """
    ready = [c.volume for c in cands if c.track == track and c.runnable]
    if not ready:
        return None
    add = " ".join(f"--add {v}" for v in ready)
    return f"autogeoref queue --track {track} {add}"


def _renumbering_guidance(city: CityConfig | None) -> str:
    """The era stanza's city-specific parenthetical, from CONFIG, never code.

    ``renumbering_note`` is the city's own wording (which book, which
    districts); a city that sets only ``renumbering_year`` gets a generic line
    built from it; a city config the caller did not supply gets nothing. The
    stanza itself only renders for era-BLOCKED volumes, which exist only where
    a renumbering table is configured (``config.era_undeclared``) — so a city
    that never renumbered shows none of this.
    """
    if city is None:
        return ""
    note = city.renumbering_note
    if note is None and city.renumbering_year is not None:
        note = f"this city renumbered in {city.renumbering_year}"
    return f" ({note})" if note else ""


def _render_place(place: list[Candidate]) -> list[str]:
    """The place table with its spend estimate — blocked rows marked, never hidden."""
    out = ["RUNNABLE — place (spends model budget: one volume at a time)"]
    if not place:
        out.append("  (nothing: no volume has sheets on disk and no results)")
        return out
    out.append(f"  {'volume':<24} {'sheets':>6} {'cached':>6} {'est. calls':>11}  {'year':>4}")
    for c in place:  # already runnable-first (see `candidates`)
        mark = "  " if c.runnable else "! "  # "!" = a run REFUSES it; see the era stanza
        out.append(
            f"{mark}{c.volume:<24} {c.sheets:>6} {c.cached_reads:>6} "
            f"{c.calls!s:>11}  {c.year or '—':>4}"
        )
        out.extend(f"  {'':<24} · {note}" for note in c.notes)
    ready = [c for c in place if c.runnable and c.calls]
    out.append("")
    out.append(
        f"  ESTIMATE, not a budget: ~{sum(c.calls.low for c in ready if c.calls)}-"
        f"{sum(c.calls.ceiling for c in ready if c.calls)} model calls to place all "
        f"{len(ready)} runnable volume(s), net of cached reads. State the estimate and "
        "get sign-off before spending."
    )
    return out


def _render_era_stanza(place: list[Candidate], city: CityConfig | None) -> list[str]:
    """The per-volume TOML stanzas for the era-blocked pool, or nothing."""
    blocked = [c for c in place if not c.runnable and c.blocked and "addresses_modern" in c.blocked]
    if not blocked:
        return []
    # NOT in the paste-ready command above, and NOT given a suggested value
    # here. The engine will not infer an address era from an edition year and
    # neither will this. Both lines ship commented out, so pasting this stanza
    # blind declares nothing and the run keeps refusing, which is the safe
    # failure. Pre-filling `= true` would be the measured hazard in a helpful
    # voice, vetoing correct sheets on every pre-renumbering volume.
    out = [""]
    out.append(
        f"  ! {len(blocked)} volume(s) will REFUSE to run until you declare the address "
        "era. The addresses channel is the only one allowed to REFUTE, and an undeclared "
        "era means MODERN — on a volume printed before the city renumbered, that reads "
        "its numerals against today's grid and vetoes CORRECT sheets. Pick ONE line per "
        f"volume in the city TOML{_renumbering_guidance(city)}:"
    )
    for c in sorted(blocked, key=lambda c: c.volume):
        year = f"  # LOC catalog year: {c.year}" if c.year else "  # year unknown"
        # ALWAYS quoted: a volume id can carry a dot, and a bare dotted key is
        # a NESTED table in TOML — it would declare an era for a volume that
        # does not exist while the operator believes they fixed the one that
        # does. Quoting is valid for the plain ids too.
        out.append(f'      [volumes."{c.volume}"]{year}')
        out.append("      # addresses_modern = true    # printed numbers ARE today's numbers")
        out.append(
            "      # addresses_modern = false   # predates the renumbering; convert via the table"
        )
    return out


def _render_serve(serve: list[Candidate], cmd: str | None) -> list[str]:
    """The serve table with its paste-ready line and the no-budget reassurance."""
    out = ["RUNNABLE — serve (spends no model budget; already placed, not yet on the map)"]
    if not serve:
        out.append("  (nothing: every placed volume is already served, or is a human placement)")
        return out
    for c in serve:
        out.append(f"  {c.volume:<24} {c.sheets:>6} sheets")
        out.extend(f"  {'':<24} · {note}" for note in c.notes)
    if cmd:
        out.append("")
        out.append("  " + cmd)
    out.append("")
    out.append(
        "  These were placed before one enqueue ran both legs; serving them is free and "
        "spends no model budget. A volume added today goes to the map on its own "
        "(`autogeoref queue --add <v>`) — pass --review to stop it for a look first."
    )
    return out


def render_candidates(cands: list[Candidate], city: CityConfig | None = None) -> str:
    """The backlog as text. Blocked rows are LISTED, never in the paste-ready line."""
    place = [c for c in cands if c.track == "place"]
    serve = [c for c in cands if c.track == "serve"]

    out = _render_place(place)
    cmd = add_command(cands, "place")
    if cmd:
        out.append("")
        out.append("  " + cmd)

    out.extend(_render_era_stanza(place, city))

    # No bounds stanza: a missing bounds source is a per-row NOTE (`candidates`),
    # not a refusal — the run bootstraps its own (bounds_bootstrap).
    out.append("")
    out.extend(_render_serve(serve, add_command(cands, "serve")))
    return "\n".join(out) + "\n"
