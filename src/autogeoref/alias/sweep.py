"""The alias sweep: scan volumes, propose, auto-write the clean tier, mark.

One idempotent pass over a city's processed volumes that closes historic
street-name alias gaps end to end:

1. **Scan.** Measure every annotated volume through the shipped tripwire's own
   counter, so a sweep number and a run number are the same number. Volumes
   above the bars, on the skip list, already marked, or in a city with no
   rename source are reported and left alone, each with its reason.
2. **Propose and tier** (:mod:`autogeoref.alias.propose`).
3. **Write the clean tier** into the volume's alias file, MERGED: an existing
   entry is never overwritten, because those carry owner sign-off this does not.

The file's bytes are a pure function of its inputs, so a re-run rewrites none.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..bounds_bootstrap import persisted_bounds
from ..centerlines import CenterlineIndex
from ..name_match import (
    HIGH_ZERO_CANDIDATE_SHARE,
    LOW_MATCH_RATE,
    MIN_PAGES_FOR_NOTE,
    MIN_READS_FOR_NOTE,
    count_name_matches,
)
from ..names import load_aliases
from ..paths import VolumePaths, atomic_write_text
from ..run_inputs import NoBoundsSourceError, resolve_bounds
from ..sheet_inputs import load_sheet_inputs
from .propose import (
    ACCEPT_LOCALITY_M,
    MARGIN_M,
    MARGIN_RATIO,
    MIN_SHEETS,
    Proposal,
    propose_volume,
)
from .source import RenameSourceError
from .validate import twin_hold, validate_table

if TYPE_CHECKING:
    from ..config.model import CityConfig, VolumeConfig

logger = logging.getLogger(__name__)

#: The guide every write cites, so a reader of a generated alias file lands on
#: the tier rules and their bars rather than on this module.
BASIS_RECORD = "docs/INTERNALS.md"

#: Where the per-entry sheet check the held tier is queued for is described.
HELD_METHOD_RECORD = "docs/OPERATIONS.md"

#: Whether a volume can be re-placed for free. The report points an operator
#: here rather than deciding for them.
RERUN_PRECONDITIONS_RECORD = "docs/OPERATIONS.md"

#: Why a volume was left alone. Every one appears in the report: a sweep that
#: skips silently is a sweep nobody can audit.
SKIP_ABOVE_BARS = "above the tripwire bars"
SKIP_DECLARED = "on the city's alias_sweep_skip list"
SKIP_NO_SOURCE = "city has no rename source configured"
SKIP_MARKED = "already swept (marker present; --force to redo)"
SKIP_NO_BOUNDS = "no resolvable bounds"


@dataclass(frozen=True)
class VolumeOutcome:
    """What the sweep did with one volume."""

    volume: str
    #: True once the volume's proposals were actually computed — the condition
    #: for writing a marker. A skipped volume is measured, not processed.
    processed: bool = False
    #: Match rate before the sweep's writes, and the rate the merged table
    #: reaches. Equal when nothing was written; None with no street reads.
    match_rate_before: float | None = None
    match_rate_after: float | None = None
    reads: int = 0
    zero_candidate_pages: int = 0
    results: int = 0
    skipped: str | None = None
    written: Mapping[str, str] = field(default_factory=dict)
    held: tuple[Proposal, ...] = ()
    refused: tuple[Proposal, ...] = ()
    no_candidate: tuple[Proposal, ...] = ()
    aborted: tuple[str, ...] = ()
    #: Written keys that a preserved comment in the same file had dispositioned
    #: as deliberately-not-aliased. The sweep still writes them (the evidence
    #: cleared the tier) but a stale prose disposition now contradicts the
    #: table, and only a human can rewrite prose.
    supersedes: tuple[str, ...] = ()
    error: str | None = None

    @property
    def flagged(self) -> bool:
        """Does the shipped tripwire flag this volume, on either half?

        Both halves use the tripwire's own bars AND its own sample floors, so
        this command cannot flag a volume the run-time note would not — a
        two-sheet index item reading 0 of 2 is not a funnel measurement.
        """
        rate_low = (
            self.match_rate_before is not None
            and self.match_rate_before < LOW_MATCH_RATE
            and self.reads >= MIN_READS_FOR_NOTE
        )
        zero_high = (
            self.results >= MIN_PAGES_FOR_NOTE
            and self.zero_candidate_pages / self.results >= HIGH_ZERO_CANDIDATE_SHARE
        )
        return rate_low or zero_high

    def document(self) -> dict[str, Any]:
        """The marker's shape."""
        return {
            "volume": self.volume,
            "match_rate_before": self.match_rate_before,
            "match_rate_after": self.match_rate_after,
            "reads": self.reads,
            "zero_candidate_pages": self.zero_candidate_pages,
            "results": self.results,
            "entries_written": dict(self.written),
            "entries_held": len(self.held),
            "entries_refused": len(self.refused),
            "no_candidate_keys": len(self.no_candidate),
            "supersedes_dispositions": list(self.supersedes),
            "aborted": list(self.aborted),
            "error": self.error,
        }


@dataclass(frozen=True)
class SweepResult:
    """One sweep run over a city."""

    city: str
    city_path: Path
    run_date: str
    outcomes: tuple[VolumeOutcome, ...]

    @property
    def written(self) -> tuple[VolumeOutcome, ...]:
        return tuple(o for o in self.outcomes if o.written)

    @property
    def aborted(self) -> tuple[VolumeOutcome, ...]:
        return tuple(o for o in self.outcomes if o.aborted or o.error)


def annotated_volumes(work: Path) -> list[str]:
    """Volumes under ``work/`` with a sheet manifest and at least one bare read.

    The bare ``p<N>.json`` is the matcher's input; the dotted siblings
    (``p<N>.annotation.*``, ``p<N>.escalated.*``) are cache records.
    """
    if not work.is_dir():
        return []
    out = []
    for directory in sorted(p for p in work.iterdir() if p.is_dir()):
        paths = VolumePaths(root=directory)
        if not paths.manifest.is_file():
            continue
        if any("." not in p.stem for p in paths.annotations.glob("p*.json")):
            out.append(directory.name)
    return out


def marker_path(paths: VolumePaths) -> Path:
    return paths.root / "alias-sweep.marker.json"


def _index(
    city: CityConfig,
    bounds: tuple[float, float, float, float],
    features: list[dict[str, Any]],
    aliases: Mapping[str, str],
) -> CenterlineIndex:
    """A volume-bounded index over an explicit alias table.

    The sweep needs three of these per volume — the table on disk, the merged
    table, and none at all — so it builds them from a table it holds rather
    than through ``run_inputs.build_index``, which reads the file.
    """
    return CenterlineIndex(
        features,
        aliases=dict(aliases),
        bounds_4326=bounds,
        name_property=city.centerline_name_property,
        type_property=city.centerline_type_property,
    )


def _bounds(
    city: CityConfig, vol: VolumeConfig, paths: VolumePaths, viewer_manifest: Path | None
) -> tuple[float, float, float, float] | None:
    """The bounds a run would use, including a previous run's derivation.

    Bounds provenance matters here and is not a detail: the record's negative
    suite shows value-in-bounds verdicts flip with it, so the proposer must see
    the bounds the run saw.
    """
    try:
        return resolve_bounds(city, vol, viewer_manifest)
    except NoBoundsSourceError:
        return persisted_bounds(paths)


def _zero_candidate_pages(paths: VolumePaths) -> tuple[int, int]:
    """``(pages with no match candidate, result records)``."""
    zero = total = 0
    for path in sorted(paths.results.glob("p*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        total += 1
        if record.get("n_candidates") == 0:
            zero += 1
    return zero, total


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def alias_document(
    volume: str,
    citation: str,
    table: Mapping[str, str],
    clean: Sequence[Proposal],
    held: Sequence[Proposal],
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The alias file this sweep writes, provenance comments included.

    Byte-stable by construction: no run id and no date, so re-running over
    unchanged inputs produces the same file and the fixture manifest stays
    still. Comment keys the volume already had are PRESERVED, first — they
    carry a human build pass's reasoning and this command cannot re-derive it.
    """
    document: dict[str, Any] = {
        key: value for key, value in (previous or {}).items() if key.startswith("_")
    }
    by_key = {p.key: p for p in clean}
    generated = [
        f"Entries below were written by `autogeoref alias-sweep` for {volume} from the"
        f" city's pinned rename source: {citation}."
        f" Tier rules and their bars: {BASIS_RECORD}."
        f" An auto-written entry is either a UNIQUE in-bounds candidate whose locality"
        f" is within {ACCEPT_LOCALITY_M:.0f} m on at least {MIN_SHEETS} sheets, or an"
        f" ambiguous one the locality decided by {MARGIN_RATIO:g}x / {MARGIN_M:.0f} m"
        f" AND printed address numerals corroborated."
    ]
    for key in sorted(table):
        proposal = by_key.get(key)
        if proposal is None:  # a pre-existing entry, already documented above
            continue
        winner = proposal.winner
        line = " ".join((winner.source_line if winner else "").split())[:200]
        generated.append(
            f"{key} -> {table[key]}: {proposal.reads} unmatched read(s);"
            f" {proposal.reason}. Source: {line}"
        )
    document["_generated"] = generated
    # The keys this command wrote, as data rather than prose: the qualification
    # instrument has to be able to subtract its own output from a volume's table,
    # or it drifts into scoring the sweep against itself.
    document["_generated_keys"] = sorted(k for k in table if k in by_key)
    if held:
        # ASCII only: every other alias table in the tree is, and a JSON escape
        # for an em-dash is noise in a file a human reads.
        document["_generated_held"] = [
            f"HELD, not aliased: {p.key} ({p.reads} read(s)),"
            f" best candidate {p.value} -- {p.reason}"
            for p in sorted(held, key=lambda p: (-p.reads, p.key))
        ]
    document.update(table)
    return document


def _superseded(previous: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, ...]:
    """Written keys a preserved prose comment NAMES as deliberately not aliased.

    Whole-token matching, because substrings lie: a bare ``in`` test reports
    ``MILTON`` as superseded on any file whose prose mentions ``HAMILTON CT``.
    A cheap net over one file, and only that — a disposition that names no key,
    or one living only in a dated record, is out of reach. Silence here is NOT
    evidence that nothing was superseded.
    """
    prose = " ".join(
        str(value)
        for key, value in previous.items()
        if key.startswith("_") and isinstance(value, str)
    ).upper()
    return tuple(
        key
        for key in sorted(keys)
        if re.search(rf"(?<![A-Z0-9]){re.escape(key)}(?![A-Z0-9])", prose)
    )


def sweep_volume(
    volume: str,
    city: CityConfig,
    work: Path,
    features: list[dict[str, Any]],
    viewer_manifest: Path | None,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> VolumeOutcome:
    """Scan one volume and, if it qualifies, write and validate its clean tier."""
    vol = city.volume(volume)
    paths = VolumePaths(root=work / volume)
    reason = city.alias_sweep_skip.get(volume)
    if reason is not None:
        # The declared skip wins over everything, including --force: it is the
        # city's statement that this volume's flag is not an alias gap.
        return VolumeOutcome(volume=volume, skipped=f"{SKIP_DECLARED}: {reason}")
    bounds = _bounds(city, vol, paths, viewer_manifest)
    if bounds is None:
        return VolumeOutcome(volume=volume, skipped=SKIP_NO_BOUNDS)

    sheets = load_sheet_inputs(paths)
    aliases_path = city.aliases_path(volume)
    previous = _read_json(aliases_path)
    existing = load_aliases(aliases_path)
    live = _index(city, bounds, features, existing)
    before = count_name_matches(sheets, live)
    zero, results = _zero_candidate_pages(paths)
    outcome = VolumeOutcome(
        volume=volume,
        match_rate_before=before.match_rate,
        match_rate_after=before.match_rate,
        reads=before.reads,
        zero_candidate_pages=zero,
        results=results,
    )
    if not force and marker_path(paths).is_file():
        # Measured first, then latched: an already-swept volume still shows its
        # reads and match rate in the scan table, because that table is the
        # audit surface and a row of "n/a" audits nothing.
        return dataclasses.replace(outcome, skipped=SKIP_MARKED)
    if not outcome.flagged:
        return dataclasses.replace(outcome, skipped=SKIP_ABOVE_BARS)
    if city.rename_source is None:
        return dataclasses.replace(outcome, skipped=SKIP_NO_SOURCE)

    alias_free = _index(city, bounds, features, {})

    def guard(key: str, value: str) -> Sequence[str]:
        """The structural rules, applied to one candidate entry on its own.

        On its own, not merged: an entry judged inside the landed table would
        inherit that table's history, and a new entry is entitled to none of
        it.
        """
        return twin_hold(key, alias_free) or validate_table(
            {key: value},
            alias_free,
            city.centerline_name_property,
            city.centerline_type_property,
        )

    try:
        proposal = propose_volume(
            volume,
            city.rename_source,
            live,
            sheets,
            grid=city.address_grid,
            block=float(city.address_block_size),
            era_modern=vol.addresses_modern is True,
            existing=existing,
            entry_guard=guard,
            counts=before,
        )
    except RenameSourceError as exc:
        return dataclasses.replace(outcome, error=str(exc))

    outcome = dataclasses.replace(
        outcome,
        processed=True,
        held=proposal.held,
        refused=proposal.refused,
        no_candidate=proposal.no_candidate,
    )
    table = proposal.table
    if not table:
        return outcome

    merged = {**existing, **table}
    failures = validate_table(
        merged,
        alias_free,
        city.centerline_name_property,
        city.centerline_type_property,
    )
    if failures:
        # Abort THIS volume and write nothing: a table that fails the structural
        # rules can re-key centerlines or shadow a surviving street, and both
        # are silent at match time.
        return dataclasses.replace(outcome, aborted=tuple(failures))

    if not dry_run:
        document = alias_document(
            volume,
            city.rename_source.citation,
            merged,
            proposal.clean,
            proposal.held,
            previous=previous,
        )
        atomic_write_text(aliases_path, json.dumps(document, indent=2) + "\n")

    after = count_name_matches(sheets, _index(city, bounds, features, merged))
    return dataclasses.replace(
        outcome,
        written=table,
        match_rate_after=after.match_rate,
        supersedes=_superseded(previous, list(table)),
    )


def run_sweep(
    city: CityConfig,
    city_path: Path,
    work: Path,
    *,
    volumes: Sequence[str] | None = None,
    viewer_manifest: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    today: date | None = None,
) -> SweepResult:
    """Sweep a city's volumes, marking the ones whose proposals were computed."""
    features = json.loads(city.centerlines_path.read_text())["features"]
    run_date = (today or datetime.now().astimezone().date()).isoformat()
    names = list(volumes) if volumes else annotated_volumes(work)
    known = set(annotated_volumes(work))
    outcomes: list[VolumeOutcome] = []
    for volume in names:
        if volumes and volume not in known:
            # A typo in --volumes must not read as "scanned, nothing to do".
            outcomes.append(
                VolumeOutcome(
                    volume=volume,
                    error=f"no annotated volume {volume!r} under {work} (typo in --volumes?)",
                )
            )
            continue
        try:
            outcome = sweep_volume(
                volume, city, work, features, viewer_manifest, force=force, dry_run=dry_run
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
            # One unreadable volume must not end a corpus pass; the report names
            # what failed and the marker is not written, so a retry re-does it.
            logger.warning("%s: alias sweep failed (%s)", volume, exc)
            outcome = VolumeOutcome(volume=volume, error=f"{type(exc).__name__}: {exc}")
        outcomes.append(outcome)
        # A marker on an ABORT would latch the failure: the next run would skip
        # the volume, report nothing, and exit 0 on a defect the command calls a
        # failure. Same for an exception. Only a completed volume is marked.
        if outcome.processed and not outcome.aborted and not outcome.error and not dry_run:
            atomic_write_text(
                marker_path(VolumePaths(root=work / volume)),
                json.dumps(
                    {"run_date": run_date, "city_config": str(city_path), **outcome.document()},
                    indent=2,
                )
                + "\n",
            )
    return SweepResult(
        city=city.name, city_path=city_path, run_date=run_date, outcomes=tuple(outcomes)
    )


def _rivals(proposal: Proposal) -> str:
    return " | ".join(
        f"{c.value} {c.locality_m:.0f}m/{c.sheets}sh"
        if c.locality_m is not None
        else f"{c.value} no-locality"
        for c in proposal.candidates[:4]
    )


def render_report(result: SweepResult, *, dry_run: bool = False) -> str:
    """The sweep-level report: what was written, and the whole held tier."""
    out: list[str] = [f"# Alias sweep — {result.city} — {result.run_date}", ""]
    if dry_run:
        out += ["**DRY RUN: nothing was written.**", ""]
    out += [
        f"City config: `{result.city_path}`. Tier rules and their bars:",
        f"`{BASIS_RECORD}`.",
        "",
        "## Scan",
        "",
        "| Volume | reads | match before | match after | written | held |"
        " refused | no-cand | note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for o in result.outcomes:
        before = f"{o.match_rate_before:.1%}" if o.match_rate_before is not None else "n/a"
        after = f"{o.match_rate_after:.1%}" if o.match_rate_after is not None else "n/a"
        note = o.error or o.skipped or ("ABORTED by the validator" if o.aborted else "")
        out.append(
            f"| {o.volume} | {o.reads} | {before} | {after} | {len(o.written)} |"
            f" {len(o.held)} | {len(o.refused)} | {len(o.no_candidate)} | {note} |"
        )
    out += ["", "## Written", ""]
    if not result.written:
        out += ["Nothing written.", ""]
    for o in result.written:
        out += [f"### {o.volume}", ""]
        for key in sorted(o.written):
            out.append(f"- `{key}` -> `{o.written[key]}`")
        if o.supersedes:
            out += [
                "",
                "**Supersedes a recorded disposition** — the file's preserved prose still"
                f" says these keys were deliberately not aliased: {', '.join(o.supersedes)}."
                " Rewrite that comment by hand.",
            ]
        out.append("")
    if result.aborted:
        out += ["## Aborted — nothing written for these volumes", ""]
        for o in result.aborted:
            out += [f"### {o.volume}", ""]
            if o.error:
                out.append(f"- error: {o.error}")
            out += [f"- {failure}" for failure in o.aborted]
            out.append("")
    out += [
        "## Held — candidates that exist but did not clear the bar",
        "",
        "Held is not discarded: each row is a candidate for an agent pass with the",
        f"per-entry sheet-check method in `{HELD_METHOD_RECORD}`.",
        "",
    ]
    if not any(o.held for o in result.outcomes):
        out += ["Nothing held.", ""]
    for o in result.outcomes:
        if not o.held:
            continue
        out += [f"### {o.volume}", ""]
        for p in sorted(o.held, key=lambda p: (-p.reads, p.key)):
            out.append(f"- `{p.key}` [{p.reads} reads] {_rivals(p)}")
            out.append(f"  - {p.reason}")
        out.append("")
    if any(o.refused for o in result.outcomes):
        out += [
            "## Refused — no candidate is reachable",
            "",
            "Not a queue. Every candidate for these keys either has no measurable",
            "locality or lies beyond the accept bar, which is what a rename looks like",
            "when its modern value has no in-bounds geometry near the reads (the",
            "out-of-city edge grid). More corroboration cannot make them writable.",
            "",
        ]
        for o in result.outcomes:
            if not o.refused:
                continue
            out += [f"### {o.volume}", ""]
            for p in sorted(o.refused, key=lambda p: (-p.reads, p.key)):
                out.append(f"- `{p.key}` [{p.reads} reads] {_rivals(p)}")
                out.append(f"  - {p.reason}")
            out.append("")
    out += [
        "## Next step (operator, spend-aware)",
        "",
        "The writes above change what MATCHES. Turning that into placements is a",
        "separate step this command deliberately does not take, because whether it is",
        "free depends on the volume: an escalation-free re-place demotes",
        "escalation-won accepts, and leaving escalation on spends fresh wherever the",
        "cached tiers no longer match the city ladder. Check the volume against",
        f"`{RERUN_PRECONDITIONS_RECORD}` first, then per volume with writes:",
        "",
    ]
    for o in result.written:
        out.append(
            f"    uv run autogeoref run {o.volume} --city {result.city_path}"
            " --work work --no-annotate"
        )
    out.append("")
    return "\n".join(out) + "\n"


__all__ = [
    "BASIS_RECORD",
    "HELD_METHOD_RECORD",
    "RERUN_PRECONDITIONS_RECORD",
    "SKIP_ABOVE_BARS",
    "SKIP_DECLARED",
    "SKIP_MARKED",
    "SKIP_NO_BOUNDS",
    "SKIP_NO_SOURCE",
    "SweepResult",
    "VolumeOutcome",
    "alias_document",
    "annotated_volumes",
    "marker_path",
    "render_report",
    "run_sweep",
    "sweep_volume",
]
