"""Propose, declare, and read a volume's printed-address era.

This module owns the era vocabulary (:data:`AddressEra`) and THE one
config->era mapping the pipeline reads (:func:`era_from_config`). The rest is
the operator tool: it compares an operator-confirmed catalog year with the
configured renumbering year. It never teaches the pipeline to infer an era,
refuses untrusted years, and reloads the TOML after writing to verify the
declaration.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .config.load import load_city_config
from .loc import catalog_year

if TYPE_CHECKING:
    from .config.model import CityConfig

logger = logging.getLogger(__name__)

#: The volume's printed-address era: "modern" = printed numbers are today's
#: numbers; "renumbered" = the city renumbered after the volume was printed
#: and numbers convert through the published table;
#: "unknown" = abstain.
AddressEra = Literal["modern", "renumbered", "unknown"]


def era_from_config(
    addresses_modern: bool | None,
    *,
    volume: str = "?",
    city_renumbered: bool = False,
) -> AddressEra:
    """Map the per-volume config flag to an address era — THE one mapping.

    **Undeclared is MODERN**: most cities never renumbered, so demanding a
    declaration would break the base case. It must NOT mean ``"renumbered"`` —
    that pushes modern numbers through a table and lands them blocks away. The
    residual risk is a city that DID renumber whose old volume declared no era:
    its numerals read against the modern grid, and because addresses is the ONLY
    channel that may REFUTE it can then veto correct placements with total
    confidence. ``city_renumbered`` makes that loud.
    """
    if addresses_modern is None:
        if city_renumbered:
            logger.warning(
                "%s: no addresses_modern declared, and this city declares a renumbering "
                "table — assuming MODERN. If this volume predates the renumbering its "
                "printed numbers are NOT today's numbers, and the addresses channel (the "
                "only one that may REFUTE) can veto correct sheets. Declare it.",
                volume,
            )
        return "modern"
    return "modern" if addresses_modern else "renumbered"


class EraError(RuntimeError):
    """The declaration cannot be made, or cannot be made safely."""


@dataclass(frozen=True)
class EraProposal:
    """What this volume's era should be, and why — or why we will not say."""

    volume: str
    year: int | None
    #: The proposal: True = modern numbering, False = predates the renumbering.
    #: None means REFUSED — ``refusal`` says why, and nothing will be written.
    modern: bool | None
    refusal: str | None = None
    #: Already declared in the TOML; there is nothing to do.
    declared: bool | None = None
    #: WHERE the year came from: ``"date"`` = LOC's structured field; anything else
    #: (a regex over the free-text description, or a catalog that does not say) is
    #: not a catalogued date. See :attr:`year_is_trusted`.
    year_source: str | None = None

    @property
    def ok(self) -> bool:
        return self.modern is not None and self.refusal is None

    @property
    def year_is_trusted(self) -> bool:
        """Did this year come from LOC's structured ``date`` field?

        **Fails CLOSED, and that is the point.** Only the exact string
        ``"date"`` counts, so an absent ``year_source`` reads as untrusted and
        gets the warning rather than rendering like a catalogued date. Warning
        on ``== "description"`` instead would fail OPEN: rename the key
        upstream and the warning silently vanishes with every test still green.
        """
        return self.year_source == "date"


def propose(city: CityConfig, catalog: dict[str, dict[str, Any]], volume: str) -> EraProposal:
    """Compare the volume's LOC edition year against the CITY's renumbering year."""
    declared = city.volumes.get(volume)
    if declared is not None and declared.addresses_modern is not None:
        return EraProposal(
            volume=volume,
            year=catalog_year(catalog, volume),
            modern=declared.addresses_modern,
            declared=declared.addresses_modern,
        )
    if city.renumbering_table_path is None:
        return EraProposal(
            volume=volume,
            year=catalog_year(catalog, volume),
            modern=None,
            refusal=(
                f"{city.name} ships no renumbering table, so no volume needs an era "
                "declaration — an undeclared era means MODERN, which is correct for a city "
                "that never renumbered. `autogeoref run` does not refuse these."
            ),
        )
    if city.renumbering_year is None:
        return EraProposal(
            volume=volume,
            year=catalog_year(catalog, volume),
            modern=None,
            refusal=(
                f"{city.name} ships a renumbering table but declares no `renumbering_year`, "
                "so there is nothing to compare an edition year against. Add it to [city] — "
                "the year the table took effect. This tool will not assume one: a city's "
                "calendar is the city's."
            ),
        )
    year = catalog_year(catalog, volume)
    if year is None:
        return EraProposal(
            volume=volume,
            year=None,
            modern=None,
            refusal=(
                "no edition year could be read from the LOC catalog for this volume — either "
                "the record has none, or its date could not be corroborated from the catalog. "
                "Read the sheet's own title page and set `addresses_modern` by hand."
            ),
        )
    # The whole arithmetic. A volume printed IN the renumbering year is the one case
    # this cannot settle from a year alone — the sheets could predate the switch by
    # months — so it refuses that too rather than coin-flip a channel that can REFUTE.
    if year == city.renumbering_year:
        return EraProposal(
            volume=volume,
            year=year,
            modern=None,
            refusal=(
                f"this volume is dated {year}, the very year {city.name} renumbered. A year "
                "cannot say whether its sheets were printed before or after the switch. Read "
                "the sheet's printed numbers against a known address and declare it by hand."
            ),
        )
    return EraProposal(
        volume=volume,
        year=year,
        modern=year > city.renumbering_year,
        year_source=_year_source(catalog, volume),
    )


def _year_source(catalog: dict[str, dict[str, Any]], volume: str) -> str | None:
    source = catalog.get(volume, {}).get("year_source")
    return source if isinstance(source, str) else None


def render(proposal: EraProposal, city: CityConfig) -> str:
    """The proposal as the operator must see it: the YEAR, then the consequence.

    The year is the point. The whole reason this is a confirm and not a batch job is
    that a human should read "1896 -> pre-1909 -> false" and recognise a wrong year
    before it becomes a wrong veto.
    """
    if proposal.declared is not None:
        return (
            f"{proposal.volume}: already declares addresses_modern = "
            f"{str(proposal.declared).lower()} — nothing to do."
        )
    if proposal.refusal is not None:
        return f"{proposal.volume}: REFUSED — {proposal.refusal}"
    modern = proposal.modern
    verdict = "AFTER" if modern else "BEFORE"
    lines = [
        f"{proposal.volume}: LOC catalog year {proposal.year} — printed {verdict} "
        f"{city.name}'s {city.renumbering_year} renumbering",
        f"    addresses_modern = {str(bool(modern)).lower()}",
    ]
    if not proposal.year_is_trusted:
        # The year came out of a free-text blurb, or a catalog that does not
        # say where it got it. Almost certainly right; also a regex over a
        # sentence, and this is the moment a human can say otherwise. The
        # message must NOT claim the date field was empty — the fallback also
        # covers a present-but-unparseable one, and a fact we did not check
        # makes the confirm a rubber stamp.
        lines.append(
            "    ⚠ THIS YEAR IS NOT A CATALOGUED DATE. The record's structured `date` "
            "field carried no usable year, so this was read out of its description text "
            "(or came from a catalog that does not say). Check the volume's own title "
            "page before you confirm."
        )
    if modern:
        lines.append(
            "    (the printed numbers ARE today's numbers; the addresses channel reads "
            "them directly)"
        )
    else:
        lines.append(
            "    (the printed numbers PREDATE the renumbering; they convert through the "
            "city's table)"
        )
        lines.append(
            "    CHECK WHICH BOOK. This volume will convert through "
            f"{_table_name(city)}. A city can renumber its districts on different "
            "dates, and two such books disagree by hundreds of metres on every old "
            "number they both claim — a volume from another district needs "
            "`renumbering_table` pointing at its own register. This tool cannot "
            "know a volume's district; you can."
        )
    return "\n".join(lines)


def refuse_untrusted(proposals: list[EraProposal]) -> str | None:
    """Why ``--yes`` cannot confirm these proposals — or ``None`` if it can.

    ``--yes`` CANNOT rubber-stamp a year that is not a catalogued date. Over
    the backlog it would otherwise write an ``addresses_modern`` derived from a
    regex over a catalogue blurb, silently, into the config key that arms the
    only channel allowed to REFUTE. A scraped year is proposed, never silent —
    and that is worth nothing if only prose enforces it. So ``--yes`` covers
    the catalogued years and refuses the rest, at one keypress each.
    """
    untrusted = [p for p in proposals if not p.year_is_trusted]
    if not untrusted:
        return None
    return (
        f"--yes will not confirm {len(untrusted)} volume(s) whose year is not "
        "a catalogued date — it was read out of the record's description text, and "
        "the whole point of the confirm is that a human looks at THOSE: "
        f"{', '.join(p.volume for p in untrusted)}.\n"
        "  Re-run them without --yes and answer the prompt, or declare them by "
        "hand from the volume's title page."
    )


def _table_name(city: CityConfig) -> str:
    return city.renumbering_table_path.name if city.renumbering_table_path else "(none)"


def declare(city_path: Path, proposal: EraProposal) -> None:
    """Write ``addresses_modern`` for one volume into the city TOML, and VERIFY it.

    Text editing, because the dependency set has no TOML writer — and then a reload
    through :func:`config.load_city_config` to prove the file still parses and the value
    came back as the one proposed. A tool that mis-declared an era would be worse than
    the wall it replaces: the wall fails loudly, a wrong declaration vetoes correct
    sheets in the confident language of evidence. On any mismatch the original is
    restored and this raises.
    """
    if proposal.declared is not None:
        # Already declared. Inserting a second `addresses_modern` into the block would
        # be a DUPLICATE KEY: tomllib rejects it, the reload below restores the file,
        # and the operator gets a TOMLDecodeError about a column number instead of the
        # sentence they need. `ok` is True for a declared proposal (it has a value and
        # no refusal), so this cannot be folded into the check below.
        raise EraError(
            f"{proposal.volume}: already declares addresses_modern = "
            f"{str(proposal.declared).lower()}. Change it by hand if it is wrong — this "
            "tool adds a declaration, it does not overrule one."
        )
    if not proposal.ok or proposal.modern is None:
        raise EraError(f"{proposal.volume}: refused, nothing to write")
    original = city_path.read_text()
    updated = _with_declaration(original, proposal.volume, proposal.modern)
    _write_atomic(city_path, updated)
    try:
        reloaded = load_city_config(city_path)
        got = reloaded.volume(proposal.volume).addresses_modern
        if got is not proposal.modern:
            raise EraError(
                f"{proposal.volume}: wrote addresses_modern = "
                f"{str(proposal.modern).lower()} but the config reloads as {got!r}. "
                "The file has been restored; declare it by hand."
            )
    except Exception:
        _write_atomic(city_path, original)  # never leave a config we could not verify
        raise


def _write_atomic(path: Path, text: str) -> None:
    """Same-directory temp + replace: a kill mid-write must not truncate a city config.

    ``write_text`` truncates and then writes, so a process killed in that window leaves
    the config half-written — and this function is called twice per declaration, once to
    write and once to RESTORE, so the naive version has that window on the recovery path
    too. The config is the file that says which volumes may run at all.
    """
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _with_declaration(text: str, volume: str, modern: bool) -> str:
    """Insert ``addresses_modern`` into the volume's block, or append a new block.

    The section header is ALWAYS quoted, because an id containing a dot is a
    NESTED table in bare TOML: it would declare an era for a volume that does
    not exist while the operator believed they fixed the one that does. Quoting
    is valid for the plain ids too, so there is no case to get right.
    """
    value = f"addresses_modern = {str(modern).lower()}"
    header_variants = (f'[volumes."{volume}"]', f"[volumes.{volume}]")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() in header_variants:
            lines.insert(i + 1, value)
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    block = f'\n[volumes."{volume}"]\n{value}\n'
    return text.rstrip("\n") + "\n" + block
