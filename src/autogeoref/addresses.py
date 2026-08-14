"""Address-numeral evidence against modern centerline ranges.

A numeral and its street identify a position along a segment; numerals on
crossing streets constrain the other axis. Pre-renumbering numerals require a
published conversion table, while modern-era numerals pass through unchanged.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from shapely.geometry import LineString

from .names import Aliases, normalize

logger = logging.getLogger(__name__)

#: Default house numbers per city block. Cities with a different convention set
#: ``address_block_size`` in their TOML.
BLOCK_SIZE = 100

#: Address-channel tolerance as a fraction of a city block. The block size is
#: supplied by city configuration.
ADDR_TOL_BLOCK_RATIO = 0.75

Side = Literal["l", "r"]


@dataclass(frozen=True)
class AddressNumeral:
    """A printed frontage address numeral read off a sheet.

    An annotation-schema extension: the annotator already reads text and
    bboxes, and this adds the numeral feature class. ``bbox`` is
    ``(x0, y0, x1, y1)`` in the annotation ("small") frame like every other
    annotation bbox — convert via the sheet-manifest scale before any full-res
    use. ``street_hint`` is the street the numeral fronts on when the annotator
    can tell, else None.
    """

    value: int
    bbox: tuple[float, float, float, float]
    street_hint: str | None = None


@dataclass(frozen=True)
class AddressMatch:
    """One centerline segment whose address range contains a numeral.

    ``side`` is which frontage's range matched, parity-checked when the source
    declares it. ``from_add``/``to_add`` are that side's range and ``fraction``
    is the numeral's linear position within it, along the segment's digitized
    direction, interpolated to ``point_4326``. ``geometry`` is the segment as a
    single LineString, MultiLineString parts concatenated in order.
    """

    side: Side
    from_add: int
    to_add: int
    fraction: float
    point_4326: tuple[float, float]
    geometry: LineString
    properties: Mapping[str, Any]


def _parse_add(raw: Any) -> int | None:
    """Parse a centerline address-range field."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = int(float(text))
    except ValueError:
        return None
    return value if value > 0 else None


def address_range_sides(props: Mapping[str, Any]) -> dict[Side, tuple[int, int]]:
    """Both frontage address ranges of one centerline feature, parsed.

    THE owner of the ``l``/``r`` ``{side}_f_add``/``{side}_t_add`` field
    parsing, shared by :func:`match_address` and the verified-accept segment
    builder. A side missing either endpoint is dropped.
    """
    out: dict[Side, tuple[int, int]] = {}
    sides: tuple[Side, ...] = ("l", "r")
    for side in sides:
        f_add = _parse_add(props.get(f"{side}_f_add"))
        t_add = _parse_add(props.get(f"{side}_t_add"))
        if f_add is not None and t_add is not None:
            out[side] = (f_add, t_add)
    return out


def line_coords(geometry: Mapping[str, Any]) -> list[tuple[float, float]] | None:
    """(Multi)LineString geometry dict -> one coordinate run, parts in order."""
    gtype = geometry.get("type")
    if gtype == "LineString":
        parts = [geometry["coordinates"]]
    elif gtype == "MultiLineString":
        parts = [p for p in geometry["coordinates"] if p]
    else:
        return None
    coords: list[tuple[float, float]] = []
    for part in parts:
        for c in part:
            pt = (float(c[0]), float(c[1]))
            if not coords or coords[-1] != pt:
                coords.append(pt)
    return coords if len(coords) >= 2 else None


def match_address(
    numeral: AddressNumeral,
    street_name: str,
    centerline_features: Iterable[Mapping[str, Any]],
    aliases: Aliases | None = None,
) -> list[AddressMatch]:
    """Find centerline segments of a street whose address range contains a numeral.

    The street is matched by normalized name against the ``street_nam`` field,
    with the volume's alias table; ``street_name`` is passed raw and normalized
    internally. A segment matches when the numeral falls inside either frontage
    side's range, and when that side declares a parity the numeral's parity must
    agree — odd numbers belong to the odd frontage. Returns every matching
    segment, one entry per matching side, possibly none.
    """
    key = normalize(street_name, aliases)
    value = numeral.value
    out: list[AddressMatch] = []
    for f in centerline_features:
        props = f.get("properties") or {}
        name = props.get("street_nam")
        if not name or normalize(str(name), aliases) != key:
            continue
        geometry = f.get("geometry")
        coords = line_coords(geometry) if geometry else None
        if coords is None:
            continue
        line: LineString | None = None
        for side, (f_add, t_add) in address_range_sides(props).items():
            lo, hi = min(f_add, t_add), max(f_add, t_add)
            if not lo <= value <= hi:
                continue
            parity = str(props.get(f"{side}_parity") or "").strip().upper()
            if parity in {"O", "E"} and value % 2 != (1 if parity == "O" else 0):
                continue
            if line is None:
                line = LineString(coords)
            span = t_add - f_add
            fraction = 0.5 if span == 0 else (value - f_add) / span
            fraction = min(max(fraction, 0.0), 1.0)
            pt = line.interpolate(fraction, normalized=True)
            out.append(
                AddressMatch(
                    side=side,
                    from_add=f_add,
                    to_add=t_add,
                    fraction=fraction,
                    point_4326=(float(pt.x), float(pt.y)),
                    geometry=line,
                    properties=props,
                )
            )
    logger.debug("numeral %d on %r -> %d segment matches", value, street_name, len(out))
    return out


@dataclass(frozen=True)
class RenumberingEntry:
    """One address-renumbering rule: a street's old range -> new range.

    Ranges are stored in pairing order — ``old_range[0]`` is the printed
    partner of ``new_range[0]`` — so on streets where the old numbers ran the
    opposite way (Cullom Av: old numbers descend westward while new ascend)
    ``old_range`` is stored descending. The conversion is a deterministic
    step from the paired endpoint: ``new = new_range[0] +/- (old -
    old_range[0])``, the sign following the ranges' relative direction.
    """

    street: str
    old_range: tuple[int, int]
    new_range: tuple[int, int]


def ambiguity_tol_numbers(block_size: int = BLOCK_SIZE) -> float:
    """Contradiction tolerance in house numbers between two table answers.

    Two entries claiming one old number contradict when their answers differ by
    more than the address channel's in-block tolerance. Pass
    ``CityConfig.address_block_size`` to keep this threshold aligned with the
    channel.

    Contradictory table entries abstain. A wrong conversion can fabricate a
    supporting vote, so ambiguity must not be resolved by file order.
    """
    return ADDR_TOL_BLOCK_RATIO * block_size


@dataclass(frozen=True)
class RenumberingTable:
    """A published address-renumbering conversion table.

    Pre-renumbering numerals must be converted before matching modern ranges;
    modern numerals pass through unchanged. An empty table returns ``None``::

        [{"street": "HERMITAGE", "old_range": [2400, 2458],
          "new_range": [4300, 4358]}, ...]

    Streets are compared after :func:`autogeoref.names.normalize`.
    """

    entries: tuple[RenumberingEntry, ...] = ()

    @classmethod
    def from_json(cls, path: Path) -> RenumberingTable:
        """Load a table from a JSON list of entry objects (see class docstring)."""
        raw = json.loads(path.read_text())
        entries = tuple(
            RenumberingEntry(
                street=str(e["street"]),
                old_range=(int(e["old_range"][0]), int(e["old_range"][1])),
                new_range=(int(e["new_range"][0]), int(e["new_range"][1])),
            )
            for e in raw
            # Suffixed numbers belong to a distinct numbering scheme; plain
            # numerals must not be converted through those entries.
            if not e.get("old_suffix")
        )
        return cls(entries=entries)

    def convert(
        self,
        street: str,
        old_number: int,
        aliases: Aliases | None = None,
        *,
        block_size: int = BLOCK_SIZE,
    ) -> int | None:
        """Convert a pre-renumbering number to its modern equivalent.

        Returns ``None`` when no entry covers the street/number — the numeral
        must then be treated as unmatchable, never passed through unconverted —
        and also when the covering entries CONTRADICT each other
        (func:`ambiguity_tol_numbers` of ``block_size``, the city's
        ``address_block_size``).
        """
        answers = self.covering_answers(street, old_number, aliases)
        if not answers:
            return None
        if max(answers) - min(answers) > ambiguity_tol_numbers(block_size):
            return None
        return answers[0]

    def covering_answers(
        self, street: str, old_number: int, aliases: Aliases | None = None
    ) -> list[int]:
        """Every answer this table can give for one old number, in file order.

        More than one is possible when normalization folds separately listed
        street sections onto one key without positional disambiguation.
        """
        key = normalize(street, aliases)
        out: list[int] = []
        for e in self.entries:
            lo, hi = min(e.old_range), max(e.old_range)
            if normalize(e.street, aliases) == key and lo <= old_number <= hi:
                old_dir = -1 if e.old_range[1] < e.old_range[0] else 1
                new_dir = -1 if e.new_range[1] < e.new_range[0] else 1
                out.append(e.new_range[0] + old_dir * new_dir * (old_number - e.old_range[0]))
        return out


#: Default table when no conversion data is configured.
EMPTY_RENUMBERING = RenumberingTable()


def modern_numeral(
    street: str,
    number: int,
    table: RenumberingTable | None = None,
    aliases: Aliases | None = None,
    *,
    block_size: int = BLOCK_SIZE,
) -> int | None:
    """Numeral in the modern (post-renumbering) system, or ``None`` if unknowable.

    Modern-era volumes pass ``table=None``: the printed number is modern.
    Pre-renumbering volumes pass their renumbering table; a numeral outside
    every table entry returns ``None``. ``block_size`` is the city's
    ``address_block_size``, scaling the table's contradiction tolerance.
    """
    if table is None:
        return number
    return table.convert(street, number, aliases, block_size=block_size)


def _bboxes_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def consensus_numerals(
    per_model: Mapping[str, Iterable[AddressNumeral]],
    aliases: Aliases | None = None,
    min_agree: int = 2,
) -> list[AddressNumeral]:
    """Cross-model consensus: the numerals >= ``min_agree`` models agree on.

    Two readings agree when they carry the same value, normalized street hint,
    and overlapping bboxes; single-model readings do not vote. ``per_model``
    maps model name to that model's readings for ONE sheet. Returns one
    representative numeral per agreed reading — the first seen, street hint as
    originally read.
    """
    keyed = [
        (
            model,
            [
                (n.value, normalize(n.street_hint, aliases) if n.street_hint else None, n)
                for n in nums
            ],
        )
        for model, nums in per_model.items()
    ]
    consensus: list[tuple[int, str | None, AddressNumeral]] = []
    for i, (_model, nums) in enumerate(keyed):
        for value, street, numeral in nums:
            if any(
                cv == value and cs == street and _bboxes_overlap(cn.bbox, numeral.bbox)
                for cv, cs, cn in consensus
            ):
                continue
            agree = 1 + sum(
                any(
                    v2 == value and s2 == street and _bboxes_overlap(n2.bbox, numeral.bbox)
                    for v2, s2, n2 in other_nums
                )
                for j, (_m2, other_nums) in enumerate(keyed)
                if j != i
            )
            if agree >= min_agree:
                consensus.append((value, street, numeral))
    return [n for _v, _s, n in consensus]


__all__ = [
    "ADDR_TOL_BLOCK_RATIO",
    "BLOCK_SIZE",
    "EMPTY_RENUMBERING",
    "AddressMatch",
    "AddressNumeral",
    "RenumberingEntry",
    "RenumberingTable",
    "address_range_sides",
    "ambiguity_tol_numbers",
    "consensus_numerals",
    "line_coords",
    "match_address",
    "modern_numeral",
]
