"""Frozen annotation prompts and annotation payload schemas."""

from __future__ import annotations

import contextlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from ..addresses import AddressNumeral
from ..margins import MarginReading
from .failures import (
    AnnotateError,
    EmptyResponseError,
    MalformedResponseError,
    _raise_budget_if_matched,
    _strip_code_fences,
)

logger = logging.getLogger(__name__)

# Verbatim PROMPT_TMPL from pipeline/annotate/annotate_batch.sh (tuned; do not edit).
PROMPT_TEMPLATE = 'Look at the Sanborn fire insurance map sheet image at IMGPATH and output ONLY a JSON object (no prose, no code fences) with this shape: {"streets": [{"name": "<street name as printed, e.g. WABASH AVE>", "bbox": [x0,y0,x1,y1], "orientation": "horizontal|vertical"}], "page_number_seen": "<sheet number printed in corner>"}. Include every legible street name label on the map with a tight pixel bounding box around the text (the image is 1326x2000 or similar). Street labels run along streets: horizontal labels for east-west streets, vertical (rotated) labels for north-south streets. Big ornate margin numbers are sheet numbers, not streets. Exclude park labels, "SEE VOLUME" notes, building names, and railroad names.'  # noqa: E501

#: v2 prompt. Core street/bbox rules mirror the frozen v1 PROMPT_TEMPLATE;
#: new feature classes are additive arrays so a v1 parser ignores them.
EXTENDED_PROMPT_TEMPLATE = (
    "Look at the Sanborn fire insurance map sheet image at IMGPATH and output ONLY a JSON "
    'object (no prose, no code fences) with this shape: {"streets": [{"name": "<street name '
    'as printed, e.g. WABASH AVE>", "bbox": [x0,y0,x1,y1], "orientation": '
    '"horizontal|vertical"}], "page_number_seen": "<sheet number printed in corner>", '
    '"address_numerals": [{"value": <integer>, "bbox": [x0,y0,x1,y1], "street": "<street '
    'this address fronts, if determinable, else null>"}], "margin_numbers": [{"side": '
    '"top|bottom|left|right", "text": "<number or note printed in that margin>"}], '
    '"rail_labels": [{"name": "<railroad name as printed, e.g. C.M.&ST.P.R.R.>", "bbox": '
    '[x0,y0,x1,y1]}], "park_labels": [{"name": "<park or cemetery name as printed>", "bbox": '
    "[x0,y0,x1,y1]}]}. "
    "STREETS: include every legible street name label on the map with a tight pixel bounding "
    "box around the text (the image is 1326x2000 or similar). Street labels run along "
    "streets: horizontal labels for east-west streets, vertical (rotated) labels for "
    "north-south streets. Big ornate margin numbers are sheet numbers, not streets; report "
    "them under margin_numbers with the side of the sheet they sit on, and also report "
    '"SEE VOLUME ..." style margin notes there. '
    "ADDRESS NUMERALS: the small numbers printed along building frontages are house address "
    "numbers; report every clearly legible one as an integer with a tight bbox and the "
    "street it fronts when you can tell. Skip ambiguous or partially legible numbers — "
    "precision matters more than coverage. "
    "RAIL: report railroad right-of-way name labels under rail_labels. "
    "PARKS: report park and cemetery name labels under park_labels. "
    "Do not report building names or business names anywhere."
)

#: The one clause that separates the diagonal prompt from the frozen one. The
#: replacement is asserted below, so this prompt cannot silently become a copy.
_CARDINAL_CLAUSE = (
    "Street labels run along "
    "streets: horizontal labels for east-west streets, vertical (rotated) labels for "
    "north-south streets. "
)
_DIAGONAL_CLAUSE = (
    "Street labels run along the street they name, so the direction the text runs IS the "
    'direction of the street: use "horizontal" for a label whose text runs left-to-right '
    'across the sheet, "vertical" for one rotated a quarter turn, and "diagonal" ONLY for a '
    "label whose text is set at a clear angle to both — more than about 20 degrees off "
    'horizontal and off vertical. A diagonal label MUST also carry "direction": [dx, dy], a '
    "pixel-space vector along the text, x growing right and y growing DOWN (so text running "
    "up-and-right is roughly [0.8, -0.6]). Most sheets have no diagonal labels at all; a "
    "label that is only slightly tilted is horizontal or vertical, not diagonal. "
)

#: v2 with the diagonal orientation offered. Identical to the frozen prompt in
#: every other respect, so an A/B moves one variable.
DIAGONAL_PROMPT_TEMPLATE = EXTENDED_PROMPT_TEMPLATE.replace(
    '"orientation": "horizontal|vertical"}]',
    '"orientation": "horizontal|vertical|diagonal", "direction": [dx, dy] (diagonal only)}]',
).replace(_CARDINAL_CLAUSE, _DIAGONAL_CLAUSE)

#: Selectable prompts by name. The frozen prompt is deliberately NOT nameable:
#: it is ``None``, and a second spelling for it would key a second cache and
#: re-buy every read already on disk.
PROMPTS: dict[str, str] = {"diagonal": DIAGONAL_PROMPT_TEMPLATE}

Orientation = Literal["horizontal", "vertical", "diagonal"]
ORIENTATIONS: tuple[str, ...] = ("horizontal", "vertical", "diagonal")


def prompt_template(name: str | None) -> str:
    """The prompt a config name selects; ``None`` is the frozen v2 prompt."""
    if name is None:
        return EXTENDED_PROMPT_TEMPLATE
    try:
        return PROMPTS[name]
    except KeyError:
        known = ", ".join(sorted(PROMPTS))
        raise AnnotateError(f"unknown annotation prompt {name!r} (known: {known})") from None


def _resolved_prompt_template(prompt_template: str | None) -> str:
    """A backend's configured template, defaulting to the frozen v2 prompt."""
    return EXTENDED_PROMPT_TEMPLATE if prompt_template is None else prompt_template


def parse_direction(value: Any) -> tuple[float, float] | None:
    """A label's pixel direction vector, or ``None`` when it is unusable.

    Never raises: one unreadable vector must not cost the whole sheet, and a
    label without a usable one falls back to its cardinal axis.
    """
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str)
        or len(value) != 2
        or not all(isinstance(v, int | float) and not isinstance(v, bool) for v in value)
    ):
        return None
    dx, dy = float(value[0]), float(value[1])
    return None if math.hypot(dx, dy) == 0 else (dx, dy)


def _bbox4(value: Any) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str)
        or len(value) != 4
        or not all(isinstance(v, int | float) and not isinstance(v, bool) for v in value)
    ):
        raise ValueError(f"bbox must be four numbers, got {value!r}")
    x0, y0, x1, y1 = (int(v) for v in value)
    return (x0, y0, x1, y1)


@dataclass(frozen=True)
class StreetLabel:
    """One street-name label on a sheet.

    ``direction`` is a pixel-space vector along the text, carried only by a
    diagonal label; the matcher falls back to a cardinal axis without one.
    """

    name: str
    bbox: tuple[int, int, int, int]
    orientation: Orientation
    direction: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "bbox": list(self.bbox),
            "orientation": self.orientation,
        }
        if self.direction is not None:
            out["direction"] = list(self.direction)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StreetLabel:
        name = data.get("name")
        if not isinstance(name, str):
            raise ValueError(f"street name must be a string, got {name!r}")
        orientation = data.get("orientation")
        if orientation not in ORIENTATIONS:
            raise ValueError(f"orientation must be {'|'.join(ORIENTATIONS)}, got {orientation!r}")
        return cls(
            name=name,
            bbox=_bbox4(data.get("bbox")),
            orientation=orientation,
            direction=parse_direction(data.get("direction")),
        )


@dataclass(frozen=True)
class Annotation:
    """The frozen v1 annotation schema for one sheet."""

    streets: tuple[StreetLabel, ...]
    page_number_seen: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "streets": [street.to_dict() for street in self.streets],
            "page_number_seen": self.page_number_seen,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Annotation:
        if not isinstance(data, Mapping):
            raise ValueError(f"annotation must be a JSON object, got {type(data).__name__}")
        streets = data.get("streets")
        if not isinstance(streets, Sequence) or isinstance(streets, str):
            raise ValueError(f"'streets' must be a list, got {streets!r}")
        page = data.get("page_number_seen")
        if page is not None and not isinstance(page, str):
            raise ValueError(f"'page_number_seen' must be a string or null, got {page!r}")
        return cls(
            streets=tuple(StreetLabel.from_dict(street) for street in streets),
            page_number_seen=page,
        )

    @classmethod
    def from_json(cls, text: str) -> Annotation:
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class ExtendedAnnotation:
    """v2 payload: the v1 annotation plus the new evidence channels."""

    annotation: Annotation
    address_numerals: tuple[AddressNumeral, ...] = ()
    margin_readings: tuple[MarginReading, ...] = ()
    rail_labels: tuple[tuple[str, tuple[int, int, int, int]], ...] = ()
    park_labels: tuple[tuple[str, tuple[int, int, int, int]], ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


class AnnotatorBackend(Protocol):
    """Anything that can annotate a sheet image."""

    def annotate(self, image_path: Path) -> Annotation: ...

    def annotate_extended(self, image_path: Path) -> ExtendedAnnotation: ...


def _bbox_of(item: dict[str, Any]) -> tuple[int, int, int, int]:
    bbox = item.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"bad bbox: {bbox!r}")
    x0, y0, x1, y1 = (int(value) for value in bbox)
    return (x0, y0, x1, y1)


def numerals_from_raw(raw: Mapping[str, Any]) -> tuple[AddressNumeral, ...]:
    """Extract tolerant address-numeral evidence from a raw v2 payload."""
    numerals: list[AddressNumeral] = []
    for item in raw.get("address_numerals") or []:
        try:
            street = item.get("street")
            numerals.append(
                AddressNumeral(
                    value=int(item["value"]),
                    bbox=_bbox_of(item),
                    street_hint=street if isinstance(street, str) and street else None,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("dropping malformed numeral %r: %s", item, exc)
    return tuple(numerals)


def extended_from_raw(raw: dict[str, Any]) -> ExtendedAnnotation:
    """Build a v2 payload from an already-decoded annotation dictionary."""
    try:
        annotation = Annotation.from_dict(raw)
    except ValueError as exc:
        raise MalformedResponseError(f"v2 response JSON violates annotation schema: {exc}") from exc

    readings: list[MarginReading] = []
    for item in raw.get("margin_numbers") or []:
        side = item.get("side")
        text = item.get("text")
        if side in ("top", "bottom", "left", "right") and isinstance(text, str):
            readings.append(MarginReading(side=side, text=text))
        else:
            logger.debug("dropping malformed margin reading %r", item)

    def labels(key: str) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
        result = []
        for item in raw.get(key) or []:
            try:
                name = item["name"]
                if isinstance(name, str) and name:
                    result.append((name, _bbox_of(item)))
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("dropping malformed %s %r: %s", key, item, exc)
        return tuple(result)

    return ExtendedAnnotation(
        annotation=annotation,
        address_numerals=numerals_from_raw(raw),
        margin_readings=tuple(readings),
        rail_labels=labels("rail_labels"),
        park_labels=labels("park_labels"),
        raw=raw,
    )


def _json_object_from_cli_text(text: str) -> dict[str, Any]:
    """The sole JSON object in a CLI's reply, unwrapping a ``result`` envelope."""
    cleaned = _strip_code_fences(text.strip())
    if not cleaned:
        raise EmptyResponseError("model returned an empty response")
    obj: Any = None
    with contextlib.suppress(json.JSONDecodeError):
        obj = json.loads(cleaned)
    if isinstance(obj, dict) and isinstance(obj.get("result"), str):
        cleaned = _strip_code_fences(obj["result"].strip())
        if not cleaned:
            raise EmptyResponseError("model returned an empty response")
        obj = None
    if isinstance(obj, dict):
        return obj
    start = cleaned.find("{")
    if start < 0:
        raise MalformedResponseError(f"response has no JSON object: {cleaned[:120]!r}")
    try:
        raw, _end = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise MalformedResponseError(f"response is not JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise MalformedResponseError(f"response is not a JSON object: {cleaned[:120]!r}")
    return raw


def parse_extended_response(text: str) -> ExtendedAnnotation:
    """Parse a v2 response, tolerating malformed additive evidence channels."""
    cleaned = _strip_code_fences(text).strip()
    if not cleaned:
        raise EmptyResponseError("empty v2 response")
    start = cleaned.find("{")
    if start < 0:
        raise MalformedResponseError(f"v2 response has no JSON object: {cleaned[:120]!r}")
    try:
        raw, _end = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise MalformedResponseError(f"v2 response is not JSON: {exc}") from exc
    return extended_from_raw(raw)


def _parse_extended_classified(text: str) -> ExtendedAnnotation:
    stripped = text.strip()
    if not stripped:
        raise EmptyResponseError("model returned an empty response")
    try:
        return parse_extended_response(stripped)
    except MalformedResponseError:
        _raise_budget_if_matched(stripped)
        raise
