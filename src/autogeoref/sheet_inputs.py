"""Source-frame matcher inputs built from cached annotations and manifests."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from .frames import rotate_annotation
from .volume import SheetInput

if TYPE_CHECKING:
    from .paths import VolumePaths

logger = logging.getLogger(__name__)


def annotation_in_source_frame(ann: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    """Map an annotation on an upright small back to the source scan frame."""
    rotation = int(info.get("rotation_applied", 0))
    if not rotation:
        return ann
    upright_size = (int(info["small_size"][0]), int(info["small_size"][1]))
    return rotate_annotation(ann, (360 - rotation) % 360, upright_size)


#: Annotation keys whose entries carry a `bbox` a matcher stage unpacks.
_LABEL_KEYS = ("streets", "rail_labels", "park_labels", "address_numerals")


def _bbox_is_four_numbers(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    bbox = item.get("bbox")
    return (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(v, int | float) and not isinstance(v, bool) for v in bbox)
    )


def drop_malformed_labels(page: str, ann: dict[str, Any]) -> dict[str, Any]:
    """Drop labels whose ``bbox`` is not four numbers.

    A model occasionally wraps a bbox one level too deep. Consumers unpack it
    positionally stages later, so an unchecked one aborts the run with a
    ValueError naming neither page nor volume.

    Dropping matches the typed schema for rail, park and numeral labels. For a
    STREET the schema is stricter and refuses the whole read; this is the
    softer landing for a cached annotation that predates it.
    """
    cleaned = ann
    for key in _LABEL_KEYS:
        items = ann.get(key)
        if not isinstance(items, list):
            continue
        kept = [item for item in items if _bbox_is_four_numbers(item)]
        if len(kept) == len(items):
            continue
        if cleaned is ann:
            cleaned = dict(ann)
        cleaned[key] = kept
        logger.warning(
            "p%s: dropped %d %s with a malformed bbox", page, len(items) - len(kept), key
        )
    return cleaned


def sheet_input_from(page: str, ann: dict[str, Any], info: dict[str, Any]) -> SheetInput:
    """Build one matcher input, applying source-frame and legacy-scale rules."""
    ann = annotation_in_source_frame(drop_malformed_labels(page, ann), info)
    return SheetInput(
        page=page,
        annotation=ann,
        full_size=(float(info["full_size"][0]), float(info["full_size"][1])),
        scale=float(ann.get("scale", info["scale"])),
    )


def load_sheet_inputs(paths: VolumePaths) -> list[SheetInput]:
    """Load cached annotations that have a matching manifest entry."""
    manifest = json.loads(paths.manifest.read_text())
    sheets: list[SheetInput] = []
    for ann_path in sorted(paths.annotations.glob("p*.json")):
        # cache records, not matcher inputs — the bare p<N>.json is the read
        if ".annotation." in ann_path.name:
            continue
        page = ann_path.stem.removeprefix("p")
        info = manifest.get(f"p{page}")
        if info is None:
            logger.warning("p%s: annotation without manifest entry, skipping", page)
            continue
        sheets.append(sheet_input_from(page, json.loads(ann_path.read_text()), info))
    return sheets
