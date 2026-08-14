"""IIIF Georeference Annotation (Allmaps) writer.

A pure serializer over data the pipeline already has — GCPs, mask, IIIF image id and
dimensions — plus :func:`export_volume`, the volume-level assembly behind
``autogeoref allmaps <volume>``. Proven end to end: an annotation built this way
renders warped and correctly placed in the Allmaps viewer.

Coordinate frame: GCP pixels here are FULL-RES pixels — the same frame as the recorded
result GCPs and, for LOC-hosted volumes, the IIIF ``info.json`` frame. A caller holding
annotation-frame ("small") pixels must rescale first.

Hosting: the Allmaps viewer proxies annotation URLs server-side, so ``localhost`` fails —
annotations must live at public URLs.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .paths import VolumePaths

logger = logging.getLogger(__name__)

#: JSON-LD context of the IIIF georeference extension.
GEOREF_CONTEXT = "http://iiif.io/api/extension/georef/1/context.json"
#: Default border-mask inset fraction (1.5% per side — the proven default).
DEFAULT_MASK_INSET = 0.015
#: Data licence of the coordinates, as a Web Annotation ``rights`` URI. Carried
#: by the page AND by every annotation on it, because a consumer that splits a
#: page into its items keeps only the item.
RIGHTS = "https://creativecommons.org/publicdomain/zero/1.0/"

#: (px, py) in full-res pixels.
PixelPoint = tuple[float, float]
#: (px, py, lng, lat): full-res pixel + WGS84 lng/lat.
GcpLngLat = tuple[float, float, float, float]


class AnnotationError(ValueError):
    """Invalid inputs for a Georeference Annotation."""


def border_mask(
    width: int, height: int, inset: float = DEFAULT_MASK_INSET
) -> list[tuple[int, int]]:
    """Rectangular resource mask inset from the sheet border (the proven default).

    Args:
        width: Full-res image width in pixels.
        height: Full-res image height in pixels.
        inset: Fraction of each dimension to inset per side.

    Returns:
        Four corner points (clockwise from top-left) in full-res pixels.
    """
    x0, y0 = round(width * inset), round(height * inset)
    x1, y1 = round(width * (1 - inset)), round(height * (1 - inset))
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _mask_svg(width: int, height: int, mask_pixels: Sequence[PixelPoint]) -> str:
    """SVG selector string for a polygon resource mask (probe-identical format)."""
    points = " ".join(f"{x},{y}" for x, y in mask_pixels)
    return f'<svg width="{width}" height="{height}"><polygon points="{points}" /></svg>'


def _min_gcps(order: int) -> int:
    """Minimum GCP count for a 2D polynomial of the given order."""
    return (order + 1) * (order + 2) // 2


def georef_annotation(
    *,
    iiif_image_id: str,
    image_width: int,
    image_height: int,
    gcps: Sequence[GcpLngLat],
    mask_pixels: Sequence[PixelPoint] | None = None,
    transformation_order: int = 1,
    annotation_id: str | None = None,
) -> dict[str, Any]:
    """Build a Georeference Annotation (AnnotationPage with one annotation).

    The body is a ``georeferencing`` FeatureCollection with a polynomial transformation and one
    Point feature per GCP; the target is a ``SpecificResource`` over an IIIF ``ImageService2``
    with an ``SvgSelector`` resource mask. ``image_width``/``image_height`` must match the
    service ``info.json``; ``gcps`` are ``(px, py, lng, lat)`` in FULL-RES pixels and WGS84,
    passed through unchanged; ``mask_pixels`` is a full-res polygon, or None to omit the
    selector so Allmaps uses the whole image. Returns a plain JSON-serializable dict, and raises
    ``AnnotationError`` on bad dimensions, too few GCPs for the order, or a degenerate mask.
    """
    if image_width <= 0 or image_height <= 0:
        raise AnnotationError(f"non-positive image size {image_width}x{image_height}")
    if transformation_order < 1:
        raise AnnotationError(f"transformation_order must be >= 1, got {transformation_order}")
    need = _min_gcps(transformation_order)
    if len(gcps) < need:
        raise AnnotationError(
            f"polynomial order {transformation_order} needs >= {need} GCPs, got {len(gcps)}"
        )
    if mask_pixels is not None and len(mask_pixels) < 3:
        raise AnnotationError(f"mask needs >= 3 points, got {len(mask_pixels)}")

    features = [
        {
            "type": "Feature",
            "properties": {"resourceCoords": [px, py]},
            "geometry": {"type": "Point", "coordinates": [lng, lat]},
        }
        for px, py, lng, lat in gcps
    ]
    target: dict[str, Any] = {
        "type": "SpecificResource",
        "source": {
            "id": iiif_image_id,
            "type": "ImageService2",
            "width": image_width,
            "height": image_height,
        },
    }
    if mask_pixels is not None:
        target["selector"] = {
            "type": "SvgSelector",
            "value": _mask_svg(image_width, image_height, mask_pixels),
        }
    item: dict[str, Any] = {"@context": GEOREF_CONTEXT}
    if annotation_id is not None:
        item["id"] = annotation_id
    item.update(
        {
            "type": "Annotation",
            "motivation": "georeferencing",
            "rights": RIGHTS,
            "target": target,
            "body": {
                "type": "FeatureCollection",
                "transformation": {
                    "type": "polynomial",
                    "options": {"order": transformation_order},
                },
                "features": features,
            },
        }
    )
    logger.debug("built georef annotation: %d GCPs, order %d", len(gcps), transformation_order)
    return {
        "@context": GEOREF_CONTEXT,
        "type": "AnnotationPage",
        "rights": RIGHTS,
        "items": [item],
    }


def export_volume(
    paths: VolumePaths,
    *,
    page_services: Mapping[str, str],
    transformation_order: int = 1,
    mask_inset: float = DEFAULT_MASK_INSET,
) -> dict[str, Any]:
    """AnnotationPage with one georeferencing Annotation per committed sheet.

    Reads a placed volume's recorded results — GCPs are already full-res and seam-adjusted in
    place — and its sheet manifest. ``page_services`` maps lower-cased page ids to IIIF image
    service ids. Read-only. Raises ``AnnotationError`` when the manifest is missing, when a
    committed sheet lacks a manifest entry or a service id (an export must never silently drop
    a placed sheet), or when nothing is committed.
    """
    from .paths import iter_results
    from .slugs import page_sort_key
    from .volume import is_committed
    from .warp import gcps_from_feature_collection

    if not paths.manifest.is_file():
        raise AnnotationError(f"no sheet manifest at {paths.manifest}")
    manifest = json.loads(paths.manifest.read_text())

    items: list[dict[str, Any]] = []
    missing: list[str] = []
    for page, record, _path in iter_results(paths, sort_key=lambda p: page_sort_key(p.stem)):
        if not is_committed(record):
            continue
        entry = manifest.get(f"p{page}")
        service = page_services.get(page.lower())
        if not isinstance(entry, dict) or service is None:
            missing.append(page)
            continue
        width, height = entry["full_size"]
        sheet = georef_annotation(
            iiif_image_id=service,
            image_width=width,
            image_height=height,
            gcps=gcps_from_feature_collection(record["gcps_geojson"]),
            mask_pixels=border_mask(width, height, mask_inset),
            transformation_order=transformation_order,
        )
        items.extend(sheet["items"])
    if missing:
        raise AnnotationError(
            "committed sheets without a manifest entry or IIIF service id: " + ", ".join(missing)
        )
    if not items:
        raise AnnotationError("no committed results to export")
    logger.info("exported %d committed sheets as georef annotations", len(items))
    return {
        "@context": GEOREF_CONTEXT,
        "type": "AnnotationPage",
        "rights": RIGHTS,
        "items": items,
    }
