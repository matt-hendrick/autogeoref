"""Dependency-light views of persisted placement records."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .rescue import is_rescue_model_corner
from .seam import shift_gcps_geojson


def pinned_orientation(record: Mapping[str, Any]) -> bool:
    """True when this placement's rotation and scale were PINNED, not fitted.

    Usually the volume's; a sheet drawn on its own grid carries its own instead
    (``rescue_pin_rotation_deg``). Either way a provenance fact, not a defect: the
    sheet's rotation error against its neighbours is unmeasured, a fit residual
    over its GCPs is the model measured against itself, and ``seam.build_ties``
    skips its synthetic corners so the seam solve cannot correct it. Read off the
    recorded GCPs through :func:`rescue.is_rescue_model_corner`, not a stamped
    field, so it holds for records written before this predicate existed.
    """
    features = (record.get("gcps_geojson") or {}).get("features") or []
    return any(is_rescue_model_corner(ft) for ft in features)


def preseam_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with a recorded seam shift reversed when one is present."""
    seam_adjusted = record.get("seam_adjusted")
    if not seam_adjusted:
        return record
    previous = copy.deepcopy(record)
    shift_gcps_geojson(
        previous.get("gcps_geojson") or {},
        -float(seam_adjusted["dx_m"]),
        -float(seam_adjusted["dy_m"]),
    )
    return previous
