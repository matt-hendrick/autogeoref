"""Persisted and configured volume scale/rotation resolution."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config.model import VolumeConfig
    from .paths import VolumePaths


def persisted_constants(paths: VolumePaths) -> tuple[float, float] | None:
    """``(scale_m_per_px, rotation_deg)`` from persisted constants, or None."""
    if not paths.constants.exists():
        return None
    recorded = json.loads(paths.constants.read_text())
    scale, rotation = recorded.get("scale_m_per_px"), recorded.get("rotation_deg")
    if scale is None or rotation is None:
        return None
    return float(scale), float(rotation)


def resolve_constants(
    paths: VolumePaths, vol: VolumeConfig, *, prefer_persisted: bool = False
) -> tuple[float, float] | None:
    """Resolve config pins and persisted constants with the caller's precedence."""
    pinned = None
    if vol.scale_m_per_px is not None and vol.rotation_deg is not None:
        pinned = (vol.scale_m_per_px, vol.rotation_deg)
    persisted = persisted_constants(paths)
    first, second = (persisted, pinned) if prefer_persisted else (pinned, persisted)
    return first if first is not None else second
