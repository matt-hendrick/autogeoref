"""The review op algebra: one op's world-space linear map.

Its own module because :mod:`.sidecars` validates ops and
:mod:`.materialize` composes them; sharing a base keeps those two off each
other.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..errors import ReviewError


def op_linear_offset(op: Mapping[str, Any]) -> tuple[Any, Any]:
    """``(L, t)`` of one op's world map ``world' = L @ world + t``.

    Raises :class:`~autogeoref.errors.ReviewError` on an unknown op type or a
    non-positive scale factor, which is what makes it double as the validator
    :func:`.sidecars.sidecar_from_dict` runs over a loaded op log.
    """
    kind = op.get("type")
    if kind == "translate":
        return np.eye(2), np.array([float(op["dx_m"]), float(op["dy_m"])])
    if kind == "scale":
        factor = float(op["factor"])
        if not (factor > 0 and math.isfinite(factor)):
            raise ReviewError(f"scale factor must be positive/finite, got {factor}")
        linear = np.eye(2) * factor
    elif kind == "rotate":
        radians = math.radians(float(op["deg"]))
        linear = np.array(
            [[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]]
        )
    else:
        raise ReviewError(f"unknown op type {kind!r}")
    cx, cy = (float(value) for value in op["center_3857"])
    center = np.array([cx, cy])
    return linear, center - linear @ center
