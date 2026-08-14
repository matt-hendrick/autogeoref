"""Small shared validation primitives for configuration and CLI boundaries."""

from __future__ import annotations

import argparse
import math
import re
from typing import Any

MIN_TILE_ZOOM = 0
MAX_TILE_ZOOM = 30

_SAFE_VOLUME_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def volume_id(value: Any) -> str:
    """Return a queue/config-safe volume identifier or raise ``ValueError``."""
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or _SAFE_VOLUME_ID.fullmatch(value) is None
    ):
        raise ValueError("must be a non-empty safe volume identifier")
    return value


def volume_argument(value: str) -> str:
    """``argparse`` type for a volume identifier that cannot escape its root."""
    try:
        return volume_id(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def finite_number(value: Any) -> float:
    """Return a finite TOML number, rejecting bool because it is an ``int`` subclass."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError("must be a finite number")
    return float(value)


def nonnegative_int(value: str) -> int:
    """``argparse`` type for budget caps that cannot use slice semantics."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def positive_int(value: str) -> int:
    """``argparse`` type for worker and lane counts."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    """``argparse`` type for wait intervals."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def port(value: str) -> int:
    """``argparse`` type for a valid TCP port."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer from 1 through 65535") from exc
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be an integer from 1 through 65535")
    return parsed


def tile_zoom(value: str) -> int:
    """``argparse`` type for supported web-mercator tile zooms."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"must be an integer from {MIN_TILE_ZOOM} through {MAX_TILE_ZOOM}"
        ) from exc
    if not MIN_TILE_ZOOM <= parsed <= MAX_TILE_ZOOM:
        raise argparse.ArgumentTypeError(
            f"must be an integer from {MIN_TILE_ZOOM} through {MAX_TILE_ZOOM}"
        )
    return parsed
