"""The value coercions both the loader and the escalation resolver need.

Everything else in `load` is exclusively the loader's; these two are not, and a
second copy of either would be a second answer to the same question.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .model import ConfigError

if TYPE_CHECKING:
    from pathlib import Path


def require_str_list(
    val: Any, message: str, *, allow_empty_items: bool = False, stripped: bool = False
) -> None:
    """Reject anything but a list of strings; the caller owns the message text."""
    if isinstance(val, list) and all(
        isinstance(item, str) and bool(allow_empty_items or (item.strip() if stripped else item))
        for item in val
    ):
        return
    raise ConfigError(message)


def model_variant(table: dict[str, Any], key: str, where: str, path: Path) -> str | None:
    val = table.get(key)
    if val is None:
        return None
    if not isinstance(val, str):
        raise ConfigError(f"{path}: {where} {key} must be a variant string")
    return val or None  # "" explicitly requests the provider's default effort
