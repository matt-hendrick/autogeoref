"""The escalation ladder a table declares, resolved against the city default.

Inheritance is presence-based, and a volume's escalation keys are one unit: it
can always turn the default-on stage off, or name one cheaper tier, without the
city ladder coming back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .fields import model_variant, require_str_list
from .model import ConfigError, EscalationResolution

if TYPE_CHECKING:
    from pathlib import Path


def _model_ladder(table: dict[str, Any], where: str, path: Path) -> tuple[str, ...]:
    val = table.get("escalation_models")
    if val is None:
        return ()
    require_str_list(val, f"{path}: {where} escalation_models must be a list of model names")
    return tuple(val)


def _escalation_model(table: dict[str, Any], where: str, path: Path) -> str | None:
    val = table.get("escalation_model")
    if val is None:
        return None
    if not isinstance(val, str):
        raise ConfigError(f"{path}: {where} escalation_model must be a model name string")
    return val


def model_variants(
    table: dict[str, Any], key: str, models: tuple[str, ...], where: str, path: Path
) -> tuple[str | None, ...]:
    val = table.get(key)
    if val is None:
        return (None,) * len(models)
    if (
        not isinstance(val, list)
        or len(val) != len(models)
        or not all(isinstance(variant, str) for variant in val)
    ):
        raise ConfigError(
            f"{path}: {where} {key} must be one variant string per escalation_models "
            'entry ("" = provider default)'
        )
    return tuple(variant or None for variant in val)


def _require_variant_pairing(table: dict[str, Any], where: str, path: Path) -> None:
    if "escalation_variant" in table and "escalation_model" not in table:
        raise ConfigError(f"{path}: {where} escalation_variant requires escalation_model")
    if "escalation_variants" in table and "escalation_models" not in table:
        raise ConfigError(f"{path}: {where} escalation_variants requires escalation_models")


def resolve_escalation(
    table: dict[str, Any], where: str, path: Path, city: EscalationResolution | None = None
) -> EscalationResolution:
    """Parse a table's escalation keys, resolved against the city default.

    The one home of the inheritance rule. Presence-based, NOT ``or``, and the
    volume's escalation keys are ONE unit: naming either inherits NOTHING from
    the city, so a volume can always turn the default-ON stage off or name one
    cheaper tier without a city-wide ladder resurrecting it. Both empty
    spellings are an explicit off. ``city=None`` parses the city block itself,
    which has nothing to inherit. Validation order differs between the two
    branches and both are pinned.
    """
    model = _escalation_model(table, where, path)
    if city is None:
        models = _model_ladder(table, where, path)
        _require_variant_pairing(table, where, path)
        return EscalationResolution(
            model=model,
            variant=model_variant(table, "escalation_variant", where, path),
            models=models,
            variants=model_variants(table, "escalation_variants", models, where, path),
        )
    _require_variant_pairing(table, where, path)
    names_a_ladder = "escalation_models" in table or "escalation_model" in table
    models = _model_ladder(table, where, path)
    variants = model_variants(table, "escalation_variants", models, where, path)
    variant = model_variant(table, "escalation_variant", where, path)
    return EscalationResolution(
        model=(model or None) if names_a_ladder else city.model,
        variant=variant if names_a_ladder else city.variant,
        models=(
            models
            if "escalation_models" in table
            else (model,)
            if model
            else ()
            if names_a_ladder
            else city.models
        ),
        variants=(
            variants
            if "escalation_models" in table
            else (variant,)
            if model
            else ()
            if names_a_ladder
            else city.variants
        ),
    )
