"""A city TOML, read once into frozen dataclasses. Nothing is re-exported here.

- `model` — `CityConfig`, `VolumeConfig`, `EscalationResolution`, `ConfigError`.
- `fields` — the two value coercions `escalation` and `load` both need.
- `escalation` — the ladder a table declares, resolved against the city's.
- `load` — `load_city_config`, and the helpers only it uses.

The direction is `load` -> `escalation` -> `fields` -> `model`, and only that
way: the loader's helpers belong to it alone, and a dataclass reaching back for
one would be the cycle this layout exists to prevent.
"""
