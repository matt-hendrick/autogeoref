"""Shared per-run inputs, parsed at most once each.

:class:`RunContext` is the domain object one ``run`` invocation threads through
the stage plan (:mod:`autogeoref.runplan`). It lives here, not in the CLI,
so the pipeline never imports the CLI module to name its own input type.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

from .run_inputs import build_index

if TYPE_CHECKING:
    from .addresses import RenumberingTable
    from .centerlines import CenterlineIndex
    from .config.model import CityConfig, VolumeConfig
    from .era import AddressEra
    from .paths import VolumePaths

logger = logging.getLogger(__name__)


@dataclass
class RunContext:
    """One ``run`` invocation's shared inputs, parsed at most once each.

    ``cached_property`` replaces the hand-rolled caches the old ``_cmd_run``
    carried (closure state dicts plus a centerline keep-alive condition that
    had to enumerate every consuming stage and ended in a ``RuntimeError``
    tripwire): the citywide centerline GeoJSON is parsed once and simply
    stays cached for the run's lifetime — tens of MB in one bounded process.
    """

    args: argparse.Namespace
    city: CityConfig
    vol: VolumeConfig
    paths: VolumePaths
    bounds: tuple[float, float, float, float]
    #: filled by the street-index stage, read by rescue (log-only check)
    index_windows: dict[str, Any] | None = None

    @cached_property
    def centerline_features(self) -> list[dict[str, Any]]:
        if (
            self.args.dry_run
            and self.city.centerlines_from_osm
            and not self.city.centerlines_path.exists()
        ):
            # a dry run spends NOTHING — no model budget and no network
            # fetch — so an absent OSM cache is fine here (a real run fetches)
            logger.info(
                "dry-run: OSM centerline cache %s absent; a real run will fetch it",
                self.city.centerlines_path,
            )
            return []
        features: list[dict[str, Any]] = json.loads(self.city.centerlines_path.read_text())[
            "features"
        ]
        return features

    @cached_property
    def index(self) -> CenterlineIndex:
        index = build_index(self.city, self.vol, self.bounds, features=self.centerline_features)
        logger.info("%s: %d centerline streets in bounds", self.vol.identifier, len(index.by_name))
        return index

    @cached_property
    def clipped_features(self) -> list[dict[str, Any]]:
        """Volume-clipped features, shared by street-index and verified-accept."""
        from .geometry import clip_features_4326

        return clip_features_4326(self.centerline_features, self.bounds)

    @cached_property
    def renumbering(self) -> RenumberingTable:
        # volumes predating the city's renumbering convert printed numbers
        # through the published table when the city ships one; otherwise
        # the empty table (abstain). A volume may carry its own table: a
        # district renumbered on a different date needs a different book
        # (a district renumbered on its own date) and must NOT convert through the city's.
        from .addresses import EMPTY_RENUMBERING, RenumberingTable

        path = self.vol.renumbering_table_path or self.city.renumbering_table_path
        if path is not None and path.exists():
            return RenumberingTable.from_json(path)
        return EMPTY_RENUMBERING

    @cached_property
    def city_renumbered(self) -> bool:
        """Did this city renumber? It said so by shipping a table."""
        return self.city.renumbering_table_path is not None

    @cached_property
    def era(self) -> AddressEra:
        # THE one config->era mapping. Undeclared is MODERN; in a city that
        # renumbered, era_from_config warns rather than guessing (G2 finding 3:
        # undeclared must never mean renumbered — that converts MODERN numbers
        # ~19 blocks away).
        from .era import era_from_config

        return era_from_config(
            self.vol.addresses_modern,
            volume=self.vol.identifier,
            city_renumbered=self.city_renumbered,
        )

    @cached_property
    def index_renumbering(self) -> RenumberingTable | None:
        from .addresses import EMPTY_RENUMBERING

        era = self.era
        if era == "modern":
            return None  # printed numbers ARE modern numbers
        if era == "renumbered":
            return self.renumbering
        return EMPTY_RENUMBERING  # unknown era: every entry abstains

    @cached_property
    def rail_index(self) -> Any | None:
        path = self.city.rail_geojson_path
        gazetteer_path = self.city.rail_gazetteer_path
        if path is None:
            return None
        from .rail import RailIndex

        if gazetteer_path is None:
            raise RuntimeError("rail_geojson configured without rail_gazetteer")
        rail = RailIndex.from_json(path, gazetteer_path=gazetteer_path)
        logger.info(
            "rail reference: %d groups, %d gazetteer rows",
            len(rail.groups),
            len(rail.gazetteer or {}),
        )
        return rail
