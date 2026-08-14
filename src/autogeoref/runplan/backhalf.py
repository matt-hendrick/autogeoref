"""Back-half stage construction: warp -> mask -> mosaic -> tile.

Serving only. Nothing here places a sheet; every stage consumes records that
are already committed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..bake.masks import stage_masks
from ..bake.mosaic import stage_mosaic
from ..bake.tiles import stage_tiles, write_tile_params
from ..bake.warp import stage_warp
from ..dag import Stage

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ..paths import VolumePaths


class BackHalfArgs(Protocol):
    """CLI surface required to bake already-committed records."""

    volume: str
    force: bool
    dry_run: bool
    max_zoom: int | None


def build_back_half_stages(
    args: BackHalfArgs,
    paths: VolumePaths,
    *,
    content_masks: bool = False,
    content_mask_exempt: tuple[str, ...] = (),
    overview_pages: tuple[str, ...] = (),
    page_scale_multiples: Mapping[str, float] | None = None,
    enabled: Callable[[], bool] = lambda: True,
) -> list[Stage]:
    """Build the committed-record serving stages used by both run modes.

    ``content_masks`` / ``content_mask_exempt`` / ``overview_pages`` /
    ``page_scale_multiples`` are the volume's declared mask style, overview
    class, and per-page scale declarations (``VolumeConfig``) — the ONLY city
    configuration the back half consumes (both callers resolve them from the
    same --city config; everything else stays structural so --warp-only keeps
    clear of placement dependencies). The multiples exist here solely for mask
    QA's ``outside_window`` re-check; they influence no mask geometry.
    """
    return [
        Stage(
            name="warp",
            run=lambda: stage_warp(paths, args.volume, force=args.force),
            enabled=enabled,
        ),
        Stage(
            name="mask",
            run=lambda: stage_masks(
                paths,
                args.volume,
                content_masks=content_masks,
                content_mask_exempt=content_mask_exempt,
                overview_pages=overview_pages,
                page_scale_multiples=page_scale_multiples,
            ),
            enabled=enabled,
        ),
        Stage(
            name="mosaic",
            run=lambda: stage_mosaic(paths, overview_pages=overview_pages),
            enabled=enabled,
        ),
        Stage(
            name="tile-params",
            run=lambda: write_tile_params(paths, max_zoom=args.max_zoom),
            outputs=[paths.root / "tiles-params.json"],
            always_run=True,
            enabled=enabled,
        ),
        Stage(
            name="tile",
            run=lambda: stage_tiles(paths, args.volume, max_zoom=args.max_zoom),
            inputs=[
                # mosaic-overview.tif is deliberately NOT listed: the runner
                # fails a non-fresh stage on a missing input, and the file is
                # absent on every volume without declared overview pages. The
                # partition manifest inside stage_mosaic makes a declaration
                # change rebuild mosaic.tif, so this list still goes stale on
                # every path that changes the companion.
                paths.root / "mosaic.tif",
                paths.warped / "warp-summary.json",
                paths.masks / "masks.geojson",
                paths.root / "tiles-params.json",
            ],
            outputs=[paths.root / f"{args.volume}.pmtiles"],
            enabled=enabled,
        ),
    ]
