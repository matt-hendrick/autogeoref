"""The back half: committed placements to a served archive. Nothing is re-exported.

- `layers` — which sheets a bake may touch, read off the committed records.
- `warp` — those sheets to COGs.
- `masks` — detect the drawn content, heal the quilt, write the cutlines.
- `mosaic` — pack the warped, masked sheets into one raster.
- `tiles` — pack the mosaic as PMTiles.

Each is a file target the runner sequences; none calls another, and only
`warp` and `masks` share anything, which is `layers`.
"""
