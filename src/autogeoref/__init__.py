"""autogeoref — auto-georeferencing for scanned Sanborn fire-insurance maps.

Core method: street-label vision annotations -> normalized-name intersection
matching against modern centerlines -> constrained RANSAC affine -> gatekeeping
-> rescue (disjoint-pair rule) -> neighbor corroboration -> seam adjustment ->
GDAL warp -> tiles. Every output sheet is either provably placed or honestly
flagged.
"""

__version__ = "0.1.0"
