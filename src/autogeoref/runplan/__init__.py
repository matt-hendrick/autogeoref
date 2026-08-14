"""How a run's stage list is built. Nothing is re-exported here.

- `backhalf` — warp, mask, mosaic, tile: serving, from committed records only.
- `placement` — everything that decides where a sheet goes, and it appends the
  back half when the run asks for it.

The direction is `placement` -> `backhalf`, never the reverse: serving must
stay runnable on its own, which is what `--warp-only` is.
"""
