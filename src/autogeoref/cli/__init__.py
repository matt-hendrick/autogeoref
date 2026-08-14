"""The command-line surface, one module per command family.

Each module owns both its argparse declaration and its handler, because those
are what change together. Nothing is re-exported here.

- `run` — `run`, `prep`: the placement pipeline and the pre-spend look at it.
- `report` — `report`, `score`, `allmaps`, `status`, `dashboard`: read-only views.
- `queue` — `queue`, which delegates to the console.
- `data` — `era`, `alias-sweep`, `discover`: city-data production.
- `viewer` — `viewer-manifest`, `publish`, `deploy-bundle`.
- `review` — `review`: render, apply, or serve the localhost UI.
- `parser` — the shared flag groups and the one `build_parser`.
- `entry` — the console-script entry point; named `entry` so that
  `from autogeoref.cli import main` fails instead of binding a module.
"""
