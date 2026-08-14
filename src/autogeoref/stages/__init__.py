"""The pipeline's stages: one module each, all reading and writing under `VolumePaths`.

Every stage is idempotent and resumable. Four of the five wrap the pure core
module of the same name — `stages.seam` is the file-target stage, `seam` is the
solve — and `match` wraps `volume.match_sheet`.

- `match` — fit each sheet's street reads; persist the volume constants.
- `rescue` — translation-only placement of what match rejected.
- `corroborate` — reinstate a revoked rescue its neighbours vouch for.
- `seam` — one joint translation solve over the committed sheets.
- `report` — assemble `report.json` and `report.md`.
"""
