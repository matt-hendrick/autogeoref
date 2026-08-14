# Chicago configuration

- `chicago.toml` — the city config (volumes, evidence channels, eras).
  Relative paths inside it resolve against this directory.
- `renumbering-chicago-1909.json` — old→new address conversion, citywide
  outside the Loop (1909 renumbering).
- `renumbering-chicago-1911-loop.json` — the Loop's 1911 register.
- `renumbering-chicago-loop-merged.json` — both books merged for the Loop
  volumes `_017`/`_018`, colliding old numbers removed. Rebuilt by
  `scripts/make_loop_renumbering_table.py`.
- `rail-gazetteer-chicago.json` — gazetteer anchors for rail rescue.
- `aliases/` — historical→modern street renames, one file per volume. See its
  README for the sources and what the files do and do not carry.
