# Aliases for Crystal Lake

Historical street names that differ from modern ones go here, one JSON file
per volume, named `aliases-<volume-id>.json` — for example
`aliases-sanborn01810_007.json`:

```json
{
  "_note": "keys starting with an underscore are comments",
  "OLD NAME": "MODERN NAME"
}
```

The directory is empty on purpose. A missing file means "no aliases", which is
the honest state before any rename has been curated, and 19 of the 19 street
names sampled off this volume's sheets are still current in OpenStreetMap. See
`docs/ADDING-A-CITY.md` §5 for how to find the ones that are not.
