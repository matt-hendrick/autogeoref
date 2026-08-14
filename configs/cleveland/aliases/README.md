# Aliases for Cleveland

Historical street names that differ from modern ones go here, one JSON file
per volume, named `aliases-<volume-id>.json` — for example
`aliases-sanborn06648_072.json`:

```json
{
  "_note": "keys starting with an underscore are comments",
  "OLD NAME": "MODERN NAME"
}
```

There is no table here, because this city's turned out to do nothing, and the
reason is worth knowing before writing another.

Cleveland prints its quadrant after the street type — `MILES AV. S.E.` — where
the normalizer originally expected a direction in front. So the volume was given
a mechanically generated table of 115 entries, one per spelling, each pointing
at the same name with the type and quadrant taken off. The normalizer has since
learned to strip a trailing quadrant itself, which made every one of those
entries a restatement of what it already did.

Measured before the file was removed: over all 921 street reads on disk, all
11,159 centerline features, and the volume's own bounded index, **not one key
differed** with the table applied or absent. 21 of the 115 pointed at names with
no geometry anywhere in the city — misreadings of streets the table already
covered under their correct spelling.

The lesson for a new city: **reach for the normalizer before the alias table.**
A table that restates a spelling rule is dead weight the day the rule lands, and
it hides the streets that genuinely were renamed. `autogeoref alias-sweep`
proposes entries from a documented rename source, which is the kind this
directory is for; `scripts/validate_alias_files.py` reports any entry that
cannot change an outcome, so a table that goes stale this way says so.

See `docs/ADDING-A-CITY.md` §5.
