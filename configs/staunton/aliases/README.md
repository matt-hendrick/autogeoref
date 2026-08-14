# Aliases for Staunton

Historical street names that differ from modern ones go here, one JSON file
per volume, named `aliases-<volume-id>.json` — for example
`aliases-sanborn02165_007.json`:

```json
{
  "_note": "keys starting with an underscore are comments",
  "OLD NAME": "MODERN NAME"
}
```

The directory is empty because the volume runs well without one, not because it
has nothing to gain. 87% of its street reads (227 of 260) already resolve
against modern OpenStreetMap, which is what makes it a good first config: a
first run should exercise the matcher rather than teach the alias system.

There is a real table to write here, and the volume's own printed index is the
answer key. Most of the time the sheets label a renamed street with both names —
`LAUREL (VIRGINIA)`, `WOOD (WALL)`, `MADISON (ASHLAND)` — so the modern half
matches and the index's "see" line is only a convenience for a reader who knows
the older one. But not every sheet prints both, and the annotator often reads
the two halves as separate labels, so the old name arrives on its own and
matches nothing. `VIRGINIA`, `EDWARD`, `ASHLAND`, `S. ANNA`, `WOOD WALL`,
`HILLSBORO RD.` and the bare parenthesised forms account for **13 of the 33
unmatched reads**, and the index names the modern street for every one:
*Virginia, see Laurel*; *Edward, see Walnut*; *Ashland, see Madison*; *Anna, see
Maple*; *Wall, see Wood*; *Short, see Wabash*; *Washington, see N. Easton*;
*Hillsboro Road, see E. Main*.

Writing them would lift the match rate to roughly 92%. It would not change which
sheets place — the one flagged detail sheet reads `E. MAIN` separately and fails
on candidate geometry, not on a name — so it is left as the exercise
`docs/ADDING-A-CITY.md` §5 describes.

The rest of the misses are not aliases and no table fixes them: model misreads
(`N. DEEN` for N. Deneen, `HOUSTON` for Huston), subdivision and owner names the
annotator picked up as streets (`J. E. SOUTHWICK'S`, `MRS. E. KING'S`), and
`SIOUX RD`, which the sheet itself prints as *"SIOUX R'D — ARBITRARY"* — a
surveyor's invented name that exists in no modern source.
