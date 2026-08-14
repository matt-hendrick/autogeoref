# Aliases for Chicago

Historical street names that differ from modern ones, one JSON file per volume,
named `aliases-<volume-id>.json`. Keys starting with an underscore are
comments; each file's `_comment` records how its entries were arrived at and
which candidates were deliberately rejected.

A table is volume-scoped on purpose: `LAKE` means different streets in
different parts of the city, so a rename valid for one volume can be wrong for
its neighbour. `scripts/validate_alias_files.py` enforces the rules in
`docs/INTERNALS.md` — values must be exact index keys, no key may shadow an
in-bounds street, no chains, and no centerline may be re-keyed.

## Where the pairings come from

Each entry pairs a name printed on a Library of Congress Sanborn sheet with a
name in the City of Chicago's street centerline file. Some pairings were found
by geometric inference from this pipeline's own placements, and some by desk
research against **"Chicago Streets", compiled by William Martin, 1948**, which
the Chicago History Museum hosts at
`http://chsmedia.org/househistory/nameChanges/start.pdf`. What these files
carry is the factual pairing and nothing else from that source — none of its
prose, none of its address ranges.

Each file's own `_comment` names the route its entries were drawn from.
