# autogeoref exports

Georeferencing for scanned Sanborn fire-insurance sheets, placed automatically
and committed only where the pipeline's evidence gates accepted them. Flagged
sheets are not here.

Nine sheets are the exception, and they carry it on the record: a
`reviewer_review` block and the status `OK (reviewer-verified)` mean a person
placed or corrected that sheet by hand rather than a gate accepting it. Filter
on `status` to take only the automated placements.

## Licence

These coordinates are released under
[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/), a
public domain dedication with no attribution required. The same statement is machine
readable as the `rights` property of every annotation and annotation page.

The licence covers this tree only. It is not a licence to the scans, which are
the Library of Congress's, nor to any reference dataset named below.

## What is here

One directory per volume:

- `gcps/p<N>.json` is the per-sheet result record. `gcps_geojson` holds the
  ground control points as full-resolution image pixels paired with WGS84
  coordinates, seam-adjusted in place.
- `allmaps.json` is an IIIF Georeference AnnotationPage. It targets the Library
  of Congress's own IIIF image services, so a viewer such as
  [Allmaps](https://allmaps.org/) warps the original scans straight from LOC
  with no imagery from this project. Note that the Allmaps viewer fetches
  annotation URLs server-side, so a `localhost` URL will not load.

## Where the coordinates come from

Most of the published points are street junctions computed against whatever open
street geometry the city publishes. Chicago's centerline file is released under
the MIT Licence, and the city's data portal asks that modifications be
disclaimed: these coordinates are modified from that file, and the city does not
vouch for them.

**Credit © OpenStreetMap contributors if you reuse the points derived from it.**
Those are the rail crossings everywhere, plus the street junctions in any city
with no municipal centerline file. No OpenStreetMap data is redistributed here,
only coordinates computed against it.

## Regenerating

This tree is rewritten by `autogeoref publish` from each volume's work tree, so
it cannot drift from what is served, and this file is rewritten with it. Editing
anything here by hand will be overwritten. Serialization is deterministic, so
re-publishing an unchanged volume produces no diff.
