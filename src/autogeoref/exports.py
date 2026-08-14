"""Researcher data exports: committed placements as reusable tracked files.

``exports/<volume>/`` is written by ``autogeoref publish``, so the tracked data can never
drift from what is served. Committed sheets only; a flagged sheet must not appear.

- ``gcps/p<N>.json``: the recorded per-sheet result records, copied verbatim
  (``gcps_geojson`` carries FULL-RES pixel GCPs, seam-adjusted in place). Never re-derived.
- ``allmaps.json``: one IIIF Georeference AnnotationPage assembled by
  :func:`autogeoref.allmaps.export_volume`, renderable from any public URL.
- ``README.md``: the tree's licence and provenance note, rewritten with it.

Serialization is deterministic (stable key and page order, no timestamps), so
re-exporting an unchanged volume is byte-identical, and a diff under ``exports/``
only ever shows a real placement change.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Mapping
from pathlib import Path

from .paths import VolumePaths, iter_results, write_if_changed

logger = logging.getLogger(__name__)

#: The tree's human-readable licence and provenance note. The authoritative copy
#: of the licence is the ``rights`` property on every annotation. This one exists
#: because people read READMEs and not JSON-LD, and it names the reference
#: sources because they do not share a rights story.
README = """# autogeoref exports

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
"""


def write_exports_readme(exports_root: Path) -> Path:
    """Write the tree's licence and provenance note, if it is not already right.

    Content is constant, so this is idempotent and mtime-stable: a publish that
    changes nothing else leaves no diff here either.
    """
    exports_root.mkdir(parents=True, exist_ok=True)
    return write_if_changed(exports_root / "README.md", README)


def volume_page_services(volume: str, *, item_json: Path | None, cache_dir: Path) -> dict[str, str]:
    """Lower-cased page id -> IIIF image service id for one volume.

    Reads a local LOC item JSON when given one; otherwise goes through the cached, rate-limited
    :class:`autogeoref.loc.LOCClient` (an already-placed volume's item document is a cache hit).

    A local document naming a DIFFERENT item is refused: page numbering restarts at 1 in every
    Sanborn volume, so a wrong volume's item would "match" every page and the export would point
    correct GCPs at the wrong volume's imagery.
    """
    from .loc import LOCClient, sheet_iiif_services

    if item_json is not None:
        item = json.loads(item_json.read_text(encoding="utf-8"))
        item_id = str((item.get("item") or {}).get("id") or "")
        ident = item_id.rstrip("/").rsplit("/", 1)[-1]
        if ident and ident != volume:
            raise ValueError(f"{item_json} is the item document of {ident}, not {volume}")
    else:
        client = LOCClient(cache_dir=cache_dir)
        try:
            item = client.item(volume)
        finally:
            client.close()
    return sheet_iiif_services(item)


def stage_export(paths: VolumePaths, *, page_services: Mapping[str, str], out_dir: Path) -> int:
    """Write one volume's export tree into ``out_dir``; returns the sheet count.

    ``out_dir`` is a staging directory: the caller lands it atomically (the
    publish transaction renames it into ``exports/<volume>``). The
    AnnotationPage is assembled first because ``export_volume`` raises rather
    than silently dropping a committed sheet; nothing is written unless the
    whole volume can be exported.
    """
    from .allmaps import export_volume
    from .slugs import page_sort_key
    from .volume import is_committed

    page = export_volume(paths, page_services=page_services)
    if out_dir.exists():
        # stale residue from a crashed publish must be replaced, never merged
        shutil.rmtree(out_dir)
    gcps_dir = out_dir / "gcps"
    gcps_dir.mkdir(parents=True)
    for _page, record, record_path in iter_results(paths, sort_key=lambda p: page_sort_key(p.stem)):
        if not is_committed(record):
            continue
        shutil.copyfile(record_path, gcps_dir / record_path.name)
    (out_dir / "allmaps.json").write_text(json.dumps(page, indent=1) + "\n", encoding="utf-8")
    logger.info("staged export: %d committed sheets -> %s", len(page["items"]), out_dir)
    return len(page["items"])
