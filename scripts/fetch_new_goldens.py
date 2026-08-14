"""Golden-testbed expansion: acquire volunteer-pinned volumes from LOC.

`VOLUMES` names the testbeds to fetch; pick them for era spread, since the
validated corpus is what every accuracy claim rests on.

Annotating a full volume is a large model spend, so each testbed freezes an
EVENLY-SPACED subset of the PINNED pages — the ones with human GCPs are the
only scoreable ones — recorded in `subset.json` beside the images.

Pixel frames: the volunteer GCPs live in the LOC jp2 NATIVE resolution, so this
downloads image/jp2 masters, not the pct:25 derivatives. A volume is an hour or
more of polite spacing, so run it in the background.

    nice uv run python scripts/fetch_new_goldens.py
"""

import json
import re
import shutil
from pathlib import Path

from autogeoref.loc import LOCClient, LOCError
from autogeoref.prep import prep_volume

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "goldens"
# the frozen ground-truth layer exports live in the fixture tree
LIVE_PROD_CACHE = ROOT / "fixtures" / "ground-truth"

VOLUMES = ["sanborn01790_040", "sanborn01790_110"]
SUBSET_SIZE = 40

_PAGE_STEM = re.compile(r"-(titl|ind\d+|\d{4})(?:\.|/|$)")


def page_of_url(url: str) -> str | None:
    """LOC sheet URL -> page id ('titl', 'ind1', '0'..'114')."""
    m = _PAGE_STEM.search(url)
    if not m:
        return None
    tag = m.group(1)
    return tag if not tag.isdigit() else str(int(tag))


def pinned_pages(vid: str) -> list[str]:
    """Pages with volunteer GCPs, from the live repo's read-only cache."""
    layers = json.loads((LIVE_PROD_CACHE / f"api-layers-{vid}.json").read_text())
    pages = []
    for lay in layers if isinstance(layers, list) else []:
        if not (lay.get("gcps_geojson") or {}).get("features"):
            continue
        slug = str(lay.get("slug") or "")
        m = re.search(r"_p(\d+[a-zS]*)$", slug)
        if m:
            pages.append(m.group(1))
    return sorted(pages, key=lambda p: (len(p), p))


def main() -> None:
    # generous timeout + retries: the tile server drops large jp2 bodies
    # mid-transfer under its default pacing (observed read timeouts at 30 s)
    client = LOCClient(cache_dir=ROOT / "cache" / "loc", timeout=180.0, max_retries=6)
    for vid in VOLUMES:
        vol_out = OUT / vid
        regions = vol_out / "regions"
        regions.mkdir(parents=True, exist_ok=True)
        # freeze the GT alongside (copied from the live repo's cache, read-only)
        gt_dst = vol_out / f"api-layers-{vid}.json"
        if not gt_dst.exists():
            shutil.copy2(LIVE_PROD_CACHE / f"api-layers-{vid}.json", gt_dst)

        pinned = pinned_pages(vid)
        step = max(1, len(pinned) // SUBSET_SIZE)
        subset = pinned[::step][:SUBSET_SIZE]
        (vol_out / "subset.json").write_text(
            json.dumps({"pinned": pinned, "subset": subset, "step": step}, indent=1)
        )
        print(f"{vid}: {len(pinned)} pinned pages, subset of {len(subset)}: {subset}")

        item = client.item(vid)
        jp2_urls = LOCClient.sheet_image_urls(item, mimetype="image/jp2")
        by_page = {}
        for u in jp2_urls:
            p = page_of_url(u)
            if p is not None:
                by_page[p] = u
        missing = [p for p in subset if p not in by_page]
        if missing:
            print(f"{vid}: WARNING pages without a jp2 URL: {missing}")
        failures = []
        for p in subset:
            url = by_page.get(p)
            if url is None:
                continue
            dest = regions / f"{vid}_p{p}.jp2"
            if dest.exists():
                continue
            print(f"{vid}: downloading p{p}", flush=True)
            try:
                client.download(url, dest)
            except LOCError as exc:
                # the tile server intermittently 520s/drops large bodies; a
                # missing page only shrinks the subset — rerun to retry
                failures.append(p)
                print(f"{vid}: p{p} failed ({exc}); continuing", flush=True)
        if failures:
            print(f"{vid}: {len(failures)} pages failed this pass: {failures}", flush=True)
        # smalls + manifest in the production pixel frame (jp2 native)
        prep_volume(regions, vol_out / "sheets")
        print(f"{vid}: prepped {len(list((vol_out / 'sheets').glob('p*_small.jpg')))} smalls")


if __name__ == "__main__":
    main()
