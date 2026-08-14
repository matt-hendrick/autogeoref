"""Acquire the digitized special single-sheet items from LOC.

A handful of catalogued single-sheet items, each digitized as several jp2
segments rather than as pages of a bound book. `ITEMS` names them and their
segment tags.

Downloads the jp2 masters into the standard per-volume layout, since the
pipeline reads jp2 scans directly. Page ids follow the import convention:
numeric tags strip zero padding, word tags stay verbatim. Placement is a
separate, later step.

Everything goes through autogeoref.loc.LOCClient. Failed segments are reported
and skipped — rerun to retry; existing files are never re-fetched.
"""

from pathlib import Path

from autogeoref.loc import LOCClient, LOCError, page_of_sheet_url

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"

VOLUMES = [
    "sanborn01790_188",
    "sanborn01790_189",
    "sanborn01790_190",
    "sanborn01790_016",
    "sanborn01790_013",
]


def page_of_url(url: str) -> str | None:
    """Sheet URL -> page id, via the one parser `loc` owns.

    This used to carry its own copy of the rule. Verified equivalent on all five
    items before the switch — same page ids for every segment, matching what is
    already on disk — so nothing re-keys."""
    return page_of_sheet_url(url)


def main() -> None:
    # generous timeout + retries: the tile server drops large jp2 bodies
    # mid-transfer under its default pacing
    client = LOCClient(cache_dir=ROOT / "cache" / "loc", timeout=180.0, max_retries=6)
    for vid in VOLUMES:
        regions = WORK / vid / "regions"
        regions.mkdir(parents=True, exist_ok=True)
        item = client.item(vid)
        urls = LOCClient.sheet_image_urls(item, mimetype="image/jp2")
        if not urls:
            print(f"{vid}: WARNING no jp2 URLs in item record", flush=True)
            continue
        failures = []
        for url in urls:
            page = page_of_url(url)
            if page is None:
                # a segment whose filename carries no page id: naming it "pNone"
                # would put an unrecognizable file where prep refuses to guess
                print(f"{vid}: SKIP {url.rpartition('/')[2]} — no page id", flush=True)
                continue
            dest = regions / f"{vid}_p{page}.jp2"
            if dest.exists():
                print(f"{vid}: p{page} already on disk", flush=True)
                continue
            print(f"{vid}: downloading p{page}", flush=True)
            try:
                client.download(url, dest)
            except LOCError as exc:
                failures.append(page)
                print(f"{vid}: p{page} failed ({exc}); continuing", flush=True)
        got = sorted(regions.glob(f"{vid}_p*.jp2"))
        print(f"{vid}: {len(got)}/{len(urls)} segments on disk", flush=True)
        if failures:
            print(f"{vid}: failed this pass: {failures} — rerun to retry", flush=True)


if __name__ == "__main__":
    main()
