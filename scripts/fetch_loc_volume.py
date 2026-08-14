"""Acquire one or more LOC volumes into this repo's per-volume work-tree layout.

THE fetch. Usable on its own, and what the queue's `fetch` track shells out to,
so there is one polite LOC lane. Per volume it writes both stores: `jp2/` holds
the LOC master byte-for-byte, `regions/` the pipeline's input derived at
IDENTICAL pixel dimensions.

The masters are the frame of record — volunteer GCPs live in the jp2 native
resolution, so a derivative would silently break every accuracy score. Keyed on
the LOC ITEM ID, never a slug another system pinned: pairing one edition's
pixels with another's GCPs measures an order of magnitude worse.

Every request goes through `autogeoref.loc.LOCClient`. A failed page is skipped
rather than retried in a tight loop, and existing files are never re-fetched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autogeoref.loc import BodyStalledError, LOCClient, LOCError, page_of_sheet_url
from autogeoref.paths import (
    VolumeBusyError,
    VolumePaths,
    atomic_output_path,
    regions_by_page,
    volume_lock,
)
from autogeoref.pillow import unlimited_image_pixels
from autogeoref.validation import volume_argument

ROOT = Path(__file__).resolve().parent.parent

#: Quality of the derived full-res JPEG, matched to what the EXISTING corpus is
#: encoded at: every acceptance threshold here was measured against inputs at
#: this quality, so a fetched volume must not arrive at another. Accuracy never
#: depended on it — what carries a GCP is the frame, which is the master's
#: either way — and raising it costs disk that compounds, since `regions/` is
#: the store a re-bake reads and therefore never prunes.
JPEG_QUALITY = 75

#: Generous, because the tile server drops large jp2 bodies mid-transfer under
#: its default pacing (read timeouts observed at 30 s during the golden fetch).
#: It bounds the GAP between two chunks, which is why BODY_BUDGET_S exists.
TIMEOUT_S = 180.0

#: Wall-clock ceiling for ONE master, and the guard a gap timeout cannot be:
#: this host answers a sustained pull by strangling the body rather than
#: refusing it, and a body still arriving is a body that never times out. A
#: ~6 MB master under 300 s is a ~20 kB/s floor — far below any transfer worth
#: waiting for, so a page that trips it is reported and skipped, not waited on.
BODY_BUDGET_S = 300.0

#: Spacing between requests. The published rate limit is a REQUEST count and
#: was never the binding constraint — each of these bodies is megabytes, and
#: what the host meters is sustained BYTES. This default sits well under the
#: measured refill rate. Raise it with --min-interval if a fetch strangles
#: anyway; lowering it is what this constant exists to discourage.
MIN_INTERVAL_S = 30.0

MAX_RETRIES = 6

#: Consecutive budget-exhausted pages that mean "the host is shaping us", not
#: "this page is unlucky". Three, because two in a row is within the noise of a
#: bad object or a dropped route and the whole point is to stop hammering.
STALL_ABORT = 3


class FetchStalledError(RuntimeError):
    """LOC is shaping this client's bandwidth: stop the pass, do not grind it."""


def derive_region(master: Path, dest: Path, *, quality: int = JPEG_QUALITY) -> tuple[int, int]:
    """Write ``master`` as a full-res JPEG at ``dest``. Returns its pixel size.

    No resampling, no rotation, no cropping: the pipeline's pixel frame is the
    master's frame (`sheets/manifest.json` records it at prep time), and a
    resize here would move every GCP that frame defines.
    """
    from PIL import Image

    with unlimited_image_pixels(), Image.open(master) as image:
        # JPEG has no palette/alpha; a jp2 in any other mode converts rather than
        # raising halfway through an encode.
        source = image if image.mode in ("RGB", "L") else image.convert("RGB")
        with atomic_output_path(dest) as temporary:
            source.save(temporary, format="JPEG", quality=quality)
        return source.size


def fetch_volume(
    volume: str,
    *,
    client: LOCClient,
    work: Path,
    dry_run: bool = False,
    quality: int = JPEG_QUALITY,
) -> int:
    """Fetch every page of one volume. Returns the number of pages that FAILED.

    A nonzero return is what stops the queue promoting a half-fetched volume to
    a model, so "failed" must mean every way this pass did not deliver a usable
    volume — including an item whose masters this repo cannot name, which would
    otherwise report success over an empty tree. Holds the volume lock for the
    whole pass, since a second fetcher on the same tree is what it refuses.
    """
    paths = VolumePaths(work / volume)
    item = client.item(volume)
    urls = LOCClient.sheet_image_urls(item, mimetype="image/jp2")
    if not urls:
        # A catalogued-but-never-scanned item: a fact about LOC's holdings,
        # not a fetch that went wrong. Many items are like this.
        print(f"{volume}: no jp2 masters in the item record — nothing to fetch", flush=True)
        return 0

    by_page: dict[str, str] = {}
    unpageable: list[str] = []
    for url in urls:
        page = page_of_sheet_url(url)
        if page is None:
            unpageable.append(url.rpartition("/")[2])
            continue
        # A duplicate page id would put two scans under one name. Report it and
        # keep the first: guessing which is the real page 12 is not a fetcher's
        # call, and prep refuses a duplicate outright anyway.
        if page in by_page:
            print(f"{volume}: WARNING two jp2 masters claim page {page}; keeping the first")
            continue
        by_page[page] = url
    for name in unpageable:
        print(f"{volume}: SKIP {name} — no page id in the filename", flush=True)
    if unpageable and not by_page:
        # Every master dropped: the volume IS digitized and this repo's page
        # grammar cannot name any of it. Reporting 0 failures here would promote
        # an empty tree to the place queue and spend a model call finding out.
        print(
            f"{volume}: FAILED — {len(unpageable)} master(s) and not one page id this repo "
            f"recognizes. Add the tag form to loc._SHEET_TAG_RE rather than working around it.",
            flush=True,
        )
        return len(unpageable)

    masters = paths.root / "jp2"
    if dry_run:
        pending = [p for p in by_page if not (masters / f"{volume}_p{p}.jp2").exists()]
        print(
            f"{volume}: {len(by_page)} pages, {len(pending)} to download "
            f"(>={round(len(pending) * client.min_interval)} s of LOC pacing), "
            f"{len(by_page) - len(pending)} already on disk",
            flush=True,
        )
        return 0

    failures: list[str] = []
    stalled = 0
    with volume_lock(paths, f"fetch {volume}"):
        # An EXISTING scan for a page wins, whatever its format. Five volumes were
        # acquired by scripts/fetch_special_sheets.py, which puts the jp2 in
        # regions/ as the only copy — deriving a .jpg beside it would give one
        # sheet two page-addressable files, and prep refuses that outright
        # (DuplicatePageError). Nothing here overwrites or second-guesses a scan
        # already on disk.
        existing = regions_by_page(paths.regions)
        masters.mkdir(parents=True, exist_ok=True)
        paths.regions.mkdir(parents=True, exist_ok=True)
        for page in sorted(by_page, key=lambda p: (len(p), p)):
            master = masters / f"{volume}_p{page}.jp2"
            region = paths.regions / f"{volume}_p{page}.jpg"
            held = existing.get(page)
            if held is not None and held != region:
                print(
                    f"{volume}: p{page} already has a scan in a different format "
                    f"({held.name}) — leaving it alone",
                    flush=True,
                )
                continue
            if master.exists() and region.exists():
                continue
            try:
                if not master.exists():
                    print(f"{volume}: downloading p{page}", flush=True)
                    client.download(by_page[page], master)
                    # The master is now the durable copy, so the client's cached
                    # body is a second copy of the same ~6 MB — tens of GB over
                    # the acquisition program, against a streaming design whose
                    # whole argument is disk (runbook 4.1). Re-running stays free
                    # without it: the exists() check above never asks.
                    client.forget_cached(by_page[page])
                if not region.exists():
                    size = derive_region(master, region, quality=quality)
                    print(f"{volume}: p{page} region {size[0]}x{size[1]}", flush=True)
            except (LOCError, OSError, ValueError) as exc:
                # One page's failure only shrinks this pass; re-run to retry. Both
                # writes are atomic, so nothing partial is left to be re-read as
                # complete — but a master whose DERIVE failed is suspect (that is
                # how a corrupt body shows up), so it goes with the failure.
                master.unlink(missing_ok=True)
                failures.append(page)
                print(f"{volume}: p{page} failed ({exc}); continuing", flush=True)
                if isinstance(exc, BodyStalledError):
                    stalled += 1
                    if stalled >= STALL_ABORT:
                        # Shaping is a property of the HOST's view of us, not of
                        # this page, so the pages after it will be shaped too.
                        # Continuing would spend the full retry ladder on every
                        # one of them — hours of pulling on a host that has
                        # already said no, which is the opposite of polite.
                        raise FetchStalledError(
                            f"{volume}: {stalled} pages in a row ran out of body budget — "
                            f"LOC is shaping this client's bandwidth, not failing it. "
                            f"Stopping the pass; wait for the block to age out (LOC "
                            f"publishes 1 hour) and re-run. Nothing already on disk is "
                            f"re-fetched."
                        ) from exc
                else:
                    stalled = 0

    ready = len(regions_by_page(paths.regions))
    print(f"{volume}: {ready}/{len(by_page)} pages ready under {paths.regions}", flush=True)
    if failures:
        print(f"{volume}: {len(failures)} failed this pass: {failures} — re-run to retry")
    return len(failures)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download every page of one or more LOC volumes into work/<vid>/.",
        epilog="Masters land in work/<vid>/jp2/; the pipeline's full-res inputs in "
        "work/<vid>/regions/. Re-run to retry failed pages; nothing already on disk "
        "is re-fetched.",
    )
    parser.add_argument("volumes", nargs="+", type=volume_argument, metavar="VOLUME")
    parser.add_argument("--work", type=Path, default=ROOT / "work", help="work-tree root")
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "cache" / "loc", help="LOC client response cache"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count the pages and the pacing floor; download nothing",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=JPEG_QUALITY,
        help=f"quality of the derived full-res region JPEG (default {JPEG_QUALITY})",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=MIN_INTERVAL_S,
        help=f"seconds between LOC requests (default {MIN_INTERVAL_S:g}); raise it if a "
        f"fetch strangles mid-body",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 1 <= args.jpeg_quality <= 100:
        print("error: --jpeg-quality must be between 1 and 100", file=sys.stderr)
        return 2
    if args.min_interval < MIN_INTERVAL_S:
        # Below the floor is how the strangling started; a caller who means it
        # can still say so, but not by accident.
        print(
            f"warning: --min-interval {args.min_interval:g}s is under the {MIN_INTERVAL_S:g}s "
            f"floor this fetcher was measured at",
            file=sys.stderr,
        )
    failures = 0
    refused: list[str] = []
    with LOCClient(
        cache_dir=args.cache,
        timeout=TIMEOUT_S,
        body_budget=BODY_BUDGET_S,
        min_interval=args.min_interval,
        max_retries=MAX_RETRIES,
    ) as client:
        for volume in args.volumes:
            try:
                failures += fetch_volume(
                    volume,
                    client=client,
                    work=args.work,
                    dry_run=args.dry_run,
                    quality=args.jpeg_quality,
                )
            except FetchStalledError as exc:
                # Host-wide, not volume-wide: the next volume would be shaped the
                # same way, so the remaining ones are not attempted. Nonzero exit
                # keeps the queue from promoting any of them.
                failures += 1
                print(f"{exc}", file=sys.stderr)
                remaining = args.volumes[args.volumes.index(volume) + 1 :]
                if remaining:
                    print(f"not attempted: {remaining}", file=sys.stderr)
                break
            except VolumeBusyError as exc:
                # Somebody else owns that tree right now, so this volume was not
                # fetched. It counts toward a NONZERO exit even though nothing went
                # wrong, because the alternative is worse: exiting 0 would have the
                # queue promote an unfetched volume to the place queue and spend a
                # model call discovering the collision. A queued drain therefore
                # fails the entry — re-add it once the sibling finishes.
                refused.append(volume)
                print(f"{volume}: SKIP — {exc}", file=sys.stderr)
            except LOCError as exc:
                # The item lookup itself failed (a bad id, or LOC is down): the
                # whole volume, not one page.
                failures += 1
                print(f"{volume}: FAILED — {exc}", file=sys.stderr)
    if refused:
        print(f"{len(refused)} volume(s) skipped as busy: {refused}", file=sys.stderr)
    # A nonzero exit is what makes the queue's fetch leg fail rather than promote
    # a half-fetched volume to placement.
    return 1 if failures or refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
