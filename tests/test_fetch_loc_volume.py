"""Contract tests for the LOC volume fetcher — the one acquisition implementation.

No network: the ``LOCClient`` is built on an ``httpx.MockTransport`` serving real
jp2 bytes, so the frame assertions are measured rather than asserted about a stub.

What these pin is what section 4.6 of the source-independence runbook calls
non-negotiable: masters, not derivatives; keyed on the LOC item id; both stores
per volume; a failed page reported and skipped; and never two fetchers on one
work tree.
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from autogeoref.loc import LOCClient
from autogeoref.paths import VolumePaths, volume_lock


def _module() -> Any:
    path = Path(__file__).parents[1] / "scripts" / "fetch_loc_volume.py"
    spec = importlib.util.spec_from_file_location("fetch_loc_volume", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FETCH = _module()
VOLUME = "sanborn01790_040"
STORAGE = "https://tile.loc.gov/storage-services/service/gmd/x/01790_21_1918-{tag}.jp2"
#: The pct:25 derivative LOC also lists. Fetching this would silently break every
#: accuracy score: the volunteer GCPs live in the master's frame.
DERIVATIVE = "https://tile.loc.gov/image-services/iiif/x-{tag}/full/pct:25/0/default.jpg"

MASTER_SIZE = (48, 72)


@pytest.fixture(autouse=True)
def _root_off_the_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the script's root at tmp for every test in this file.

    ``--work`` and ``--cache`` both default to an absolute path under the repo
    root, so a ``main()`` call omitting either writes into the checkout — which
    is invisible while the checkout is writable and ``cache/`` is gitignored.
    Most calls below pass the flags anyway; the stalled-pass one deliberately
    does not, so something fails if this stops working.
    """
    monkeypatch.setattr(FETCH, "ROOT", tmp_path / "script-root")


def _jp2(size: tuple[int, int] = MASTER_SIZE) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (17, 34, 51)).save(buffer, format="JPEG2000")
    return buffer.getvalue()


def _item(*tags: str) -> dict[str, Any]:
    """An item document shaped like LOC's: each sheet group carries a jp2 master and
    the jpeg derivatives, and the fetcher must pick the master."""
    return {
        "resources": [
            {
                "files": [
                    [
                        {
                            "mimetype": "image/jpeg",
                            "url": DERIVATIVE.format(tag=tag),
                            "height": 18,
                        },
                        {"mimetype": "image/jp2", "url": STORAGE.format(tag=tag), "height": 72},
                    ]
                    for tag in tags
                ]
            }
        ]
    }


class _Transport:
    """Serves the item JSON and the masters, and records every URL requested."""

    def __init__(self, item: dict[str, Any], *, fail: set[str] | None = None) -> None:
        self.item = item
        self.fail = fail or set()
        self.urls: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.urls.append(url)
        if "?fo=json" in url:
            return httpx.Response(200, json=self.item)
        if any(tag in url for tag in self.fail):
            return httpx.Response(404)
        return httpx.Response(200, content=_jp2())


def _client(tmp_path: Path, transport: _Transport, min_interval: float = 0.0) -> LOCClient:
    return LOCClient(
        tmp_path / "cache",
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
        # zero pacing so the test does not sit through the conduct floor; the floor
        # itself is `test_loc`'s contract, not this module's
        min_interval=min_interval,
        max_retries=0,
    )


def _fetch(tmp_path: Path, transport: _Transport, min_interval: float = 0.0, **kwargs: Any) -> int:
    exit_code: int = FETCH.fetch_volume(
        VOLUME,
        client=_client(tmp_path, transport, min_interval),
        work=tmp_path / "work",
        **kwargs,
    )
    return exit_code


def test_both_stores_land_with_the_master_frame_intact(tmp_path: Path) -> None:
    """The master is the frame of record, and the pipeline's input must share it
    EXACTLY — a resize here would move every GCP the frame defines."""
    transport = _Transport(_item("0001", "0002"))

    assert _fetch(tmp_path, transport) == 0

    paths = VolumePaths(tmp_path / "work" / VOLUME)
    master = paths.root / "jp2" / f"{VOLUME}_p1.jp2"
    region = paths.regions / f"{VOLUME}_p1.jpg"
    assert master.is_file() and region.is_file()
    assert master.read_bytes() == _jp2(), "the master lands byte-for-byte"
    with Image.open(region) as image:
        assert image.size == MASTER_SIZE
        assert image.format == "JPEG"
    assert sorted(p.name for p in paths.regions.iterdir()) == [
        f"{VOLUME}_p1.jpg",
        f"{VOLUME}_p2.jpg",
    ], "regions holds only the pipeline inputs — a jp2 there would double-read the sheet"


def test_only_the_jp2_masters_are_ever_requested(tmp_path: Path) -> None:
    """Not the pct:25 derivative, whatever else the item lists."""
    transport = _Transport(_item("0001"))

    _fetch(tmp_path, transport)

    assert not any("pct:25" in url for url in transport.urls)
    assert [u for u in transport.urls if ".jp2" in u] == [STORAGE.format(tag="0001")]


def test_the_item_is_looked_up_by_its_loc_id(tmp_path: Path) -> None:
    """Keyed on the LOC item id, never a slug another system pinned — the retraction
    in the runbook's section 2 measured 67.5 m for the alternative."""
    transport = _Transport(_item("0001"))

    _fetch(tmp_path, transport)

    assert transport.urls[0] == f"https://www.loc.gov/item/{VOLUME}/?fo=json"


def test_a_failed_page_is_reported_and_skipped_leaving_no_partial_master(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One page's failure only shrinks the pass. A half-written master would be read
    as complete on the re-run that is supposed to retry it, so it goes."""
    transport = _Transport(_item("0001", "0002", "0003"), fail={"0002"})

    failures = _fetch(tmp_path, transport)

    assert failures == 1
    paths = VolumePaths(tmp_path / "work" / VOLUME)
    assert sorted(p.name for p in paths.regions.iterdir()) == [
        f"{VOLUME}_p1.jpg",
        f"{VOLUME}_p3.jpg",
    ]
    assert not (paths.root / "jp2" / f"{VOLUME}_p2.jp2").exists()
    out = capsys.readouterr().out
    assert "p2 failed" in out and "re-run to retry" in out


def test_nothing_already_on_disk_is_re_fetched(tmp_path: Path) -> None:
    """A re-run costs only what it never got — the whole reason a failure is survivable."""
    transport = _Transport(_item("0001", "0002"), fail={"0002"})
    assert _fetch(tmp_path, transport) == 1
    first = len([u for u in transport.urls if ".jp2" in u])

    retry = _Transport(_item("0001", "0002"))
    assert _fetch(tmp_path, retry) == 0

    assert first == 2
    assert [u for u in retry.urls if ".jp2" in u] == [STORAGE.format(tag="0002")]


def test_a_dry_run_downloads_nothing_and_prices_the_pacing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    transport = _Transport(_item("0001", "0002"))

    # The quote has to track the pacing the client is ACTUALLY configured with:
    # priced against a hardcoded interval it silently understates every fetch run
    # at a slower floor, which is the number an operator plans days around.
    assert _fetch(tmp_path, transport, min_interval=7.0, dry_run=True) == 0

    assert not any(".jp2" in url for url in transport.urls)
    assert not (tmp_path / "work" / VOLUME / "jp2").exists()
    out = capsys.readouterr().out
    assert "2 pages, 2 to download" in out and ">=14 s" in out


def test_two_masters_claiming_one_page_keep_the_first_and_say_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Guessing which scan is the real page 12 is not a fetcher's call, and prep
    refuses a duplicate page id outright anyway."""
    item = _item("0007")
    other = STORAGE.replace("01790_21_1918", "01790_21_1918_rescan").format(tag="0007")
    item["resources"][0]["files"].append([{"mimetype": "image/jp2", "url": other, "height": 72}])

    assert _fetch(tmp_path, transport := _Transport(item)) == 0

    assert [u for u in transport.urls if ".jp2" in u] == [STORAGE.format(tag="0007")]
    assert "two jp2 masters claim page 7" in capsys.readouterr().out


def test_a_volume_someone_else_owns_is_refused_not_fetched(tmp_path: Path) -> None:
    """Never fetch a volume already being fetched: two writers of one work tree. The
    per-volume lock is the same one a placement run takes."""
    from autogeoref.paths import VolumeBusyError

    paths = VolumePaths(tmp_path / "work" / VOLUME)
    paths.root.mkdir(parents=True)
    transport = _Transport(_item("0001"))

    with volume_lock(paths, "a sibling fetch"), pytest.raises(VolumeBusyError):
        _fetch(tmp_path, transport)

    assert not any(".jp2" in url for url in transport.urls)


def test_an_item_with_no_masters_is_not_a_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A catalogued-but-never-scanned item has nothing to fetch. That is a fact about
    LOC's holdings, not a fetch that went wrong."""
    assert _fetch(tmp_path, _Transport({"resources": []})) == 0
    assert "nothing to fetch" in capsys.readouterr().out


def test_an_unpageable_master_is_named_rather_than_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Prep refuses to silently skip a sheet it cannot name, and so does this: a
    master that vanishes without a word is a missing page nobody looks for."""
    item = _item("0001")
    weird = "https://tile.loc.gov/storage-services/service/gmd/x/unrecognizable.jp2"
    item["resources"][0]["files"].append([{"mimetype": "image/jp2", "url": weird, "height": 72}])

    assert _fetch(tmp_path, _Transport(item)) == 0

    assert "SKIP unrecognizable.jp2" in capsys.readouterr().out


def test_a_volume_whose_every_master_is_unpageable_is_a_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The difference that matters to the queue. A digitized item this repo cannot
    name a single page of must NOT report success: a zero exit promotes an empty
    work tree to the place queue, and the model call is how you find out."""
    item = {
        "resources": [
            {
                "files": [
                    [{"mimetype": "image/jp2", "url": f"https://tile.loc.gov/x/no{n}.jp2"}]
                    for n in (1, 2)
                ]
            }
        ]
    }
    transport = _Transport(item)

    assert _fetch(tmp_path, transport) == 2

    assert not any(".jp2" in url for url in transport.urls), "it never downloads either"
    assert "not one page id this repo recognizes" in capsys.readouterr().out


def test_an_uppercase_word_tag_is_a_page_not_a_dropped_file(tmp_path: Path) -> None:
    """`_016` ships `-CBDa`/`-CBDb` where `_188`/`_189` ship `-cbd1`/`-cbd2`, and
    both spellings are pages under this repo's grammar. A lowercase-only tag
    pattern dropped every master of that volume and called the fetch a success."""
    base = "https://tile.loc.gov/storage-services/service/gmd/x/01790_16_1903-{tag}.jp2"
    item = {
        "resources": [
            {
                "files": [
                    [{"mimetype": "image/jp2", "url": base.format(tag=t), "height": 72}]
                    for t in ("CBDa", "CBDb")
                ]
            }
        ]
    }

    assert _fetch(tmp_path, _Transport(item)) == 0

    regions = VolumePaths(tmp_path / "work" / VOLUME).regions
    assert sorted(p.name for p in regions.iterdir()) == [
        f"{VOLUME}_pCBDa.jpg",
        f"{VOLUME}_pCBDb.jpg",
    ]


def test_a_page_already_held_in_another_format_is_left_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The five special-sheet volumes store the jp2 in `regions/` as the ONLY copy
    (scripts/fetch_special_sheets.py). Deriving a .jpg beside it would give one
    sheet two page-addressable files, which prep refuses outright — so a scan
    already on disk wins whatever its format, and nothing here overwrites one."""
    from autogeoref.prep import prep_volume

    regions = VolumePaths(tmp_path / "work" / VOLUME).regions
    regions.mkdir(parents=True)
    held = regions / f"{VOLUME}_p1.jp2"
    held.write_bytes(_jp2())
    transport = _Transport(_item("0001", "0002"))

    assert _fetch(tmp_path, transport) == 0

    assert sorted(p.name for p in regions.iterdir()) == [
        f"{VOLUME}_p1.jp2",  # untouched, and no p1.jpg beside it
        f"{VOLUME}_p2.jpg",
    ]
    assert held.read_bytes() == _jp2()
    assert [u for u in transport.urls if ".jp2" in u] == [STORAGE.format(tag="0002")]
    assert "already has a scan in a different format" in capsys.readouterr().out
    # the real assertion: prep still accepts the tree
    prep_volume(regions, tmp_path / "sheets")


def test_the_master_is_not_kept_twice(tmp_path: Path) -> None:
    """The client caches every body, so a master would land in `jp2/` AND in
    `cache/loc/` — a second copy of tens of gigabytes, against a streaming design
    whose whole argument is disk. Idempotence does not depend on the cache here:
    the on-disk check short-circuits before any request."""
    transport = _Transport(_item("0001"))
    cache = tmp_path / "cache"

    FETCH.fetch_volume(
        VOLUME, client=_client(tmp_path, transport), work=tmp_path / "work", dry_run=False
    )

    assert (tmp_path / "work" / VOLUME / "jp2" / f"{VOLUME}_p1.jp2").is_file()
    assert not list(cache.glob("*.bin")), "the cached body is dropped once the master lands"
    assert list(cache.glob("*.json")), "the item document stays — small, and re-read constantly"

    # and a re-run still costs nothing: it never asks
    again = _Transport(_item("0001"))
    assert _fetch(tmp_path, again) == 0
    assert not any(".jp2" in url for url in again.urls)


def test_a_master_is_written_atomically(tmp_path: Path) -> None:
    """A body this large is minutes of transfer, and the console's Stop button
    SIGTERMs the process group. A non-atomic write would leave a truncated file
    that every later pass reads as a complete master — the one file the docstring
    calls the frame of record."""
    calls: list[Path] = []
    real = FETCH.derive_region

    def _spy(master: Path, dest: Path, **kw: Any) -> tuple[int, int]:
        # while the derive runs, the master must ALREADY be at its final name and
        # complete: no partial file was ever visible under it
        calls.append(master)
        assert master.read_bytes() == _jp2()
        derived: tuple[int, int] = real(master, dest, **kw)
        return derived

    FETCH.derive_region = _spy
    try:
        assert _fetch(tmp_path, _Transport(_item("0001"))) == 0
    finally:
        FETCH.derive_region = real
    assert len(calls) == 1
    # no temporary left behind beside it
    assert [p.name for p in calls[0].parent.iterdir()] == [f"{VOLUME}_p1.jp2"]


def test_the_derived_region_carries_no_resize_or_rotation(tmp_path: Path) -> None:
    """derive_region on its own: same pixels, same orientation, JPEG container."""
    master = tmp_path / "m.jp2"
    Image.new("RGB", (61, 23), (200, 100, 50)).save(master, format="JPEG2000")
    dest = tmp_path / "out" / "r.jpg"

    assert FETCH.derive_region(master, dest) == (61, 23)

    with Image.open(dest) as image:
        assert image.size == (61, 23) and image.format == "JPEG"


def test_the_default_encoding_is_the_one_the_corpus_already_uses(tmp_path: Path) -> None:
    """75 is a MEASURED value, not a taste: deriving `_066` p10's master at this
    quality reproduces the region jpg already on disk byte-for-byte, so a fetched
    volume reaches the gates in the same format every threshold was measured against.
    Raising it also doubles the one store a prune never reclaims. Change it with a
    dated record, not in passing.
    """
    assert FETCH.JPEG_QUALITY == 75
    assert FETCH.parse_args(["v"]).jpeg_quality == 75

    # and the flag actually reaches the encoder, so the override is real
    master = tmp_path / "m.jp2"
    Image.new("RGB", (200, 200)).save(master, format="JPEG2000")
    low = tmp_path / "low.jpg"
    high = tmp_path / "high.jpg"
    FETCH.derive_region(master, low, quality=20)
    FETCH.derive_region(master, high, quality=95)
    assert low.stat().st_size < high.stat().st_size


def test_a_grayscale_master_converts_rather_than_failing_mid_encode(tmp_path: Path) -> None:
    """A mode JPEG cannot hold must be converted BEFORE the encode, not raise halfway."""
    master = tmp_path / "m.jp2"
    Image.new("L", (12, 8), 128).save(master, format="JPEG2000")

    assert FETCH.derive_region(master, tmp_path / "r.jpg") == (12, 8)


def test_the_cli_refuses_a_quality_outside_the_encoder_range(tmp_path: Path) -> None:
    argv = [VOLUME, "--work", str(tmp_path), "--cache", str(tmp_path / "cache")]
    assert FETCH.main([*argv, "--jpeg-quality", "0"]) == 2
    assert FETCH.main([*argv, "--jpeg-quality", "101"]) == 2
    assert not (tmp_path / VOLUME).exists()


def test_the_cli_defaults_are_absolute_paths_under_the_scripts_root(tmp_path: Path) -> None:
    """Neither default is relative to the working directory, so `chdir` cannot
    keep a run off the checkout — only passing the flags, or moving the root."""
    root = tmp_path / "script-root"
    parsed = FETCH.parse_args([VOLUME])

    assert parsed.cache == root / "cache" / "loc"
    assert parsed.work == root / "work"


def test_a_volume_id_that_could_escape_its_root_is_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        FETCH.parse_args(["../etc"])
    assert "safe volume identifier" in capsys.readouterr().err


class _StallingTransport(_Transport):
    """Serves the item JSON, then shapes every master into a trickle.

    What tile.loc.gov actually did after ~150 MB in one pass: 200 OK, bytes
    still arriving, at a rate no gap timeout can distinguish from progress.
    """

    def __init__(self, item: dict[str, Any], clock: Any) -> None:
        super().__init__(item)
        self.clock = clock

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.urls.append(url)
        if "?fo=json" in url:
            return httpx.Response(200, json=self.item)

        def trickle() -> Any:
            for _ in range(50):
                self.clock.now += 40.0
                yield b"x" * 8

        return httpx.Response(200, content=trickle())


def _stalling_client(tmp_path: Path, transport: _StallingTransport) -> LOCClient:
    return LOCClient(
        tmp_path / "cache",
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
        clock=transport.clock,
        sleep=transport.clock.sleep,
        min_interval=0.0,
        body_budget=100.0,
        max_retries=1,
    )


def test_a_shaped_host_stops_the_pass_instead_of_grinding_every_page(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Shaping is the host's view of this CLIENT, so the next page is shaped too.

    Grinding the retry ladder through the rest of the volume is hours of pulling
    on a host that already said no. Three in a row stops the pass.
    """

    class _Clock:
        def __init__(self) -> None:
            self.now = 1000.0

        def __call__(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += seconds

    transport = _StallingTransport(_item(*[f"{n:04d}" for n in range(1, 21)]), _Clock())
    with pytest.raises(FETCH.FetchStalledError) as caught:
        FETCH.fetch_volume(
            VOLUME,
            client=_stalling_client(tmp_path, transport),
            work=tmp_path / "work",
        )

    assert "shaping this client's bandwidth" in str(caught.value)
    # The remedy has to be in the message: waiting is the only thing that works,
    # and a re-run costs nothing it already has.
    assert "re-run" in str(caught.value)
    # Stopped at the third stalled page, not the twentieth.
    assert len([u for u in transport.urls if ".jp2" in u]) == FETCH.STALL_ABORT * 2
    assert not list((tmp_path / "work" / VOLUME / "jp2").glob("*.jp2"))
    assert "p1 failed" in capsys.readouterr().out


def test_a_stalled_pass_does_not_attempt_the_volumes_behind_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One shaped volume means the whole pass is shaped — and a nonzero exit is
    what keeps the queue from promoting an unfetched volume to a model."""
    attempted: list[str] = []

    def stalling_fetch(volume: str, **kwargs: Any) -> int:
        attempted.append(volume)
        raise FETCH.FetchStalledError(f"{volume}: shaping this client's bandwidth")

    monkeypatch.setattr(FETCH, "fetch_volume", stalling_fetch)
    # No --cache on purpose: this is the one call that builds the client, so it
    # is what proves the fixture above is holding the default off the checkout.
    code = FETCH.main([VOLUME, "sanborn01790_041", "sanborn01790_042", "--work", str(tmp_path)])

    assert code != 0
    assert attempted == [VOLUME]
    err = capsys.readouterr().err
    assert "sanborn01790_041" in err and "not attempted" in err
    assert (tmp_path / "script-root" / "cache" / "loc").is_dir()
