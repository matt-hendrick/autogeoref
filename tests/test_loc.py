"""Tests for the LOC acquisition client (mocked transport, fake clock)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

from autogeoref.loc import (
    USER_AGENT,
    HostNotAllowedError,
    LOCClient,
    LOCRequestError,
    MethodNotAllowedError,
    iiif_service_id,
    page_of_sheet_url,
    sheet_iiif_services,
)
from autogeoref.viewer.sources import loc_titles

WORK_ITEM_FIXTURE = (
    Path(__file__).resolve().parent.parent / "work" / "loc-item-sanborn01790_024.json"
)


class FakeClock:
    """Injectable monotonic clock; sleeping advances it instead of waiting."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.now += seconds


class RecordingHandler:
    """MockTransport handler that records every request it sees."""

    def __init__(self, clock: FakeClock, responder: Any = None) -> None:
        self.clock = clock
        self.requests: list[httpx.Request] = []
        self.times: list[float] = []
        self.responder = responder

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.times.append(self.clock.now)
        if self.responder is not None:
            result: httpx.Response | Exception = self.responder(request, len(self.requests))
            if isinstance(result, Exception):
                raise result
            return result
        return httpx.Response(200, json={"ok": True, "url": str(request.url)})


def make_client(
    tmp_path: Path, handler: RecordingHandler, clock: FakeClock, **kwargs: Any
) -> LOCClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return LOCClient(
        tmp_path / "cache",
        http_client=http,
        clock=clock,
        sleep=clock.sleep,
        **kwargs,
    )


def test_sends_honest_user_agent(tmp_path: Path) -> None:
    clock = FakeClock()
    handler = RecordingHandler(clock)
    client = make_client(tmp_path, handler, clock)
    client.get_json("https://www.loc.gov/search/?q=x&fo=json")
    assert handler.requests[0].headers["User-Agent"] == USER_AGENT
    # Honest-UA contract: identifies the tool and carries a contact field.
    # The contact value comes from AUTOGEOREF_CONTACT (repo URL fallback),
    # so assert the shape, not a literal.
    assert USER_AGENT.startswith("autogeoref/")
    assert "(contact: " in USER_AGENT and USER_AGENT.rstrip().endswith(")")
    contact = USER_AGENT.split("(contact: ", 1)[1].rstrip(")")
    assert contact.strip()


def test_ua_contact_never_anonymous(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty/blank AUTOGEOREF_CONTACT must fall back to the repo URL —
    a "(contact: )" header would silently violate the honest-UA contract."""
    from autogeoref.loc import _contact_from_env

    for blank in ("", "   "):
        monkeypatch.setenv("AUTOGEOREF_CONTACT", blank)
        assert _contact_from_env().strip()
    monkeypatch.setenv("AUTOGEOREF_CONTACT", "ops@example.org")
    assert _contact_from_env() == "ops@example.org"
    monkeypatch.delenv("AUTOGEOREF_CONTACT")
    assert _contact_from_env().startswith("https://")


def test_every_request_has_explicit_timeout(tmp_path: Path) -> None:
    clock = FakeClock()
    handler = RecordingHandler(clock)
    client = make_client(tmp_path, handler, clock, timeout=42.0)
    client.get_json("https://www.loc.gov/item/x/?fo=json")
    timeout = handler.requests[0].extensions["timeout"]
    assert timeout["connect"] == 42.0
    assert timeout["read"] == 42.0


def test_requests_are_spaced_at_least_five_seconds(tmp_path: Path) -> None:
    clock = FakeClock()
    handler = RecordingHandler(clock)
    client = make_client(tmp_path, handler, clock)
    for i in range(4):
        client.get_json(f"https://www.loc.gov/item/x{i}/?fo=json")
    assert len(handler.times) == 4
    gaps = [b - a for a, b in zip(handler.times, handler.times[1:], strict=False)]
    assert all(gap >= 5.0 for gap in gaps), gaps


def test_clients_sharing_a_cache_share_the_request_lane(tmp_path: Path) -> None:
    """The shared cache directory, not a client instance, owns request pacing."""
    clock = FakeClock()
    handler = RecordingHandler(clock)
    first = make_client(tmp_path, handler, clock)
    second = make_client(tmp_path, handler, clock)

    first.get_json("https://www.loc.gov/item/first/?fo=json")
    second.get_json("https://www.loc.gov/item/second/?fo=json")

    assert len(handler.times) == 2
    assert handler.times[1] - handler.times[0] >= 5.0
    assert clock.sleeps == [5.0]


def test_concurrent_same_url_cache_miss_fetches_once(tmp_path: Path) -> None:
    """The cache lane covers a miss through atomic publication for all clients."""
    clock = FakeClock()
    request_started = threading.Event()
    release_request = threading.Event()

    def responder(request: httpx.Request, n: int) -> httpx.Response:
        assert n == 1
        request_started.set()
        assert release_request.wait(timeout=5)
        return httpx.Response(200, json={"url": str(request.url), "ok": True})

    handler = RecordingHandler(clock, responder)
    first = make_client(tmp_path, handler, clock)
    second = make_client(tmp_path, handler, clock)
    url = "https://www.loc.gov/item/shared/?fo=json"
    start = threading.Barrier(2)
    results: list[Any] = []
    errors: list[BaseException] = []

    def fetch(client: LOCClient) -> None:
        try:
            start.wait()
            results.append(client.get_json(url))
        except BaseException as exc:  # communicate failures from the worker thread
            errors.append(exc)

    threads = [threading.Thread(target=fetch, args=(client,)) for client in (first, second)]
    for thread in threads:
        thread.start()
    assert request_started.wait(timeout=5)
    release_request.set()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert not errors
    assert len(handler.requests) == 1
    assert results == [{"url": url, "ok": True}] * 2
    cached = list((tmp_path / "cache").glob("*.json"))
    assert len(cached) == 1
    assert json.loads(cached[0].read_text()) == results[0]


def test_backoff_on_429_then_success(tmp_path: Path) -> None:
    def responder(request: httpx.Request, n: int) -> httpx.Response:
        if n <= 2:
            return httpx.Response(429)
        return httpx.Response(200, json={"ok": True})

    clock = FakeClock()
    handler = RecordingHandler(clock, responder)
    client = make_client(tmp_path, handler, clock)
    payload = client.get_json("https://www.loc.gov/item/y/?fo=json")
    assert payload == {"ok": True}
    assert len(handler.requests) == 3
    # Backoff delays grow (5, 10) and the 5 s floor still holds between attempts.
    gaps = [b - a for a, b in zip(handler.times, handler.times[1:], strict=False)]
    assert all(gap >= 5.0 for gap in gaps), gaps
    assert gaps[1] > gaps[0]


def test_retries_exhausted_raises(tmp_path: Path) -> None:
    def responder(request: httpx.Request, n: int) -> httpx.Response:
        return httpx.Response(503)

    clock = FakeClock()
    handler = RecordingHandler(clock, responder)
    client = make_client(tmp_path, handler, clock, max_retries=2)
    with pytest.raises(LOCRequestError):
        client.get_json("https://www.loc.gov/item/z/?fo=json")
    assert len(handler.requests) == 3  # initial + 2 retries


def test_transport_error_is_retried(tmp_path: Path) -> None:
    def responder(request: httpx.Request, n: int) -> Any:
        if n == 1:
            return httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json={"ok": True})

    clock = FakeClock()
    handler = RecordingHandler(clock, responder)
    client = make_client(tmp_path, handler, clock)
    assert client.get_json("https://www.loc.gov/item/w/?fo=json") == {"ok": True}
    assert len(handler.requests) == 2


def test_non_retryable_http_error_raises_immediately(tmp_path: Path) -> None:
    def responder(request: httpx.Request, n: int) -> httpx.Response:
        return httpx.Response(404)

    clock = FakeClock()
    handler = RecordingHandler(clock, responder)
    client = make_client(tmp_path, handler, clock)
    with pytest.raises(LOCRequestError):
        client.get_json("https://www.loc.gov/item/missing/?fo=json")
    assert len(handler.requests) == 1


def test_json_cache_hit_issues_no_network_request(tmp_path: Path) -> None:
    clock = FakeClock()
    handler = RecordingHandler(clock)
    client = make_client(tmp_path, handler, clock)
    url = "https://www.loc.gov/item/cached/?fo=json"
    first = client.get_json(url)
    assert len(handler.requests) == 1
    second = client.get_json(url)
    assert len(handler.requests) == 1  # no second network request
    assert first == second
    # The raw JSON dump landed in the cache dir.
    cached = list((tmp_path / "cache").glob("*.json"))
    assert len(cached) == 1
    assert json.loads(cached[0].read_text()) == first


def test_bytes_cache_hit_and_download(tmp_path: Path) -> None:
    def responder(request: httpx.Request, n: int) -> httpx.Response:
        return httpx.Response(200, content=b"JPEGDATA")

    clock = FakeClock()
    handler = RecordingHandler(clock, responder)
    client = make_client(tmp_path, handler, clock)
    url = "https://tile.loc.gov/image-services/iiif/x/full/pct:25/0/default.jpg"
    assert client.get_bytes(url) == b"JPEGDATA"
    assert client.get_bytes(url) == b"JPEGDATA"
    assert len(handler.requests) == 1
    dest = tmp_path / "out" / "sheet.jpg"
    assert client.download(url, dest) == dest
    assert dest.read_bytes() == b"JPEGDATA"
    assert len(handler.requests) == 1  # download also served from cache


def _trickle(clock: FakeClock, chunks: int, seconds_per_chunk: float) -> Any:
    """A body that arrives a chunk at a time, spending clock between chunks.

    The shape the image host actually produced: bytes keep coming, so no gap is
    ever long enough to trip the read timeout, and the transfer never ends.
    """

    def body() -> Any:
        for _ in range(chunks):
            clock.now += seconds_per_chunk
            yield b"x" * 8

    return body()


def test_trickled_body_exceeds_its_budget_and_is_retried(tmp_path: Path) -> None:
    """A body no gap timeout can catch must still end the attempt.

    Without the budget this hangs for as long as the server keeps dribbling —
    measured against tile.loc.gov at ~390 B/s with 17-24 s between reads, which
    is hours for one 6 MB master and no log line to say so.
    """

    def responder(request: httpx.Request, n: int) -> httpx.Response:
        return httpx.Response(200, content=_trickle(clock, chunks=100, seconds_per_chunk=20.0))

    clock = FakeClock()
    handler = RecordingHandler(clock, responder)
    client = make_client(tmp_path, handler, clock, body_budget=60.0, max_retries=2)
    with pytest.raises(LOCRequestError):
        client.get_bytes("https://tile.loc.gov/storage-services/service/gmd/x.jp2")
    # Retried like any other transport failure, then given up on: the caller
    # gets a failed page it can report and skip, not a hang.
    assert len(handler.requests) == 3
    assert not list((tmp_path / "cache").glob("*.bin"))


def test_body_within_its_budget_is_returned_whole(tmp_path: Path) -> None:
    def responder(request: httpx.Request, n: int) -> httpx.Response:
        return httpx.Response(200, content=_trickle(clock, chunks=4, seconds_per_chunk=5.0))

    clock = FakeClock()
    handler = RecordingHandler(clock, responder)
    client = make_client(tmp_path, handler, clock, body_budget=60.0)
    assert client.get_bytes("https://tile.loc.gov/storage-services/service/gmd/y.jp2") == b"x" * 32
    assert len(handler.requests) == 1


def test_body_budget_is_per_attempt_not_per_call(tmp_path: Path) -> None:
    """A slow first attempt must not spend the budget of the one that recovers."""

    def responder(request: httpx.Request, n: int) -> httpx.Response:
        if n == 1:
            return httpx.Response(200, content=_trickle(clock, chunks=100, seconds_per_chunk=20.0))
        return httpx.Response(200, content=b"JP2")

    clock = FakeClock()
    handler = RecordingHandler(clock, responder)
    client = make_client(tmp_path, handler, clock, body_budget=60.0, max_retries=3)
    assert client.get_bytes("https://tile.loc.gov/storage-services/service/gmd/z.jp2") == b"JP2"
    assert len(handler.requests) == 2


def test_decoded_body_is_not_decoded_twice(tmp_path: Path) -> None:
    """The rebuilt response must not carry the ENCODED body's framing headers.

    iter_bytes has already un-gzipped the body, so a surviving content-encoding
    would have httpx try to un-gzip the plain bytes and fail — silently turning
    a good master into a failed page.
    """
    import gzip

    payload = b"JP2 MASTER BYTES"

    def responder(request: httpx.Request, n: int) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip", "content-type": "image/jp2"},
            content=gzip.compress(payload),
        )

    clock = FakeClock()
    handler = RecordingHandler(clock, responder)
    client = make_client(tmp_path, handler, clock)
    url = "https://tile.loc.gov/storage-services/service/gmd/gz.jp2"
    assert client.get_bytes(url) == payload


def test_refuses_non_loc_hosts(tmp_path: Path) -> None:
    clock = FakeClock()
    handler = RecordingHandler(clock)
    client = make_client(tmp_path, handler, clock)
    for url in (
        "https://oldinsurancemaps.net/api/volumes",
        "https://example.com/",
        "https://evil-loc.gov.attacker.io/",
        "https://notloc.gov/",
        "http://www.loc.gov/item/x/?fo=json",  # non-https refused too
    ):
        with pytest.raises(HostNotAllowedError):
            client.get_json(url)
    assert handler.requests == []


def test_allows_loc_subdomains(tmp_path: Path) -> None:
    clock = FakeClock()
    handler = RecordingHandler(clock)
    client = make_client(tmp_path, handler, clock)
    client.get_json("https://loc.gov/item/a/?fo=json")
    client.get_json("https://www.loc.gov/item/b/?fo=json")
    client.get_json("https://tile.loc.gov/image-services/c")
    assert len(handler.requests) == 3


def test_refuses_write_methods(tmp_path: Path) -> None:
    clock = FakeClock()
    handler = RecordingHandler(clock)
    client = make_client(tmp_path, handler, clock)
    for method in ("POST", "PUT", "DELETE", "PATCH", "post"):
        with pytest.raises(MethodNotAllowedError):
            client.request(method, "https://www.loc.gov/item/x/?fo=json")
    assert handler.requests == []


def test_search_builds_catalog_url(tmp_path: Path) -> None:
    clock = FakeClock()
    handler = RecordingHandler(clock)
    client = make_client(tmp_path, handler, clock)
    client.search("sanborn chicago", per_page=50)
    url = str(handler.requests[0].url)
    assert url.startswith("https://www.loc.gov/search/?")
    assert "q=sanborn+chicago" in url
    assert "fo=json" in url
    assert "c=50" in url


def test_catalog_results_expose_the_digitized_flag_and_follow_pagination(
    tmp_path: Path,
) -> None:
    """The census, and the file it is made of.

    A city with no `loc_catalog` publishes a layer with no title and no era,
    and nothing produced one. This search IS the material: its raw items are
    the shape `viewer.sources.loc_titles` parses, so writing them out —
    pagination and all — is what `discover --out` does.
    """

    def item(ident: str, date: str, digitized: bool, **extra: Any) -> dict[str, Any]:
        return {
            "id": f"http://www.loc.gov/item/{ident}/",
            "title": f"Sanborn Chicago {ident}",
            "date": date,
            "digitized": digitized,
            **extra,
        }

    page1 = {
        "results": [
            item(
                "sanborn01790_024",
                "1917",
                True,
                description=["Vol. 7, 1917. 118 sheet(s). Bound."],
                resources=[{"files": 118, "url": "https://www.loc.gov/resource/x/"}],
            ),
            # never scanned: the catalog lists it, callers must skip it
            item("sanborn01790_099", "1922", False, resources=[]),
        ],
        "pagination": {"next": "https://www.loc.gov/search/?q=x&fo=json&sp=2"},
    }
    page2 = {
        "results": [item("sanborn01790_034", "1925", True)],
        "pagination": {"next": None},
    }

    def responder(request: httpx.Request, n: int) -> httpx.Response:
        if request.url.params.get("sp") == "2":
            return httpx.Response(200, json=page2)
        return httpx.Response(200, json=page1)

    clock = FakeClock()
    handler = RecordingHandler(clock, responder)
    client = make_client(tmp_path, handler, clock)
    results = client.catalog_results("sanborn chicago")
    census = [client.parse_result(item) for item in results]
    assert [v.item_id for v in census] == [
        "sanborn01790_024",
        "sanborn01790_099",
        "sanborn01790_034",
    ]
    # a caller skips a never-scanned volume on this flag alone
    assert [v.digitized for v in census] == [True, False, True]
    assert (census[0].sheet_count, census[1].sheet_count) == (118, None)
    assert census[0].resource_urls == ("https://www.loc.gov/resource/x/",)
    assert census[2].date == "1925"
    # and the UNPARSED items are a usable loc_catalog, which is the whole
    # reason they are handed out as well as the records
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(results), encoding="utf-8")
    titled = loc_titles(catalog, "Chicago, Ill.")["sanborn01790_024"]
    assert (titled["year"], titled["volume_number"]) == (1917, "7")


@pytest.fixture(scope="module")
def item_fixture() -> dict[str, Any]:
    if not WORK_ITEM_FIXTURE.is_file():
        pytest.skip("cached LOC item fixture not present")
    data: dict[str, Any] = json.loads(WORK_ITEM_FIXTURE.read_text(encoding="utf-8"))
    return data


def test_item_fetch_parses_real_cached_response(
    tmp_path: Path, item_fixture: dict[str, Any]
) -> None:
    def responder(request: httpx.Request, n: int) -> httpx.Response:
        assert request.url.path == "/item/sanborn01790_024/"
        assert request.url.params["fo"] == "json"
        return httpx.Response(200, json=item_fixture)

    clock = FakeClock()
    handler = RecordingHandler(clock, responder)
    client = make_client(tmp_path, handler, clock)
    item = client.item("sanborn01790_024")
    assert item["item"]["digitized"] is True
    assert "sanborn01790_024" in item["item"]["id"]


def test_sheet_image_urls_pick_largest_jpeg_per_sheet(item_fixture: dict[str, Any]) -> None:
    urls = LOCClient.sheet_image_urls(item_fixture)
    n_groups = len(item_fixture["resources"][0]["files"])
    assert len(urls) == n_groups == 118
    assert all(u.startswith("https://tile.loc.gov/") for u in urls)
    assert all(u.endswith("default.jpg") for u in urls)
    # First group's largest jpeg variant is the pct:25 rendition (1830 px tall).
    assert "pct:25" in urls[0]


def test_iiif_service_id_recodes_the_storage_path() -> None:
    url = (
        "https://tile.loc.gov/storage-services/service/gmd/gmd410m/g4104m/"
        "g4104cm/g01790191707/01790_07_1917-0001.jp2"
    )
    assert iiif_service_id(url) == (
        "https://tile.loc.gov/image-services/iiif/"
        "service:gmd:gmd410m:g4104m:g4104cm:g01790191707:01790_07_1917-0001"
    )
    # anything outside the tile.loc.gov storage tree has no image service
    assert iiif_service_id("https://tile.loc.gov/image-services/iiif/x") is None
    assert iiif_service_id("https://example.com/storage-services/a/b.jp2") is None


def test_sheet_iiif_services_maps_pages_from_the_real_item(
    item_fixture: dict[str, Any],
) -> None:
    services = sheet_iiif_services(item_fixture)
    # every sheet group (118) carries a jp2 with a recognizable page tag
    assert len(services) == 118
    assert services["1"] == (
        "https://tile.loc.gov/image-services/iiif/"
        "service:gmd:gmd410m:g4104m:g4104cm:g01790191707:01790_07_1917-0001"
    )
    assert "titl" in services  # word pages ride along; never looked up


def test_sheet_iiif_services_page_tag_grammar() -> None:
    base = "https://tile.loc.gov/storage-services/service/gmd/x/vol-{tag}.jp2"

    def item(*tags: str) -> dict[str, Any]:
        files = [[{"mimetype": "image/jp2", "url": base.format(tag=t), "height": 1}] for t in tags]
        return {"resources": [{"files": files}]}

    services = sheet_iiif_services(item("0023", "0005S", "0000a", "cbd2", "ind1"))
    # paste-up 'S' and continuation letters lower-cased to match local page ids
    assert set(services) == {"23", "5s", "0a", "cbd2", "ind1"}


def test_page_of_sheet_url_preserves_the_case_that_carries_meaning() -> None:
    """Case IS meaning in a page id: an uppercase `S` marks a page that MAY be a
    skeleton twin (`slugs.skeleton_pages` settles it) and a lowercase letter a
    continuation sheet, so the ONE parser every fetcher shares must not fold it.
    `sheet_iiif_services` lower-cases for its own lookup keys, which is a
    different job — hence a test each."""
    base = "https://tile.loc.gov/storage-services/service/gmd/x/01790_49_1950-{tag}.jp2"

    assert page_of_sheet_url(base.format(tag="0023")) == "23"
    assert page_of_sheet_url(base.format(tag="0005S")) == "5S"
    assert page_of_sheet_url(base.format(tag="0000a")) == "0a"
    assert page_of_sheet_url(base.format(tag="titl")) == "titl"
    assert page_of_sheet_url(base.format(tag="cbd2")) == "cbd2"
    # no trailing tag at all: nothing to name the page after
    assert page_of_sheet_url("https://tile.loc.gov/x/nosuchtag.jp2") is None


def test_fetch_index_sheet_downloads_title_sheet(
    tmp_path: Path, item_fixture: dict[str, Any]
) -> None:
    def responder(request: httpx.Request, n: int) -> httpx.Response:
        return httpx.Response(200, content=b"INDEXSHEET")

    clock = FakeClock()
    handler = RecordingHandler(clock, responder)
    client = make_client(tmp_path, handler, clock)
    dest = tmp_path / "index.jpg"
    result = client.fetch_index_sheet(item_fixture, dest)
    assert result == dest
    assert dest.read_bytes() == b"INDEXSHEET"
    fetched = str(handler.requests[0].url)
    assert "titl" in fetched or "index" in fetched.lower()


# ----------------------------------------------------------------------
# Redirect conduct: hops are host-checked BEFORE the request and re-enter
# the rate-limit lane
# ----------------------------------------------------------------------


def test_redirect_to_disallowed_host_never_reaches_the_network(tmp_path: Path) -> None:
    """A loc.gov redirect pointing off-host must be refused BEFORE the hop is
    fetched — following it and checking afterwards already leaked a GET."""

    def responder(request: httpx.Request, n: int) -> httpx.Response:
        if request.url.host == "www.loc.gov":
            return httpx.Response(302, headers={"location": "https://evil.example.com/x"})
        return httpx.Response(200, json={"ok": True})

    clock = FakeClock()
    handler = RecordingHandler(clock, responder)
    client = make_client(tmp_path, handler, clock)
    with pytest.raises(HostNotAllowedError):
        client.get_json("https://www.loc.gov/item/redir/?fo=json")
    hosts = {r.url.host for r in handler.requests}
    assert hosts == {"www.loc.gov"}, f"a request escaped to {hosts - {'www.loc.gov'}}"


def test_redirect_hops_re_enter_the_rate_limit_lane(tmp_path: Path) -> None:
    """Each redirect hop is a real request to loc.gov and must honor the
    >= min_interval spacing like any other request."""

    def responder(request: httpx.Request, n: int) -> httpx.Response:
        if "old-path" in str(request.url):
            return httpx.Response(
                301, headers={"location": "https://www.loc.gov/item/new-path/?fo=json"}
            )
        return httpx.Response(200, json={"ok": True})

    clock = FakeClock()
    handler = RecordingHandler(clock, responder)
    client = make_client(tmp_path, handler, clock)
    body = client.get_json("https://www.loc.gov/item/old-path/?fo=json")
    assert body == {"ok": True}
    assert len(handler.times) == 2
    gaps = [b - a for a, b in zip(handler.times, handler.times[1:], strict=False)]
    assert all(gap >= 5.0 for gap in gaps), f"redirect hop not lane-spaced: {gaps}"


def test_relative_redirect_resolves_against_current_url(tmp_path: Path) -> None:
    def responder(request: httpx.Request, n: int) -> httpx.Response:
        if str(request.url).endswith("/moved/"):
            return httpx.Response(302, headers={"location": "../item/final/?fo=json"})
        return httpx.Response(200, json={"here": str(request.url)})

    clock = FakeClock()
    handler = RecordingHandler(clock, responder)
    client = make_client(tmp_path, handler, clock)
    body = client.get_json("https://www.loc.gov/x/moved/")
    assert body["here"].endswith("/item/final/?fo=json")


def test_endless_redirect_chain_raises(tmp_path: Path) -> None:
    def responder(request: httpx.Request, n: int) -> httpx.Response:
        return httpx.Response(302, headers={"location": f"https://www.loc.gov/loop/{n}/"})

    clock = FakeClock()
    handler = RecordingHandler(clock, responder)
    client = make_client(tmp_path, handler, clock)
    with pytest.raises(LOCRequestError, match="redirect hops"):
        client.get_json("https://www.loc.gov/loop/0/")
