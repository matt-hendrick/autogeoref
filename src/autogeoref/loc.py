"""Library of Congress acquisition client for Sanborn map volumes.

Wraps the loc.gov JSON API and tile.loc.gov image services behind a polite, cache-first
HTTP client. Conduct contract:

* ONE request lane per cache directory: every request is serialized and spaced at least
  ``min_interval`` apart, across cooperating processes.
* Capped exponential backoff on 429 / 5xx / transport errors, still spaced.
* A wall-clock BUDGET on every response body — the read timeout bounds the gap between
  chunks and cannot end a transfer that trickles forever.
* Every GET is cached on disk; a cache hit issues no request.
* GET only, against loc.gov / tile.loc.gov, redirects followed MANUALLY with each hop
  host-checked and re-entering the request lane.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlencode, urlsplit

import httpx

from .paths import atomic_output_path, atomic_write_text

logger = logging.getLogger(__name__)


def _contact_from_env() -> str:
    """Honest-UA contact (LOC conduct contract): AUTOGEOREF_CONTACT env var
    (a reachable email/URL), falling back to the repo URL — an empty or
    blank value falls back too, so the header is never anonymous."""
    contact = (os.environ.get("AUTOGEOREF_CONTACT") or "").strip()
    return contact or "https://github.com/matt-hendrick/autogeoref"


USER_AGENT = f"autogeoref/0.1 (contact: {_contact_from_env()})"
ALLOWED_HOSTS = ("loc.gov", "tile.loc.gov")
SEARCH_URL = "https://www.loc.gov/search/"
ITEM_URL = "https://www.loc.gov/item/{item_id}/?fo=json"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})
MAX_REDIRECT_HOPS = 5
#: Headers that describe the body as it arrived on the wire. A body read chunk
#: by chunk is already decoded and re-framed, so these must not ride along onto
#: the rebuilt response.
_BODY_FRAMING_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding"})


class _RequestLane:
    """The in-process half of a cache-directory request lane."""

    def __init__(self) -> None:
        self.lock = threading.Lock()


_REQUEST_LANES: dict[Path, _RequestLane] = {}
_REQUEST_LANES_LOCK = threading.Lock()


def _request_lane(cache_dir: Path) -> _RequestLane:
    key = cache_dir.resolve()
    with _REQUEST_LANES_LOCK:
        lane = _REQUEST_LANES.get(key)
        if lane is None:
            lane = _RequestLane()
            _REQUEST_LANES[key] = lane
        return lane


@contextmanager
def _locked_request_lane(cache_dir: Path) -> Iterator[Path]:
    """Hold the cross-process cache lane and yield its request timestamp file."""
    lock_path = cache_dir / ".loc-request.lock"
    timestamp_path = cache_dir / ".loc-request-at"
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield timestamp_path
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _boot_id() -> str:
    """Identify the monotonic-clock epoch on Linux for persisted lane times."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def _load_request_timestamp(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        saved_boot_id, saved_timestamp = path.read_text(encoding="utf-8").split()
        if saved_boot_id != _boot_id():
            logger.info("%s: LOC request timestamp has a different boot id; resetting lane", path)
            return None
        return float(saved_timestamp)
    except ValueError:
        logger.warning("%s: invalid LOC request timestamp; resetting lane", path)
        return None


class LOCError(Exception):
    """Base class for all LOC client errors."""


class HostNotAllowedError(LOCError):
    """URL does not point at an allowed loc.gov host."""


class MethodNotAllowedError(LOCError):
    """A non-GET (write-capable) HTTP method was requested."""


class LOCRequestError(LOCError):
    """The request failed after exhausting all retries."""


class _BodyBudgetExceeded(httpx.ReadTimeout):
    """A body that outran ``body_budget``, raised INSIDE one attempt.

    A ``ReadTimeout`` on purpose: the retry ladder already treats every
    transport error as retryable, and a shaped body deserves the same backoff
    as a dead one. The subclass exists only so the ladder can remember WHICH
    kind of timeout ended each attempt.
    """


class BodyStalledError(LOCRequestError):
    """Every attempt ran out of ``body_budget`` mid-body.

    Distinct from its parent because the remedy is: the host is answering, and
    it is shaping the bytes rather than refusing them, so this is not a
    transient one URL will recover from on the next try. A caller pulling many
    large bodies should STOP the pass rather than spend the retry ladder on
    every remaining one.
    """


class LOCResponseError(LOCError):
    """The response body could not be interpreted (e.g. bad JSON)."""


@dataclass(frozen=True)
class VolumeRecord:
    """One volume from a catalog census.

    ``item_id`` is the LOC item identifier when it could be derived from the
    record URL, else the raw id URL. ``digitized`` says whether LOC actually
    scanned this volume — the catalog lists volumes that were never scanned,
    and callers must skip those.
    """

    item_id: str
    title: str
    date: str
    digitized: bool
    sheet_count: int | None
    resource_urls: tuple[str, ...]


def _derive_item_id(raw_id: str) -> str:
    """Reduce a catalog id URL like ``.../item/<slug>/`` to its slug."""
    parts = [p for p in urlsplit(raw_id).path.split("/") if p]
    if len(parts) >= 2 and parts[-2] == "item":
        return parts[-1]
    return raw_id


def catalog_year(catalog: Mapping[str, dict[str, Any]], volume: str) -> int | None:
    """Edition year from a catalog entry — stored as an int or a digit string."""
    year = catalog.get(volume, {}).get("year")
    return int(year) if isinstance(year, int | str) and str(year).isdigit() else None


class LOCClient:
    """Cache-first, cache-lane-rate-limited, GET-only client for loc.gov.

    ``http_client``, ``clock`` and ``sleep`` are injectable for tests; a
    caller-supplied client stays the caller's to close. ``timeout`` bounds the
    gap between two chunks of a body, NOT the body — the image host leaves long
    gaps inside a large body, so it must be generous, and a generous gap
    timeout cannot tell a trickle from progress. ``body_budget`` closes that
    hole; exceeding it raises ``httpx.ReadTimeout``, so a strangled transfer
    retries like any other transport failure.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        http_client: httpx.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        min_interval: float = 5.0,
        timeout: float = 30.0,
        body_budget: float = 300.0,
        max_retries: int = 4,
        backoff_base: float = 5.0,
        backoff_cap: float = 120.0,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._owns_client = http_client is None
        # Redirects are checked and rate-limited one hop at a time.
        self._client = (
            http_client if http_client is not None else httpx.Client(follow_redirects=False)
        )
        self._clock = clock
        self._sleep = sleep
        self._min_interval = min_interval
        self._timeout = timeout
        self._body_budget = body_budget
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._lane = _request_lane(self.cache_dir)
        self._last_request_at: float | None = None
        self._request_timestamp_path: Path | None = None

    @property
    def min_interval(self) -> float:
        """Seconds this client spaces requests by — the pacing floor a caller
        quotes when it prices a fetch."""
        return self._min_interval

    def close(self) -> None:
        """Close the underlying HTTP client if this instance owns it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Conduct guards
    # ------------------------------------------------------------------

    @staticmethod
    def _check_url(url: str) -> None:
        """Raise unless ``url`` is https on an allowed loc.gov host."""
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        allowed = any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS)
        if parts.scheme != "https" or not allowed:
            raise HostNotAllowedError(
                f"refusing request to {url!r}: only https URLs under "
                f"{', '.join(ALLOWED_HOSTS)} are permitted"
            )

    def request(self, method: str, url: str) -> httpx.Response:
        """Issue a rate-limited request. Only ``GET`` is ever allowed.

        This is the single choke point for network traffic: every public
        method routes through it. Non-GET methods raise
        :class:`MethodNotAllowedError` before any request is built, and
        non-loc.gov URLs raise :class:`HostNotAllowedError`.
        """
        if method.upper() != "GET":
            raise MethodNotAllowedError(
                f"refusing {method!r} request: this client is read-only (GET only)"
            )
        self._check_url(url)
        with self._request_lane():
            return self._get_with_retries(url)

    @contextmanager
    def _request_lane(self) -> Iterator[None]:
        """Serialize cache misses and network requests for this cache directory."""
        with self._lane.lock, _locked_request_lane(self.cache_dir) as timestamp_path:
            self._last_request_at = _load_request_timestamp(timestamp_path)
            self._request_timestamp_path = timestamp_path
            try:
                yield
            finally:
                self._request_timestamp_path = None

    def _wait_for_lane(self) -> None:
        if self._last_request_at is not None:
            elapsed = self._clock() - self._last_request_at
            remaining = self._min_interval - elapsed
            if remaining > 0:
                self._sleep(remaining)

    def _follow_get(self, url: str) -> httpx.Response:
        """One GET, following redirects manually: every hop is host-checked
        BEFORE its request is issued and re-enters the rate-limit lane."""
        current = url
        for _ in range(MAX_REDIRECT_HOPS + 1):
            self._check_url(current)
            self._wait_for_lane()
            self._last_request_at = self._clock()
            if self._request_timestamp_path is None:
                raise RuntimeError("LOC request attempted outside its request lane")
            atomic_write_text(self._request_timestamp_path, f"{_boot_id()} {self._last_request_at}")
            response = self._budgeted_get(current)
            location = response.headers.get("location")
            if response.status_code in REDIRECT_STATUS and location:
                current = str(httpx.URL(current).join(location))
                logger.debug("redirect %d -> %s", response.status_code, current)
                continue
            return response
        raise LOCRequestError(f"GET {url}: more than {MAX_REDIRECT_HOPS} redirect hops")

    def _budgeted_get(self, url: str) -> httpx.Response:
        """One GET whose BODY is read under ``body_budget``, not just ``timeout``.

        Streams so the clock can be checked between chunks: the timeout bounds
        the gap between two of them, so a body that keeps trickling resets it
        forever. Exceeding the budget raises ``httpx.ReadTimeout`` — the same
        class a dead gap raises — so the retry ladder above needs no new case.

        Returns an ordinary fully-read response, so every caller keeps reading
        ``.content`` / ``.text`` / ``.headers`` as before.
        """
        request = self._client.build_request(
            "GET",
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=self._timeout,
        )
        response = self._client.send(request, stream=True)
        try:
            started = self._clock()
            chunks: list[bytes] = []
            received = 0
            for chunk in response.iter_bytes():
                chunks.append(chunk)
                received += len(chunk)
                elapsed = self._clock() - started
                if elapsed > self._body_budget:
                    raise _BodyBudgetExceeded(
                        f"body of {url} exceeded the {self._body_budget:.0f}s budget "
                        f"({received} bytes in {elapsed:.0f}s)",
                        request=request,
                    )
        finally:
            response.close()
        # Rebuilt rather than returned as-is, because a streamed response cannot
        # expose .content. Headers carry over so charset and Location survive,
        # minus the three that describe the ENCODED body: iter_bytes already
        # decoded it, so a surviving content-encoding would have httpx decode
        # the decoded bytes a second time.
        headers = [
            (name, value)
            for name, value in response.headers.multi_items()
            if name.lower() not in _BODY_FRAMING_HEADERS
        ]
        return httpx.Response(
            response.status_code,
            headers=headers,
            content=b"".join(chunks),
            request=request,
        )

    def _get_with_retries(self, url: str) -> httpx.Response:
        last_error: str = "no attempts made"
        # Whether the LAST attempt died on the body budget, which decides the
        # class of the give-up error: a shaped body is a different situation
        # from a flaky one and the caller has to be able to tell (BodyStalled).
        stalled = False
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                delay = min(self._backoff_base * 2 ** (attempt - 1), self._backoff_cap)
                logger.warning(
                    "retrying %s in %.1fs (attempt %d): %s", url, delay, attempt + 1, last_error
                )
                self._sleep(delay)
            try:
                response = self._follow_get(url)
            except _BodyBudgetExceeded as exc:
                last_error = f"transport error: {exc}"
                stalled = True
                continue
            except httpx.TransportError as exc:
                last_error = f"transport error: {exc}"
                stalled = False
                continue
            if response.status_code in RETRYABLE_STATUS:
                last_error = f"HTTP {response.status_code}"
                continue
            if response.status_code >= 400:
                raise LOCRequestError(f"GET {url} failed: HTTP {response.status_code}")
            if response.status_code >= 300:
                # a redirect without a Location header (or one an injected
                # follow_redirects client failed to resolve) is not content
                raise LOCRequestError(
                    f"GET {url} answered {response.status_code} redirect; refusing to "
                    f"treat it as content (location: {response.headers.get('location')})"
                )
            # belt-and-braces: an injected client that follows redirects
            # itself must still land on an allowed host
            self._check_url(str(response.url))
            return response
        give_up = BodyStalledError if stalled else LOCRequestError
        raise give_up(f"GET {url} failed after {self._max_retries + 1} attempts: {last_error}")

    # ------------------------------------------------------------------
    # Disk cache
    # ------------------------------------------------------------------

    def _cache_path(self, url: str, suffix: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}{suffix}"

    def get_json(self, url: str) -> Any:
        """GET a JSON document, serving from the disk cache when present.

        The raw response text is dumped to the cache per catalog request;
        a cache hit issues no network request.
        """
        self._check_url(url)
        path = self._cache_path(url, ".json")
        if path.is_file():
            logger.debug("cache hit for %s", url)
            text = path.read_text(encoding="utf-8")
        else:
            with self._request_lane():
                if path.is_file():
                    logger.debug("cache filled while waiting for %s", url)
                    text = path.read_text(encoding="utf-8")
                else:
                    response = self._get_with_retries(url)
                    text = response.text
                    try:
                        json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise LOCResponseError(f"GET {url} returned invalid JSON: {exc}") from exc
                    atomic_write_text(path, text)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LOCResponseError(f"cached body for {url} is invalid JSON: {exc}") from exc

    def get_bytes(self, url: str) -> bytes:
        """GET a binary body (image file), serving from the disk cache when present."""
        self._check_url(url)
        path = self._cache_path(url, ".bin")
        if path.is_file():
            logger.debug("cache hit for %s", url)
            return path.read_bytes()
        with self._request_lane():
            if path.is_file():
                logger.debug("cache filled while waiting for %s", url)
                return path.read_bytes()
            response = self._get_with_retries(url)
            with atomic_output_path(path) as temporary:
                temporary.write_bytes(response.content)
            return response.content

    def download(self, url: str, dest: Path) -> Path:
        """Download ``url`` to ``dest`` (through the cache) and return ``dest``.

        The write is atomic: a body this large is minutes of transfer, and a
        process killed mid-write would otherwise leave a truncated file that a
        later ``dest.exists()`` check reads as a complete download.
        """
        data = self.get_bytes(url)
        with atomic_output_path(dest) as temporary:
            temporary.write_bytes(data)
        return dest

    def forget_cached(self, url: str) -> bool:
        """Drop ``url``'s cached body. True if there was one. Never touches JSON.

        For a caller that has already written the bytes somewhere durable and
        does its own "is it on disk?" check — a whole-corpus image fetch, where
        the cache copy would be a second copy of every master. Idempotence does
        not depend on the cache there: the fetcher skips a page whose file
        exists. Catalog JSON stays; it is small and makes a re-run free.
        """
        path = self._cache_path(url, ".bin")
        try:
            path.unlink()
        except OSError:
            return False
        return True

    # ------------------------------------------------------------------
    # Catalog API
    # ------------------------------------------------------------------

    def search(self, query: str, *, page: int | None = None, per_page: int = 100) -> Any:
        """Run one catalog search request and return the parsed JSON."""
        params: dict[str, str] = {"q": query, "fo": "json", "c": str(per_page)}
        if page is not None:
            params["sp"] = str(page)
        return self.get_json(f"{SEARCH_URL}?{urlencode(params)}")

    def item(self, item_id: str) -> Any:
        """Fetch item metadata JSON for ``item_id``."""
        return self.get_json(ITEM_URL.format(item_id=item_id))

    def catalog_results(self, query: str, *, max_pages: int = 10) -> list[dict[str, Any]]:
        """The catalog's raw ``results`` items for ``query``, pagination followed
        up to ``max_pages`` pages.

        Unparsed, because this IS the shape a ``loc_catalog`` file holds —
        ``viewer.sources.loc_titles`` reads these items directly, so writing the
        list out produces one. :meth:`parse_result` turns them into records.
        """
        out: list[dict[str, Any]] = []
        payload = self.search(query)
        for page in range(max_pages):
            out.extend(dict(r) for r in payload.get("results") or [])
            next_url = (payload.get("pagination") or {}).get("next")
            # don't fetch (and then discard) a page beyond the cap
            if not next_url or page == max_pages - 1:
                break
            payload = self.get_json(str(next_url))
        return out

    @staticmethod
    def parse_result(result: Mapping[str, Any]) -> VolumeRecord:
        """One catalog item as a volume record. It carries the ``digitized``
        flag: the catalog lists volumes that were never scanned, and callers
        must skip those."""
        resources = result.get("resources") or []
        sheet_count: int | None = None
        urls: list[str] = []
        for resource in resources:
            if not isinstance(resource, Mapping):
                continue
            files = resource.get("files")
            if sheet_count is None and isinstance(files, int):
                sheet_count = files
            url = resource.get("url")
            if isinstance(url, str):
                urls.append(url)
        date = result.get("date")
        return VolumeRecord(
            item_id=_derive_item_id(str(result.get("id", ""))),
            title=str(result.get("title", "")),
            date=str(date) if date is not None else "",
            digitized=bool(result.get("digitized", False)),
            sheet_count=sheet_count,
            resource_urls=tuple(urls),
        )

    # ------------------------------------------------------------------
    # Sheet images
    # ------------------------------------------------------------------

    @staticmethod
    def sheet_image_urls(
        item_json: Mapping[str, Any], *, mimetype: str = "image/jpeg"
    ) -> list[str]:
        """Extract one download URL per sheet from an item JSON document.

        For each file group under ``resources[*].files`` the largest
        variant of ``mimetype`` (by pixel height) is chosen.
        """
        urls: list[str] = []
        for resource in item_json.get("resources") or []:
            for group in resource.get("files") or []:
                best_url: str | None = None
                best_height = -1
                for variant in group:
                    if not isinstance(variant, Mapping):
                        continue
                    if variant.get("mimetype") != mimetype:
                        continue
                    url = variant.get("url")
                    if not isinstance(url, str):
                        continue
                    height = variant.get("height")
                    h = int(height) if isinstance(height, int | float) else 0
                    if h > best_height:
                        best_height = h
                        best_url = url
                if best_url is not None:
                    urls.append(best_url)
        return urls

    def fetch_index_sheet(self, item_json: Mapping[str, Any], dest: Path) -> Path:
        """Download the volume's index/title sheet to ``dest``.

        Chooses the first sheet whose URL stem mentions ``index`` or
        ``titl`` (LOC's naming for title/index pages), falling back to the
        first sheet of the volume.
        """
        urls = self.sheet_image_urls(item_json)
        if not urls:
            raise LOCResponseError("item JSON contains no image files")
        chosen = next(
            (u for u in urls if "index" in u.lower() or "titl" in u.lower()),
            urls[0],
        )
        return self.download(chosen, dest)


#: IIIF Image API root of the tile.loc.gov image service.
IIIF_SERVICE_ROOT = "https://tile.loc.gov/image-services/iiif/"

#: Trailing page tag of a LOC sheet file stem: numbered sheets (``0023``,
#: paste-up ``0005S``, continuation ``0000a``) and word pages (``titl``,
#: ``ind1``, ``cbd2``, ``CBDa``). The word form is CASE-INSENSITIVE because
#: LOC's own filenames are not, and `slugs._PAGE_RE` already admits both
#: spellings; a lowercase-only pattern made a whole volume look as though it
#: had no fetchable page at all.
_SHEET_TAG_RE = re.compile(r"-(\d{4}[A-Za-z]?|[A-Za-z]+\d*)$")


def iiif_service_id(url: str) -> str | None:
    """``tile.loc.gov`` storage-services file URL -> IIIF Image API service id.

    The image service exposes every storage path with slashes recoded as
    colons (verified by the rendered Allmaps proof); returns ``None`` for
    URLs outside that storage tree.
    """
    parts = urlsplit(url)
    prefix = "/storage-services/"
    if parts.hostname != "tile.loc.gov" or not parts.path.startswith(prefix):
        return None
    head, _, name = parts.path[len(prefix) :].rpartition("/")
    stem = name.rsplit(".", 1)[0]
    return IIIF_SERVICE_ROOT + ":".join([*head.split("/"), stem])


def page_of_sheet_url(url: str) -> str | None:
    """LOC sheet file URL -> this repo's page id, or ``None`` if it carries none.

    ``-0023.jp2`` -> ``'23'``, ``-0005S.jp2`` -> ``'5S'``, ``-titl.jp2`` ->
    ``'titl'``. Zero padding is stripped from the numeric part; **case is
    preserved**, because case is meaning in a page id — an uppercase ``S`` marks
    a skeleton twin and a lowercase letter a continuation sheet. THE one place a
    LOC URL becomes a page id, so the fetchers and the IIIF lookup cannot
    disagree about which page a file is.
    """
    stem = urlsplit(url).path.rpartition("/")[2].rsplit(".", 1)[0]
    match = _SHEET_TAG_RE.search(stem)
    if match is None:
        return None
    tag = match.group(1)
    return f"{int(tag[:4])}{tag[4:]}" if tag[:4].isdigit() else tag


def sheet_iiif_services(item_json: Mapping[str, Any]) -> dict[str, str]:
    """Lower-cased page id -> IIIF service id, from an item's jp2 variants.

    Page ids follow the local grammar (``'23'``, ``'5S'`` -> ``'5s'``,
    ``'cbd2'``, ``'CBDa'`` -> ``'cbda'``); non-map pages (``titl``, ``ind1``,
    ...) appear too and are simply never looked up.
    """
    services: dict[str, str] = {}
    for url in LOCClient.sheet_image_urls(item_json, mimetype="image/jp2"):
        service = iiif_service_id(url)
        if service is None:
            continue
        page = page_of_sheet_url(url)
        if page is None:
            continue
        services[page.lower()] = service
    return services
