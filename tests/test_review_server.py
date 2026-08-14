"""Route-table coverage for the review server transport (review/server.py).

The real ReviewHandler is driven over a real loopback socket; only the app
behind it is a stub, because the subject here is dispatch — every _GET_ROUTES
arm, the /api/volumes any-length quirk, the _SAFE_FILE re-checks (404, never
400), the Host guard, the ReviewError -> 400 mapping — not the payloads.
tests/test_review_app.py owns the payload/save behaviour on the real ReviewApp.
"""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from autogeoref.errors import ReviewError
from autogeoref.local_server import loopback_server
from autogeoref.paths import VolumePaths
from autogeoref.review.server import ReviewHandler

JPEG = b"\xff\xd8jpegbytes"


class _City:
    name = "Testville"


class StubApp:
    """The transport-facing surface of ReviewApp, with no model behind it.

    media_path vs resolve_media matters: the real media_path is what records
    an overlay as shown, so the stub logs which one each route called.
    """

    include_ok = False
    city = _City()

    def __init__(self, root: Path) -> None:
        self.root = root
        self.ui_dir = root / "ui"
        self.vendor_dir = root / "vendor"
        self.volumes = ["volX"]
        self.media_calls: list[tuple[str, str]] = []
        self.resolve_calls: list[tuple[str, str]] = []

    def paths(self, volume: str) -> VolumePaths:
        return VolumePaths(self.root / volume)

    def media_path(self, volume: str, filename: str) -> Path:
        self.media_calls.append((volume, filename))
        return self.root / volume / "sheets" / filename

    def resolve_media(self, volume: str, filename: str) -> Path:
        self.resolve_calls.append((volume, filename))
        return self.root / volume / "sheets" / filename

    def placed_payload(self, volume: str) -> dict[str, Any]:
        return {"volume": volume, "placed": []}

    def sheet_payload(self, volume: str, page: str) -> dict[str, Any]:
        if page == "999":
            raise ReviewError("no result for page 999")
        return {"volume": volume, "page": page}

    def centerlines_payload(self, volume: str) -> dict[str, Any]:
        assert volume == "volX"
        return {"type": "FeatureCollection", "features": []}


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[int, StubApp]]:
    app = StubApp(tmp_path)
    app.ui_dir.mkdir()
    (app.ui_dir / "index.html").write_text("<html>review</html>")
    app.vendor_dir.mkdir()
    (app.vendor_dir / "leaflet.js").write_text("// vendor")
    sheets = tmp_path / "volX" / "sheets"
    sheets.mkdir(parents=True)
    (sheets / "p4_small.jpg").write_bytes(JPEG)
    (tmp_path / "volX" / "results").mkdir()  # review_queue reads an empty pool
    handler = type("BoundReviewHandler", (ReviewHandler,), {"app": app})
    with loopback_server(handler, 0) as httpd:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            yield httpd.server_address[1], app
        finally:
            httpd.shutdown()


def _request(
    port: int, path: str, *, host: str | None = None, method: str = "GET"
) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        conn.putheader("Host", f"127.0.0.1:{port}" if host is None else host)
        conn.endheaders()
        resp = conn.getresponse()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
    finally:
        conn.close()


# --------------------------------------------------------------- route arms


def test_root_serves_the_index(served: tuple[int, StubApp]) -> None:
    port, _ = served
    status, headers, body = _request(port, "/")
    assert status == 200
    assert headers["content-type"].startswith("text/html")
    assert body == b"<html>review</html>"


def test_vendor_serves_a_static_file(served: tuple[int, StubApp]) -> None:
    port, _ = served
    status, headers, body = _request(port, "/vendor/leaflet.js")
    assert status == 200
    assert headers["content-type"].startswith("text/javascript")
    assert body == b"// vendor"


def test_vendor_missing_file_is_404(served: tuple[int, StubApp]) -> None:
    port, _ = served
    status, _, body = _request(port, "/vendor/missing.js")
    assert status == 404
    assert json.loads(body) == {"error": "not found"}


def test_media_records_the_overlay_and_is_never_cached(served: tuple[int, StubApp]) -> None:
    port, app = served
    status, headers, body = _request(port, "/media/volX/p4_small.jpg")
    assert status == 200
    assert body == JPEG
    assert headers["cache-control"] == "no-store"  # serving IS the overlay-shown proof
    assert app.media_calls == [("volX", "p4_small.jpg")]


def test_refmedia_is_cacheable_and_records_nothing(served: tuple[int, StubApp]) -> None:
    port, app = served
    status, headers, body = _request(port, "/refmedia/volX/p4_small.jpg")
    assert status == 200
    assert body == JPEG
    assert "cache-control" not in headers  # neighbor context: browser MAY cache
    assert app.resolve_calls == [("volX", "p4_small.jpg")]
    assert app.media_calls == []  # never counts as an overlay shown


def test_api_volumes_payload(served: tuple[int, StubApp]) -> None:
    port, _ = served
    status, _, body = _request(port, "/api/volumes")
    assert status == 200
    assert json.loads(body) == {"volumes": ["volX"], "city": "Testville"}


@pytest.mark.parametrize("path", ["/api/volumes/anything", "/api/volumes/a/b/c/d"])
def test_api_volumes_keeps_its_any_length_match(served: tuple[int, StubApp], path: str) -> None:
    # count=None in _GET_ROUTES: extra segments still dispatch (historical quirk)
    port, _ = served
    status, _, body = _request(port, path)
    assert status == 200
    assert json.loads(body) == {"volumes": ["volX"], "city": "Testville"}


def test_api_placed(served: tuple[int, StubApp]) -> None:
    port, _ = served
    status, _, body = _request(port, "/api/placed/volX")
    assert status == 200
    assert json.loads(body) == {"volume": "volX", "placed": []}


def test_api_queue_runs_the_real_queue_reader(served: tuple[int, StubApp]) -> None:
    port, _ = served
    status, _, body = _request(port, "/api/queue/volX")
    assert status == 200
    assert json.loads(body) == []  # empty results dir -> empty reviewable pool


def test_api_sheet(served: tuple[int, StubApp]) -> None:
    port, _ = served
    status, _, body = _request(port, "/api/sheet/volX/4")
    assert status == 200
    assert json.loads(body) == {"volume": "volX", "page": "4"}


def test_api_centerlines(served: tuple[int, StubApp]) -> None:
    port, _ = served
    status, _, body = _request(port, "/api/centerlines/volX")
    assert status == 200
    assert json.loads(body) == {"type": "FeatureCollection", "features": []}


# ----------------------------------------------------------- refusal shapes


@pytest.mark.parametrize(
    "path",
    [
        "/vendor/bad%20name.js",
        "/media/volX/..%2Fp4_small.jpg",
        "/refmedia/volX/..%2Fp4_small.jpg",
    ],
)
def test_unsafe_filenames_are_404_never_400(served: tuple[int, StubApp], path: str) -> None:
    port, app = served
    status, _, body = _request(port, path)
    assert status == 404
    assert json.loads(body) == {"error": "not found"}
    assert app.media_calls == []  # the guard fires before anything is recorded


@pytest.mark.parametrize("path", ["/", "/api/volumes"])
def test_nonloopback_host_is_refused(served: tuple[int, StubApp], path: str) -> None:
    port, _ = served
    status, _, body = _request(port, path, host="evil.example.com")
    assert status == 403
    assert json.loads(body) == {"error": "loopback only"}


def test_review_error_maps_to_400(served: tuple[int, StubApp]) -> None:
    port, _ = served
    status, _, body = _request(port, "/api/sheet/volX/999")
    assert status == 400
    assert json.loads(body) == {"error": "no result for page 999"}


@pytest.mark.parametrize(
    "path",
    [
        "/nope",
        "/nope/nope/nope",
        "/api/nope/volX",
        "/vendor",  # segment-count mismatch on a known prefix
        "/media/volX",
        "/media/volX/p4_small.jpg/extra",
        "/api/queue/volX/extra",
        "/api/sheet/volX",
    ],
)
def test_unknown_paths_are_404(served: tuple[int, StubApp], path: str) -> None:
    port, _ = served
    status, _, body = _request(port, path)
    assert status == 404
    assert json.loads(body) == {"error": "not found"}


def test_head_is_unsupported(served: tuple[int, StubApp]) -> None:
    port, _ = served
    status, _, _ = _request(port, "/", method="HEAD")
    assert status == 501  # no do_HEAD; BaseHTTPRequestHandler refuses the method
