"""The localhost-only review HTTP server (stdlib; no new dependency).

Transport only: this module stays importable without the review model
(:mod:`.app`, :mod:`.materialize` — numpy/pyproj), so the console can inherit
:class:`ReviewHandler` at module level without paying for a map renderer it
may never draw. The model imports below are function-local: by the time a
request runs, :func:`serve` (or the console's) has already paid them, so they
are dict lookups.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import parse_qs, urlparse

from ..errors import ReviewError
from ..local_server import host_allowed, loopback_server
from ..paths import VolumeBusyError

if TYPE_CHECKING:
    from .app import ReviewApp

logger = logging.getLogger(__name__)

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

#: Media-filename validation — a different subject from the volume grammar
#: (validation.volume_id) and the page grammar (slugs.valid_review_page).
_SAFE_FILE = re.compile(r"^[A-Za-z0-9._-]+$")

#: Loose files of the review page itself, served from ``ReviewApp.ui_dir`` next
#: to its index. Declared, not globbed, so the UI directory is not a document
#: root: only what is named here is reachable.
_UI_ASSETS = frozenset({"affine.js"})


class ReviewHandler(BaseHTTPRequestHandler):
    """Routes; all state lives on the class-attached :class:`ReviewApp`."""

    app: ClassVar[ReviewApp]
    server_version = "autogeoref-review"

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s %s", self.address_string(), fmt % args)

    def _send_json(self, code: int, obj: Any) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text: str) -> None:
        data = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, *, store: bool = True) -> None:
        if not path.is_file():
            self._send_json(404, {"error": "not found"})
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type", _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
        )
        self.send_header("Content-Length", str(len(data)))
        if not store:
            # The sheet image IS the ghost overlay, and serving it is what proves a
            # sheet was shown (ReviewApp.overlay_shown). A browser that satisfied
            # the re-open from its own cache would paint the overlay while this
            # server learned nothing — and the verdict that followed would be
            # refused, on a sheet the operator is looking at. So: never cached.
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # ------------------------------------------------------------- GET routes

    def _get_index(self) -> None:
        self._send_file(self.app.ui_dir / "index.html")

    def _get_vendor(self, filename: str) -> None:
        if not _SAFE_FILE.match(filename):
            self._send_json(404, {"error": "not found"})
            return
        self._send_file(self.app.vendor_dir / filename)

    def _get_media(self, volume: str, filename: str) -> None:
        # validates the volume AND records the overlay as shown: this is the
        # ghost raster, so serving it is what `save` demands to have happened
        if not _SAFE_FILE.match(filename):
            self._send_json(404, {"error": "not found"})
            return
        self._send_file(self.app.media_path(volume, filename), store=False)

    def _get_refmedia(self, volume: str, filename: str) -> None:
        # neighbor-context raster: same files, but NEVER records an
        # overlay as shown (and so may be browser-cached)
        if not _SAFE_FILE.match(filename):
            self._send_json(404, {"error": "not found"})
            return
        self._send_file(self.app.resolve_media(volume, filename))

    def _get_placed(self, volume: str) -> None:
        self._send_json(200, self.app.placed_payload(volume))

    def _get_volumes(self) -> None:
        self._send_json(200, {"volumes": self.app.volumes, "city": self.app.city.name})

    def _get_queue(self, volume: str) -> None:
        from .app import review_queue

        query = parse_qs(urlparse(self.path).query)
        include_ok = self.app.include_ok or query.get("all") == ["1"]
        self._send_json(200, review_queue(self.app.paths(volume), volume, include_ok=include_ok))

    def _get_sheet(self, volume: str, page: str) -> None:
        self._send_json(200, self.app.sheet_payload(volume, page))

    def _get_centerlines(self, volume: str) -> None:
        self._send_json(200, self.app.centerlines_payload(volume))

    #: Route table with prefix resolution: fixed prefix -> (total segment count,
    #: handler method), and the segments past the prefix become the method's positional
    #: args. ``None`` keeps ``/api/volumes``'s any-length match, and the file-serving
    #: methods re-check ``_SAFE_FILE`` themselves. Hazard: on a count mismatch at
    #: prefix_len=2 the resolver returns None rather than falling through, so a
    #: 2-segment key whose first segment equals a 1-segment key shadows it.
    _GET_ROUTES: ClassVar[dict[tuple[str, ...], tuple[int | None, str]]] = {
        ("vendor",): (2, "_get_vendor"),
        ("media",): (3, "_get_media"),
        ("refmedia",): (3, "_get_refmedia"),
        ("api", "placed"): (3, "_get_placed"),
        ("api", "volumes"): (None, "_get_volumes"),
        ("api", "queue"): (3, "_get_queue"),
        ("api", "sheet"): (4, "_get_sheet"),
        ("api", "centerlines"): (3, "_get_centerlines"),
    }

    def _get_route(self, parts: list[str]) -> Callable[[], None] | None:
        """The handler for this GET path, or ``None`` for a 404."""
        if not parts:
            return self._get_index
        if len(parts) == 1 and parts[0] in _UI_ASSETS:
            name = parts[0]
            return lambda: self._send_file(self.app.ui_dir / name)
        for prefix_len in (2, 1):
            spec = self._GET_ROUTES.get(tuple(parts[:prefix_len]))
            if spec is None:
                continue
            count, name = spec
            if count is not None and len(parts) != count:
                return None
            args = parts[prefix_len:] if count is not None else []
            method: Callable[..., None] = getattr(self, name)
            return lambda: method(*args)
        return None

    def do_GET(self) -> None:
        if not host_allowed(self.headers.get("Host", "")):
            self._send_json(403, {"error": "loopback only"})
            return
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        try:
            route = self._get_route(parts)
            if route is None:
                self._send_json(404, {"error": "not found"})
                return
            route()
        except ReviewError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception:
            logger.exception("GET %s failed", self.path)
            self._send_json(500, {"error": "internal error (see server log)"})

    def do_POST(self) -> None:
        from .apply import apply_reviews

        if not host_allowed(self.headers.get("Host", "")):
            self._send_json(403, {"error": "loopback only"})
            return
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        try:
            if parts[:2] == ["api", "sidecar"] and len(parts) == 4:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                code, payload = self.app.save(parts[2], parts[3], body)
                self._send_json(code, payload)
            elif parts[:2] == ["api", "apply"] and len(parts) == 3:
                # same route shape and semantics as the console's apply: `do_warp=False` because
                # an HTTP request is no place to spend ten minutes in gdalwarp — the summary's
                # rerun_hint says what is still owed. The Content-Type bar mirrors the console's
                # CSRF guard: unlike /api/sidecar (whose base sha a hostile page cannot read),
                # this route needs no secret, so a bare cross-site POST would fire it blind;
                # application/json makes the browser preflight, which this server never grants.
                if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                    self._send_json(
                        415, {"error": "POST bodies must be application/json (CSRF guard)"}
                    )
                    return
                summary = apply_reviews(self.app.paths(parts[2]), parts[2], do_warp=False)
                self._send_json(200, summary)
            else:
                self._send_json(404, {"error": "not found"})
        except VolumeBusyError as exc:
            # apply_reviews takes the volume lock; a run/prep owns the tree
            self._send_json(409, {"error": str(exc)})
        except ReviewError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception:
            logger.exception("POST %s failed", self.path)
            self._send_json(500, {"error": "internal error (see server log)"})


def serve(app: ReviewApp, port: int = 8765) -> None:
    """Serve the review UI on localhost (never any other interface)."""
    handler = type("BoundReviewHandler", (ReviewHandler,), {"app": app})
    with loopback_server(handler, port) as httpd:
        logger.info("review UI: http://127.0.0.1:%d/  (Ctrl-C to stop)", port)
        print(f"review UI: http://127.0.0.1:{port}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("review UI stopped")
