"""The one page: the console's HTTP handler, its CSRF rule, and the server."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlparse

from ..cli_context import apply_reviews_locked, packaged_ui
from ..errors import ReviewError
from ..local_server import host_allowed, loopback_server
from ..logfiles import tail_log
from ..paths import VolumeBusyError
from ..queue.store import TRACKS, QueueError
from ..review.server import ReviewHandler

if TYPE_CHECKING:
    from .actions import ConsoleActions

logger = logging.getLogger(__name__)


def csrf_safe(sec_fetch_site: str | None, content_type: str | None) -> bool:
    """May a MUTATING request with these headers proceed? The console's CSRF rule.

    The loopback bind plus the Host header is not enough once the routes MUTATE and spend
    money: on a DIRECT cross-origin ``fetch`` the browser sets ``Host`` to this server — the
    real target, not the attacker's domain — so that check passes and cannot help. Two
    independent bars, EITHER sufficient: ``Sec-Fetch-Site`` must be ``same-origin`` or ``none``
    (missing is inconclusive), or ``Content-Type`` must be ``application/json``, which is not
    CORS-safelisted and so forces a preflight this server answers 501. A `curl` with neither
    still passes, but `curl` here is the trusted operator; the vector closed is the browser.
    """
    if sec_fetch_site is not None and sec_fetch_site not in ("same-origin", "none"):
        return False
    ctype = (content_type or "").split(";")[0].strip().lower()
    return ctype == "application/json"


#: Path-safety allowlist for a volume id in a log URL (same alphabet the queue's
#: own board uses).
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

#: A log is written per (volume, LEG), and the legs are the things that actually
#: run — `all` is a track you enqueue on, never a log that exists. `queue.store.log_path`
#: is keyed the same way (`queue.command._run_leg`), one leg per track.
_LOG_LEGS = TRACKS

#: The console page's own scripts, served BEFORE the review handler is asked
#: for anything. The inherited fallthrough 404s whenever there is no review app,
#: which is exactly the configuration an operator runs for a bare queue — and a
#: scriptless board looks like a broken page, not a missing city.
_CONSOLE_ASSETS = frozenset({"board.js"})


class ConsoleRoutes(ReviewHandler):
    """Console routes first; everything else is the review server's, unchanged.

    Inherits :class:`review.server.ReviewHandler` directly — a transport-only
    module, so the base import costs no numpy/pyproj and ``queue --candidates``
    never pays for a map renderer it never draws. The review MODEL stays
    deferred. The five values the handler used to close over are class
    attributes, bound by :func:`serve`'s ``type()`` call.
    """

    server_version = "autogeoref-console"

    work: Path
    console_ui: Path
    build_board: Callable[[], dict[str, Any]]
    actions: ConsoleActions
    review_app: Any = None

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(body, dict):
            raise ReviewError("expected a JSON object")
        return body

    def _guard(self) -> bool:
        if host_allowed(self.headers.get("Host", "")):
            return True
        self._send_json(403, {"error": "loopback only"})
        return False

    def _mutation_guard(self) -> bool:
        """Refuse a cross-site POST — the Host guard cannot, once routes MUTATE.

        The decision is :func:`csrf_safe`; this only turns a `False` into the right
        status. A cross-site label is a 403; a wrong content type is a 415, because
        those are different fixes (one is "you are not us", the other "send JSON").
        """
        site = self.headers.get("Sec-Fetch-Site")
        if site is not None and site not in ("same-origin", "none"):
            self._send_json(403, {"error": "cross-site request refused"})
            return False
        if not csrf_safe(site, self.headers.get("Content-Type")):
            self._send_json(415, {"error": "POST bodies must be application/json (CSRF guard)"})
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802
        if not self._guard():
            return
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        try:
            if not parts:
                self._send_file(self.console_ui / "index.html")
                return
            if len(parts) == 1 and parts[0] in _CONSOLE_ASSETS:
                self._send_file(self.console_ui / parts[0])
                return
            # the review UI, on this port and this origin, so its absolute
            # /api/… and /media/… fetches land on the routes below unchanged
            if parts == ["review"] and self.review_app is not None:
                self._send_file(self.review_app.ui_dir / "index.html")
                return
            if parts == ["api", "board"]:
                self._send_json(200, self.build_board())
                return
            if (
                len(parts) == 4
                and parts[:2] == ["api", "log"]
                and _SAFE_NAME.match(parts[2])
                and parts[3] in _LOG_LEGS
            ):
                self._send_text(tail_log(self.work, parts[2], parts[3]))
                return
        except ReviewError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        if self.review_app is None:
            self._send_json(404, {"error": "not found"})
            return
        super().do_GET()  # /vendor, /media, /api/{volumes,queue,sheet,centerlines}

    # -------------------------------------------------------------- POST routes

    def _post_enqueue(self) -> None:
        b = self._body()
        # then_serve defaults True (end to end); the UI's "review first"
        # box sends then_serve=false, which parks it at needs-review.
        self._send_json(
            200,
            self.actions.enqueue(
                b["volume"],
                str(b.get("track") or "place"),
                then_serve=bool(b.get("then_serve", True)),
            ),
        )

    def _post_dequeue(self) -> None:
        b = self._body()
        self._send_json(200, self.actions.dequeue(b["volume"], b.get("track") or None))

    def _post_retry_failed_reads(self) -> None:
        b = self._body()
        self._send_json(
            200,
            self.actions.retry_failed_reads(b["volume"], str(b.get("track") or "place")),
        )

    def _post_drain(self) -> None:
        b = self._body()
        self._send_json(
            200,
            self.actions.start_drain(
                str(b.get("target") or "both"),
                annotate_jobs=b.get("annotate_jobs"),
                serve_lanes=b.get("serve_lanes"),
            ),
        )

    def _post_drain_stop(self) -> None:
        b = self._body()
        self._send_json(200, self.actions.stop_drain(str(b.get("target") or "both")))

    def _post_apply(self, volume: str) -> None:
        # The reviewer's verdicts, materialised by review.apply's OWN apply_reviews — the
        # do_warp=False rationale and the busy contract live on
        # cli_context.apply_reviews_locked. `paths()` validates the volume id
        # (a bad one is a 400, not a 500). The summary says what is still
        # owed; nothing here re-implements any of it.
        summary = apply_reviews_locked(self.review_app.paths(volume), volume, do_warp=False)
        self._send_json(200, summary)

    _POST_ROUTES: ClassVar[dict[tuple[str, ...], str]] = {
        ("api", "enqueue"): "_post_enqueue",
        ("api", "dequeue"): "_post_dequeue",
        ("api", "retry-failed-reads"): "_post_retry_failed_reads",
        ("api", "drain"): "_post_drain",
        ("api", "drain", "stop"): "_post_drain_stop",
    }

    def _post_route(self, parts: list[str]) -> Callable[[], None] | None:
        """The handler for this POST path, or ``None`` for the review fall-through."""
        name = self._POST_ROUTES.get(tuple(parts))
        if name is not None:
            handler: Callable[[], None] = getattr(self, name)
            return handler
        if parts[:2] == ["api", "apply"] and len(parts) == 3 and self.review_app is not None:
            volume = parts[2]
            return lambda: self._post_apply(volume)
        return None

    @contextmanager
    def _json_errors(self) -> Iterator[None]:
        """One JSON error contract for every POST route — the arms name different fixes."""
        try:
            yield
        except KeyError as exc:
            self._send_json(400, {"error": f"missing field {exc}"})
        except QueueError as exc:
            # the queue refused: already queued, a drain is live, nothing to serve.
            # 409 — the request was well-formed and the WORLD said no.
            self._send_json(409, {"error": str(exc)})
        except VolumeBusyError as exc:
            # apply_reviews takes the volume lock; a run/prep owns the tree
            self._send_json(409, {"error": str(exc)})
        except ReviewError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception:
            logger.exception("POST %s failed", self.path)
            self._send_json(500, {"error": "internal error (see server log)"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._guard() or not self._mutation_guard():
            return
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        route = self._post_route(parts)
        if route is not None:
            with self._json_errors():
                route()
            return
        if self.review_app is None:
            self._send_json(404, {"error": "not found"})
            return
        super().do_POST()  # /api/sidecar — the overlay gate lives in review.save


def serve(
    work: Path,
    *,
    build_board: Any,
    actions: ConsoleActions,
    review_app: Any = None,
    ui_dir: Path | None = None,
    port: int = 8766,
) -> None:
    """ONE page that shows the state and drives it.

    The review pane is not a rebuild of the review UI — it IS the review UI, served from this
    port and talking to the review server's own routes, which :class:`ConsoleRoutes` inherits
    rather than restating. A verdict recorded here lands in the same sidecar and is materialised
    by the same ``--apply``. ``review_app`` is optional: with no city config there is nothing to
    review against, so the console serves its board and says the pane is unavailable rather than
    offering a review surface that would be wrong.
    """
    console_ui = ui_dir or packaged_ui("queue_ui")

    # `app` is what the inherited review routes read. It is bound only when there IS
    # one: with no city there is no review pane, every review route 404s before it
    # could reach `super()`, and a placeholder app would just be a way to get a
    # confusing error instead of an honest absence.
    attrs: dict[str, Any] = {
        "work": work,
        "console_ui": console_ui,
        # staticmethod, or storing a plain function in the class dict would
        # rebind it as a method and hand it `self`
        "build_board": staticmethod(build_board),
        "actions": actions,
        "review_app": review_app,
    }
    if review_app is not None:
        attrs["app"] = review_app
    handler = type("BoundConsoleHandler", (ConsoleRoutes,), attrs)
    with loopback_server(handler, port) as httpd:
        print(f"console: http://127.0.0.1:{port}/")
        if actions.can_act:
            print("  acting ENABLED — enqueue, drain, review and serve from the page")
        else:
            print(
                "  READ-ONLY: started without --city, so every action button is DISABLED.\n"
                "  To act (enqueue / drain / review), restart with a city config:\n"
                f"    autogeoref queue --serve --city <city.toml> --work {work}"
            )
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("console stopped")
