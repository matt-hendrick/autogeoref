"""What the console server hands a browser, and what it refuses.

The page is packaged with the wheel, so a script tag it does not carry is a
board that renders nothing — and the script route has to work on a console with
no review app behind it, which is exactly how a bare queue is run. Nothing else
is served from that tree. The write routes spend money, so a cross-site post is
refused on the fetch metadata rather than on the Host header alone.
"""

from __future__ import annotations

from pathlib import Path

from autogeoref import cli_context
from autogeoref.console import server as console_server
from autogeoref.queue import store as qstore


def test_the_board_page_carries_no_city_fact(tmp_path: Path) -> None:
    """The same rule the viewer's HTML lives under: no city name, no path to a city's
    TOML, no volume id baked into the page. The board renders whatever `/api/board`
    hands it — including the `--candidates` command, which the SERVER composes from
    the roots it was actually started with (`console.cli._candidates_command`)."""
    from autogeoref.cli_context import packaged_ui

    html = (packaged_ui("queue_ui") / "index.html").read_text()
    for fact in ("chicago", "sanborn", "configs/", "fixtures/"):
        assert fact not in html.lower(), f"city fact {fact!r} leaked into the admin page"


def test_the_board_page_can_tail_an_open_running_job_log() -> None:
    """Live log display reuses the bounded read-only `/api/log` endpoint."""
    from autogeoref.cli_context import packaged_ui

    html = (packaged_ui("queue_ui") / "index.html").read_text()
    assert 'data-live-log="true"' in html
    assert "Live tail: updates every 5 seconds" in html
    assert 'fetch("/api/log/" + requested.path, { cache: "no-store" })' in html
    assert "openLog === requested" in html


def test_a_cross_site_post_is_refused_even_with_a_loopback_host() -> None:
    """The CSRF bar the Host guard cannot provide once the routes spend money.

    On a hostile page's direct cross-origin fetch the browser sends the REAL Host
    (this server), so `host_allowed` passes — the second bar is what stops the page
    enqueueing a volume and starting a drain. Either signal alone suffices, and a
    request carrying neither (a `curl`, i.e. the trusted operator) is not the vector.
    """
    # a browser's cross-site request: labelled, whatever its content type
    assert not console_server.csrf_safe("cross-site", "application/json")
    assert not console_server.csrf_safe("same-site", "application/json")
    # the CORS-safelisted type that would skip preflight is refused
    assert not console_server.csrf_safe(None, "text/plain")
    assert not console_server.csrf_safe(None, "application/x-www-form-urlencoded")
    assert not console_server.csrf_safe(None, None)
    # the legitimate client: same-origin OR a direct navigation, sending JSON
    assert console_server.csrf_safe("same-origin", "application/json")
    assert console_server.csrf_safe("none", "application/json")
    assert console_server.csrf_safe(None, "application/json; charset=utf-8")


def test_console_parallelism_help_names_annotation_and_escalation() -> None:
    page = cli_context.packaged_ui("queue_ui") / "index.html"
    assert "pages annotated or escalated at once" in page.read_text()


def test_the_board_page_ships_the_script_it_asks_for() -> None:
    """A script tag the package does not carry is a board that renders nothing
    and throws on its first poll."""
    ui = cli_context.packaged_ui("queue_ui")
    assert '<script src="board.js"></script>' in (ui / "index.html").read_text()
    assert (ui / "board.js").is_file()


def test_the_board_script_is_served_by_a_console_that_has_no_city(tmp_path: Path) -> None:
    """The trap this route exists to close: everything the console does not
    handle itself falls through to the review server, and that fallthrough 404s
    whenever there is no review app — which is exactly the configuration an
    operator runs for a bare queue. Both the packaged-resource test above and
    the wheel smoke pass while the board page is scriptless.
    """
    from autogeoref.console.server import ConsoleRoutes

    handler = ConsoleRoutes.__new__(ConsoleRoutes)
    handler.console_ui = tmp_path
    handler.review_app = None
    handler.path = "/board.js"
    handler.headers = {"Host": "127.0.0.1:8010"}  # type: ignore[assignment]
    sent: list[Path] = []
    handler._send_file = sent.append  # type: ignore[method-assign, assignment]
    refused: list[tuple[int, dict[str, str]]] = []
    handler._send_json = lambda code, body: refused.append((code, body))  # type: ignore[assignment]

    handler.do_GET()
    assert sent == [tmp_path / "board.js"], refused

    # and nothing else: the console serves its own page's scripts, not a tree
    for path in ("/server.py", "/board.js/x", "/../etc/passwd"):
        sent.clear()
        refused.clear()
        handler.path = path
        handler.do_GET()
        assert sent == [], path
        assert refused and refused[0][0] == 404, path


def test_the_page_has_a_dot_a_panel_and_a_target_for_every_queue() -> None:
    """The page loops over the tracks the SERVER sends, so a queue with no element to
    render into shows nothing. Assert the elements exist rather than trusting the loop."""
    page = (cli_context.packaged_ui("queue_ui") / "index.html").read_text()
    for track in qstore.TRACKS:
        assert f'id="ddot-{track}"' in page and f'id="dtext-{track}"' in page
        assert f'<div class="panel" id="{track}"></div>' in page
        assert f'<option value="{track}">' in page
