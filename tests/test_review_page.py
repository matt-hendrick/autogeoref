"""What the review server hands a browser, and what it refuses to hand anyone.

The page's asset route is an allowlist and not a document root, and the host
guard closes the DNS-rebinding vector. The page itself must escape every free
string it takes off disk — a result status and a sidecar verdict have no
grammar, and both are written into rows built with markup.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from autogeoref.review.server import ReviewHandler


def test_host_header_guard() -> None:
    from autogeoref.local_server import host_allowed

    assert host_allowed("127.0.0.1:8765")
    assert host_allowed("localhost:8765")
    assert host_allowed("localhost")
    assert host_allowed("[::1]:8765")
    assert not host_allowed("evil.example.com:8765")  # DNS-rebinding vector
    assert not host_allowed("127.0.0.1.evil.example.com")
    assert not host_allowed("")


def test_the_page_asset_route_serves_only_what_it_declares(tmp_path: Path) -> None:
    """The review page loads its affine mirror as a sibling file, so the server
    grew a route for it. That route is an allowlist and not a document root:
    the UI directory holds the operator's page, and anything not named is a
    404 — including a traversal and a longer path under the same first
    segment."""
    handler = ReviewHandler.__new__(ReviewHandler)
    handler.app = SimpleNamespace(ui_dir=tmp_path)  # type: ignore[assignment]
    sent: list[Path] = []
    handler._send_file = sent.append  # type: ignore[method-assign, assignment]

    route = handler._get_route(["affine.js"])
    assert route is not None
    route()
    assert sent == [tmp_path / "affine.js"]

    for parts in (["index.html"], ["server.py"], ["..", "etc"], ["affine.js", "x"]):
        assert handler._get_route(parts) is None, parts


def test_no_free_string_off_disk_reaches_the_review_page_as_markup() -> None:
    """A result's status and a sidecar's verdict have no grammar — the reader
    writes ``str(r.get("status", ""))`` — and both are written into rows built
    with ``innerHTML``, where a stray ``<`` eats the rest of the row. The
    sibling console page decided this class was real and fixed it; this is the
    absent half, which no rendering test can see."""
    page = (
        Path(__file__).resolve().parent.parent / "src" / "autogeoref" / "review_ui" / "index.html"
    ).read_text(encoding="utf-8")
    raw_sites = (
        "${q.status}",
        "${q.verdict}",
        "${sheet.sidecar.verdict}",
        "[sheet.status]",
        # a number today (`round(rmse, 2)`), which is exactly the value that
        # becomes a string later without anyone revisiting this line
        "${sheet.rmse_vs_human_m}",
    )
    for raw in raw_sites:
        assert raw not in page, f"{raw} reaches innerHTML unescaped"
    for escaped in ("esc(q.status)", "esc(q.verdict)", "esc(sheet.status)"):
        assert escaped in page
    # the warn markers in the status line ARE meant to render, so the fix is
    # escaping at the point of entry and not switching the join to textContent
    assert '$("sheet-status").innerHTML = bits.join' in page


def test_the_review_page_ships_the_script_it_asks_for() -> None:
    """A script tag the package does not carry is a page that loads and then
    throws on its first use of the affine."""
    ui = Path(__file__).resolve().parent.parent / "src" / "autogeoref" / "review_ui"
    page = (ui / "index.html").read_text(encoding="utf-8")
    assert 'src="affine.js"' in page
    assert (ui / "affine.js").is_file()
