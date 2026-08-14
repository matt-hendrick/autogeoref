"""Shared safeguards for local-only HTTP administration servers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def host_allowed(host: str) -> bool:
    """Only loopback Host headers are served (DNS-rebinding guard)."""
    name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
    return name in ("127.0.0.1", "localhost", "[::1]")


@contextmanager
def loopback_server(
    handler: type[BaseHTTPRequestHandler], port: int
) -> Iterator[ThreadingHTTPServer]:
    """Bind a threaded HTTP server to IPv4 loopback only."""
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        yield httpd
