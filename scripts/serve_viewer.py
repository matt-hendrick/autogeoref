"""Static file server with HTTP Range support, for viewing the viewer locally.

PMTiles archives are read by byte-range requests; the stdlib
``http.server`` ignores the ``Range`` header, so pmtiles.js would have to
download whole multi-hundred-MB archives. This adds the minimal
single-range handling (``bytes=start-end``) the pmtiles client needs.

Run from the repo root and open http://localhost:8123/viewer/ :

    .venv/bin/python scripts/serve_viewer.py [port] [root]

No dependency, no cache, loopback by default; ``AUTOGEOREF_SERVE_HOST=0.0.0.0``
reaches it from outside, which is what a container needs. For real hosting use
object storage + CDN (``autogeoref deploy-bundle``).
"""

from __future__ import annotations

import os
import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, AnyStr

if TYPE_CHECKING:
    from _typeshed import SupportsRead, SupportsWrite

_RANGE = re.compile(r"bytes=(\d*)-(\d*)$")


class RangeHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + single-range GET (RFC 9110 §14.2)."""

    def send_head(self):  # type: ignore[no-untyped-def]  # stdlib parent is untyped
        range_header = self.headers.get("Range")
        match = _RANGE.match(range_header) if range_header else None
        if match is None:
            return super().send_head()

        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self.send_error(404, "File not found")
            return None
        size = path.stat().st_size
        start_s, end_s = match.groups()
        if start_s:
            start = int(start_s)
            end = min(int(end_s), size - 1) if end_s else size - 1
        elif end_s:  # suffix range: last N bytes
            start = max(size - int(end_s), 0)
            end = size - 1
        else:
            self.send_error(400, "Bad range")
            return None
        if start >= size or start > end:
            self.send_error(416, "Range not satisfiable")
            self.send_header("Content-Range", f"bytes */{size}")
            return None

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()

        f = path.open("rb")
        f.seek(start)
        self._range_remaining: int | None = end - start + 1
        return f

    def copyfile(self, source: SupportsRead[AnyStr], outputfile: SupportsWrite[AnyStr]) -> None:
        remaining = getattr(self, "_range_remaining", None)
        if remaining is None:
            super().copyfile(source, outputfile)
            return
        self._range_remaining = None
        while remaining > 0:
            chunk = source.read(min(65536, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
    root = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path.cwd()
    # Loopback by default: this serves the whole repo root, and a dev server
    # that reaches the LAN the moment someone runs it is not a default anyone
    # asked for. `AUTOGEOREF_SERVE_HOST=0.0.0.0` opts in, which is what a
    # container needs — inside one, loopback is the container's own and a
    # published port reaches nothing.
    host = os.environ.get("AUTOGEOREF_SERVE_HOST", "127.0.0.1")
    handler = partial(RangeHandler, directory=str(root))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"serving {root} at http://{host}:{port}/ (Range-capable; Ctrl-C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
