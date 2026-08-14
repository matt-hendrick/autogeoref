"""Load a page in a headless browser and read what it actually drew.

The class of defect neither a source grep nor a unit test reaches: `the page
loaded and drew nothing`. No new toolchain and no browser download — the system
browser, its DevTools protocol over a socket, and the repo's own Range-capable
static server.

Driven over the protocol rather than with ``--dump-dom``, for two things that
flag cannot give: a real wait (its virtual-time budget expires about a second
after launch, while a map is still assembling) and console severity (every page
message reaches stderr as ``INFO:CONSOLE``, warnings and errors alike).

DNS is blocked for every load (127.0.0.1 excluded), so a page that reaches for
a third party fails inside the browser and nothing leaves the host.
"""

from __future__ import annotations

import base64
import contextlib
import importlib.util
import json
import os
import re
import secrets
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import warnings
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Whichever is installed. `chromium` is the snap on the development checkout;
#: `google-chrome` is what a GitHub runner ships.
BROWSERS = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable")

_DEVTOOLS = re.compile(r"DevTools listening on (ws://\S+)")

#: SwiftShader's driver chatter, in case a GL backend routes it through the
#: page console rather than chromium's own log. A test flaky from birth is
#: worse than no test. `CONTEXT_LOST_WEBGL` belongs here for the same reason and
#: not because a page may lose its context: the software rasteriser drops one
#: under load, eight workers deep, and no page can do anything about that.
_DRIVER_NOISE = ("GL Driver Message", "GPU stall", "CONTEXT_LOST_WEBGL")


def _load_script(name: str) -> ModuleType:
    """Import a `scripts/` module by path — `scripts/` is not an installed package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class BrowserStartError(RuntimeError):
    """A browser is installed but never announced its DevTools endpoint."""


#: What proved the installed browser unusable, once one launch has. The rest of
#: the tier then reports that instead of re-paying for the same failure per
#: test. Per worker process, so a run distributed by file re-pays it per file.
_unstartable: str | None = None


def _no_browser(reason: str) -> NoReturn:
    """Skip this tier, or fail it on CI where a browser is installed.

    Loud on CI rather than silent: the runner image ships both Chrome and
    Chromium, so a skip there means the whole tier stopped running and nobody
    would see it. If nothing is installed there, the fix is
    `browser-actions/setup-chrome` in the workflow — a package install, not a
    browser download. If one is installed and refuses to start, the message
    carries what it printed and the fix is wherever that points.
    """
    if os.environ.get("CI"):
        raise AssertionError(reason)
    pytest.skip(reason)


def browser() -> str:
    """The browser this tier can drive, or skip — the same shape as the node skip.

    Two ways to have none, and only the first is a question PATH can answer:
    nothing installed, or something installed that will not start. Both skip
    here and both fail on CI.
    """
    if _unstartable is not None:
        _no_browser(_unstartable)
    for name in BROWSERS:
        found = shutil.which(name)
        if found:
            return found
    _no_browser(f"no headless browser on PATH (tried {', '.join(BROWSERS)})")


@contextmanager
def serve(root: Path) -> Iterator[str]:
    """Serve ``root`` on a loopback port with Range support; yields the base URL.

    Range matters: pmtiles.js reads an archive by byte range, and the stdlib
    handler would make it download the whole archive instead.
    """
    handler = partial(_load_script("serve_viewer").RangeHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


# ---------------------------------------------------------------------------
# a WebSocket client, small enough to read: DevTools is the only peer
# ---------------------------------------------------------------------------


class _Socket:
    """RFC 6455 text frames over a plain socket. Client-to-server frames are masked."""

    def __init__(self, url: str, timeout: float) -> None:
        parsed = urlparse(url)
        self.sock = socket.create_connection((parsed.hostname, parsed.port), timeout=timeout)
        self.sock.settimeout(timeout)
        key = base64.b64encode(secrets.token_bytes(16)).decode()
        path = parsed.path + ("?" + parsed.query if parsed.query else "")
        self.sock.sendall(
            f"GET {path} HTTP/1.1\r\nHost: {parsed.hostname}:{parsed.port}\r\n"
            f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode()
        )
        self._buffer = b""
        while b"\r\n\r\n" not in self._buffer:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise AssertionError("DevTools closed the connection during the handshake")
            self._buffer += chunk
        head, self._buffer = self._buffer.split(b"\r\n\r\n", 1)
        if b"101" not in head.split(b"\r\n")[0]:
            raise AssertionError(f"DevTools refused the upgrade: {head!r}")

    def _read(self, count: int) -> bytes:
        while len(self._buffer) < count:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise AssertionError("DevTools closed the connection")
            self._buffer += chunk
        out, self._buffer = self._buffer[:count], self._buffer[count:]
        return out

    def send(self, text: str) -> None:
        payload = text.encode()
        header = bytearray([0x81])
        size = len(payload)
        if size < 126:
            header.append(0x80 | size)
        elif size < 1 << 16:
            header.append(0x80 | 126)
            header += struct.pack(">H", size)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", size)
        mask = secrets.token_bytes(4)
        header += mask
        self.sock.sendall(bytes(header) + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

    def recv(self) -> str:
        """The next text message, reassembling continuation frames."""
        parts: list[bytes] = []
        while True:
            first, second = self._read(2)
            opcode, size = first & 0x0F, second & 0x7F
            if size == 126:
                size = struct.unpack(">H", self._read(2))[0]
            elif size == 127:
                size = struct.unpack(">Q", self._read(8))[0]
            body = self._read(size)
            if opcode == 0x9:  # ping: answer it or the peer eventually hangs up
                self.sock.sendall(bytes([0x8A, 0x80]) + secrets.token_bytes(4))
                continue
            if opcode == 0x8:
                raise AssertionError("DevTools closed the connection")
            parts.append(body)
            if first & 0x80:
                return b"".join(parts).decode()

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.sock.close()


# ---------------------------------------------------------------------------
# the page load
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsoleMessage:
    level: str
    text: str


@dataclass(frozen=True)
class PageLoad:
    """What one page load produced."""

    dom: str
    console: tuple[ConsoleMessage, ...] = field(default=())
    #: values of the caller's `capture` expressions, read in the loaded page
    captured: dict[str, object] = field(default_factory=dict)

    @property
    def errors(self) -> tuple[ConsoleMessage, ...]:
        """Error-severity messages, uncaught exceptions included."""
        return tuple(m for m in self.console if m.level == "error")

    def element(self, marker: str) -> str:
        """The first line of the rendered DOM containing ``marker``."""
        for line in self.dom.splitlines():
            if marker in line:
                return line.strip()
        raise AssertionError(f"no element matching {marker!r} in the rendered DOM")


class _Session:
    """One DevTools conversation with one tab."""

    def __init__(self, ws_url: str, timeout: float) -> None:
        self.socket = _Socket(ws_url, timeout)
        self.next_id = 0
        self.console: list[ConsoleMessage] = []

    def call(self, method: str, **params: object) -> dict[str, Any]:
        self.next_id += 1
        wanted = self.next_id
        self.socket.send(json.dumps({"id": wanted, "method": method, "params": params}))
        while True:
            message = json.loads(self.socket.recv())
            if "id" not in message:
                self._note(message)
                continue
            if message["id"] != wanted:
                continue
            if "error" in message:
                raise AssertionError(f"{method} failed: {message['error']}")
            result: dict[str, Any] = message.get("result", {})
            return result

    def _note(self, message: dict[str, Any]) -> None:
        method, params = message.get("method"), message.get("params", {})
        if method == "Runtime.consoleAPICalled":
            level = {"warning": "warning", "error": "error"}.get(params.get("type", ""), "log")
            text = " ".join(
                str(arg.get("value", arg.get("description", ""))) for arg in params.get("args", [])
            )
            self.console.append(ConsoleMessage(level, text))
        elif method == "Runtime.exceptionThrown":
            details = params.get("exceptionDetails", {})
            text = details.get("exception", {}).get("description") or details.get("text", "")
            self.console.append(ConsoleMessage("error", text))
        elif method == "Log.entryAdded":
            entry = params.get("entry", {})
            self.console.append(
                ConsoleMessage(entry.get("level", "log"), str(entry.get("text", "")))
            )


#: How long a page gets to satisfy its condition. Generous because it is a
#: DEADLINE, not a sleep: a passing page returns as soon as it is ready, and
#: the whole suite runs eight workers deep, so a browser assembling a map can
#: be starved for a long time by the seven other jobs on the machine.
DEFAULT_TIMEOUT_S = 90.0


def load_page(
    url: str,
    *,
    until: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    settle_s: float = 0.0,
    capture: Mapping[str, str] | None = None,
    block_dns: bool = True,
    viewport: tuple[int, int] | None = None,
    host_alias: str | None = None,
) -> PageLoad:
    """``url``, loaded until ``until`` evaluates truthy. Returns the DOM and console.

    ``until`` is a JS expression polled against the wall clock — the page's own
    observable state, never elapsed time; a page that never satisfies it fails
    with the DOM it did reach. ``settle_s`` then holds the page open longer, for
    the one thing a condition cannot express: that something never appears.
    ``capture`` is `{name: JS expression}` read in the loaded page rather than
    grepped from the DOM. ``viewport`` `(w, h)` sets a smaller layout, and
    ``host_alias`` resolves that hostname to the loopback server, so a decision
    read from ``location.hostname`` is exercised as a public host."""
    with _launched(browser(), block_dns=block_dns, host_alias=host_alias) as ws_url:
        session = _Session(ws_url, timeout=timeout_s)
        try:
            session.call("Runtime.enable")
            session.call("Log.enable")
            session.call("Page.enable")
            if viewport is not None:
                session.call(
                    "Emulation.setDeviceMetricsOverride",
                    width=viewport[0],
                    height=viewport[1],
                    deviceScaleFactor=1,
                    mobile=viewport[0] < 500,
                )
            session.call("Page.navigate", url=url)
            deadline = time.monotonic() + timeout_s
            ready = False
            while time.monotonic() < deadline:
                result = session.call(
                    "Runtime.evaluate", expression=f"Boolean({until})", returnByValue=True
                )
                if result.get("result", {}).get("value") is True:
                    ready = True
                    break
                time.sleep(0.15)
            if ready and settle_s:
                time.sleep(settle_s)
            dom = session.call(
                "Runtime.evaluate",
                expression="document.documentElement.outerHTML",
                returnByValue=True,
            )["result"]["value"]
            # awaitPromise, or an async expression returns its Promise and
            # serialises to `{}`. Iterating that yields nothing, so every
            # assertion over the result passes without reading the page.
            captured = {
                name: session.call(
                    "Runtime.evaluate",
                    expression=expression,
                    returnByValue=True,
                    awaitPromise=True,
                )["result"].get("value")
                for name, expression in (capture or {}).items()
            }
            console = tuple(
                message
                for message in session.console
                if not any(noise in message.text for noise in _DRIVER_NOISE)
            )
            if not ready:
                raise AssertionError(
                    f"{url} never satisfied `{until}` in {timeout_s:.0f}s.\n"
                    f"console: {console}\nDOM:\n{dom[:4000]}"
                )
            return PageLoad(dom=dom, console=console, captured=captured)
        finally:
            session.socket.close()


@contextmanager
def _launched(executable: str, *, block_dns: bool, host_alias: str | None = None) -> Iterator[str]:
    """A headless browser with one blank tab; yields that tab's DevTools URL.

    ``ignore_cleanup_errors`` because the profile is a throwaway: failing to
    delete it says nothing about the page, and must not fail the caller's test.
    """
    with tempfile.TemporaryDirectory(
        prefix="autogeoref-browser-", ignore_cleanup_errors=True
    ) as profile:
        argv = [
            executable,
            "--headless",
            "--no-sandbox",
            "--disable-gpu-sandbox",
            # SwiftShader: the vendored MapLibre needs a real WebGL context and
            # there is no GPU here
            "--enable-unsafe-swiftshader",
            "--disable-dev-shm-usage",
            "--window-size=1280,900",
            f"--user-data-dir={profile}",
            "--remote-debugging-port=0",
        ]
        # the alias rule leads: `MAP *` matches everything, so a later entry
        # for one host never gets read
        rules = [f"MAP {host_alias} 127.0.0.1"] if host_alias else []
        if block_dns:
            rules += ["MAP * ~NOTFOUND", "EXCLUDE 127.0.0.1"]
        if rules:
            argv.append("--host-resolver-rules=" + ", ".join(rules))
        argv.append("about:blank")
        # Its own session, and so its own process group: that is what lets
        # `_stop` signal the children the browser forks without also signalling
        # the test runner they would otherwise share a group with.
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            browser_ws = _devtools_or_no_browser(proc, executable)
            yield _first_page_target(urlparse(browser_ws).port)
        finally:
            _stop(proc)


def _devtools_or_no_browser(proc: subprocess.Popen[str], executable: str) -> str:
    """The DevTools URL, or treat the browser as missing — skip, or fail on CI.

    Reports what the browser printed, because the cause is always outside this
    repository — a confinement rule, a missing runtime directory, a profile it
    could not write. Only a browser that DIED is remembered; one that is merely
    slow raises from `_wait_for_devtools` and stays a failure.
    """
    global _unstartable
    try:
        return _wait_for_devtools(proc)
    except BrowserStartError as unstartable:
        _unstartable = f"{executable} will not start: {unstartable}"
        _no_browser(_unstartable)


def _stop(proc: subprocess.Popen[Any], timeout_s: float = 15.0) -> None:
    """Stop the browser and every process it forked, and wait for them to go.

    Any stream type: this reads the pid and the exit status, never the pipes.

    The profile directory is deleted the moment this returns, so a child still
    flushing storage into it races that delete. Asking the browser to stop is
    not enough: whether it exits gracefully or is killed, it can leave children
    behind, and only a group signal reaches those. Raises nothing — this runs
    in a ``finally`` and must never replace the caller's result.
    """
    proc.terminate()
    _exited(proc.pid, timeout_s=timeout_s)
    # Unreaped either way by now, so the pid — and the group it names — is
    # still ours, and this cannot land on a stranger that inherited the pid.
    with contextlib.suppress(OSError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=timeout_s)
    if not _drained(proc.pid, timeout_s=5.0):
        # Loud, because the alternative is a runner quietly filling with
        # browsers. Suppressed so the notice itself cannot mask a result.
        with contextlib.suppress(Exception):
            warnings.warn(f"browser process group {proc.pid} survived SIGKILL", stacklevel=2)


def _exited(pid: int, *, timeout_s: float) -> bool:
    """Whether ``pid`` exited within ``timeout_s``, leaving it unreaped either way.

    Unreaped is the point: the pid names the process group, and reaping frees
    it for reuse — after which a group signal could reach a stranger.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT | os.WNOHANG) is not None:
                return True
        except OSError:  # never our child, or already reaped
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _drained(pgid: int, *, timeout_s: float) -> bool:
    """Whether every process in ``pgid`` has exited, waiting up to ``timeout_s``.

    Signal 0 only, since the caller has reaped the leader by now: a recycled
    pid would cost a wait here, never a signal to something else.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _wait_for_devtools(proc: subprocess.Popen[str], timeout_s: float = 30.0) -> str:
    """The endpoint the browser announces on stderr, or raise carrying that stderr.

    Two ways to fail, and they are not the same fault. Stderr at end of file
    means the browser is gone and will go again — `BrowserStartError`, which the
    caller reads as an unusable browser. Reaching the deadline with it still
    running means it was slow, which eight workers deep says nothing about the
    next launch, so that stays an ordinary failure and is never remembered.
    Either way the last few lines travel with it, or the caller can only report
    that nothing was announced, naming no cause and no fix.
    """
    deadline = time.monotonic() + timeout_s
    assert proc.stderr is not None
    said: deque[str] = deque(maxlen=6)
    # Waited on rather than read straight, or the budget is only consulted
    # between lines and a browser that starts and then goes quiet hangs here.
    while (remaining := deadline - time.monotonic()) > 0:
        if not select.select([proc.stderr], [], [], remaining)[0]:
            break
        line = proc.stderr.readline()
        if not line:
            tail = "\n".join(said) or "(it printed nothing)"
            raise BrowserStartError(f"no DevTools endpoint announced. It said:\n{tail}")
        said.append(line.rstrip())
        found = _DEVTOOLS.search(line)
        if found:
            return found.group(1)
    raise AssertionError(
        f"the browser announced no DevTools endpoint in {timeout_s:.0f}s and is "
        f"still running. It said:\n" + ("\n".join(said) or "(it printed nothing)")
    )


def _first_page_target(port: int | None, timeout_s: float = 20.0) -> str:
    """The blank tab's DevTools URL, polled until the browser opens one.

    A missed read is retried like an empty answer: a loaded machine can lose a
    single request to its own timeout while the browser is still coming up, and
    that is what the budget here is for. The last one is reported if it runs out.
    """
    deadline = time.monotonic() + timeout_s
    missed = "it opened none"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/list", timeout=5
            ) as response:
                targets = json.loads(response.read())
        except (OSError, ValueError) as exc:  # transport, or a body cut short
            missed = f"{type(exc).__name__}: {exc}"
        else:
            for target in targets:
                if target.get("type") == "page" and target.get("webSocketDebuggerUrl"):
                    return str(target["webSocketDebuggerUrl"])
        time.sleep(0.1)
    raise AssertionError(f"the browser opened no page target ({missed})")
