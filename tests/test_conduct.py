"""Conduct contracts: politeness, timeouts, and the failure taxonomy.

These pin the behavioural contracts that keep the tool a polite, read-only citizen of
loc.gov and make vision failures diagnosable:

1. The LOC client never issues two requests less than 5 s apart.
2. Every external call carries an explicit timeout.
3. An empty vision response is classified distinctly from a budget-limit message.
4. No write request reaches a production host, and non-LOC hosts are refused.
5. No test can reach the network through httpx's default transports — the autouse guard
   in conftest trips before any connection is attempted.
6. No tracked file names a machine-local absolute path.

All transports, clocks, and model backends are mocked; no test sleeps for real.
"""

from __future__ import annotations

import ast
import asyncio
import re
import socket
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from autogeoref.annotate.failures import AnnotationCallError, BudgetLimitError, EmptyResponseError
from autogeoref.annotate.invocation import ClaudeCLIBackend
from autogeoref.loc import HostNotAllowedError, LOCClient, MethodNotAllowedError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE = REPOSITORY_ROOT / "src" / "autogeoref"
#: The two `annotate` modules allowed to reach a model: `api_call` holds the SDK
#: backends, `cli_call` spawns the executables. All else routes through them —
#: `invocation` included, which picks a backend and touches no SDK.
MODEL_FACING_MODULES = frozenset({"api_call.py", "cli_call.py"})
SCRIPTS_DIRECTORY = REPOSITORY_ROOT / "scripts"
EXPERIMENTS_DIRECTORY = SCRIPTS_DIRECTORY / "experiments"
MODEL_EXECUTABLES = frozenset({"claude", "claude.exe", "codex", "codex.exe"})


_SPAWN_FUNCTIONS = frozenset({"run", "Popen", "call", "check_call", "check_output"})
_ASYNC_SPAWN_FUNCTIONS = frozenset({"create_subprocess_exec", "create_subprocess_shell"})


def _bindings(tree: ast.AST) -> dict[str, list[ast.expr]]:
    """Collect simple assignments so static executable names cannot hide in a variable."""
    bindings: dict[str, list[ast.expr]] = {}
    for node in ast.walk(tree):
        value: ast.expr | None = None
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            value = node.value
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            value = node.value
            targets = [node.target]
        if value is not None:
            for target in targets:
                if isinstance(target, ast.Name):
                    bindings.setdefault(target.id, []).append(value)
    return bindings


def _string_values(
    node: ast.expr,
    bindings: dict[str, list[ast.expr]],
    resolving: frozenset[str] = frozenset(),
) -> set[str]:
    """Return statically knowable string values for a command expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Name) and node.id not in resolving:
        return {
            value
            for binding in bindings.get(node.id, [])
            for value in _string_values(binding, bindings, resolving | {node.id})
        }
    if isinstance(node, ast.IfExp):
        return _string_values(node.body, bindings, resolving) | _string_values(
            node.orelse, bindings, resolving
        )
    if isinstance(node, ast.JoinedStr):
        values = {""}
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                part_values = {part.value}
            elif isinstance(part, ast.FormattedValue):
                part_values = _string_values(part.value, bindings, resolving)
            else:
                return set()
            values = {prefix + suffix for prefix in values for suffix in part_values}
        return values
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
        left = _string_values(node.left, bindings, resolving)
        right = _string_values(node.right, bindings, resolving)
        separator = "" if isinstance(node.op, ast.Add) else "/"
        return {
            left_value + separator + right_value for left_value in left for right_value in right
        }
    if isinstance(node, ast.Call) and node.args:
        function = node.func
        if isinstance(function, ast.Name) and function.id in {"Path", "PurePath", "str"}:
            return _string_values(node.args[0], bindings, resolving)
        if isinstance(function, ast.Attribute) and function.attr == "join":
            values = {""}
            for argument in node.args:
                argument_values = _string_values(argument, bindings, resolving)
                values = {
                    f"{prefix}/{suffix}" if prefix else suffix
                    for prefix in values
                    for suffix in argument_values
                }
            return values
        if isinstance(function, ast.Attribute) and function.attr == "split":
            return _string_values(function.value, bindings, resolving)
    return set()


def _command_values(
    node: ast.expr, bindings: dict[str, list[ast.expr]], resolving: frozenset[str] = frozenset()
) -> set[str]:
    if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
        return _string_values(node.elts[0], bindings)
    if isinstance(node, ast.Name) and node.id not in resolving:
        return {
            command
            for binding in bindings.get(node.id, [])
            for command in _command_values(binding, bindings, resolving | {node.id})
        }
    return _string_values(node, bindings)


def _api_aliases(
    tree: ast.AST,
    module: str,
    api_functions: frozenset[str],
    bindings: dict[str, list[ast.expr]],
) -> tuple[set[str], set[str]]:
    """Resolve normal imports and assignment aliases for a process API."""
    modules = {module}
    function_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module:
                    modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == module:
            for alias in node.names:
                if alias.name == "*":
                    function_aliases.update(api_functions)
                elif alias.name in api_functions:
                    function_aliases.add(alias.asname or alias.name)

    changed = True
    while changed:
        changed = False
        for name, values in bindings.items():
            if name in function_aliases:
                continue
            if any(
                _is_api_reference(value, modules, function_aliases, api_functions, bindings)
                for value in values
            ):
                function_aliases.add(name)
                changed = True
    return modules, function_aliases


def _is_api_reference(
    node: ast.expr,
    module_aliases: set[str],
    function_aliases: set[str],
    api_functions: frozenset[str],
    bindings: dict[str, list[ast.expr]],
) -> bool:
    if isinstance(node, ast.Name):
        return node.id in function_aliases
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in module_aliases
        and node.attr in api_functions
    ):
        return True
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in module_aliases
    ):
        return bool(_string_values(node.args[1], bindings) & api_functions)
    return False


def _is_api_call(
    node: ast.Call,
    module_aliases: set[str],
    function_aliases: set[str],
    api_functions: frozenset[str],
    bindings: dict[str, list[ast.expr]],
) -> bool:
    return _is_api_reference(node.func, module_aliases, function_aliases, api_functions, bindings)


def _spawn_command(node: ast.Call) -> ast.expr | None:
    if node.args:
        return node.args[0]
    for keyword in node.keywords:
        if keyword.arg == "args":
            return keyword.value
    return None


def _is_model_sdk(module: str | None) -> bool:
    return module is not None and (
        module == "anthropic" or module.startswith(("anthropic.", "openai"))
    )


def _conduct_paths() -> tuple[Path, ...]:
    """Every production module and script subject to model conduct.

    ``scripts/`` covers ``experiments/`` too, so promoting a harness out of it
    cannot drop the harness out of this guard.
    """
    return (*SOURCE_PACKAGE.rglob("*.py"), *SCRIPTS_DIRECTORY.rglob("*.py"))


def _model_conduct_violations(source: str, path: Path) -> list[str]:
    """Return forbidden SDK imports and statically detectable model-process launches."""
    violations: list[str] = []
    is_transport = path.parent == SOURCE_PACKAGE / "annotate" and path.name in MODEL_FACING_MODULES
    tree = ast.parse(source, filename=str(path))
    bindings = _bindings(tree)
    subprocess_modules, subprocess_functions = _api_aliases(
        tree, "subprocess", _SPAWN_FUNCTIONS, bindings
    )
    os_modules, os_functions = _api_aliases(tree, "os", frozenset({"system"}), bindings)
    asyncio_modules, asyncio_functions = _api_aliases(
        tree, "asyncio", _ASYNC_SPAWN_FUNCTIONS, bindings
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports_sdk = any(_is_model_sdk(alias.name) for alias in node.names)
            if imports_sdk and not is_transport:
                violations.append(f"{path}:{node.lineno}: model SDK import")
        elif isinstance(node, ast.ImportFrom):
            if _is_model_sdk(node.module) and not is_transport:
                violations.append(f"{path}:{node.lineno}: model SDK import")
        elif isinstance(node, ast.Call) and (
            _is_api_call(
                node,
                subprocess_modules,
                subprocess_functions,
                _SPAWN_FUNCTIONS,
                bindings,
            )
            or _is_api_call(node, os_modules, os_functions, frozenset({"system"}), bindings)
            or _is_api_call(
                node,
                asyncio_modules,
                asyncio_functions,
                _ASYNC_SPAWN_FUNCTIONS,
                bindings,
            )
        ):
            command = _spawn_command(node)
            commands = _command_values(command, bindings) if command is not None else set()
            invokes_model = any(
                Path(value.split(maxsplit=1)[0]).name in MODEL_EXECUTABLES for value in commands
            )
            if invokes_model and not is_transport:
                violations.append(f"{path}:{node.lineno}: model executable spawn")
    return violations


class FakeClock:
    """Injectable clock: sleeping advances fake time instead of waiting."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class RecordingHandler:
    """Mock transport handler recording each request and its fake timestamp."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.requests: list[httpx.Request] = []
        self.times: list[float] = []
        self.status_codes: list[int] = [200] * 100

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.times.append(self.clock.now)
        status = self.status_codes[len(self.requests) - 1]
        return httpx.Response(status, json={"ok": True})


def make_client(tmp_path: Path, **kwargs: Any) -> tuple[LOCClient, RecordingHandler, FakeClock]:
    clock = FakeClock()
    handler = RecordingHandler(clock)
    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = LOCClient(
        tmp_path / "cache",
        http_client=http,
        clock=clock,
        sleep=clock.sleep,
        **kwargs,
    )
    return client, handler, clock


# ----------------------------------------------------------------------
# 0. Model transport choke point
# ----------------------------------------------------------------------


def test_model_sdk_and_executable_spawning_are_confined_to_the_two_transports() -> None:
    violations = [
        violation
        for path in _conduct_paths()
        for violation in _model_conduct_violations(path.read_text(encoding="utf-8"), path)
    ]
    assert violations == []


def test_model_conduct_scans_every_script() -> None:
    """Every script is scanned, not just the ones under ``experiments/``.

    Asserting the experiments subtree alone still passes when the scan is
    narrowed back to it, because that subtree sits inside ``scripts/`` — which
    would silently un-guard the promoted tools and the one that spends budget.
    """
    script_paths = set(SCRIPTS_DIRECTORY.rglob("*.py"))
    experiment_paths = set(EXPERIMENTS_DIRECTORY.rglob("*.py"))
    assert experiment_paths and script_paths > experiment_paths
    assert SCRIPTS_DIRECTORY / "annotate_volume.py" in script_paths
    assert script_paths <= set(_conduct_paths())


def test_conduct_guard_rejects_a_model_spawn_outside_the_two_transports() -> None:
    source = 'import subprocess\nsubprocess.run(["claude", "-p", "prompt"])\n'
    assert _model_conduct_violations(source, SOURCE_PACKAGE / "annotate" / "schema.py")


@pytest.mark.parametrize(
    "source",
    [
        'from subprocess import run\nrun(["claude", "-p", "prompt"])\n',
        'import subprocess as sp\nsp.run(["codex", "exec", "prompt"])\n',
        (
            'import subprocess\nexecutable = "/opt/models/" + "claude"\n'
            'command = [executable, "-p", "prompt"]\nsubprocess.run(command)\n'
        ),
        (
            "from pathlib import Path\nfrom subprocess import Popen as spawn\n"
            'executable = Path("/opt/models") / "codex"\nspawn(args=[executable, "exec"])\n'
        ),
        ('import subprocess\nrunner = subprocess.run\nrunner(["claude", "-p", "prompt"])\n'),
        (
            'import subprocess as sp\nmethod = "run"\n'
            'getattr(sp, method)(["codex", "exec", "prompt"])\n'
        ),
        'import os as operating\noperating.system("claude -p prompt")\n',
        'import asyncio as aio\naio.create_subprocess_exec("codex", "exec", "prompt")\n',
    ],
)
def test_conduct_guard_rejects_subprocess_aliases_and_static_command_expressions(
    source: str,
) -> None:
    violations = _model_conduct_violations(source, SOURCE_PACKAGE / "annotate" / "schema.py")
    assert any("model executable spawn" in violation for violation in violations)


def test_conduct_guard_allows_an_unrelated_subprocess_alias() -> None:
    source = 'from subprocess import run\nrun(["gdalwarp", "input.tif", "output.tif"])\n'
    assert _model_conduct_violations(source, SOURCE_PACKAGE / "warp.py") == []


@pytest.mark.parametrize("source", ["import openai\n", "from openai import OpenAI\n"])
def test_conduct_guard_rejects_openai_sdk_imports(source: str) -> None:
    violations = _model_conduct_violations(source, SOURCE_PACKAGE / "annotate" / "schema.py")
    assert any("model SDK import" in violation for violation in violations)


# ----------------------------------------------------------------------
# 1. One request lane: >= 5 s between ANY two requests
# ----------------------------------------------------------------------


def test_loc_client_never_issues_two_requests_less_than_5s_apart(tmp_path: Path) -> None:
    client, handler, _clock = make_client(tmp_path)
    # Mix of catalog and image requests, plus a retried failure in the middle.
    handler.status_codes[2] = 429  # third network request must be retried
    client.get_json("https://www.loc.gov/search/?q=a&fo=json")
    client.get_json("https://www.loc.gov/item/one/?fo=json")
    client.get_json("https://www.loc.gov/item/two/?fo=json")  # 429 then retry
    client.get_bytes("https://tile.loc.gov/image-services/x/default.jpg")
    assert len(handler.times) == 5
    gaps = [b - a for a, b in zip(handler.times, handler.times[1:], strict=False)]
    assert all(gap >= 5.0 for gap in gaps), f"requests too close together: {gaps}"


def test_cache_hits_do_not_touch_the_network_at_all(tmp_path: Path) -> None:
    client, handler, _clock = make_client(tmp_path)
    url = "https://www.loc.gov/item/cached/?fo=json"
    client.get_json(url)
    assert len(handler.requests) == 1
    for _ in range(3):
        client.get_json(url)
    assert len(handler.requests) == 1


# ----------------------------------------------------------------------
# 2. Every external call carries a timeout
# ----------------------------------------------------------------------


def test_every_external_call_carries_an_explicit_timeout(tmp_path: Path) -> None:
    client, handler, _clock = make_client(tmp_path, timeout=30.0)
    client.get_json("https://www.loc.gov/search/?q=a&fo=json")
    client.get_bytes("https://tile.loc.gov/image-services/x/default.jpg")
    assert len(handler.requests) == 2
    for request in handler.requests:
        timeout = request.extensions.get("timeout")
        assert timeout is not None, "request issued without an explicit timeout"
        assert timeout["connect"] == 30.0
        assert timeout["read"] == 30.0
        assert timeout["write"] == 30.0


# ----------------------------------------------------------------------
# 3. Empty vision response vs budget-limit message: distinct failures
# ----------------------------------------------------------------------


def _cli_backend_with_output(stdout: str) -> ClaudeCLIBackend:
    def runner(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    return ClaudeCLIBackend(runner=runner)


def test_empty_response_and_budget_message_are_distinct_failures(tmp_path: Path) -> None:
    image = tmp_path / "p1_small.jpg"
    image.write_bytes(b"x")

    with pytest.raises(EmptyResponseError) as empty_info:
        _cli_backend_with_output("").annotate(image)
    with pytest.raises(BudgetLimitError) as budget_info:
        _cli_backend_with_output(
            "Claude usage limit reached. Your limit will reset at 3am."
        ).annotate(image)

    empty_exc, budget_exc = empty_info.value, budget_info.value
    assert type(empty_exc) is not type(budget_exc)
    assert not isinstance(empty_exc, BudgetLimitError)
    assert not isinstance(budget_exc, EmptyResponseError)
    # Both are still annotation failures, catchable as a family.
    assert isinstance(empty_exc, AnnotationCallError)
    assert isinstance(budget_exc, AnnotationCallError)


def test_zero_streets_parse_is_not_a_failure(tmp_path: Path) -> None:
    image = tmp_path / "p2_small.jpg"
    image.write_bytes(b"x")
    backend = _cli_backend_with_output('{"streets": [], "page_number_seen": null}')
    annotation = backend.annotate(image)  # must NOT raise
    assert annotation.streets == ()


# ----------------------------------------------------------------------
# 4. No write requests to any production host
# ----------------------------------------------------------------------


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
def test_loc_client_refuses_write_method_usage(tmp_path: Path, method: str) -> None:
    client, handler, _clock = make_client(tmp_path)
    with pytest.raises(MethodNotAllowedError):
        client.request(method, "https://www.loc.gov/item/x/?fo=json")
    assert handler.requests == [], "a non-GET request reached the transport"


@pytest.mark.parametrize(
    "url",
    [
        "https://oldinsurancemaps.net/",
        "https://oldinsurancemaps.net/api/documents/123",
        "https://api.anthropic.com/v1/messages",
        "https://example.com/loc.gov/",
        "http://www.loc.gov/item/x/?fo=json",  # plain http refused as well
    ],
)
def test_loc_client_refuses_non_loc_hosts(tmp_path: Path, url: str) -> None:
    client, handler, _clock = make_client(tmp_path)
    with pytest.raises(HostNotAllowedError):
        client.get_json(url)
    with pytest.raises(HostNotAllowedError):
        client.get_bytes(url)
    with pytest.raises(HostNotAllowedError):
        client.request("GET", url)
    assert handler.requests == [], "a refused URL reached the transport"


# ----------------------------------------------------------------------
# 5. The no-network guard trips at the httpx transport boundary
# ----------------------------------------------------------------------


def test_default_httpx_get_trips_the_guard_before_any_connection() -> None:
    """A default-transport request fails on the tripwire, not on a socket.

    A listening loopback socket stands in for "the network": the guard must
    raise before httpx opens a connection, so the listener never sees one.
    """
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(0)
        port = listener.getsockname()[1]
        with pytest.raises(AssertionError, match="default network transport"):
            httpx.get(f"http://127.0.0.1:{port}/", timeout=1.0)
        with pytest.raises((BlockingIOError, TimeoutError)):
            listener.accept()


def test_default_async_httpx_request_trips_the_guard() -> None:
    async def attempt() -> None:
        async with httpx.AsyncClient() as client:
            await client.get("http://127.0.0.1:9/", timeout=1.0)

    with pytest.raises(AssertionError, match="default network transport"):
        asyncio.run(attempt())


def test_explicit_mock_transport_is_not_tripped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert client.get("https://www.loc.gov/item/x/?fo=json").json() == {"ok": True}


# ----------------------------------------------------------------------
# 6. No tracked file names a machine-local absolute path
# ----------------------------------------------------------------------

#: A path that resolves on one person's machine and nowhere else: a home
#: directory, a mounted Windows drive, a UNC share, or an agent scratch root.
#: Write a home path as ``~/`` and an example path relative to the checkout.
LOCAL_PATHS = (
    re.compile(r"/home/[A-Za-z0-9._-]+/"),
    re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    re.compile(r"/mnt/[a-z]/"),
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:\\"),
    re.compile(r"(?<!:)[\\/]{2}wsl[.$]"),
    re.compile(r"/tmp/claude-[0-9]"),
)
#: Escape marker, per line or on a fenced block's opening fence, never per
#: file, so a real leak elsewhere in the same file still fails. The standing
#: exemptions are the UNC branch in the git hook and the two docs warning a
#: reader off one, the codex auth line that already reads ``<user>``, a
#: redacted placeholder shaped like a home path on purpose, and paths that
#: live inside a container image rather than on anybody's machine.
LOCAL_PATH_OK = "local-path-ok"
#: Text files only; a tracked binary or an image has nothing to read.
UNSCANNED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".pmtiles", ".woff2", ".ico"})


def _local_path_in(line: str) -> str | None:
    """Return the machine-local path a line names, or None if it is portable."""
    if LOCAL_PATH_OK in line:
        return None
    for pattern in LOCAL_PATHS:
        found = pattern.search(line)
        if found is not None:
            return found.group(0)
    return None


def _tracked_files() -> list[Path]:
    listing = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if listing.returncode != 0:
        return []
    return [REPOSITORY_ROOT / name for name in listing.stdout.split("\0") if name]


def test_no_tracked_file_names_a_machine_local_path() -> None:
    """A path only its author can resolve is noise to every other reader.

    Scans tracked files; mark a deliberate one with ``local-path-ok`` on its
    own line or on a fenced block's opening fence. Passes where git cannot
    list files, as in a source tarball.
    """
    leaks: list[str] = []
    for path in _tracked_files():
        if path.suffix.lower() in UNSCANNED_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        relative = path.relative_to(REPOSITORY_ROOT)
        exempt_fence = False
        for number, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith("```"):
                exempt_fence = LOCAL_PATH_OK in line if not exempt_fence else False
                continue
            if exempt_fence:
                continue
            found = _local_path_in(line)
            if found is not None:
                leaks.append(f"{relative}:{number}: {found}")
    assert not leaks, (
        f"{len(leaks)} tracked line(s) name a machine-local absolute path; rewrite as "
        f"'~/' or a checkout-relative path, or mark the line {LOCAL_PATH_OK!r}:\n"
        + "\n".join(leaks)
    )


def test_the_guard_lists_tracked_files() -> None:
    """A guard that scans nothing passes forever — except where nothing is what
    there is, as in a source tarball or the container's copy of the tree."""
    assert len(_tracked_files()) > 100 or not (REPOSITORY_ROOT / ".git").exists()


@pytest.mark.parametrize(
    "line",
    [
        "WORK = Path('/home/someone/autogeoref/work')",  # local-path-ok
        "cp /Users/someone/notes.txt .",  # local-path-ok
        "the checkout at /mnt/c/Repositories/thing",  # local-path-ok
        "the checkout at D:\\Repositories\\thing",  # local-path-ok
        "//wsl.localhost/Ubuntu/home/someone/autogeoref",  # local-path-ok
        "cache at /tmp/claude-1000/session/scratchpad",  # local-path-ok
    ],
)
def test_the_guard_catches_each_shape_and_honours_the_escape(line: str) -> None:
    assert _local_path_in(line) is not None
    assert _local_path_in(f"{line}  # {LOCAL_PATH_OK}") is None


@pytest.mark.parametrize(
    "line",
    [
        "WORK = Path(os.environ.get('AUTOGEOREF_WORK', str(ROOT / 'work')))",
        "the survey ran against ~/autogeoref/work",
        "python scripts/experiments/water.py --arms /tmp/seam-arms/sample",
        "https://www.loc.gov/item/sanborn01790_001/",
        "LEAK_TOKENS = ('localhost', 'titiler', 'wsl.localhost')",
        "the ratio is 3:1 and the run took 2:30",
    ],
)
def test_the_guard_passes_portable_lines(line: str) -> None:
    assert _local_path_in(line) is None
