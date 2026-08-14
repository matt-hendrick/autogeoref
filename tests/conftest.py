"""Shared fixtures: paths into the frozen fixture tree, and the network tripwires.

The fixture tree (see FIXTURES.md) is a read-only copy of the live pipeline's
recorded inputs and outputs — it is the spec. Tests must never write into it.

One opt-in fixture lives here too, because the queue test modules share it:
`_publish_without_viewer_io`. It is requested by name, never autouse, so no
other test has viewer publication silently stubbed out from under it.
"""

import importlib.util
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn

import httpx
import pytest

import autogeoref.annotate.cli_call as cli_call
import autogeoref.annotate.invocation as invocation
from autogeoref.annotate.providers import CLI_PROVIDERS
from autogeoref.paths import VolumePaths

#: every binary a backend could spawn — registry-derived, so new backends are
#: guarded without anyone updating this
MODEL_CLI_BINARIES = frozenset(
    name
    for provider in CLI_PROVIDERS.values()
    for name in (provider.default_executable, f"{provider.default_executable}.exe")
)

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
FIXTURES = ROOT / "fixtures"
# A tracked standalone probe cache creates fixtures/ in clean CI checkouts.
# This file is present only in the complete frozen fixture tree.
FIXTURE_TREE_SENTINEL = FIXTURES / "reference" / "street_center_lines.geojson"


def load_script(relative: str) -> ModuleType:
    """Import a `scripts/` library module by path — `scripts/` is not a package.

    `relative` is a path under `scripts/`. The module is registered in
    `sys.modules` under its bare stem before it runs, which is both what
    `@dataclass` needs (it reads its own module back out of there) and what
    lets the module's siblings import it by name, the way a script run from
    `scripts/` does.
    """
    path = FIXTURES.parent / "scripts" / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


#: Epoch seconds for :func:`antedate` — long before any file a test writes.
ANTEDATED = 1000.0


def antedate(*paths: Path, when: float = ANTEDATED) -> None:
    """Stamp `paths` at a fixed past mtime so a freshness check cannot ride the clock.

    A directory argument stamps everything under it. Freshness compares
    wall-clock mtimes and the wall clock is not monotonic — it runs fast and is
    stepped backwards on a host time sync, so two writes seconds apart can be
    recorded out of order. Pin whichever side a test needs to be older.
    """
    for path in paths:
        for target in sorted(path.rglob("*")) if path.is_dir() else [path]:
            os.utime(target, (when, when))


def antedated(paths: VolumePaths, when: float = ANTEDATED) -> VolumePaths:
    """`antedate` a whole volume tree and return it, to wrap a stage call's argument.

    A stage given a pinned tree can only be re-run by an input it rewrites
    itself, which is what makes a rebuild-or-skip assertion clock-independent.
    """
    antedate(paths.root, when=when)
    return paths


@pytest.fixture(autouse=True, scope="session")
def never_sleep_between_retries() -> Iterator[None]:
    """Retry backoff is real seconds; no suite may pay them.

    `annotate_with_retry` spaces attempts so a provider blip cannot burn a
    page's whole allowance at once. Stage callers do not thread the injectable
    sleep, so every test driving a retry through a stage would wait for real —
    measured at 94s on one unmarked module, which lands in the fast suite.
    The base is read at call time, so zeroing it here reaches every path.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(invocation, "RETRY_BASE_S", 0.0)
        yield


@pytest.fixture(autouse=True, scope="session")
def never_spawn_a_model() -> Iterator[None]:
    """No model call in tests, EVER (conduct contract) — enforced, not trusted.

    Every backend takes an injected runner, so no test needs the real one — but escalation is
    default-ON where a ladder resolves, so a test driving `autogeoref run` against a real city
    TOML would reach the model CLI for real, and the warn-and-continue path would hide it. The
    default runner is therefore a tripwire: any test reaching it fails loudly, naming itself.
    SESSION scope, and that is load-bearing: function-scoped, pytest sets this up AFTER every
    higher-scoped fixture, so a module-scoped fixture driving the CLI ran its whole body with
    the real runner installed while still reporting PASS.
    """
    with pytest.MonkeyPatch.context() as mp:

        def tripwire(argv: list[str], timeout: float) -> NoReturn:
            raise AssertionError(
                "a test reached the real model CLI runner — no test may spend "
                f"model budget or touch the network: {argv[:2]}"
            )

        mp.setattr(cli_call, "_default_runner", tripwire)
        yield


@pytest.fixture(autouse=True, scope="session")
def never_spawn_a_subprocess_model() -> Iterator[None]:
    """Belt-and-braces for the tripwire above: a bare backend spawn also fails.

    Patching the module attribute above misses a caller holding its own reference, and
    OpenCode's isolated spawn is chosen by identity against that attribute so it routes past
    the patch. Guard the spawn itself, and let every non-model subprocess through untouched.

    SESSION scope for the reason given above — this is the guard that actually caught nothing
    during the measured 40-spawn escape, because it was not yet installed when the module-scoped
    fixture ran.
    """
    real_run = subprocess.run

    def guarded(*args: Any, **kwargs: Any) -> Any:
        argv = args[0] if args else kwargs.get("args", [])
        first = str(argv[0]) if isinstance(argv, (list, tuple)) and argv else str(argv)
        if Path(first).name in MODEL_CLI_BINARIES:
            raise AssertionError(
                f"a test tried to spawn the real `{Path(first).name}` CLI — no test "
                "may spend model budget or touch the network"
            )
        return real_run(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(subprocess, "run", guarded)
        yield


@pytest.fixture(autouse=True, scope="session")
def never_touch_the_network() -> Iterator[None]:
    """No HTTP request in tests, EVER — enforced at the httpx transport boundary.

    The model tripwires above only cover model-CLI spawns, while production defaults elsewhere
    construct live ``httpx`` clients, so a new test could reach the real network without
    tripping them. Patching the default transports' handlers fails such a test before any
    connection is attempted, while explicit ``httpx.MockTransport`` clients keep working.
    SESSION scope: this guard had the same escape as the two above, since a module-scoped
    fixture is set up before any function-scoped one. No such escape was observed, but a
    boundary guard a fixture can outrun is not a boundary.
    """

    def tripwire(self: object, request: httpx.Request) -> NoReturn:
        raise AssertionError(
            "a test reached httpx's default network transport "
            f"({request.method} {request.url}) — tests never use the network; "
            "inject an httpx.MockTransport client or a fake fetcher instead"
        )

    async def async_tripwire(self: object, request: httpx.Request) -> NoReturn:
        tripwire(self, request)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(httpx.HTTPTransport, "handle_request", tripwire)
        mp.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_tripwire)
        yield


AUTO_VOLUMES = [
    "sanborn01790_024",
    "sanborn01790_021",
    "sanborn01790_034",
    "sanborn01790_038",
]


@pytest.fixture(scope="session")
def auto_volumes() -> list[str]:
    """The fixture volumes with full recorded results trees."""
    return list(AUTO_VOLUMES)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    if not FIXTURE_TREE_SENTINEL.is_file():
        pytest.skip(
            "FIXTURE TREE NOT PRESENT — every golden contract (replay, warp, "
            "masks, rescue, seam, corroboration, parity, integrity) is "
            "being SKIPPED; a green run here "
            "does NOT certify the port. See FIXTURES.md for provenance."
        )
    return FIXTURES


@pytest.fixture(scope="session")
def ground_truth_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "ground-truth" / "api-layers-sanborn01790_006.5.json"


@pytest.fixture(scope="session")
def centerlines_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "reference" / "street_center_lines.geojson"


@pytest.fixture(scope="session")
def aliases_dir() -> Path:
    """Chicago's alias tables, which are tracked and so need no skip guard.

    Deliberately independent of ``fixtures_dir``: a test that reads only alias
    tables runs on a bare clone. One that also needs fixture data asks for
    ``fixtures_dir`` itself and picks the guard up there.
    """
    return CONFIGS / "chicago" / "aliases"


@pytest.fixture(scope="session")
def ref_annotations_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / "ref-volume" / "annotations"


@pytest.fixture
def _publish_without_viewer_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Queue tests exercise state transitions; viewer publication is tested separately."""
    from autogeoref.viewer import publish as viewer_publish

    monkeypatch.setattr(
        viewer_publish, "publish_volume", lambda *_args, **_kwargs: Path("published.pmtiles")
    )


@pytest.fixture(scope="module")
def viewer_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A complete, servable viewer bundle: the real page files over a real archive.

    Here rather than in one browser module because two of them load it, and a
    fixture imported by name collides with the parameter that requests it.
    Module-scoped, so file-based test distribution builds it once per file.
    """
    from viewer_browser_support import build_viewer_bundle

    return build_viewer_bundle(tmp_path_factory.mktemp("viewer-bundle"))
