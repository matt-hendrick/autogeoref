"""Harness for the deploy shell scripts: a temporary data root and stub tools.

Both scripts are run for real with `rclone`, `curl` and `npx` stubbed onto PATH,
so no test reaches the network and none asserts on what the source text says.
The stubs record every invocation and answer from files the test writes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from viewer_support import page_stub

ROOT = Path(__file__).resolve().parent.parent
PUSH = ROOT / "scripts" / "push_archives.sh"
DEPLOY = ROOT / "scripts" / "deploy_site.sh"

#: What deploy-bundle copies into a bundle and refuses to build without.
PAGE_FILES = (
    "index.html",
    "404.html",
    "app.css",
    "lib.js",
    "app.js",
    "walkthrough.html",
    "walkthrough.css",
    "walkthrough.js",
    "favicon.ico",
    "favicon.svg",
    "apple-touch-icon.png",
    "safari-pinned-tab.svg",
    "_headers",
)

CITY_TOML = '[city]\nname = "Test City"\ncenterlines = "streets.geojson"\naliases_dir = "aliases"\n'

#: `lsf` on an empty prefix and on a mistyped bucket both exit 3; `lsd` is what
#: tells them apart, and STUB_NO_BUCKET is the mistyped one.
RCLONE_STUB = """#!/bin/bash
set -u
printf 'rclone %s\\n' "$*" >> "$STUB_LOG"
prev=""
for arg in "$@"; do
  case "$prev" in --files-from) sed 's/^/  from: /' "$arg" >> "$STUB_LOG" ;; esac
  prev="$arg"
done
case "${1:-}" in
  lsf)
    if [ -f "$STUB_PUBLISHED" ] && [ -z "${STUB_NO_BUCKET:-}" ]; then
      cat "$STUB_PUBLISHED"
    else
      exit 3
    fi
    ;;
  lsd)
    if [ -n "${STUB_NO_BUCKET:-}" ]; then exit 3; fi
    ;;
esac
exit 0
"""

#: A first range request misses the edge cache and a later one hits it, so the
#: deploy's gate is only satisfied by the repeat fetch. The deployed page serves
#: whatever STUB_PAGE names, which is the bundle this run built.
CURL_STUB = """#!/bin/bash
set -u
printf 'curl %s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *purge_cache*)
    if [ -n "${STUB_PURGE_FAILS:-}" ]; then
      printf '{"success":false,"errors":[{"message":"no"}]}\\n'
    else
      printf '{"result":{"id":"z"},"success":true,"errors":[]}\\n'
    fi
    exit 0
    ;;
  *manifest.json*)
    if [ -f "${STUB_PAGE:-}" ]; then cat "$STUB_PAGE"; else exit 22; fi
    exit 0
    ;;
esac
count=0
if [ -f "$STUB_LOG.curls" ]; then count="$(cat "$STUB_LOG.curls")"; fi
count=$((count + 1))
printf '%s' "$count" > "$STUB_LOG.curls"
if [ "$count" -eq 1 ]; then
  sed 's/^cf-cache-status:.*/cf-cache-status: MISS/I' "$STUB_HEADERS"
elif [ -n "${STUB_SECOND_UNCACHED:-}" ]; then
  grep -vi '^cf-cache-status:' "$STUB_HEADERS"
else
  cat "$STUB_HEADERS"
fi
"""

NPX_STUB = """#!/bin/bash
set -u
printf 'npx %s\\n' "$*" >> "$STUB_LOG"
printf 'https://example-project.pages.dev\\n'
"""

GOOD_HEADERS = "\r\n".join(
    [
        "HTTP/2 206",
        "content-range: bytes 0-99/1024",
        'etag: "abc"',
        "access-control-allow-origin: https://example-project.pages.dev",
        "access-control-expose-headers: etag,content-range",
        "cache-control: public, max-age=31536000, immutable",
        "cf-cache-status: HIT",
        "",
        "",
    ]
)


class Deployment:
    """A temporary data root, the stub tools, and the environment they need."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.log = tmp_path / "calls.log"
        self.published = tmp_path / "published.txt"
        self.headers = tmp_path / "headers.txt"
        self.data = tmp_path / "data"
        self.city = tmp_path / "city.toml"
        self.city.write_text(CITY_TOML)
        self.viewer = self.data / "viewer"
        self.manifest = self.viewer / "test-city" / "manifest.json"
        self.manifest.parent.mkdir(parents=True)
        for name in PAGE_FILES:
            (self.viewer / name).write_text(page_stub(name))
        self.tiles = self.data / "deploy" / "tiles"
        (self.tiles / "autogeoref").mkdir(parents=True)
        (self.tiles / "basemap").mkdir(parents=True)
        self.headers.write_text(GOOD_HEADERS)
        self._write_stubs()

    def _write_stubs(self) -> None:
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        for name, body in (("rclone", RCLONE_STUB), ("curl", CURL_STUB), ("npx", NPX_STUB)):
            stub = bin_dir / name
            stub.write_text(body)
            stub.chmod(0o755)
        self.bin = bin_dir

    def archive(self, directory: str, name: str, size: int) -> Path:
        path = self.tiles / directory / f"{name}.pmtiles"
        path.write_bytes(b"\0" * size)
        return path

    def write_manifest(self, volumes: list[str], basemap: str | None = None) -> None:
        doc: dict[str, object] = {
            "volumes": [
                {"id": v, "era": "1890s", "pmtiles": f"../../deploy/tiles/autogeoref/{v}.pmtiles"}
                for v in volumes
            ],
            "site": {"name": "Test City"},
        }
        if basemap is not None:
            site = doc["site"]
            assert isinstance(site, dict)
            site["basemap"] = {
                "type": "vector",
                "pmtiles": f"../deploy/tiles/basemap/{basemap}.pmtiles",
            }
        self.manifest.write_text(json.dumps(doc, indent=1))

    def publish(self, entries: dict[str, int]) -> None:
        """Declare what the bucket prefix already holds: name -> byte size."""
        self.published.write_text(
            "".join(f"{name}.pmtiles|{size}\n" for name, size in entries.items())
        )

    def env(self, **overrides: str) -> dict[str, str]:
        env = dict(os.environ)
        # the console script may only be on PATH via the running interpreter's
        # environment; the stubs must still win
        interpreter = Path(sys.executable).parent
        env["PATH"] = os.pathsep.join([str(self.bin), str(interpreter), env["PATH"]])
        env.update(
            {
                "AUTOGEOREF_DATA_ROOT": str(self.data),
                "AUTOGEOREF_TILES_BASE": "https://tiles.example.com/testcity",
                "AUTOGEOREF_R2_BUCKET": "example-bucket",
                "AUTOGEOREF_PAGES_PROJECT": "example-project",
                "AUTOGEOREF_MAPBOX_TOKEN": "pk.testtoken",
                "AUTOGEOREF_CF_ZONE_ID": "zone123",
                "CLOUDFLARE_API_TOKEN": "cftoken",
                # the optional ones too, or a maintainer who exports what
                # docs/OPERATIONS.md documents fails these tests on their own machine
                "AUTOGEOREF_R2_REMOTE": "r2",
                "AUTOGEOREF_R2_CHUNK_SIZE": "64M",
                "AUTOGEOREF_R2_TRANSFERS": "4",
                "AUTOGEOREF_SITE_URL": "https://atlas.example.com",
                "AUTOGEOREF_PAGES_ORIGIN": "https://example-project.pages.dev",
                # never the maintainer's own .env: these tests set every value
                "AUTOGEOREF_ENV_FILE": str(self.root / "absent.env"),
                "STUB_LOG": str(self.log),
                "STUB_PUBLISHED": str(self.published),
                "STUB_HEADERS": str(self.headers),
                # what the deployed page answers with: the bundle this run wrote
                "STUB_PAGE": str(self.data / "deploy" / "test-city" / "manifest.json"),
            }
        )
        for key, value in overrides.items():
            if value:
                env[key] = value
            else:
                env.pop(key, None)
        return env

    def run(self, script: Path, *args: str, **overrides: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(script), *args],
            cwd=ROOT,
            env=self.env(**overrides),
            capture_output=True,
            text=True,
            check=False,
        )

    def scripts_without_an_environment(self) -> Path:
        """A copy of the deploy scripts whose checkout root has no `.venv`.

        The scripts resolve their own root from `BASH_SOURCE`, so copying them
        is all it takes to reach the branch that refuses to run without the
        project interpreter.
        """
        target = self.root / "bare" / "scripts"
        target.mkdir(parents=True)
        for name in ("push_archives.sh", "deploy_site.sh", "deploy_lib.sh"):
            copied = target / name
            copied.write_text((ROOT / "scripts" / name).read_text(encoding="utf-8"))
            copied.chmod(0o755)
        return target

    def calls(self) -> str:
        return self.log.read_text() if self.log.exists() else ""
