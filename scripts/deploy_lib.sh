# Shared plumbing for push_archives.sh and deploy_site.sh. Sourced, not run.
#
# Holds the three things both need: the loud check for deployment settings, the
# two city facts only the package can answer, and the manifest read that says
# which archives the page will actually fetch.
#
# No credential is read here. The R2 access key stays in rclone's own config;
# these scripts name the remote and let rclone resolve the rest.

# The checkout these scripts belong to.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

note() {
  printf '%s\n' "$*" >&2
}

# Read KEY=value lines from an untracked settings file (.env, or whatever
# AUTOGEOREF_ENV_FILE names), without executing it and without overriding
# anything already exported: a variable you set for one run wins over the file.
# Blank lines, `#` comments, a leading `export`, and one layer of surrounding
# quotes are accepted; nothing else is interpreted.
load_env_file() {
  local file="$1" line key value
  if [ ! -f "$file" ]; then
    return 0
  fi
  while IFS= read -r line || [ -n "$line" ]; do
    # a file saved on the Windows side of the same tree ends its lines with CR,
    # which would otherwise ride into the middle of a remote name
    line="${line%$'\r'}"
    case "$line" in '' | '#'*) continue ;; esac
    line="${line#export }"
    key="${line%%=*}"
    value="${line#*=}"
    case "$key" in '' | *[!A-Za-z0-9_]*) continue ;; esac
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    if [ -z "${!key:-}" ]; then
      export "$key=$value"
    fi
  done < "$file"
}

load_env_file "${AUTOGEOREF_ENV_FILE:-$ROOT/.env}"

# Where viewer/ and deploy/ live. They are gitignored, so a worktree runs
# against the main checkout by setting AUTOGEOREF_DATA_ROOT to it.
DATA_ROOT="${AUTOGEOREF_DATA_ROOT:-$ROOT}"

# Names every missing setting at once, before any work happens.
require_env() {
  local name missing=()
  for name in "$@"; do
    if [ -z "${!name:-}" ]; then
      missing+=("$name")
    fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    die "not set: ${missing[*]} — put them in .env (see deploy.env.example) or export them"
  fi
}

require_tool() {
  if ! command -v "$1" > /dev/null 2>&1; then
    die "$1 is not on PATH${2:+ — $2}"
  fi
}

# The project interpreter, which is the only one that can import the package.
venv_python() {
  if [ -x "$ROOT/.venv/bin/python" ]; then
    printf '%s\n' "$ROOT/.venv/bin/python"
  else
    die "no $ROOT/.venv/bin/python — build the environment with make setup"
  fi
}

# The console script, preferring this checkout's environment over the PATH.
autogeoref_cli() {
  if [ -x "$ROOT/.venv/bin/autogeoref" ]; then
    printf '%s\n' "$ROOT/.venv/bin/autogeoref"
  elif command -v autogeoref > /dev/null 2>&1; then
    command -v autogeoref
  else
    die "no autogeoref command — build the environment with make setup"
  fi
}

# "<city-slug> <serving-dir>" for a city TOML: the page directory it publishes
# to and the deploy/tiles/ directory this pipeline writes its archives into.
# Both rules live in the package, so they are asked rather than restated.
city_facts() {
  local python
  # Pass the failure on by hand. `die` inside a helper reached through `$(...)`
  # exits only that subshell, and `set -e` does not fire on a failed assignment
  # in a function that is itself running inside a command substitution — so
  # without this the next line runs an empty command and reports 127 over the
  # top of the real message.
  python="$(venv_python)" || return 1
  "$python" - "$1" << 'PY'
import sys
from pathlib import Path

from autogeoref.config.load import load_city_config
from autogeoref.config.model import city_slug
from autogeoref.viewer.config import load_viewer_config

city = Path(sys.argv[1])
print(city_slug(load_city_config(city).name), load_viewer_config(city).serving_dirs[0])
PY
}

# Local paths of every archive one city's page fetches, basemap first because it
# is the small file the page cannot draw anything without.
#
# Takes the city's manifest and the viewer directory holding the page files. A
# volume's `pmtiles` is relative to the manifest; the basemap block is verbatim
# config and resolves against the page instead, which is a different directory.
# Refuses a manifest naming a file that is not on disk, or two archives sharing
# a basename: uploads are flat, so the second would replace the first.
manifest_archives() {
  local python
  python="$(venv_python 2> /dev/null || command -v python3)"
  if [ -z "$python" ]; then
    die "no python interpreter on PATH"
  fi
  "$python" - "$1" "$2" << 'PY'
import json
import sys
from pathlib import Path

manifest, viewer_dir = Path(sys.argv[1]), Path(sys.argv[2])
doc = json.loads(manifest.read_text(encoding="utf-8"))
paths = []
basemap = ((doc.get("site") or {}).get("basemap") or {}).get("pmtiles")
if basemap:
    paths.append(viewer_dir / basemap)
paths += [
    manifest.parent / v["pmtiles"] for v in doc.get("volumes") or [] if v.get("pmtiles")
]
if not paths:
    sys.exit(f"{manifest} names no pmtiles archive — bake and publish one first")
missing = [str(p) for p in paths if not p.is_file()]
if missing:
    sys.exit("the manifest names archives that are not on disk:\n  " + "\n  ".join(missing))
seen: dict[str, Path] = {}
for path in paths:
    resolved = path.resolve()
    clash = seen.get(resolved.name)
    if clash is not None and clash != resolved:
        sys.exit(f"two archives share the basename {resolved.name}: {clash} and {resolved}")
    seen[resolved.name] = resolved
for path in dict.fromkeys(seen.values()):
    print(path)
PY
}

# The path part of the public tiles base URL: the bucket prefix one city's
# archives are uploaded under, and what keeps two cities' basenames apart.
# Empty segments are dropped, the same normalisation `public_tiles_base` applies
# to the URL the manifest gets, so the upload key and the served URL agree.
tiles_prefix() {
  local rest path segment prefix=""
  rest="${1#*://}"
  path="${rest#*/}"
  if [ "$path" = "$rest" ]; then
    return 0
  fi
  local IFS=/
  for segment in $path; do
    if [ -n "$segment" ]; then
      prefix="${prefix:+$prefix/}$segment"
    fi
  done
  printf '%s' "$prefix"
}

# Refuse anything deploy-bundle would refuse, before a long upload rather than
# after it. The bundle applies the real rule; this states the same one for the
# upload, which can be run without it.
check_tiles_base() {
  local base="$1" lower authority
  lower="$(printf '%s' "$base" | tr '[:upper:]' '[:lower:]')"
  case "$lower" in
    http://* | https://*) ;;
    *) die "AUTOGEOREF_TILES_BASE must be an http(s) URL, got '$base'" ;;
  esac
  case "$base" in
    *\?* | *\#*) die "AUTOGEOREF_TILES_BASE must carry no query or fragment: '$base'" ;;
  esac
  authority="${base#*://}"
  authority="${authority%%/*}"
  case "$authority" in
    *@*) die "AUTOGEOREF_TILES_BASE must not embed credentials: '$base'" ;;
    '') die "AUTOGEOREF_TILES_BASE has no hostname: '$base'" ;;
  esac
}
