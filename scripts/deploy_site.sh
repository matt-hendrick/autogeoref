#!/bin/bash
# The whole deploy for one city: bundle, upload archives, publish the page,
# then prove a range request off the deployed archives still works.
#
# Step order matters. A newly baked volume is invisible until the page is
# republished, because the volume list lives in the manifest inside the Pages
# bundle and not beside the archives on R2 — an upload alone appears to succeed
# and changes nothing a visitor can see.
#
#   bash scripts/deploy_site.sh --city configs/<city>/<city>.toml --dry-run
#   bash scripts/deploy_site.sh --city configs/<city>/<city>.toml
#
# Settings come from the environment or .env — see deploy.env.example.
set -euo pipefail

# shellcheck source=scripts/deploy_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/deploy_lib.sh"

#: Cloudflare will not cache an object this large on any plan below Enterprise;
#: those archives answer BYPASS forever, which is the rule working. Verifying
#: the edge cache against one of them would fail for the wrong reason.
CACHEABLE_LIMIT=$((512 * 1024 * 1024))

usage() {
  cat << 'EOF'
usage: bash scripts/deploy_site.sh --city <city.toml> [options]

  --city <path>       city TOML; names the manifest, bundle and serving directory
  --replace <name>    re-upload this archive over the published one and purge
                      its URL (repeatable; passed to push_archives.sh)
  --no-purge          with --replace, skip the purge and say so loudly
  --skip-upload       page only: rebuild the bundle and deploy it, upload nothing
  --dry-run           print the plan and classify the uploads; change nothing
  -h, --help          this text
EOF
}

CITY=""
DRY_RUN=0
SKIP_UPLOAD=0
PUSH_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --city)
      CITY="${2:?--city needs a path}"
      shift 2
      ;;
    --replace)
      PUSH_ARGS+=(--replace "${2:?--replace needs an archive name}")
      shift 2
      ;;
    --no-purge)
      PUSH_ARGS+=(--no-purge)
      shift
      ;;
    --skip-upload)
      SKIP_UPLOAD=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

if [ -z "$CITY" ]; then
  usage >&2
  die "--city is required"
fi
if [ ! -f "$CITY" ]; then
  die "no city config at $CITY"
fi
if [ "$SKIP_UPLOAD" -eq 1 ] && [ "${#PUSH_ARGS[@]}" -gt 0 ]; then
  die "--skip-upload uploads nothing, so it cannot be combined with --replace/--no-purge"
fi

# Everything named before anything happens. AUTOGEOREF_MAPBOX_TOKEN is in here
# because deploy-bundle regenerates config.js from it on every run: without it
# the rebuilt page ships with address search switched off, announced only by a
# note on stderr.
require_env AUTOGEOREF_TILES_BASE AUTOGEOREF_R2_BUCKET AUTOGEOREF_PAGES_PROJECT \
  AUTOGEOREF_MAPBOX_TOKEN
check_tiles_base "$AUTOGEOREF_TILES_BASE"
require_tool curl
require_tool npx "install Node.js; npx fetches wrangler on demand"
if [ "$SKIP_UPLOAD" -eq 0 ]; then
  require_tool rclone "install it, e.g. sudo apt install rclone"
fi

TILES_BASE="${AUTOGEOREF_TILES_BASE%/}"
CITY_FACTS="$(city_facts "$CITY")"
read -r SLUG SERVE_DIR <<< "$CITY_FACTS"
MANIFEST="$DATA_ROOT/viewer/$SLUG/manifest.json"
BUNDLE="$DATA_ROOT/deploy/$SLUG"
# R2 serves an error response without CORS headers, so the browser reports a
# missing archive as a CORS failure. The verification below asks with an Origin
# to tell the two apart; a preview deployment's origin must be in the bucket
# policy too, which is what this default is about.
ORIGIN="${AUTOGEOREF_PAGES_ORIGIN:-https://${AUTOGEOREF_PAGES_PROJECT}.pages.dev}"

if [ ! -f "$MANIFEST" ]; then
  die "no manifest at $MANIFEST — publish a volume for this city first"
fi

# Read before anything is built: this refuses a manifest naming an archive that
# is not on disk, and it is what the verification below probes.
MANIFEST_ARCHIVES="$(manifest_archives "$MANIFEST" "$(dirname "$(dirname "$MANIFEST")")")"
PUBLISHED_NAMES="$(printf '%s\n' "$MANIFEST_ARCHIVES" | sed 's#.*/##')"

# An archive in the serving directory the manifest does not name is a bake that
# was never published, and the deploy would ship the page without it. The
# overview companion is not a volume and nothing serves it.
unpublished=""
for archive in "$DATA_ROOT/deploy/tiles/$SERVE_DIR"/*.pmtiles; do
  if [ ! -f "$archive" ]; then
    continue
  fi
  case "$archive" in *-overview.pmtiles) continue ;; esac
  if ! printf '%s\n' "$PUBLISHED_NAMES" | grep -qxF "$(basename "$archive")"; then
    unpublished="$unpublished  $(basename "$archive")"$'\n'
  fi
done
if [ -n "$unpublished" ]; then
  note "warning: baked but not named by $MANIFEST:"
  note "${unpublished%$'\n'}"
  note "publish each one (autogeoref publish <volume> --city $CITY) or the page ships without it"
fi

printf 'city      %s (%s)\n' "$SLUG" "$CITY"
printf 'manifest  %s\n' "$MANIFEST"
printf 'bundle    %s\n' "$BUNDLE"
printf 'tiles     %s\n' "$TILES_BASE"
printf 'pages     %s\n' "$AUTOGEOREF_PAGES_PROJECT"

if [ "$DRY_RUN" -eq 1 ]; then
  printf '\ndry run — would run:\n'
  printf '  autogeoref deploy-bundle %s --city %s --viewer %s --out %s%s\n' \
    "$TILES_BASE" "$CITY" "$DATA_ROOT/viewer" "$BUNDLE" \
    "${AUTOGEOREF_SITE_URL:+ --site-url $AUTOGEOREF_SITE_URL}"
  if [ "$SKIP_UPLOAD" -eq 0 ]; then
    printf '  bash scripts/push_archives.sh --manifest %s ...\n' "$MANIFEST"
  fi
  printf '  npx wrangler pages deploy %s --project-name=%s\n' "$BUNDLE" "$AUTOGEOREF_PAGES_PROJECT"
  printf '  then a range request against %s/<archive>\n\n' "$TILES_BASE"
  if [ "$SKIP_UPLOAD" -eq 0 ]; then
    bash "$ROOT/scripts/push_archives.sh" --manifest "$MANIFEST" --dry-run \
      ${PUSH_ARGS[@]+"${PUSH_ARGS[@]}"}
  fi
  exit 0
fi

printf '\n== 1/4 bundle\n'
CLI="$(autogeoref_cli)"
BUNDLE_ARGS=()
if [ -n "${AUTOGEOREF_SITE_URL:-}" ]; then
  BUNDLE_ARGS+=(--site-url "$AUTOGEOREF_SITE_URL")
fi
"$CLI" deploy-bundle "$TILES_BASE" \
  --city "$CITY" \
  --viewer "$DATA_ROOT/viewer" \
  --out "$BUNDLE" \
  ${BUNDLE_ARGS[@]+"${BUNDLE_ARGS[@]}"}

if [ "$SKIP_UPLOAD" -eq 0 ]; then
  printf '\n== 2/4 archives\n'
  bash "$ROOT/scripts/push_archives.sh" --manifest "$MANIFEST" \
    ${PUSH_ARGS[@]+"${PUSH_ARGS[@]}"}
else
  printf '\n== 2/4 archives — skipped (--skip-upload)\n'
fi

printf '\n== 3/4 page\n'
npx --yes wrangler pages deploy "$BUNDLE" --project-name="$AUTOGEOREF_PAGES_PROJECT"

printf '\n== 4/4 verify\n'

# The smallest archive: it is the cheapest to range-request and the one that is
# certainly under the edge cache's size limit.
smallest_archive() {
  local archive smallest="" size best=""
  while IFS= read -r archive; do
    size="$(stat -c %s "$archive")"
    if [ -z "$best" ] || [ "$size" -lt "$best" ]; then
      best="$size"
      smallest="$archive"
    fi
  done <<< "$MANIFEST_ARCHIVES"
  printf '%s\t%s\n' "$smallest" "$best"
}

# One header's value, empty when the response does not carry it. `grep` finding
# nothing is not a failure here, and under pipefail it would end the script.
header_value() {
  printf '%s\n' "$1" | { grep -i "^$2:" || true; } | tail -1 | tr -d '\r'
}

require_header() {
  if [ -z "$(header_value "$1" "$2")" ]; then
    note "$1"
    die "$3 did not answer with a $2 header"
  fi
}

PROBE_LINE="$(smallest_archive)"
IFS=$'\t' read -r PROBE PROBE_SIZE <<< "$PROBE_LINE"
PROBE_URL="$TILES_BASE/$(basename "$PROBE")"
printf 'range request: %s\n' "$PROBE_URL"

FIRST="$(curl -sS -D- -o /dev/null --max-time 60 -r 0-99 -H "Origin: $ORIGIN" "$PROBE_URL")"
if ! printf '%s\n' "$FIRST" | grep -qiE '^HTTP/[0-9.]+ 206'; then
  note "$FIRST"
  die "$PROBE_URL did not answer 206 — a 404 means the archive is not uploaded"
fi
require_header "$FIRST" "access-control-allow-origin" "$PROBE_URL"
# pmtiles.js stores the ETag from its first fetch and compares it on every range
# after: unreachable from the browser, the layer dies with EtagMismatch
require_header "$FIRST" "etag" "$PROBE_URL"
require_header "$FIRST" "access-control-expose-headers" "$PROBE_URL"
require_header "$FIRST" "cache-control" "$PROBE_URL"
printf 'ok        206, CORS, ETag and Cache-Control\n'

if [ -z "$(header_value "$FIRST" "cf-cache-status")" ]; then
  note "note: no cf-cache-status header, so the edge cache check does not apply here"
elif [ "$PROBE_SIZE" -ge "$CACHEABLE_LIMIT" ]; then
  note "note: $PROBE is over the edge cache's size limit, so it answers BYPASS"
else
  SECOND="$(curl -sS -D- -o /dev/null --max-time 60 -r 0-99 -H "Origin: $ORIGIN" "$PROBE_URL")"
  STATUS="$(header_value "$SECOND" "cf-cache-status")"
  case "$STATUS" in
    *HIT*) printf 'ok        %s\n' "$STATUS" ;;
    *DYNAMIC*)
      note "$STATUS"
      note ".pmtiles is not one of the extensions Cloudflare caches by default,"
      note "so the archives are not even eligible until the hostname has a Cache Rule."
      die "the archives are not cached at the edge"
      ;;
    "")
      die "the repeated range request answered without a cf-cache-status header"
      ;;
    *)
      note "$STATUS"
      die "a repeated range request did not hit the edge cache"
      ;;
  esac
fi

# The archives being right proves nothing about the volume LIST, which lives in
# the manifest inside the bundle. Reported rather than gated: a preview
# deployment and a production alias that has not caught up yet both look like a
# mismatch and are both fine.
LIVE_MANIFEST="$(curl -sS --max-time 30 "$ORIGIN/$(basename "$MANIFEST")" 2> /dev/null || true)"
if [ "$LIVE_MANIFEST" = "$(cat "$BUNDLE/$(basename "$MANIFEST")")" ]; then
  printf 'ok        %s serves the manifest this run built\n' "$ORIGIN"
else
  note "note: $ORIGIN is not yet serving the manifest this run built."
  note "A preview deployment or an alias that has not caught up both read like this;"
  note "check the deployment URL wrangler printed above before treating it as wrong."
fi

printf '\ndeployed %s\n' "$SLUG"
