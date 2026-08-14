#!/bin/bash
# Upload the PMTiles archives one city's page fetches to that city's bucket
# prefix. Safe to run while a bake queue is still landing volumes.
#
# It adds what is missing and never overwrites what is already there. An archive
# keeps its name across a re-bake, so a plain copy would replace a published
# object, and a cache holding byte ranges of the old one then serves them beside
# ranges of the new: pmtiles.js compares the ETag it read first, raises
# EtagMismatch and drops the layer. Nothing reports it but the visitor's console.
#
# So a re-baked archive is refused by name, and replacing one is deliberate:
# --replace uploads that archive and purges its URL from the edge cache.
#
#   bash scripts/push_archives.sh --city configs/<city>/<city>.toml --dry-run
#   bash scripts/push_archives.sh --city configs/<city>/<city>.toml
#   bash scripts/push_archives.sh --city configs/<city>/<city>.toml --replace <volume>
#
# Settings come from the environment or .env — see deploy.env.example. The R2
# access key is not among them: rclone keeps it in its own config and this names
# only the remote.
set -euo pipefail

# shellcheck source=scripts/deploy_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/deploy_lib.sh"

#: An archive's content never changes under its name, so a visitor may keep one
#: for as long as it likes. Replacing one is what --replace and the purge exist
#: for.
CACHE_CONTROL="Cache-Control: public, max-age=31536000, immutable"

#: `Timing-Allow-Origin` does NOT belong beside it, though the archives want
#: one: without it the browser reports zero for every tile's size and duration,
#: so the page cannot measure the bulk of what a visitor waits on. rclone's s3
#: backend maps only a fixed set of header names onto the PUT — cache-control,
#: content-*, x-amz-meta-* — and logs `Don't know how to set key ... on upload`
#: for anything else while still exiting 0. So passing it here would upload an
#: archive without the header and print one error per archive in a deploy that
#: reports success. It belongs on the edge instead, as a Transform Rule on the
#: tiles hostname, which also covers the archives already published.

#: Part size for multipart uploads. Not tuning: at rclone's 5Mi default a 450 MB
#: archive is ~90 parts, and R2's S3 edge fails a part occasionally, so fewer
#: parts is fewer retries. rclone buffers chunk x transfers x upload concurrency,
#: so lower this if that is too much memory.
CHUNK_SIZE="${AUTOGEOREF_R2_CHUNK_SIZE:-64M}"
TRANSFERS="${AUTOGEOREF_R2_TRANSFERS:-4}"

#: Exit code for the one refusal a caller may want to tell apart: local archives
#: differ from published ones and no --replace named them.
EXIT_NEEDS_REPLACE=2

usage() {
  cat << 'EOF'
usage: bash scripts/push_archives.sh --city <city.toml> [options]

  --city <path>       city TOML; names the manifest and the serving directory
  --manifest <path>   read this manifest instead of viewer/<city-slug>/manifest.json
  --replace <name>    re-upload this archive over the published one and purge
                      its URL (repeatable; a volume id or an archive filename)
  --no-purge          with --replace, skip the purge and say so loudly
  --dry-run           classify and print; upload nothing
  -h, --help          this text
EOF
}

CITY=""
MANIFEST=""
DRY_RUN=0
NO_PURGE=0
REPLACE=()

while [ $# -gt 0 ]; do
  case "$1" in
    --city)
      CITY="${2:?--city needs a path}"
      shift 2
      ;;
    --manifest)
      MANIFEST="${2:?--manifest needs a path}"
      shift 2
      ;;
    --replace)
      REPLACE+=("${2:?--replace needs an archive name}")
      shift 2
      ;;
    --no-purge)
      NO_PURGE=1
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

if [ -z "$CITY" ] && [ -z "$MANIFEST" ]; then
  usage >&2
  die "--city is required"
fi

require_tool rclone "install it, e.g. sudo apt install rclone"
require_env AUTOGEOREF_TILES_BASE AUTOGEOREF_R2_BUCKET
check_tiles_base "$AUTOGEOREF_TILES_BASE"
if [ "${#REPLACE[@]}" -gt 0 ] && [ "$NO_PURGE" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
  require_tool curl
  require_env AUTOGEOREF_CF_ZONE_ID CLOUDFLARE_API_TOKEN
fi

TILES_BASE="${AUTOGEOREF_TILES_BASE%/}"
PREFIX="$(tiles_prefix "$TILES_BASE")"
BUCKET_ROOT="${AUTOGEOREF_R2_REMOTE:-r2}:${AUTOGEOREF_R2_BUCKET}"
DEST="$BUCKET_ROOT"
if [ -n "$PREFIX" ]; then
  DEST="$DEST/$PREFIX"
fi

if [ -z "$MANIFEST" ]; then
  CITY_FACTS="$(city_facts "$CITY")"
  read -r SLUG _ <<< "$CITY_FACTS"
  MANIFEST="$DATA_ROOT/viewer/$SLUG/manifest.json"
fi
if [ ! -f "$MANIFEST" ]; then
  die "no manifest at $MANIFEST — publish a volume for this city first"
fi
# the page directory: a volume's path is relative to the manifest, the basemap's
# to the page beside it
VIEWER_DIR="$(dirname "$(dirname "$MANIFEST")")"

ARCHIVE_LIST="$(manifest_archives "$MANIFEST" "$VIEWER_DIR")"
mapfile -t ARCHIVES <<< "$ARCHIVE_LIST"

# `lsf` exits 3 when the prefix holds nothing yet, which is the first deploy —
# and equally when the bucket or the prefix is a typo, where taking it for an
# empty prefix would classify every archive as new and upload the corpus
# somewhere nobody serves. So that answer is only accepted if the bucket lists.
list_published() {
  local out status=0
  out="$(rclone lsf --format "ps" --separator "|" --files-only --include "*.pmtiles" "$DEST/" 2>&1)" || status=$?
  if [ "$status" -eq 3 ]; then
    if ! rclone lsd "$BUCKET_ROOT" > /dev/null 2>&1; then
      die "$DEST/ holds nothing and $BUCKET_ROOT does not list — check the bucket name in AUTOGEOREF_R2_BUCKET and the remote in rclone config"
    fi
    return 0
  fi
  if [ "$status" -ne 0 ]; then
    printf '%s\n' "$out" >&2
    die "rclone could not list $DEST/ — is the remote configured? (rclone config)"
  fi
  printf '%s\n' "$out"
}

PUBLISHED_LIST="$(list_published)"
declare -A PUBLISHED_SIZE=()
while IFS='|' read -r name size; do
  if [ -n "$name" ]; then
    PUBLISHED_SIZE["$name"]="$size"
  fi
done <<< "$PUBLISHED_LIST"

# A volume id, an archive filename, or a path all name the same archive.
wanted_replaced() {
  local name="${1%.pmtiles}" token
  for token in ${REPLACE[@]+"${REPLACE[@]}"}; do
    token="$(basename "$token")"
    if [ "${token%.pmtiles}" = "$name" ]; then
      return 0
    fi
  done
  return 1
}

mib() {
  awk -v bytes="$1" 'BEGIN { printf "%.1f MiB", bytes / 1048576 }'
}

NEW=()
REPLACING=()
DIFFERING=()
present=0
declare -A LOCAL_NAMES=()

for archive in "${ARCHIVES[@]}"; do
  name="$(basename "$archive")"
  LOCAL_NAMES["$name"]=1
  local_size="$(stat -c %s "$archive")"
  published="${PUBLISHED_SIZE[$name]:-}"
  if [ -z "$published" ]; then
    NEW+=("$archive")
    printf 'new       %-42s %s\n' "$name" "$(mib "$local_size")"
  elif wanted_replaced "$name"; then
    REPLACING+=("$archive")
    printf 'REPLACE   %-42s %s published, %s local\n' \
      "$name" "$(mib "$published")" "$(mib "$local_size")"
  elif [ "$published" != "$local_size" ]; then
    DIFFERING+=("$name")
    printf 'DIFFERS   %-42s %s published, %s local\n' \
      "$name" "$(mib "$published")" "$(mib "$local_size")"
  else
    present=$((present + 1))
  fi
done

extra=()
for name in "${!PUBLISHED_SIZE[@]}"; do
  if [ -z "${LOCAL_NAMES[$name]:-}" ]; then
    extra+=("$name")
  fi
done

printf '%d archive(s) in %s\n' "${#ARCHIVES[@]}" "$MANIFEST"
printf '  %d already published unchanged, %d new, %d differing, %d to replace\n' \
  "$present" "${#NEW[@]}" "${#DIFFERING[@]}" "${#REPLACING[@]}"
if [ "${#extra[@]}" -gt 0 ]; then
  # never deleted here: an old archive may still be open in someone's session,
  # and this script is run while a bake queue is still adding volumes
  printf '  %d published archive(s) this manifest does not name: %s\n' \
    "${#extra[@]}" "$(printf '%s ' "${extra[@]}")"
fi

if [ "${#DIFFERING[@]}" -gt 0 ]; then
  note ""
  note "refusing to overwrite ${#DIFFERING[@]} published archive(s):"
  for name in "${DIFFERING[@]}"; do
    note "  $name"
  done
  note ""
  note "A re-baked archive keeps its name, so uploading it would replace the"
  note "published object and corrupt tiles for any reader holding a cached range."
  note "If that is what you want, name each one and it will be purged after upload:"
  note "  bash scripts/push_archives.sh --city <city.toml> --replace ${DIFFERING[0]%.pmtiles}"
  exit "$EXIT_NEEDS_REPLACE"
fi

for token in ${REPLACE[@]+"${REPLACE[@]}"}; do
  matched=0
  for archive in ${REPLACING[@]+"${REPLACING[@]}"} ${NEW[@]+"${NEW[@]}"}; do
    if [ "$(basename "${archive%.pmtiles}")" = "$(basename "${token%.pmtiles}")" ]; then
      matched=1
    fi
  done
  if [ "$matched" -eq 0 ]; then
    die "--replace $token names no archive in $MANIFEST"
  fi
done

if [ "${#NEW[@]}" -eq 0 ] && [ "${#REPLACING[@]}" -eq 0 ]; then
  printf 'nothing to upload\n'
  exit 0
fi

if [ "$DRY_RUN" -eq 1 ]; then
  printf 'dry run: nothing uploaded\n'
  exit 0
fi

# The file list rclone is given, removed however the script ends: set -e leaves
# through the failing command, not through the function.
LIST_FILE="$(mktemp)"
trap 'rm -f "$LIST_FILE"' EXIT

# One rclone per source directory, in manifest order, so the basemap goes first
# and the page is drawable while the atlas archives are still uploading.
upload_new() {
  local dir archive dirs=()
  for archive in "${NEW[@]}"; do
    dir="$(dirname "$archive")"
    if [[ " ${dirs[*]-} " != *" $dir "* ]]; then
      dirs+=("$dir")
    fi
  done
  for dir in "${dirs[@]}"; do
    : > "$LIST_FILE"
    for archive in "${NEW[@]}"; do
      if [ "$(dirname "$archive")" = "$dir" ]; then
        basename "$archive" >> "$LIST_FILE"
      fi
    done
    printf '\nuploading %d new archive(s) from %s\n' "$(wc -l < "$LIST_FILE")" "$dir"
    # --ignore-existing is the safety: this pass cannot overwrite by construction,
    # whatever the classification above concluded
    rclone copy "$dir" "$DEST/" \
      --files-from "$LIST_FILE" \
      --ignore-existing \
      --header-upload "$CACHE_CONTROL" \
      --s3-chunk-size "$CHUNK_SIZE" \
      --transfers "$TRANSFERS" \
      --progress
  done
}

purge_url() {
  local url="$1" response
  response="$(curl -sS -X POST \
    "https://api.cloudflare.com/client/v4/zones/${AUTOGEOREF_CF_ZONE_ID}/purge_cache" \
    -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
    -H "Content-Type: application/json" \
    --data "{\"files\":[\"${url}\"]}")"
  if ! printf '%s' "$response" | grep -q '"success"[[:space:]]*:[[:space:]]*true'; then
    printf '%s\n' "$response" >&2
    die "the cache purge for $url failed — that archive is replaced but stale at the edge"
  fi
  printf 'purged %s\n' "$url"
}

replace_published() {
  local archive name
  for archive in "${REPLACING[@]}"; do
    name="$(basename "$archive")"
    printf '\nreplacing published %s\n' "$name"
    rclone copyto "$archive" "$DEST/$name" \
      --header-upload "$CACHE_CONTROL" \
      --s3-chunk-size "$CHUNK_SIZE" \
      --progress
    if [ "$NO_PURGE" -eq 1 ]; then
      note "NOT purging $TILES_BASE/$name (--no-purge): until the edge expires it,"
      note "readers can mix ranges of the old archive with the new one"
    else
      purge_url "$TILES_BASE/$name"
    fi
  done
}

if [ "${#NEW[@]}" -gt 0 ]; then
  upload_new
fi
if [ "${#REPLACING[@]}" -gt 0 ]; then
  replace_published
fi
printf '\nuploaded to %s/\n' "$DEST"
