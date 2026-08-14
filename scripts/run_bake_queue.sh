#!/bin/bash
# Parallel per-volume PMTiles bake queue.
#
# List file: one "<mosaic-tif>|<out-pmtiles>" per line ('#' comments ok).
# Runs N lanes (default 2) of scripts/bake_volume_pmtiles.py under nice;
# each landed archive is published through the same locked helper the queue uses.
# Already-existing
# non-empty outputs are skipped, so an interrupted queue just re-runs.
#
#   setsid bash scripts/run_bake_queue.sh work/bake/queue.list 2 \
#       > work/bake/queue.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

if [ "${1:-}" == "--one" ]; then
  src="$2"; dst="$3"
  name="$(basename "$dst" .pmtiles)"
  if [ -s "$dst" ]; then echo "[$name] exists, skipping"; exit 0; fi
  echo "[$name] baking ($(date -Is))"
  if nice -n 19 .venv/bin/python scripts/bake_volume_pmtiles.py "$src" "$dst" \
      --processes 6 > "work/bake/$name.log" 2>&1; then
    if .venv/bin/autogeoref publish "$name" --source "$dst" --city configs/chicago/chicago.toml \
        --loc-catalog fixtures/loc-catalog-chicago.json > /dev/null; then
      echo "[$name] landed + manifest regenerated ($(date -Is))"
    else
      echo "[$name] FAILED to publish — archive remains at $dst"
      exit 1
    fi
  else
    echo "[$name] FAILED — see work/bake/$name.log"
  fi
  exit 0
fi

LIST="${1:?usage: run_bake_queue.sh <list-file> [lanes]}"
LANES="${2:-2}"
mkdir -p work/bake
while IFS='|' read -r src dst; do
  case "$src" in ''|'#'*) continue;; esac
  while [ "$(jobs -rp | wc -l)" -ge "$LANES" ]; do wait -n; done
  bash "$0" --one "$src" "$dst" &
done < "$LIST"
wait
echo "queue drained ($(date -Is))"
