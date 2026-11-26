#!/bin/sh
# Container entrypoint. On first boot with an empty data disk, build the
# default season in the background so the site comes up right away.
set -e
SEASONS="${PITCHLAB_SEASONS:-2025 2026}"
DATA="${PITCHLAB_DATA_DIR:-/app/data}"
if ! ls "$DATA"/processed/meta_*.json >/dev/null 2>&1; then
  echo "No processed data yet; building $SEASONS in the background (takes ~10-20 min)"
  python -m pitchlab.build $SEASONS &
fi
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
