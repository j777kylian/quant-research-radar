#!/usr/bin/env bash
# Schedule once hourly at UTC minute 5; it only retrieves a small overlap.
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"
command -v uv >/dev/null || { echo "BLOCKED: uv is required" >&2; exit 2; }
DB="${PHASE20_DATABASE_URL:-sqlite:///$ROOT/data/phase20-market.db}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOG_DIR="outputs/phase20-collection/logs"
LOCK_DIR="outputs/phase20-collection/.market.lock"
mkdir -p "$LOG_DIR"
mkdir "$LOCK_DIR" 2>/dev/null || { echo "BLOCKED: live collector already running" >&2; exit 3; }
trap 'rmdir "$LOCK_DIR"' EXIT INT TERM
LOG="$LOG_DIR/live-$RUN_ID.log"
exec > >(tee -a "$LOG") 2>&1
uv run quant-radar phase20-market-collect live --database-url "$DB" --archive-root data/raw --code-sha "$(git rev-parse HEAD)" "$@"
