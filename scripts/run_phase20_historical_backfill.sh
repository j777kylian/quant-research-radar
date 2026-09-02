#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"
command -v uv >/dev/null || { echo "BLOCKED: uv is required" >&2; exit 2; }
[[ -f pyproject.toml ]] || { echo "BLOCKED: not a quant-research-radar checkout" >&2; exit 2; }
DB="${PHASE20_DATABASE_URL:-sqlite:///$ROOT/data/phase20-market.db}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOG_DIR="outputs/phase20-collection/logs"
LOCK_DIR="outputs/phase20-collection/.market.lock"
mkdir -p "$LOG_DIR"
mkdir "$LOCK_DIR" 2>/dev/null || { echo "BLOCKED: historical collector already running: $LOCK_DIR" >&2; exit 3; }
trap 'rmdir "$LOCK_DIR"' EXIT INT TERM
LOG="$LOG_DIR/historical-$RUN_ID.log"
AUDIT="outputs/phase20-collection/audits/historical-$RUN_ID.json"
mkdir -p "$(dirname "$AUDIT")"
exec > >(tee -a "$LOG") 2>&1
SHA="$(git rev-parse HEAD)"
printf 'RUN_ID=%s\nDATABASE_URL=%s\nMODE=ACCELERATED_RECONSTRUCTIVE_RESEARCH\nREAL_RECEIPT_PIT=NOT_CLAIMED\nLOG=%s\n' "$RUN_ID" "$DB" "$LOG"
uv run quant-radar phase20-market-collect historical --database-url "$DB" --archive-root data/raw --audit-output "$AUDIT" --code-sha "$SHA" "$@"
printf 'AUDIT=%s\n' "$AUDIT"
