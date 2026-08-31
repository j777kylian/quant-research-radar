#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

command -v uv >/dev/null || { echo "BLOCKED: uv is required" >&2; exit 2; }
[[ -f .env ]] || { echo "BLOCKED: .env is required" >&2; exit 2; }
if ! grep -Eq '^DEEPSEEK_API_KEY=[^[:space:]]+' .env; then
  echo "BLOCKED: DEEPSEEK_API_KEY is missing from .env" >&2
  exit 2
fi

SOURCE_DB="${PHASE16C_FAST_SOURCE_DB:-data/phase16b-live-20260829T111620Z-89642.db}"
[[ -f "$SOURCE_DB" ]] || { echo "BLOCKED: source archive is missing: $SOURCE_DB" >&2; exit 2; }
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ARCHIVE_DB="data/phase16c-fast-archive-$RUN_ID.db"
OUTPUT_ROOT="outputs/replay/phase16c-fast/$RUN_ID"
LOG_DIR="outputs/replay/logs"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
LOG="$LOG_DIR/phase16c-fast-$RUN_ID.log"

exec > >(tee -a "$LOG") 2>&1
printf 'MODE=%s\nPIT_BASIS=%s\nREAL_RECEIPT_PIT=%s\n' \
  'ACCELERATED_RECONSTRUCTIVE_REPLAY' \
  'SOURCE_NATIVE_AVAILABILITY_TIME' \
  'NOT_CLAIMED'
printf 'SOURCE_DB=%s\nARCHIVE_DB=%s\nOUTPUT_ROOT=%s\n' "$SOURCE_DB" "$ARCHIVE_DB" "$OUTPUT_ROOT"
cp "$SOURCE_DB" "$ARCHIVE_DB"
DATABASE_URL="sqlite:///$ROOT/$ARCHIVE_DB" uv run quant-radar collect hyperliquid \
  --history --limit 500 \
  --start 2026-08-30T10:00:00+00:00 \
  --end 2026-08-30T23:00:00+00:00
uv run quant-radar phase16c-fast \
  --database-url "sqlite:///$ROOT/$ARCHIVE_DB" \
  --output-dir "$OUTPUT_ROOT" \
  --provider deepseek
printf 'SUMMARY=%s\nLOG=%s\n' "$OUTPUT_ROOT/phase16c-fast-summary.json" "$LOG"
