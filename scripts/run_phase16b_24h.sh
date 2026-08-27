#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"
command -v uv >/dev/null || { echo "BLOCKED: uv is required" >&2; exit 2; }
[[ -f .env ]] || { echo "BLOCKED: .env is required" >&2; exit 2; }
grep -Eq '^DEEPSEEK_API_KEY=[^[:space:]]+' .env || { echo "BLOCKED: DEEPSEEK_API_KEY is missing" >&2; exit 2; }
LOCK_DIR=".phase16b-live.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then echo "BLOCKED: live smoke already active" >&2; exit 2; fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ROOT_OUT="outputs/live/phase16b/$RUN_ID"
DB="sqlite:///$ROOT/data/phase16b-live-$RUN_ID.db"
mkdir -p "$ROOT_OUT/logs"
exec > >(tee "$ROOT_OUT/logs/harness.log") 2>&1
export DATABASE_URL="$DB"
export LLM_PROVIDER=deepseek
SHA="$(git rev-parse HEAD)"
echo "LIVE_RUN_ID=$RUN_ID"
echo "DATABASE=$DB"
uv run quant-radar init-db
if ! uv run python -m quant_research_radar.cli live-cycle --database-url "$DB" --output-dir "$ROOT_OUT" --cycle 1 --code-sha "$SHA"; then
  echo "LIVE_CYCLE_STATUS=BLOCKED_OR_FAILED; not waiting for cycle 2" >&2
  exit 1
fi
echo "Waiting 24 hours before cycle 2"
sleep 86400
uv run python -m quant_research_radar.cli live-cycle --database-url "$DB" --output-dir "$ROOT_OUT" --cycle 2 --code-sha "$SHA"
uv run python - "$ROOT_OUT" "$SHA" <<'PY'
import sys
from pathlib import Path
from quant_research_radar.live import write_live_review, write_live_summary
import json
root = Path(sys.argv[1])
audits = [json.loads((root / f"cycle-{cycle}" / "audit.json").read_text()) for cycle in (1, 2)]
print(f"SUMMARY={write_live_summary(root, audits, sys.argv[2])}")
print(f"REVIEW={write_live_review(root)}")
PY
