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

SHA="$(git rev-parse HEAD)"
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOG_DIR="outputs/replay/logs"
DB="data/phase16a-replay.db"
LOCK_DIR=".phase16a-replay.lock"
mkdir -p "$LOG_DIR" data outputs/replay
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "BLOCKED: another Phase 1.6A run appears active ($LOCK_DIR)" >&2
  exit 2
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
LOG="$LOG_DIR/$RUN_ID.log"
exec > >(tee -a "$LOG") 2>&1

export DATABASE_URL="sqlite:///$ROOT/$DB"
export LLM_PROVIDER=deepseek

if [[ -n "$(git status --porcelain)" ]]; then
  echo "WARNING: repository has uncommitted changes; recording SHA $SHA"
else
  echo "Repository clean at $SHA"
fi

NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
WARMUP_START="$(uv run python -c 'from datetime import datetime,UTC,timedelta; print((datetime.now(UTC)-timedelta(days=33)).isoformat())')"
uv run quant-radar init-db
uv run quant-radar collect hyperliquid --history --limit 800
uv run quant-radar collect arxiv --limit 25 || echo "arXiv collection degraded; continuing"
uv run quant-radar collect repec --limit 25 || echo "RePEc degraded; continuing"

mapfile -t DAYS < <(uv run python -c 'from quant_research_radar.replay import replay_dates; from datetime import datetime,UTC; print("\n".join(d.isoformat() for d in replay_dates(datetime.now(UTC))))')
for DAY in "${DAYS[@]}"; do
  CUTOFF="${DAY}T23:59:59.999999+00:00"
  uv run quant-radar replay --date "$DAY" --as-of "$CUTOFF" --provider deepseek --output-dir outputs/replay --warmup-start "$WARMUP_START" --code-sha "$SHA"
done

uv run python - <<'PY'
import json
from datetime import UTC, datetime
from pathlib import Path
from quant_research_radar.replay import replay_dates, write_summary
from quant_research_radar.db import make_engine, make_session_factory
from quant_research_radar.config import get_settings

root = Path("outputs/replay")
dates = replay_dates(datetime.now(UTC))
audits = [json.loads((root / day.isoformat() / "audit.json").read_text()) for day in dates]
summary = write_summary(root, datetime.now(UTC), datetime.now(UTC), dates, datetime.now(UTC), audits, "${SHA}")
print(f"SUMMARY={summary}")
PY

echo "PASS/PARTIAL: Phase 1.6A artifacts generated"
echo "LOG=$LOG"
echo "SUMMARY=outputs/replay/phase16a-summary.json"
for DAY in "${DAYS[@]}"; do echo "REPORT=outputs/replay/$DAY/daily.md AUDIT=outputs/replay/$DAY/audit.json"; done
