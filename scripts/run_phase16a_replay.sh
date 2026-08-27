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
export PHASE16A_SHA="$SHA"
export PHASE16A_RUN_ID="$RUN_ID"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "WARNING: repository has uncommitted changes; recording SHA $SHA"
else
  echo "Repository clean at $SHA"
fi

NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
WARMUP_START="$(uv run python -c 'from quant_research_radar.replay import replay_dates; from datetime import datetime,UTC,timedelta; dates=replay_dates(datetime.now(UTC)); print((datetime.combine(min(dates), datetime.min.time(), UTC)-timedelta(days=33)).isoformat())')"
LATEST_REPLAY_CUTOFF="$(uv run python -c 'from quant_research_radar.replay import replay_dates,utc_day_cutoff; from datetime import datetime,UTC; print(utc_day_cutoff(max(replay_dates(datetime.now(UTC)))).isoformat())')"
export PHASE16A_END="$LATEST_REPLAY_CUTOFF"
uv run quant-radar init-db
uv run quant-radar collect hyperliquid --history --limit 1200 --start "$WARMUP_START" --end "$LATEST_REPLAY_CUTOFF" --phase16a-run-id "$RUN_ID"
uv run quant-radar replay --date "$(date -u +%Y-%m-%d)" --as-of "$LATEST_REPLAY_CUTOFF" --collection-end "$LATEST_REPLAY_CUTOFF" --provider fake --output-dir outputs/replay --warmup-start "$WARMUP_START" --code-sha "$SHA" --phase16a-run-id "$RUN_ID" --coverage-only
uv run python - "$WARMUP_START" "$LATEST_REPLAY_CUTOFF" <<'PY'
import sys
from datetime import datetime, UTC
from sqlalchemy import select
from quant_research_radar.db import get_phase16a_collection_run, make_engine, make_session_factory, normalize_utc
from quant_research_radar.replay import ASSETS, funding_coverage
from quant_research_radar.config import get_settings

start = normalize_utc(datetime.fromisoformat(sys.argv[1]))
end = normalize_utc(datetime.fromisoformat(sys.argv[2]))
engine = make_engine("sqlite:///data/phase16a-replay.db")
session = make_session_factory(engine)()
run = get_phase16a_collection_run(session, source="hyperliquid", phase16a_run_id=__import__("os").environ["PHASE16A_RUN_ID"], requested_start=start, requested_end=end, code_sha=__import__("os").environ["PHASE16A_SHA"])
if run is None or not run.diagnostics:
    raise SystemExit("BLOCKED: matching collection diagnostics are missing")
coverage = funding_coverage(session, start, end, run.diagnostics)
print("Funding coverage:", coverage)
if not all(item["required_warmup_satisfied"] for item in coverage.values()):
    raise SystemExit("PARTIAL: required bounded funding warm-up is unavailable")
PY
uv run quant-radar collect arxiv --limit 25 || echo "arXiv collection degraded; continuing"
uv run quant-radar collect repec --limit 25 || echo "RePEc degraded; continuing"

DAYS="$(uv run python -c 'from quant_research_radar.replay import replay_dates; from datetime import datetime,UTC; print("\n".join(d.isoformat() for d in replay_dates(datetime.now(UTC))))')"
while IFS= read -r DAY; do
  [ -n "$DAY" ] || continue
  CUTOFF="${DAY}T23:59:59.999999+00:00"
  uv run quant-radar replay --date "$DAY" --as-of "$CUTOFF" --collection-end "$LATEST_REPLAY_CUTOFF" --provider deepseek --output-dir outputs/replay --warmup-start "$WARMUP_START" --code-sha "$SHA" --phase16a-run-id "$RUN_ID"
done <<EOF
$DAYS
EOF

export PHASE16A_SHA="$SHA"
export PHASE16A_RUN_ID="$RUN_ID"
uv run python - <<'PY'
import json
from datetime import UTC, datetime
from pathlib import Path
from quant_research_radar.replay import parse_utc_timestamp, replay_dates, write_summary
from quant_research_radar.db import make_engine, make_session_factory
from quant_research_radar.config import get_settings

root = Path("outputs/replay")
dates = replay_dates(datetime.now(UTC))
audits = [json.loads((root / day.isoformat() / "audit.json").read_text()) for day in dates]
import os

requested_end = parse_utc_timestamp(os.environ["PHASE16A_END"], "PHASE16A_END")
warmup_start = parse_utc_timestamp(audits[0]["warmup_start"], "warmup_start")
summary = write_summary(root, datetime.now(UTC), datetime.now(UTC), dates, warmup_start, audits, os.environ["PHASE16A_SHA"], phase16a_run_id=os.environ["PHASE16A_RUN_ID"], requested_end=requested_end)
print(f"SUMMARY={summary}")
PY

echo "PASS/PARTIAL: Phase 1.6A artifacts generated"
echo "LOG=$LOG"
echo "SUMMARY=outputs/replay/phase16a-summary.json"
while IFS= read -r DAY; do
  [ -n "$DAY" ] || continue
  echo "REPORT=outputs/replay/$DAY/daily.md AUDIT=outputs/replay/$DAY/audit.json"
done <<EOF
$DAYS
EOF
