#!/usr/bin/env bash
# Canonical Daily operations entry point.
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
command -v uv >/dev/null || { echo "BLOCKED: uv is required" >&2; exit 2; }
[[ -f .env ]] || { echo "BLOCKED: .env is required" >&2; exit 2; }
SHA="$(git rev-parse HEAD)"
LOG_DIR="outputs/operations/logs"
mkdir -p "$LOG_DIR"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOG="$LOG_DIR/daily-$RUN_ID.log"
exec > >(tee -a "$LOG") 2>&1
printf 'MODE=daily CODE_SHA=%s LOG=%s\n' "$SHA" "$LOG"
uv run quant-radar ops daily --code-sha "$SHA" "$@"
