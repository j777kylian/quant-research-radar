#!/usr/bin/env bash
# Lightweight scheduler tick (run by the LaunchAgent every ~30 minutes).
# This is NOT a data-collection run: it makes no provider/LLM call unless a
# Daily or Weekly job is logically due in Asia/Shanghai time.
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
command -v uv >/dev/null || { echo "BLOCKED: uv is required" >&2; exit 2; }
[[ -f .env ]] || { echo "BLOCKED: .env is required" >&2; exit 2; }
SHA="$(git rev-parse HEAD)"
uv run quant-radar ops tick --code-sha "$SHA" "$@"
