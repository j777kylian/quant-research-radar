#!/usr/bin/env bash
# One-command operational health snapshot.
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
command -v uv >/dev/null || { echo "BLOCKED: uv is required" >&2; exit 2; }
uv run quant-radar ops status "$@"
