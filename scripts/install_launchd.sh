#!/usr/bin/env bash
# Install the user-level LaunchAgent that runs the scheduler tick every 30 min.
# No sudo, no system daemons, reversible via scripts/uninstall_launchd.sh.
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
LABEL="com.quantradar.scheduler"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
TICK="$ROOT/scripts/radar_scheduler_tick.sh"
LOG_DIR="$ROOT/outputs/operations/logs"

command -v uv >/dev/null || { echo "BLOCKED: uv is required" >&2; exit 2; }
[[ -x "$TICK" ]] || { echo "BLOCKED: scheduler tick script missing" >&2; exit 2; }
mkdir -p "$LOG_DIR"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$TICK</string>
    </array>
    <key>WorkingDirectory</key><string>$ROOT</string>
    <key>StartInterval</key><integer>1800</integer>
    <key>RunAtLoad</key><true/>
    <key>StandardOutPath</key><string>$LOG_DIR/scheduler.log</string>
    <key>StandardErrorPath</key><string>$LOG_DIR/scheduler.err.log</string>
</dict>
</plist>
EOF

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "INSTALLED $PLIST"
launchctl list | grep "$LABEL" || true
