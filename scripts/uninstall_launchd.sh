#!/usr/bin/env bash
# Uninstall/disable the LaunchAgent.
set -euo pipefail
LABEL="com.quantradar.scheduler"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
launchctl unload "$PLIST" 2>/dev/null || true
rm -f "$PLIST"
echo "UNINSTALLED $LABEL"
