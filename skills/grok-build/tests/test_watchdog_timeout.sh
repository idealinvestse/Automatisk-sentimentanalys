#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCHDOG="$SCRIPT_DIR/../scripts/grok-watchdog.sh"
export GROK_BUILD_WORKSPACE="${GROK_BUILD_WORKSPACE:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

TASK_ID="test-timeout-$$"
export GROK_BIN="/bin/sleep"

set +e
"$WATCHDOG" --task-id "$TASK_ID" --phase plan --timeout-secs 3 -- 10
code=$?
set -e

if [[ "$code" -ne 124 ]]; then
  echo "FAIL: expected exit 124, got $code"
  exit 1
fi

if [[ ! -f "$GROK_BUILD_WORKSPACE/memory/grok-runs/$TASK_ID/watchdog.log" ]]; then
  echo "FAIL: watchdog.log not created"
  exit 1
fi

echo "PASS test_watchdog_timeout"