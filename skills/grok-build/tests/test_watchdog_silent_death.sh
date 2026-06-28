#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCHDOG="$SCRIPT_DIR/../scripts/grok-watchdog.sh"
export GROK_BUILD_WORKSPACE="${GROK_BUILD_WORKSPACE:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

TASK_ID="test-silent-$$"
export GROK_BIN="/bin/false"

set +e
"$WATCHDOG" --task-id "$TASK_ID" --phase execute --timeout-secs 30 -- noop
code=$?
set -e

if [[ "$code" -ne 125 ]]; then
  echo "FAIL: expected exit 125 (silent death), got $code"
  exit 1
fi

echo "PASS test_watchdog_silent_death"