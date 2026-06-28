#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GROK_BUILD_WORKSPACE="${GROK_BUILD_WORKSPACE:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
REPORT="$GROK_BUILD_WORKSPACE/bin/grok-build-cost-report"
TRACK="$SCRIPT_DIR/../scripts/cost-track.sh"

TASK_ID="test-cost-$$"
"$TRACK" record --task-id "$TASK_ID" --phase plan --input-tokens 100 --output-tokens 50 --cost-usd 0.01
"$TRACK" record --task-id "$TASK_ID" --phase execute --input-tokens 200 --output-tokens 100 --cost-usd 0.02

out="$("$REPORT" --task "$TASK_ID")"
echo "$out" | grep -q "plan"
echo "$out" | grep -q "0.0300" || echo "$out" | grep -q "0.03"

echo "PASS test_cost_report"