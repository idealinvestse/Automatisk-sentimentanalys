#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE="$SCRIPT_DIR/../scripts/compose-plan.sh"
export GROK_BUILD_WORKSPACE="${GROK_BUILD_WORKSPACE:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

out="$("$COMPOSE" --task "smoke test dry run" --workdir /tmp/test --slug smoke-test --dry-run --no-auto-research)"
prompt="$(echo "$out" | sed -n 's/^DRY_RUN prompt=\([^ ]*\).*/\1/p')"
task_id="$(echo "$out" | sed -n 's/.*task_id=//p')"

if [[ ! -f "$prompt" ]]; then
  echo "FAIL: prompt file not created: $prompt"
  exit 1
fi

if [[ ! -f "$GROK_BUILD_WORKSPACE/memory/grok-runs/$task_id/outbox/dry-run-complete.json" ]]; then
  echo "FAIL: dry-run-complete.json missing"
  exit 1
fi

echo "PASS test_dry_run"