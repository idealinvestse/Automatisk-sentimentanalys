#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GROK_BUILD_WORKSPACE="${GROK_BUILD_WORKSPACE:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

for t in test_watchdog_timeout test_watchdog_silent_death test_research_dispatcher test_cost_report test_dry_run; do
  echo "==> $t"
  bash "$SCRIPT_DIR/${t}.sh"
done

echo "All grok-build tests passed."