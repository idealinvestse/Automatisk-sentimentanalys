#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISPATCHER="$SCRIPT_DIR/../scripts/research-dispatcher.sh"
export GROK_BUILD_WORKSPACE="${GROK_BUILD_WORKSPACE:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

out="$("$DISPATCHER" detect --task "add new LLM provider integration" --json)"
tier="$(echo "$out" | python3 -c 'import json,sys; print(json.load(sys.stdin)["tier"])')"

if [[ "$tier" != "HIGH" ]]; then
  echo "FAIL: expected HIGH tier, got $tier"
  exit 1
fi

echo "$out" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert "cache_path" in d and "task_id" in d'

out2="$("$DISPATCHER" detect --task "fix typo in readme" --json)"
tier2="$(echo "$out2" | python3 -c 'import json,sys; print(json.load(sys.stdin)["tier"])')"
if [[ "$tier2" != "INTERNAL" ]]; then
  echo "FAIL: expected INTERNAL tier, got $tier2"
  exit 1
fi

echo "PASS test_research_dispatcher"