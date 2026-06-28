# grok-build Memory Notes (2026-06-28)

## Resolved

- **Silent Grok failures**: use `skills/grok-build/scripts/grok-watchdog.sh` (exit 125 = silent death; check watchdog.log).
- **Shell messaging anti-pattern**: use `outbox-write.sh` + OpenClaw message tool; never `openclaw message send` via shell.

## Commands

```bash
bin/grok-build-cost-report
bin/grok-build-cost-report --task <id>
skills/grok-build/scripts/compose-plan.sh --dry-run --task "..." --workdir PATH
bash skills/grok-build/tests/run-all.sh
```

## Config

- `GROK_BUILD_TIMEOUT` (default 3600)
- `GROK_BUILD_RESEARCH_TTL_HOURS` (default 48)