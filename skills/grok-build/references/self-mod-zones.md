# Self-Modification Zones

| Zone | Paths | Gate |
|------|-------|------|
| Green | Project repos under `/root/projects/` | Standard plan → approve → execute |
| Yellow | `skills/grok-build/**`, `bin/grok-build-*`, OpenClaw workspace config | Requires explicit Alabama approval + INTERNAL tier |
| Red | `~/.config/moss/secrets.env`, production cron, MossRouter live config | Never modify via grok-build |

## Yellow-zone files (grok-build self-improve)

- `skills/grok-build/SKILL.md`
- `skills/grok-build/scripts/grok-watchdog.sh`
- `skills/grok-build/scripts/research-dispatcher.sh`
- `skills/grok-build/scripts/compose-plan.sh`
- `skills/grok-build/scripts/outbox-write.sh`
- `skills/grok-build/scripts/cost-track.sh`
- `bin/grok-build-cost-report`

Changes to yellow-zone files must:
1. Pass `skills/grok-build/tests/run-all.sh`
2. Remain backwards-compatible (opt-in flags)
3. Update `MEMORY.md` after merge