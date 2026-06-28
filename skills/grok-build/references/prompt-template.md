## Goal

{{TASK}}

## Context

- **Workdir**: `{{WORKDIR}}`
- **Date**: {{DATE}}
- **Research tier**: {{TIER}}

## Requirements

1. Read relevant workspace files before proposing changes.
2. Follow existing patterns and coding standards in the workdir.
3. List concrete file paths to create or modify.
4. Include verification steps (tests, commands).
5. Note out-of-scope items explicitly.

## Out of Scope

- Unrelated refactors
- Committed secrets or API keys
- Changes outside the declared workdir unless justified

## Verification

- All new tests pass
- No regressions in existing test suite
- Manual smoke check where applicable