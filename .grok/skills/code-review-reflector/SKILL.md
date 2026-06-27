---
name: code-review-reflector
description: Use this skill after a coding session to reflect on the changes, find undetected bugs, edge cases, inconsistencies, security issues, performance problems and missed better approaches. Produces a structured review report. Trigger with code review my session, review latest changes, reflektera över kodsessionen, hitta missade fel, post session code review, review my coding session
---

# Code Review Reflector

This skill acts as a **second, calmer pair of eyes** after a coding session. It reviews the actual changes made (diffs + context), reflects on them, and surfaces problems, risks and improvement opportunities that are easy to miss while in the flow of implementation.

It is especially powerful when used:
- Right after finishing a feature or refactor (before commit/push)
- When you feel "it works but something feels off"
- As a quality gate before handing work to another agent or deploying
- To build better coding habits over time by learning what you tend to miss

## When to Use
- Immediately after a focused coding session
- Before committing a larger set of changes
- When you want an objective "did I miss anything important?" check
- After an agent has implemented a task from `RECOMMENDED_NEXT_TASKS.md`
- Periodically on recent work to catch accumulating small issues

## Core Principles
- **Focus on what actually changed** — do not review the entire codebase, only the diff + necessary context.
- **Multiple angles of attack** — correctness, robustness, consistency, performance, security, maintainability, tests, docs, and "better ways".
- **Constructive and specific** — every finding should explain *why* it matters and ideally suggest a concrete fix or question.
- **Categorized severity** — Critical (bugs that can break things), Important, Improvement, Question.
- **Works with existing context** — uses `PROJECT_STATUS.md`, `AGENT_CONTEXT.md` and `RECOMMENDED_NEXT_TASKS.md` when available to judge consistency and fit.
- **Actionable output** — produces a clear review document that can be used to fix issues or feed new tasks back into the planning skill.

## Workflow — Execute in Order

### Phase 1: Identify the Changes from This Session
Determine what to review:

- Preferred: Uncommitted changes + recent commits since last clean state.
  - Run `git status --porcelain`
  - Run `git diff` (unstaged) and `git diff --cached` (staged)
  - If there are recent commits: `git log --oneline -10` and ask user or infer the range since last "good" state (or since last review)

- Alternative: User specifies a commit range, branch diff, or specific files.

- If nothing obvious is staged/unstaged, ask the user: "What changes from this session do you want me to review? (e.g. last 3 commits, specific files, or the whole working tree)"

Save the raw diff(s) for reference.

### Phase 2: Gather Context for the Review
Read supporting files so the review is grounded in reality:

- Read relevant parts of `AGENT_CONTEXT.md` and `PROJECT_STATUS.md` (especially architecture, coding conventions, important invariants)
- Read `RECOMMENDED_NEXT_TASKS.md` if it exists (to see if the changes align with the intended task)
- For the changed files: read the full current version + enough surrounding context (use `read_file` with limits or multiple calls)
- Understand the intent: What was the goal of this session? (from task description, commit messages, or ask user)

Key things to internalize:
- Existing patterns and conventions in this codebase
- Critical invariants and "do not break" rules
- What "good" looks like in this project

### Phase 3: Multi-Angle Code Review
Systematically review the changes from these perspectives (prioritize based on what the diff contains):

**1. Correctness & Logic**
- Off-by-one errors, wrong conditions, inverted logic
- Race conditions or missing synchronization (if relevant)
- Incorrect assumptions about data shape, nullability, types
- Wrong algorithm or formula

**2. Robustness & Error Handling**
- Missing try/except, missing validation of inputs/parameters
- Poor error messages or swallowed exceptions
- Missing handling of edge cases (empty lists, zero, negative, very large values, special characters)
- What happens on network failure, DB error, invalid user input?

**3. Consistency with Codebase**
- Does it follow the same patterns as similar code elsewhere? (naming, structure, error handling style, logging)
- Are new abstractions consistent with existing ones?
- Does it break any documented or implicit invariants?

**4. Maintainability & Readability**
- Overly complex functions or deeply nested logic
- Magic numbers/strings that should be constants
- Missing or misleading comments on non-obvious parts
- Duplication introduced (or missed opportunity to remove existing duplication)

**5. Performance & Resource Usage**
- Inefficient loops, N+1 queries, unnecessary work in hot paths
- Missing caching where it would clearly help
- Resource leaks (file handles, connections, memory in long-running processes)

**6. Security (basic but important)**
- Injection risks (SQL, command, template)
- Missing authorization checks on new endpoints/actions
- Hardcoded secrets or sensitive data in code/logs
- Unsafe deserialization or eval usage

**7. Testing Gaps**
- Are there new paths that clearly need tests?
- Existing tests that might now be broken or insufficient?
- Missing unit, integration or edge-case tests for the changed logic

**8. Documentation & Traceability**
- Did public APIs, config, or behavior change without doc updates?
- Are commit messages / task references clear enough for future readers?

**9. Missed Better Approaches**
- Is there a simpler, cleaner, or more idiomatic way to achieve the same goal?
- Did the implementation take a detour that could have been avoided with a better abstraction that already exists?
- Opportunities for small refactors that would have made this change cleaner

**10. Quantitative Code Quality Metrics**
When the changed files are Python (or adapt the approach), run the helper:
```bash
python scripts/compute_basic_metrics.py --git-changed
```
or on specific files from the diff.

Pay attention to signals such as:
- Very long functions (max_function_length significantly over 50-80 lines)
- High approximate complexity relative to LOC in the changed code
- Sudden increase in TODO/FIXME count introduced by this session
- Large modules with few functions (potential god-class smell)
- Files with both high recent churn and high complexity

Add a short "Metrics snapshot" subsection in the review report when the numbers are meaningful. This provides objective data alongside the qualitative findings.

### Phase 4: Produce the Review Report
Create a file named something like:
- `SESSION_REVIEW_YYYYMMDD_HHMM.md` (in root or `reviews/` folder)
- Or update/append to a running `CODE_REVIEWS.md`

**Recommended structure:**

```
# Code Review — Session <date/time or commit range>

**Reviewed changes:** <summary of what was changed>
**Context used:** AGENT_CONTEXT.md (date), PROJECT_STATUS.md, task X from RECOMMENDED_NEXT_TASKS.md
**Reviewer:** code-review-reflector skill

## Summary
One paragraph: Overall impression + most important findings.

## Critical Issues (must fix)
- Finding 1 with location, explanation, suggested fix
- ...

## Important Issues
- ...

## Improvements & Nice-to-Haves
- ...

## Questions & Clarifications
- Things that were unclear from the code/intent

## Positive Observations
- What was done well (important for learning)

## Recommended Next Actions
- Specific fixes to do now
- Suggested new tasks for the task planner skill (e.g. "Add tests for X edge case", "Refactor Y for consistency")
- Whether to re-run github-project-status after fixes
```

Use the template in `references/SESSION_REVIEW_TEMPLATE.md` as starting point.

### Phase 5: Present & Discuss
- Show the user the review report
- Highlight the top 2–3 most important findings
- Ask if they want to:
  - Fix critical/important issues together right now
  - Turn some findings into new tasks in `RECOMMENDED_NEXT_TASKS.md`
  - Adjust anything in the review
  - Run the review again after fixes

Be encouraging but honest — the goal is learning and quality, not criticism.

## Special Considerations
- **Large diffs**: Focus on the riskiest or most complex parts. Summarize "I reviewed the core logic in detail and spot-checked the rest".
- **Agent-written code**: Be extra thorough on consistency and edge cases — agents sometimes take shortcuts.
- **No obvious issues**: Still produce a short positive review + any small improvements or questions. Never force findings.
- **Language & tone**: Professional, specific, and kind. Explain the "why" so the developer learns.
- **Integration with other skills**:
  - After fixes, strongly recommend running `github-project-status` again.
  - Findings that represent new work can be turned into tasks for `github-repo-deep-dive`.

## Resources
- `references/SESSION_REVIEW_TEMPLATE.md` — Recommended structure for the review report
- `references/CODE_REVIEW_CHECKLIST.md` — Detailed checklist the agent can consult mentally or on screen during the multi-angle review in Phase 3
- `references/CODE_QUALITY_REPORT_TEMPLATE.md` — Template for a full-repo code quality report (run periodically on the whole project)
- `scripts/compute_basic_metrics.py` — Helper that computes LOC, function count, rough complexity, TODO count etc. (supports Python, JS/TS, Go)
- `scripts/hotspots.py` — Advanced script to find files with high churn + high complexity (great for identifying risky areas)
- The skill also benefits from fresh `AGENT_CONTEXT.md` and `PROJECT_STATUS.md` from the other skills for consistency checks

## Generating a Full-Repo Code Quality Report
To create a broad `CODE_QUALITY_REPORT.md` for the entire project (not just one session):
1. Run `python scripts/hotspots.py --since "3 months ago" --top 20`
2. Optionally run `python scripts/compute_basic_metrics.py` on core directories
3. Use `references/CODE_QUALITY_REPORT_TEMPLATE.md` as base
4. Focus on hotspot files, TODO density, and long-term maintainability signals
5. Store the report in root or `docs/` and update it every 4–6 weeks

This gives a strategic view of code health over time.

This skill turns "it works on my machine" into "I have thought through the important angles and here is what I found".
