---
name: github-repo-deep-dive
description: Use this skill to deeply explore and understand a GitHub repository, ask clarifying counter-questions about goals and constraints, and identify the most suitable next development or coding tasks. Complements github-project-status by turning current state into actionable prioritized tasks. Trigger with deep dive repo, explore github project for next tasks, find suitable development tasks, ställ motfrågor och föreslå nästa koduppgifter, repo deep analysis for task planning, understand codebase and recommend next steps
---

# GitHub Repo Deep Dive & Task Recommender

This skill performs a deep, intelligent exploration of a GitHub project to truly understand its current state, architecture, strengths, weaknesses and opportunities. It then engages in a short dialogue (asking sharp counter-questions) before proposing the **most suitable next development tasks** — the ones that are closest to the current reality, highest value, and best aligned with user goals.

It is the natural next step after running `github-project-status`. Use it when you want not just documentation, but **actionable direction** for the next coding session(s).

## When to Use
- After a `github-project-status` run, when you want to decide "what should we build or improve next?"
- When starting a new development phase or planning a sprint
- When the project feels "stuck" or you have many possible directions and need focus
- When handing over to another agent and want it to start on high-value work immediately
- To turn raw codebase understanding into a prioritized, realistic task list

## Core Principles
- **Deep understanding first** — never suggest tasks from superficial knowledge.
- **Ask before proposing** — use counter-questions to clarify priorities, constraints, risk tolerance and desired direction. Do not assume.
- **"Närmaste lämpliga" focus** — prioritize tasks that are realistic given current code state, require minimal context switching, build directly on existing work, and deliver visible progress.
- **Synergy with github-project-status** — strongly prefer that fresh `PROJECT_STATUS.md` and `AGENT_CONTEXT.md` exist before deep analysis.
- **Output is actionable for both humans and agents** — every proposed task must be specific enough that an agent can start implementing it with the existing context files.
- **Transparency** — always explain *why* a task is recommended now (impact, effort, dependencies, strategic fit).

## Workflow — Execute in Strict Order

### Phase 1: Preparation & Fresh Context (Mandatory)
1. Check if the repo root contains recent `PROJECT_STATUS.md` and `AGENT_CONTEXT.md` (look at "Last Updated" date).
2. If missing or older than ~1-2 days (or if major changes happened since last run), **instruct the user to first run the `github-project-status` skill**, or offer to do it as a first step.
3. Once fresh status/context files exist, read them fully (or key sections) using `read_file`.
4. Locate the repo root (`git rev-parse --show-toplevel`).

This gives you a strong baseline so you do not duplicate effort.

### Phase 2: Deep Codebase Exploration
Go beyond the status files. Perform targeted deep dives:

- **Recent focus areas**: `git log --oneline --since="4 weeks ago" -- . | head -30` + look at changed files in recent commits.
- **Hotspots & pain points**:
  - Files with high churn (many recent commits)
  - Large or complex files (potential refactoring candidates)
  - Areas with many TODO/FIXME comments (re-run the grep from status skill if needed)
  - Obvious code smells: duplication, long functions, missing abstraction, tight coupling (spot by reading 3-5 key modules)
- **Architecture & data flow deep dive**:
  - Read the most central files (core services, routers, models, main agent loops, config loaders, API handlers).
  - Understand the "happy path" and error paths for the main user journeys.
  - Map how data moves between components (e.g. user request → LLM router → external API → DB → response).
- **Unused or under-used code**: Look for dead code, commented-out sections, or features that exist in docs but have little implementation.
- **Extension points & modularity**: Identify where new features are *easiest* to add cleanly (good interfaces, plugin patterns, clear separation).
- **Technical debt that blocks progress**: Things that make every new feature harder (e.g. missing types, poor error handling, no tests in critical path, hardcoded values).

Use multiple focused `read_file` calls. Prioritize quality over quantity. Build a rich mental model of:
- What the system *really* does today
- Where it is strong / fragile
- What would be "cheap" vs "expensive" to change or extend

### Phase 3: Initial Understanding Summary (Internal)
Before talking to the user, silently synthesize:
- One-paragraph "Current deep understanding of the project"
- Top 5-7 observed strengths
- Top 5-7 observed limitations / opportunities / pain points
- Rough categories of possible work: new features, refactoring, performance, reliability, developer experience, new integrations, etc.

Do **not** output this yet.

### Phase 4: Ask Clarifying Counter-Questions (Critical Interactive Step)
Now engage the user with 4–7 sharp, high-signal questions. The goal is to understand **intent and constraints** so you can filter and rank tasks intelligently.

Good questions (adapt to what you learned in Phase 2 & 3):

- What is the single most important outcome or user value you want to achieve in the next 2–4 weeks?
- Which part of the current system feels most limiting or painful for you (or your users) right now?
- Are there any hard constraints I should know about? (tech choices you want to keep, deployment limits, time budget, compliance, etc.)
- Do you prefer right now: quick visible wins that deliver value soon, foundational improvements that make future work much easier, or bigger new features?
- Have you received any recent feedback, bug reports, or ideas from users or yourself that point to specific directions?
- Looking 3–6 months ahead, what direction does the project need to move in? (e.g. more automation, better UX, new markets, reliability, cost reduction)
- Any areas you *definitely do not* want to touch right now, or things we should avoid?

Ask these (or better ones based on your analysis) one or a few at a time, in natural conversation. Listen carefully to the answers. Ask follow-ups if something is unclear or contradictory.

Only proceed to task proposal when you have enough signal.

### Phase 5: Synthesize & Rank Next Tasks
Using everything you now know (deep code understanding + user answers), generate a short, high-quality list of recommended next tasks (typically 5–8 items).

For **each task** define:
- **ID** (e.g. TASK-01)
- **Title** (clear, action-oriented)
- **Why this task now** (1-2 sentences linking to current state + user goals)
- **Description** (what needs to be done, at a level an agent can start from)
- **Primary files / components** likely to be touched
- **Estimated effort** (Small / Medium / Large — with rough reasoning)
- **Dependencies / prerequisites** (e.g. "requires fresh AGENT_CONTEXT.md", "depends on TASK-03 being done first")
- **Expected impact / value**
- **Risks / things to watch**
- **Success criteria** (how do we know it's done well?)

**Ranking logic** (be explicit about it):
1. Highest value relative to effort ("bang for the buck")
2. Builds directly on recent work or existing strong foundations
3. Unblocks other valuable work
4. Reduces technical debt that slows everything down
5. Aligns with expressed user priorities from the Q&A
6. Realistic given current codebase maturity

Separate into categories if useful:
- Quick wins (can be done in one session)
- Strategic / foundational
- New capability / feature

### Phase 6: Produce Output Artifacts
Create or update these files in the repo root (or `docs/`):

1. **`RECOMMENDED_NEXT_TASKS.md`** (primary output — create fresh each time)
   - Header with date + repo + "Generated by github-repo-deep-dive skill after clarifying questions"
   - Short "Understanding & Rationale" section (what you learned + why these tasks)
   - The ranked task list with all fields above
   - "How to use this file": An agent (or you) should pick the top task, read the relevant parts of `AGENT_CONTEXT.md` + `PROJECT_STATUS.md`, then start implementing. After finishing, re-run `github-project-status` and optionally this skill again.

2. **Update `AGENT_CONTEXT.md`** (append or replace a section)
   - Add or refresh a section: `## Recommended Next Development Tasks (as of <date>)`
   - List the top 3–5 tasks with short titles + one-sentence rationale. This way any future agent loading the context file immediately sees the current direction.

3. **Optional but recommended**:
   - Lightly update `PROJECT_STATUS.md` "Recommendations for Next Development Sessions" section with the top priorities.
   - If a `BACKLOG.md` or similar exists, sync the new tasks into it.

### Phase 7: Present to User + Next Steps
- Show the user the `RECOMMENDED_NEXT_TASKS.md` (or key parts)
- Summarize: "Based on deep analysis + your answers, here are the most suitable next tasks..."
- Highlight 1–2 top recommendations with strong "why now"
- Ask: "Which task (or combination) would you like to tackle next? Or do you want to adjust any of the questions/priorities and re-run the analysis?"
- Offer to:
  - Start implementing the chosen task together (using the full context)
  - Refine the task list
  - Run the status skill again after implementation
  - Create more detailed specs for a chosen task

## Special Considerations
- **If user answers are vague**: Ask more targeted follow-up questions instead of guessing.
- **Conflicting signals**: Surface the tension and ask the user to prioritize (e.g. "You mentioned both quick wins and long-term architecture — which should take precedence right now?")
- **Large / complex repos**: Focus deep dive on the core domain logic and the areas most relevant to the user's expressed goals. Note in the output which parts were analyzed in depth.
- **No clear user answers**: Fall back to suggesting tasks that improve the *foundation* (making future work easier) and clearly state the assumption.
- **Agent handoff**: The combination of fresh `PROJECT_STATUS.md` + `AGENT_CONTEXT.md` + `RECOMMENDED_NEXT_TASKS.md` should allow another agent to pick up and start high-value work with minimal ramp-up.

## Relationship to Other Skills
- **github-project-status** — Run this first (or let this skill trigger it). This skill consumes its output and goes deeper into task recommendation.
- Future skills (e.g. implementation helpers, test generators, doc updaters) can be chained after a task is chosen.

## Success Criteria for This Skill
After running it, the user (and any agent) should be able to answer:
- "What is the single best thing to work on next in this project, and why?"
- "What context do I need to start that task successfully?"
- "What clarifying questions were asked and how did the answers shape the recommendation?"

This turns passive repo exploration into **active, aligned, high-leverage development direction**.

Run this skill whenever you want clarity and momentum instead of analysis paralysis.