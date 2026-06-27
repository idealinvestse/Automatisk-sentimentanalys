---
name: github-project-status
description: Use this skill after programming sessions to scan a GitHub project repository, create current status, update all documentation and support documents, maintain system descriptions and feature lists, and generate complete context for agents to continue development from. Trigger with update github project status, scan repo and refresh docs, generate full project context, maintain feature lists, create current status of github repo, uppdatera github projektstatus, skanna repo och uppdatera dokumentation, generera agentkontext för github projekt
---

# GitHub Project Status

This skill keeps a GitHub project's documentation, feature lists, system descriptions and agent-ready context perfectly synchronized with the live codebase. Run it after every significant development session to produce a complete, trustworthy snapshot that new agents (or you) can use immediately to continue work without missing context.

## When to Activate
- Immediately after finishing a coding, refactoring or debugging session
- Before switching context or handing development to another agent/session
- When planning next steps and needing an objective current-state overview
- To refresh stale README, FEATURES or architecture docs
- Any time you want a single source of truth AGENT_CONTEXT.md for LLM agents working on the project

## Core Principles
- Work primarily on a **local git clone** of the repo (preferred). The skill auto-detects repo root.
- If only a GitHub URL is given, perform a shallow clone to a temp location, analyze, and output ready-to-merge docs.
- Always produce **actionable, up-to-date** documents — never generic or outdated.
- Prioritize accuracy over perfection: cross-reference claims in docs against actual code.
- Focus analysis on what matters for continued development: architecture, features, invariants, open tasks, recent changes.
- The crown jewel is **AGENT_CONTEXT.md** — a concise yet complete briefing optimized for LLM context windows.

## Workflow — Execute Strictly in Order

### 1. Locate and Prepare Repository
Use `bash` tool with these commands:
- `git rev-parse --show-toplevel 2>/dev/null || echo "NOT_A_GIT_REPO"`
- If inside a git repo: `REPO_ROOT=$(git rev-parse --show-toplevel)` ; `cd "$REPO_ROOT"`
- Else if user gave GitHub URL (https://github.com/owner/repo or owner/repo):
  - Sanitize name, clone shallow: `git clone --depth 1 <url> /tmp/github-project-status-<name>`
  - Set REPO_ROOT to the clone dir
- Always: `git fetch origin 2>/dev/null || true` and note current branch + HEAD
- Record: remote origin, current branch, HEAD sha, last commit timestamp and message
- Check for uncommitted changes: `git status --porcelain` and `git diff --stat`

If working tree has changes from the just-finished session, treat them as "Work in Progress — Current Session" and include prominently.

### 2. Map Project Structure and Type
Run structure commands (prefer `tree` if available, else `find`):
- `tree -L 3 --dirsfirst -I '.git|node_modules|__pycache__|venv|env|dist|build|target' . 2>/dev/null || find . -maxdepth 3 -type f ! -path '*/.git/*' | sort | head -80`
- Identify type by scanning for manifest files (in priority order):
  - Node/TS: package.json, tsconfig.json, next.config.*, vite.config.*
  - Python: pyproject.toml, requirements*.txt, setup.py, poetry.lock, src/ layout
  - Rust: Cargo.toml, src/main.rs or lib.rs
  - Go: go.mod, cmd/ or internal/
  - Other: Dockerfile, docker-compose.yml, Makefile, .github/workflows/*.yml, composer.json, pom.xml
- Note key directories: src/, app/, lib/, core/, api/, frontend/, backend/, docs/, tests/, scripts/
- List top-level config and entrypoint files

### 3. Read Existing Documentation (Baseline)
Use `read_file` (with limits where files are long) on:
- README.md (or README.rst, README.txt) — capture claimed purpose, features, install/run instructions, badges
- Any existing PROJECT_STATUS.md, STATUS.md, FEATURES.md, ARCHITECTURE.md, DESIGN.md, SYSTEM_DESIGN.md, TODO.md, CHANGELOG.md, AGENT_CONTEXT.md (or similar names in root or docs/)
- docs/ folder contents if present: `ls -1 docs/ 2>/dev/null || true`
- Important config: .env.example, docker-compose.yml, Makefile, pyproject.toml / package.json (read key sections: scripts, dependencies, description)

Note any obvious staleness (e.g. "README says feature X but code has no trace of it").

**Optional — Code Quality Metrics**
If desired, run the metrics helpers from `code-review-reflector`:
```bash
python ../code-review-reflector/scripts/compute_basic_metrics.py --git-changed
python ../code-review-reflector/scripts/hotspots.py --since "3 months ago"
```
Include a short "Code Quality Snapshot" in PROJECT_STATUS.md with key signals (hotspot files, TODO density, complexity warnings). This is very useful for long-running projects.

### 4. Perform Targeted Codebase Analysis
Do not read every file. Prioritize ruthlessly:
- Read the 5–12 most important files for understanding the system (entrypoints, core classes/modules, main routers/handlers, config loaders, key models/schemas).
- For each: read first 80–150 lines + any obvious docstrings or header comments. Use multiple targeted `read_file` calls with offset/limit if needed.
- Extract:
  - High-level purpose of each major component
  - Tech stack signals (imports, framework usage, DB clients, external API calls)
  - How components interact (e.g. "FastAPI backend → PostgreSQL via SQLAlchemy → Redis cache")
  - Any explicit architecture notes or invariants in comments
- Run broad TODO/FIXME scan (language-aware):
  `grep -r --include="*.py" --include="*.js" --include="*.ts" --include="*.go" --include="*.rs" --include="*.java" --include="*.php" "TODO\|FIXME\|XXX\|HACK\|OPTIMIZE" . 2>/dev/null | head -40`
- Also search for feature-related comments or constants if obvious patterns exist.
- Check test coverage signals and CI setup (look in .github/workflows for test jobs).
- If the project has a clear "main" service or agent (e.g. VPS control hub, LLM router), drill one level deeper into its core logic.

Result of this step: mental model of the full system architecture and current implementation state.

### 5. Synthesize System Description
Write 1–3 tight paragraphs that answer:
- What does this system actually do for its users?
- What problem does it solve and how (high-level architecture + key technologies)?
- Who is the target user / deployment context (e.g. "self-hosted VPS agent for WooCommerce automation used by small e-comm operators in Sweden/Norway")?

This becomes the canonical "System Description" section used everywhere.

### 6. Build and Maintain Feature Lists
Create three categories (always from actual code + previous docs, reconciled):
- **Implemented / Live in Production** — features that are coded, tested or obviously working. Include brief description + key file(s) reference.
- **In Progress / Partial** — features with code started, config flags, or marked as WIP in current session changes.
- **Planned / Backlog** — items mentioned in TODO comments, old docs, or logical next steps inferred from architecture.

If `FEATURES.md` exists in root or docs/, **edit it** to reflect the new reconciled list (use `edit_file` or `write_file` + careful replace). Add or update a "Last synced" header with date.

If it does not exist, create a clean, well-structured `FEATURES.md` in the project root.

### 7. Create or Update the Central Documents (Most Important Step)
Always produce these two files in the repo root (or docs/ if the project already uses that convention). Use `write_file` for new files or when full rewrite is cleaner; prefer `edit_file` for surgical updates on existing ones.

**A. PROJECT_STATUS.md** (create if missing)
Structure exactly like this (adapt content, keep headings):

```
# Project Status — <Project Name>

**Last Updated:** <ISO date> via github-project-status skill  
**Repository:** <origin url>  
**Current Branch:** <branch> @ <short sha>  
**Working Tree:** clean / has uncommitted changes from current session

## Recent Activity
- Last 5–8 commits with dates and short messages (highlight session work)
- Summary of uncommitted changes

## System Description
[the 1–3 paragraph synthesis from step 5]

## Architecture Overview
- High-level components and responsibilities (bullet or Mermaid if simple)
- Tech stack (languages, frameworks, infra, external services)
- Key data flows and integration points
- Deployment model (VPS, Docker, serverless, etc.)

## Feature Status
### Implemented
- ...

### In Progress
- ...

### Planned
- ...

## Known Issues & Technical Debt
- Extracted TODOs/FIXMEs grouped by area + file refs
- Any obvious gaps or inconsistencies found during analysis

## Recommendations for Next Development Sessions
- Concrete priorities based on current state
- Notes for agents: "When implementing X, see Y in Z.py because..."
- Suggested refactors or improvements that would unlock future features
```

**B. AGENT_CONTEXT.md** (this is the primary deliverable for continued agent development — make it excellent)
Optimized for being pasted into LLM prompts. Keep total reasonable length but complete.

Recommended structure:

```
# AGENT CONTEXT — <Project Name>
**Generated:** <date> | Use this file as the single source of truth when continuing development.

## 1. What This System Is
[system description paragraphs]

## 2. Current Feature Inventory
[the three-category list from step 6, concise]

## 3. Architecture & Key Components
[Detailed but compact breakdown of modules, responsibilities, how they talk to each other, important design decisions and invariants]

## 4. Important File Map
- `path/to/keyfile.py` — purpose + what an agent must know before editing
- `README.md`, `FEATURES.md`, `PROJECT_STATUS.md` — always read first on new session
- Config, entrypoints, core business logic files

## 5. How to Work With This Codebase
- Run / test / build / deploy commands (from Makefile, package.json scripts, docs)
- Coding conventions observed (naming, error handling, logging, testing style)
- Where to add new features of different types

## 6. Open Tasks & Priorities
- High-priority items from TODO scan + recommendations
- Any "do not break" rules or invariants

## 7. Context for Future Agents
[Any project-specific guidance: "After every change that affects features, re-run the github-project-status skill", "We use Swedish/Norwegian localization in X", specific gotchas discovered]
```

Update any existing AGENT_CONTEXT.md by replacing its content with the fresh version (or edit sections).

### 8. Update Supporting Project Documentation
- **README.md**: Add or refresh a short "Project Status & Documentation" section near the top (after badges/description) that says:
  > For the absolute latest system description, feature status and complete agent context, see [PROJECT_STATUS.md](PROJECT_STATUS.md) and [AGENT_CONTEXT.md](AGENT_CONTEXT.md). These files are kept in sync with the codebase by the github-project-status skill. Last updated: <date>.

  Also verify that install/run instructions still match reality; if not, note it in the status file.

- For any other existing docs in docs/ or root (e.g. old design docs): do a light review. If they are clearly stale relative to code, add a note at the top of PROJECT_STATUS.md under "Documentation Health". Do **not** rewrite every file unless the discrepancy is critical — the two central files (PROJECT_STATUS + AGENT_CONTEXT) are the priority.

- If a TODO.md exists, update it with the fresh extracted list (or create one in root if many TODOs were found). Group by component/area.

### 9. Final Output to User
After all file operations:
- Summarize what was done:
  - Repo analyzed (name, last activity, size indicators)
  - Documents created/updated (list full paths)
  - Key findings (new features implemented since last known state, major open items, any documentation/code mismatches discovered)
  - Uncommitted changes highlighted
- If the analysis was done in a temp clone: give clear instructions "Review the generated PROJECT_STATUS.md and AGENT_CONTEXT.md in /tmp/... then copy them into your local clone root, commit and push."
- If edits were made directly in the user's local repo: remind them to `git status`, review diffs, then commit with a message like `docs: refresh project status, features and agent context after <session description>`
- Offer to iterate: "Want me to refine any section, add more detail on component X, or run tests/CI checks if configured?"

## Special Handling
- **Large / Monorepo projects**: Limit depth of tree and file reads. Focus on root manifests + the 2–3 most important sub-packages or services. Note "monorepo — analyzed core service X" in status.
- **No prior documentation**: The skill creates everything from scratch based on code — this is normal and valuable.
- **After session with many changes**: Emphasize the "Work in Progress" section and make sure AGENT_CONTEXT explains what was just built so the next agent can pick up exactly where left off.
- **Language**: Analysis works for any language. Adapt grep includes accordingly. Project description should reflect actual user language if obvious (e.g. Swedish/Norwegian strings in UI).
- **Safety**: Never commit or push automatically. Only propose or perform local file edits. User always reviews before git commit/push to GitHub.

## Output Quality Goals
Every run must leave the project in a state where:
- A brand new agent can be given AGENT_CONTEXT.md + the repo URL and immediately understand what to build next without asking clarifying questions about current state.
- All feature claims are truthful and traceable to code.
- Documentation debt does not accumulate.

Run this skill often — it is the "save game" mechanism for long-running agent-driven development on GitHub projects.
