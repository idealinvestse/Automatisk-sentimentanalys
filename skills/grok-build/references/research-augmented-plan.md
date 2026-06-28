# Research-Augmented Plan

When research tier is MEDIUM or HIGH, include a **Research Findings** section in the plan:

```markdown
## Research Findings ({{TIER}} tier)

- Key facts with sources
- Pricing / API compatibility notes (if relevant)
- Gaps and unknowns
```

Rules:
- INTERNAL tier: workspace context only — no web research section required.
- MEDIUM tier: 3–5 bullet findings from Serper quick search.
- HIGH tier: deeper Exa research with citations and trade-off analysis.
- Cache results in `memory/research-cache/<hash>.md` via research-dispatcher.