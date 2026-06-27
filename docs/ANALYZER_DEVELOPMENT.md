# ANALYZER DEVELOPMENT GUIDE (Uppdaterad 2026-06-27 efter A0)

## Så här lägger du till en ny analyzer på 60 sekunder (super-enkelt!)

1. Kopiera `src/analysis/templates/new_analyzer_template.py` → `src/analysis/din_nya.py`
2. Ändra namnet och `@register_analyzer("din_nya")`
3. Implementera `async def analyze(self, ctx)`
4. Lägg till i `selected_analyzers` eller låt registry hantera dependencies.
5. Klart! Den körs automatiskt i rätt ordning.

Exempel på avancerade analyzers just tillagda: `advanced_coaching` + `root_cause_insight`.

Vi har nu **extremt låg friktion** för nya värdeskapande insikter.