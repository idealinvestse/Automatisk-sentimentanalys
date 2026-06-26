
---

### TASK-05: Lägg till enhetstester för `llm_judge_panel` (särskilt filter-logik och tomma tillstånd)
**Why this task now**: Efter code review identifierades att filter-logiken ("Endast ändrade") och hantering av tomma tillstånd behöver testtäckning för att undvika regressioner när panelen utvecklas vidare.

**Description**:
- Skriv tester för `render_llm_judge_panel` och `render_llm_judge_summary`.
- Täck fall: normal data, filter "Endast ändrade", tom data, saknade fält (`segment_index`, `reasoning` etc.), edge cases för `_is_changed()`.
- Använd NiceGUI test-mönster eller enkla enhetstester på hjälpfunktionerna.

**Primary files / components**:
- `tests/test_llm_judge_panel.py` (ny fil)
- `app/nicegui_dashboard/components/llm_judge_panel.py`

**Estimated effort**: Small (1 session).

**Dependencies / prerequisites**: TASK-02 (panelen finns).

**Expected impact / value**: Medel. Ökar förtroendet för komponenten och gör framtida ändringar säkrare.

**Risks / things to watch**: NiceGUI-komponenter kan vara svåra att enhetstesta; fokusera på rena hjälpfunktioner (`_get_verdicts`, `_is_changed`).

**Success criteria**:
- Bra testtäckning på filter-logik och edge cases.
- Tester körs gröna i CI.

---

### TASK-06: Exponera webhook/circuit breaker status från `AlertEngine` till dashboard
**Why this task now**: Code review visade att statusraden i `alerts_panel.py` är statisk/hårdkodad. För att göra alerting verkligen produktionsmogen behöver vi visa verkligt tillstånd (CLOSED/OPEN, antal failures).

**Description**:
- Exponera `AlertEngine` circuit breaker state (t.ex. via en singleton, dashboard state eller enkel API-endpoint `/alerting/status`).
- Uppdatera `alerts_panel.py` att visa dynamisk status (färgkodad: grön = CLOSED, röd = OPEN).
- Eventuellt lägg till reset-knapp för circuit breaker i test_lab.

**Primary files / components**:
- `src/alerting.py`
- `app/nicegui_dashboard/components/alerts_panel.py`
- `app/nicegui_dashboard/services/nicegui_api_client.py` (om ny endpoint)
- `src/api/routers/` (valfritt)

**Estimated effort**: Medium.

**Dependencies / prerequisites**: TASK-04 (alerting polish).

**Expected impact / value**: Hög för reliability. QA-teamet ser direkt om webhooken är nere.

**Risks / things to watch**: Undvik att göra `AlertEngine` till en global singleton om det inte redan är så. Bättre att exponera via en service eller dashboard state.

**Success criteria**:
- Dashboard visar korrekt circuit breaker status i realtid.
- Status uppdateras när breaker öppnas/stängs.

---

## How to use this file
En agent (eller du) bör:
1. Välja en task (börja med TASK-01 eller TASK-02 beroende på vad som känns mest prioriterat).
2. Läsa relevanta delar av `AGENT_CONTEXT.md` + `PROJECT_STATUS.md`.
3. Implementera.
4. Efter avslut: Kör `github-project-status` igen (och eventuellt denna deep-dive skill).

**Rekommenderad ordning just nu**: TASK-01 → TASK-02 → TASK-03 (foundational + direkt nytta för dashboard).