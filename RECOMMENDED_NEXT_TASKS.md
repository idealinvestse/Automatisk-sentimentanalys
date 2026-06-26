
---

### TASK-07: Gör circuit breaker state mer robust vid multi-worker deployment
**Why this task now**: Den nuvarande implementationen i `src/api/routers/alerting.py` använder en modulnivå `_status_engine`. Detta fungerar bra för single-worker development men blir problematiskt vid flera uvicorn workers, reloads eller horisontell skalning.

**Description**:
- Utvärdera alternativ: Redis-baserad state, `app.state`-baserad lösning, eller en dedikerad liten service.
- Välj en enkel men robust lösning för produktionsanvändning.
- Uppdatera `get_alerting_status` och `reset_circuit_breaker` endpoints att använda den nya lösningen.
- Dokumentera begränsningar och rekommendationer i `docs/`.

**Primary files / components**:
- `src/api/routers/alerting.py`
- `src/alerting.py`
- `src/api/app.py` (lifespan / app.state)
- Eventuellt Redis-beroende eller enklare in-memory store

**Estimated effort**: Medium.

**Dependencies / prerequisites**: TASK-06 (alerting status).

**Expected impact / value**: Hög för production readiness och reliability.

**Risks / things to watch**: Undvik att göra alerting state för komplext. Börja med en enkel men tydlig förbättring.

**Success criteria**:
- Circuit breaker state är konsekvent även vid flera workers.
- Reset-funktionen fungerar som förväntat.
- Bra dokumentation av begränsningar.