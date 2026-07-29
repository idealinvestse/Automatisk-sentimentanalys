# DATA-01 — Corpus-specifikation för pilotgates

**Skapad:** 2026-07-17  
**Status:** Spec klar; *data* levereras externt (ej i git)  
**Relaterat:** [DEVELOPMENT.md § DATA-01](DEVELOPMENT.md), `scripts/import_domain_corpus.py`

---

## 1. Mål

Ersätt syntetiska baselines med anonymiserad svensk callcenter-data så att:

- sentiment/intent-gates blir meningsfulla
- WER/telefoni-eval kan köras (separat audio-slice)
- kundkommunikation om kvalitet blir försvarbar

---

## 2. Minsta storlek (pilot-gate)

| Dataset | Format | Minimum för `--pilot-gate` | Rekommenderat |
|---------|--------|----------------------------|---------------|
| Sentiment val | CSV `text,label` | **500** rader | 1 000+ balanserat |
| Intent val | JSONL `{"text","intent"}` | **200** rader | 500+ med ≥5 per intent |
| ASR/WER (valfritt men krävs för WER-claim) | Audio + referenstranskript | **100** samtal | **500** manuellt granskade |

Labels sentiment: `negativ` | `neutral` | `positiv` (samma som `data/callcenter_val.csv`).

Intent-labels: följ befintliga nycklar i `data/intent_val.jsonl` / callcenter-profilens intent-lista.

---

## 3. Kvalitetskrav

1. **Anonymiserad** — personnummer, telefon, e-post, namn, kontonummer borttagna eller maskerade.
2. **Manuell stickprovsreview** — automatisk PII-scan i import-skriptet är *referens*, inte juridiskt godkännande.
3. **Telefoni-realism** — för WER: 8 kHz/VoIP om det är er produktion; inkludera agent- och kundsida om möjligt.
4. **Ingen raw audio i git** — endast `data/import/*_real.*` (gitignored) eller lagring utanför repo.
5. **Laglig grund** — dokumentera i DPIA (samtycke / berättigat intresse / avtal); se decision pack.

---

## 4. Filnamn som import-skriptet känner igen

I `--source-dir` (utanför git):

**Sentiment (första träff vinner):**

- `sentiment.csv`
- `callcenter_val.csv`
- `callcenter_sentiment.csv`

**Intent:**

- `intent.jsonl`
- `intent_val.jsonl`
- `callcenter_intent.jsonl`

Efter import:

- `data/import/callcenter_val_real.csv`
- `data/import/intent_val_real.jsonl`

---

## 5. Workflow (kort)

```bash
# 1) Rådata på krypterad volym → anonymisera → lägg filer i /secure/anonymized

# 2) Import med pilot-trösklar
python scripts/import_domain_corpus.py \
  --source-dir /secure/anonymized \
  --pilot-gate

# 3) Eval mot riktig korpus
python scripts/evaluate_real_corpus.py \
  --sentiment-csv data/import/callcenter_val_real.csv \
  --intent-jsonl data/import/intent_val_real.jsonl

# 4) Jämför med reports/domain_baseline.json (synthetic) — uppdatera endast medvetet
```

Utvecklingssmok utan riktig data finns kvar i [DEVELOPMENT.md](DEVELOPMENT.md) (synthetic `data/callcenter_val.csv`).

---

## 6. Definition of done (DATA-01 för conditional go)

- [ ] ≥500 sentiment-rader importerade utan PII-scan-fail (eller dokumenterad manuell override)
- [ ] ≥200 intent-rader importerade och validerade
- [ ] `evaluate_real_corpus.py` körd; resultat arkiverat under `reports/` (lokalt, ej nödvändigtvis commit:at om känsligt)
- [ ] Intern beslut: får kvalitetsclaim göras till pilotkund? (ja/nej)

**Utanför denna spec (kräver er leverans):** faktiska ljudfiler och annotatörsbudget.
