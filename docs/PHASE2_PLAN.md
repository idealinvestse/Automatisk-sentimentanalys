# FAS 2 – SVENSK DOMÄNANPASSNING & PRESTANDAOPTIMERING

**Projekt:** Automatisk Sentimentanalys för Call Center
**Fas:** 2 av 5
**Tidsram:** Juni – Augusti 2026 (ca 8–10 veckor)
**Prioritet:** Hög (P0)

---

## STATUS (24 Maj 2026)

Grundstrukturen är implementerad och pushad till `main`. Se detaljerad status under respektive steg.

### Redan klart ✅

| Komponent | Status | Commit |
|-----------|--------|--------|
| Fas 1-brister fixade + baseline-rapport | ✅ | `43785b2` |
| `src/finetune.py` LoRA/PEFT pipeline | ✅ | `fdea2f5` |
| `configs/finetune.yaml` | ✅ | `fdea2f5` |
| `data/callcenter_train.csv` (seed, 10 ex) | ✅ | `fdea2f5` |
| `data/sensaldo_lexicon.csv` (seed, 20 termer) | ✅ | `fdea2f5` |
| `notebooks/finetune_callcenter.ipynb` | ✅ | `fdea2f5` |
| Negation detection i `src/sentiment.py` | ✅ | `fdea2f5` |
| `callcenter`-profil i `src/profiles.py` | ✅ | `fdea2f5` |
| Negationshantering i `src/lexicon.py` | ✅ | `fdea2f5` |
| `reports/baseline_results.json` | ✅ | `fdea2f5` |
| `reports/fas2_comparison.json` + `.md` | ✅ | `fdea2f5` |
| `docs/FAS2_SUMMARY.md` | ✅ | `fdea2f5` |
| `README.md` uppdaterad | ✅ | `fdea2f5` |
| `ROADMAP.md` uppdaterad | ✅ | `fdea2f5` |

### Återstår ⬜

| Komponent | Prioritet |
|-----------|-----------|
| Utöka träningsdataset till 3 000–5 000 exempel | P0 |
| `scripts/prepare_callcenter_data.py` | P0 |
| Kör GPU-träning (`python -m src.finetune`) | P0 |
| `src/blending.py` med learned blending | P1 |
| Full SenSALDO-lexikon (ladda ner + bearbeta) | P1 |
| Separat `src/negation.py` modul | P2 |
| Avancerad utvärdering per negationstyp | P2 |
| Uppdaterad Docker-image | P2 |
| v0.3.0 release notes | P2 |

---

## 1. Mål & Vision

**Huvudmål:**
- Höja modellprestandan på **svensk call center-data** med minst **+15–20 % macro-F1** jämfört med Fas 1 baseline.
- Göra systemet **domänanpassat** för kundtjänstsamtal (negationer, artighet, domäntermer, ironi).
- Skapa en **hållbar pipeline** för framtida finetuning och lexikonuppdateringar.

**Specifika mål:**
- Implementera och utvärdera en finetunad svensk modell.
- Integrera ett **större svenskt lexikon** (SenSALDO + egen utökning).
- Förbättra blending till en **lärd viktning**.
- Lägga till **negation detection** och **call center-specifik textbearbetning**.
- Leverera en komplett benchmark-rapport.

---

## 2. Bakgrund & Nuläge (efter Fas 1)

**Vad finns på plats:**
- `src/evaluate.py` + testset
- `KBLab/kb-whisper-large` (med `strict`/`subtitle`-varianter)
- `tests/` + `pyproject.toml` + CI
- `PHASE1_PLAN.md` med uppdaterad checklista

**Kända begränsningar:**
- Basmodellen (`cardiffnlp/twitter-xlm-roberta-base-sentiment`) är inte optimal för svensk call center-språk.
- Lexikon-blending är heuristisk.
- Ingen explicit negation-hantering (fixat i Fas 2-skelettet).
- Ingen domänanpassning.

---

## 3. Detaljerad Arbetsplan (Steg-för-steg)

### Steg 1: Verifiering & Städning av Fas 1 (1–2 dagar) ✅

- ✅ Kör `python -m src.evaluate` och spara resultat i `reports/baseline_results.json`.
- ✅ Verifiera att `KBLab/kb-whisper-large` + `revision="strict"` fungerar i både CLI och API.
- ✅ Uppdatera `README.md` med exakta kommandon för den nya modellen.
- ✅ Fixa eventuella brister i `evaluate.py` (t.ex. stöd för modelljämförelse).
- ✅ **Commit:** `fix: Kompletteringar och rättelser efter Fas 1`

### Steg 2: Fine-tuning Pipeline (2–3 veckor)

**Del 2.1 – Dataset-förberedelse** ⬜
- Skapa `data/callcenter_train.jsonl` och `data/callcenter_val.jsonl` (minst 3 000–5 000 exempel).
- Använd syntetisk data + augmentation (negationer, artighetsfraser, domäntermer).
- Skapa `scripts/prepare_callcenter_data.py`.

**Del 2.2 – Fine-tuning** ✅ (pipeline klar, GPU-träning återstår)
- ✅ Skapa `src/finetune.py` med stöd för:
  - `KBLab/bert-base-swedish-cased` + sentiment-head
- ✅ Använd `peft`, `transformers`, `datasets` och `accelerate`.
- ✅ Implementera:
  - LoRA-konfiguration (r=16, alpha=32)
  - Early stopping + learning rate scheduler
  - Utvärdering efter varje epoch
- ✅ Skapa `configs/finetune.yaml`
- ⬜ Kör `python -m src.finetune --config configs/finetune.yaml` på GPU

**Del 2.3 – Notebook** ✅
- ✅ Skapa `notebooks/finetune_callcenter.ipynb` med full pipeline.

### Steg 3: Större Svenskt Lexikon & Förbättrad Blending (1,5–2 veckor)

**Del 3.1 – Lexikon** 🔶 (seed finns, fullt lexikon återstår)
- ⬜ Ladda ner och bearbeta **SenSALDO** (eller motsvarande stor svensk lexikon).
- ⬜ Skapa `data/sensaldo_full.csv` (term, polarity, confidence).
- ✅ Utöka med call center-specifika termer (seed: 20 termer i `data/sensaldo_lexicon.csv`).

**Del 3.2 – Förbättrad blending** ⬜
- ⬜ Skapa `src/blending.py` med:
  - `LearnedBlender` (enkel logistic regression eller liten neural nätverk)
  - Alternativ: Bayesian blending eller viktad ensemble
- ✅ Uppdatera `src/lexicon.py` med bättre svenska tokenisering (hantera sammansatta ord).
- ✅ Implementera **negation detection** (regelbaserad i `src/sentiment.py`).

**Del 3.3 – Ny profil** ✅
- ✅ Lägg till `callcenter`-profil i `src/profiles.py` med anpassad rengöring och max_length=384.

### Steg 4: Avancerad Utvärdering & Benchmarking (1 vecka) 🔶

- ✅ Utöka `src/evaluate.py` med:
  - Jämförelse: Baseline vs KB-Whisper-large vs Finetunad modell
  - Per-profil prestanda (forum, call, news)
- ⬜ Negation-specifik utvärdering
- ✅ Skapa `reports/fas2_comparison.json` + `reports/fas2_comparison.md`
- ⬜ Generera confusion matrices och felanalys (från skarp modellkörning).

### Steg 5: Integration & Produktionsanpassning (1 vecka) ⬜

- ⬜ Uppdatera `src/sentiment.py` och `src/main.py` att använda den nya finetunade modellen som default.
- ✅ Lägg till stöd för modellväxling via `--model` och API.
- ⬜ Uppdatera Docker-image och `requirements.txt`.
- ⬜ Lägg till exempel i `samples/` för call center-användning.

### Steg 6: Dokumentation & Avslutning (3–5 dagar) ✅

- ✅ Uppdatera `README.md` med:
  - Hur man kör fine-tuning
  - Prestandajämförelse (tabell)
  - Rekommenderad konfiguration för call center
- ✅ Skapa `docs/FAS2_SUMMARY.md`
- ✅ Uppdatera `ROADMAP.md`
- ⬜ Skriv release notes för v0.3.0

---

## 4. Tekniska Specifikationer

| Komponent              | Rekommendation                          | Kommentar |
|------------------------|-----------------------------------------|---------|
| **Basmodell**          | `KBLab/bert-base-swedish-cased` + LoRA | Bäst svensk prestanda för text |
| **Fine-tuning metod**  | PEFT + LoRA (r=8, alpha=16, dropout=0.05) | Effektivt och billigt |
| **Lexikon**            | SenSALDO + egen call center-utökning   | Större täckning |
| **Blending**           | Learned weight (logistic regression)   | Bättre än heuristik |
| **Negation detection** | Regelbaserad + enkel classifier        | Kritiskt för svenska |
| **Utvärdering**        | macro-F1 + per-klass + negation-F1     | Måste visa tydlig förbättring |

---

## 5. Framgångskriterier (Definition of Done)

- [ ] Finetunad modell ger **≥ +15 % macro-F1** på call center-testset
- [ ] SenSALDO integrerat och fungerar
- [ ] Learned blending implementerat och utvärderat
- [ ] Negation detection förbättrar prestanda på negationsexempel med ≥ 10 %
- [ ] Fullständig benchmark-rapport finns
- [ ] All dokumentation uppdaterad
- [ ] CI passerar alla tester

---

## 6. Tidsplan & Resursuppskattning

| Steg                        | Varaktighet     | Resurs     | Status |
|-----------------------------|-----------------|------------|--------|
| Steg 1 (Verifiering)        | 1–2 dagar       | 1 dev      | ✅ |
| Steg 2 (Fine-tuning)        | 2–3 veckor      | 1–2 devs   | 🔶 |
| Steg 3 (Lexikon + Blending) | 1,5–2 veckor    | 1 dev      | 🔶 |
| Steg 4 (Utvärdering)        | 1 vecka         | 1 dev      | 🔶 |
| Steg 5–6 (Integration + Docs)| 1–1,5 vecka     | 1 dev      | 🔶 |
| **Totalt**                  | **8–10 veckor** | -          | |

---

## 7. Risker & Mitigering

| Risk                              | Sannolikhet | Mitigering |
|-----------------------------------|-------------|------------|
| Otillräckligt med träningsdata    | Medel       | Använd syntetisk data + augmentation |
| Fine-tuning tar för lång tid      | Hög         | Använd LoRA + mindre batch size |
| Lexikon ger marginell förbättring | Låg         | Kombinera med learned blending |
| Negation detection blir för komplex | Låg       | Börja med regelbaserad lösning |

---

## 8. Leverabler

- ✅ `src/finetune.py` + konfigurationsfiler
- ✅ `notebooks/finetune_callcenter.ipynb`
- 🔶 `data/sensaldo_full.csv` + `data/callcenter_train.jsonl` (seed finns)
- ⬜ `src/blending.py`
- 🔶 `reports/fas2_comparison.json` + `.md` (från heuristic baseline, skarp GPU behövs)
- ✅ Uppdaterad `README.md` + `ROADMAP.md`
- ⬜ Ny version `v0.3.0`

---

*Plan skapad 24 Maj 2026, uppdaterad med status.*