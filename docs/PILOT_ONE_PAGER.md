# Pilot — Automatisk callcenter-analys (svenska)

**För:** QA- och contact-center-ledare  
**Läge:** Kontrollerad pilot (inte full produktionsrullning)

---

## Vad ni får

Ett **analyslager** ovanpå era samtal — inte en ersättning för er contact-center-plattform (Genesys, NICE, m.fl.).

- Transkription på **svenska** med lokal talmodell (KB-Whisper) som standard
- Sentiment, intent, QA-stöd, insikter och agentöversikt — **QA-stöd och coachning är pilotens primärnytta**
- Djupare AI-resonemang (sammanfattning, coaching) **selektivt** enligt er kundprofil
- Bearbetning som utgår från **lokal drift först** (pilotmiljö: Windows/CUDA); externa AI-leverantörer används endast när er kundprofil uttryckligen tillåter det och underlag/resultat sparas lokalt
- Er kundorganisation identifieras och styr sin egen analys-, QA- och leverantörskonfiguration

---

## Vad som är unikt i piloten

| Fokus | Betydelse |
|-------|-----------|
| Svenska i telefoni | Optimerat för nordisk kundtjänst, inte bara engelska demos |
| Local-first | Grundtranskription och analys sker i pilotmiljön utan krav på moln |
| Kundstyrd bearbetning | Vilka data som får behandlas externt (text/råljud) och av vilka leverantörer bestäms av er kundprofil — aldrig tyst |
| Ärliga gränser | Vi lovar inte "100 % QM" eller suite-paritet utan uppmätta resultat på *er* data |

---

## Villkor i piloten (kort)

1. Transkription körs **lokalt** som standard; moln-STT används endast om er kundprofil aktiverar det uttryckligen.
2. Extern AI-behandling (om er profil tillåter den) sker med dokumenterade leverantörer och dataskyddsläge; leverantörsval och avtalsläge godkänns av pilotägaren innan aktivering.
3. Kvalitetstal (t.ex. felord i transkription) redovisas först efter mätning på **era** anonymiserade samtal.
4. Pilotens syfte är att validera värde i er miljö — inte att ersätta hela WFM-/CCaaS-svitan.
5. Automatiseringen omfattar analyser, bedömningar och rekommendationer — inte automatiskt verkställda åtgärder utanför analysverktyget.

---

## Vad vi behöver av er

- Tillgång till **anonymiserade** samtalsutdrag (text och/eller ljud) för utvärdering
- Kontaktperson för QA + IT/säkerhet (miljö, nycklar, nätverk)
- Bekräftelse om samtalen kan innehålla särskilt känsliga uppgifter (t.ex. hälsa) — då hålls analysen strikt lokal

---

## Vad ni *inte* ska förvänta er i denna fas

- Full feature-paritet med stora internationella suite-produkter
- Realtids-agent-assist med subsekunds-latens som standard (batch/post-call är primärt)
- Publicerade “globala” accuracy-siffror utan er egen mätning

---

## Nästa steg

1. Kickoff: målkö, volym, compliance-ram  
2. Teknisk smoke i er staging/GPU-miljö  
3. DATA-01: gemensam anonymiserad korpus  
4. Gemensam genomgång av resultat och go/no-go för bredare rullning  

**Kontakt / underlag internt:** `docs/PILOT_RUNBOOK.md`, `STRATEGY.md`
