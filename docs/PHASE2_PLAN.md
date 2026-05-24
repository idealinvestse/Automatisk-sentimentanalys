# Fas 2: Svensk Domänanpassning & Förbättrad Prestanda

## Mål

- Höja macro-F1 på svensk call center-data genom LoRA/PEFT fine-tuning.
- Integrera ett större svenskt lexikon (`data/sensaldo_lexicon.csv` seed).
- Förbättra negationshantering och call center-profilering.
- Skapa benchmarkrapporter för baseline och Fas 2.

## Status

- [x] Fas 1-brister kompletterade (`reports/baseline_results.json`, CI coverage 80).
- [x] Fine-tuning pipeline skapad (`src/finetune.py`, `configs/finetune.yaml`).
- [x] Syntetiskt call center-träningsdataset skapat.
- [x] SenSALDO-seedlexikon skapat.
- [x] Negation detection och `callcenter`-profil implementerad.
- [x] Fas 2-jämförelserapporter skapade.
- [ ] Kör verklig GPU-träning och uppdatera benchmark med skarpa modellresultat.

## Körning

```bash
python -m src.finetune --config configs/finetune.yaml
python -m src.evaluate evaluate --backend model --model models/callcenter-sentiment-lora --profile callcenter
```
