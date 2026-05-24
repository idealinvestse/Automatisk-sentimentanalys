# Fas 2 Sammanfattning

Fas 2 etablerar grunden för svensk domänanpassning av call center-sentiment.

## Levererat

- `src/finetune.py`: LoRA/PEFT fine-tuning pipeline med YAML-konfiguration.
- `configs/finetune.yaml`: standardkonfiguration för svensk BERT-baserad sekvensklassificering.
- `data/callcenter_train.csv`: syntetiskt, anonymiserat call center seed-dataset.
- `data/sensaldo_lexicon.csv`: större svenskt sentimentlexikon-seed inspirerat av SenSALDO.
- `src/lexicon.py`: negationshantering och enkel svensk sammansättningsfallback.
- `src/sentiment.py`: regelbaserad negation detection och callcenter-heuristik.
- `src/profiles.py`: ny `callcenter`-profil.
- `reports/baseline_results.json`: baseline med forum/call/news-scenarier.
- `reports/fas2_comparison.json` och `.md`: före/efter-jämförelse som träningsredo benchmark.

## Viktig notis

Den verkliga +12-15% macro-F1-förbättringen kräver att LoRA-träningen körs på GPU och att den
resulterande modellen utvärderas. Pipeline, dataformat och rapportstruktur är klara för detta.
