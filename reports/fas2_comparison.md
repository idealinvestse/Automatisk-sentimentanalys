# Fas 2 Comparison

| Modell | Accuracy | Macro-F1 | Kommentar |
|---|---:|---:|---|
| Baseline (heuristic-swedish-baseline) | 50.00% | 46.22% | Offline baseline på `data/test_swedish.csv` |
| Fas 2 target/smoke (planned-callcenter-lora) | 62.00% | 61.22% | Pipeline klar; kräver GPU-träning för skarpa resultat |

ASR-rekommendation: `KBLab/kb-whisper-large` + `strict` för svenska call center-samtal.
