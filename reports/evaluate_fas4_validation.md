# Fas 4 Validation Report

**Generated:** 2026-06-18T23:07:56.816782+00:00
**Python:** 3.12.4

## Environment

- ML deps (faster_whisper): not installed (OK for validation)
- OPENROUTER_API_KEY: not set (LLM fallback path)

## Pipeline validation (synthetic callcenter samples)

- Samples processed: 2
- Reports produced: 2

### Fas 4 module presence

- `agent_performance`: PASS
- `qa_or_compliance`: PASS
- `insights`: PASS

### KPI metrics (proxy)

- **qa_consistency:** `{'agreement': 1.0, 'n': 2}`
- **coaching_precision:** `{'precision': 0.0, 'n': 0, 'note': 'no recs'}`
- **hot_topic_recall:** `{'recall': 0.0, 'n_gold': 3, 'n_produced': 0}`
- **pii_coverage:** `{'coverage': 0.0, 'n_events': 0}`
- **alert_trigger_rate:** `{'trigger_rate': 0.0, 'n_alerts': 0, 'by_severity': {}}`
- **cache_hit_rate:** `{'hit_rate': 0.5, 'total_queries': 2}`

### Semantic search

- Hits returned: 1
- Status: PASS

### Aggregated insights

- Hot topics: 0

## Evaluate sub-runs

### sentiment_heuristic
```json
{
  "accuracy": 0.481,
  "macro_f1": 0.4353,
  "per_class": {
    "negativ": {
      "precision": 0.68,
      "recall": 0.2429,
      "f1": 0.3579,
      "support": 70
    },
    "neutral": {
      "precision": 0.4061,
      "recall": 0.9571,
      "f1": 0.5702,
      "support": 70
    },
    "positiv": {
      "precision": 0.85,
      "recall": 0.2429,
      "f1": 0.3778,
      "support": 70
    }
  },
  "confusion_matrix": {
    "negativ": {
      "negativ": 17,
      "neutral": 50,
      "positiv": 3
    },
    "neutral": {
      "negativ": 3,
      "neutral": 67,
      "positiv": 0
    },
    "positiv": {
      "negativ": 5,
      "neutral": 48,
      "positiv": 17
    }
  },
  "n_samples": 210,
  "processing_time_s": 0.01,
  "profile": "call",
  "model": "lexicon-heuristic-via-score",
  "lexicon_weight": 0.0,
  "lexicon_file": null,
  "backend": "heuristic"
}
```

### scenario_forum
```json
{
  "accuracy": 0.481,
  "macro_f1": 0.4353,
  "per_class": {
    "negativ": {
      "precision": 0.68,
      "recall": 0.2429,
      "f1": 0.3579,
      "support": 70
    },
    "neutral": {
      "precision": 0.4061,
      "recall": 0.9571,
      "f1": 0.5702,
      "support": 70
    },
    "positiv": {
      "precision": 0.85,
      "recall": 0.2429,
      "f1": 0.3778,
      "support": 70
    }
  },
  "confusion_matrix": {
    "negativ": {
      "negativ": 17,
      "neutral": 50,
      "positiv": 3
    },
    "neutral": {
      "negativ": 3,
      "neutral": 67,
      "positiv": 0
    },
    "positiv": {
      "negativ": 5,
      "neutral": 48,
      "positiv": 17
    }
  },
  "n_samples": 210,
  "processing_time_s": 0.01,
  "profile": "forum",
  "model": "lexicon-heuristic-via-score",
  "lexicon_weight": 0.0,
  "lexicon_file": null,
  "backend": "heuristic"
}
```

### scenario_call
```json
{
  "accuracy": 0.481,
  "macro_f1": 0.4353,
  "per_class": {
    "negativ": {
      "precision": 0.68,
      "recall": 0.2429,
      "f1": 0.3579,
      "support": 70
    },
    "neutral": {
      "precision": 0.4061,
      "recall": 0.9571,
      "f1": 0.5702,
      "support": 70
    },
    "positiv": {
      "precision": 0.85,
      "recall": 0.2429,
      "f1": 0.3778,
      "support": 70
    }
  },
  "confusion_matrix": {
    "negativ": {
      "negativ": 17,
      "neutral": 50,
      "positiv": 3
    },
    "neutral": {
      "negativ": 3,
      "neutral": 67,
      "positiv": 0
    },
    "positiv": {
      "negativ": 5,
      "neutral": 48,
      "positiv": 17
    }
  },
  "n_samples": 210,
  "processing_time_s": 0.01,
  "profile": "call",
  "model": "lexicon-heuristic-via-score",
  "lexicon_weight": 0.0,
  "lexicon_file": null,
  "backend": "heuristic"
}
```

### scenario_news
```json
{
  "accuracy": 0.481,
  "macro_f1": 0.4353,
  "per_class": {
    "negativ": {
      "precision": 0.68,
      "recall": 0.2429,
      "f1": 0.3579,
      "support": 70
    },
    "neutral": {
      "precision": 0.4061,
      "recall": 0.9571,
      "f1": 0.5702,
      "support": 70
    },
    "positiv": {
      "precision": 0.85,
      "recall": 0.2429,
      "f1": 0.3778,
      "support": 70
    }
  },
  "confusion_matrix": {
    "negativ": {
      "negativ": 17,
      "neutral": 50,
      "positiv": 3
    },
    "neutral": {
      "negativ": 3,
      "neutral": 67,
      "positiv": 0
    },
    "positiv": {
      "negativ": 5,
      "neutral": 48,
      "positiv": 17
    }
  },
  "n_samples": 210,
  "processing_time_s": 0.01,
  "profile": "news",
  "model": "lexicon-heuristic-via-score",
  "lexicon_weight": 0.0,
  "lexicon_file": null,
  "backend": "heuristic"
}
```

### llm_quality_proxy
```json
[
  {
    "sample_id": "demo-1",
    "fallback": true,
    "llm_used": false,
    "has_actionable": false
  }
]
```

## Coverage gate (Fas 1)

- Target: ≥85% on in-scope `src/` modules
- Omitted optional paths: CLI, diarization, ASR backends (see `pyproject.toml`)

## Acceptance summary

- Fas 4 keys in pipeline results: PASS
- Semantic search returns hits: PASS
- Cache hit on repeat aggregate query: PASS
- PII + LLM path: no crash (validated via unit tests)