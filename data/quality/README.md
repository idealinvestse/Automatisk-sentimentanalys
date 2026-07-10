# Quality OS data slots (DATA-01)

Placeholder paths for MQM annotations and human preference pairs.
See `configs/quality_mqm.yaml` for schema and gate settings.

## Files

| File | Schema | Purpose |
|------|--------|---------|
| `mqm_annotations.jsonl` | `MqmAnnotation` per line | Aspect/intent/emotion error typology |
| `preference_pairs.jsonl` | `PreferencePair` per line | Release gate for deep-path / coaching |

## Example lines (do not commit real PII)

```json
{"call_id":"demo-1","errors":[{"error_type":"aspect_wrong","severity":"major","span_text":"fel faktura"}],"overall_quality":0.4}
```

```json
{"call_id":"demo-1","chosen":"deep_v2","rejected":"deep_v1","annotator":"qa-1","field":"actionable_coaching"}
```

## CI

- `python scripts/evaluate_preference_gate.py` — exits 0 with skip when empty
- `python scripts/evaluate_preference_gate.py --require-corpus` — fails until DATA-01 lands
