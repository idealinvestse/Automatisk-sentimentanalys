# Svenska ljudprover

Lägg testfiler här för svensk ASR-, pipeline- och sentimentutvärdering.

## Mappstruktur

```
sv/
  callcenter/     # Kundtjänstsamtal (agent + kund)
  retail/         # Butik/reklamation (valfritt)
  custom/         # Egna kategorier
```

## Lägga till en fil

1. Placera `.wav`, `.mp3` eller `.flac` under rätt undermapp.
2. (Valfritt) Lägg en sidofil `mitt_samtal.meta.yaml` bredvid ljudfilen (`audio_smoke_v1`):

```yaml
schema: audio_smoke_v1
language: sv
expected_sentiment: negativ
scenario: billing_complaint
speakers: 2
expected_transcript_contains:
  - "faktura"
notes: "Kund klagar på felaktig faktura"
```

`expected_transcript_contains` valideras av `evaluate audio smoke` / `pipeline`. Syntetiska toner (ingen tal) sätter `skip_ml: true`.

3. Aktivera pack i [`../manifest.yaml`](../manifest.yaml) (`enabled: true` för `sv_callcenter`).
4. Validera: `python -m src.evaluate audio validate`

## L7 smoke-fixture

`callcenter/smoke_sv_billing.meta.yaml` is committed. The matching `.wav` may be
absent from git (large/binary policy) — L7 scripts skip ML when `skip_ml: true`.
Sidecars for stems `090932`…`104207` document expected labels; the audio files
themselves are local-only. Do not claim WER until anonymized recordings are present.

## Köra tester

```bash
python -m src.evaluate audio validate
python -m src.evaluate audio list --pack sv_callcenter
python -m src.evaluate audio smoke --pack sv_callcenter --device cuda
python -m src.evaluate audio run --scenario pipeline --pack sv_callcenter --device cuda
python scripts/run_pilot_gates.py --skip-l8 --skip-l9 --device cpu
```