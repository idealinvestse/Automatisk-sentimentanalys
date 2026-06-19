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
2. (Valfritt) Lägg en sidofil `mitt_samtal.meta.yaml` bredvid ljudfilen:

```yaml
expected_sentiment: negativ
scenario: billing_complaint
speakers: 2
notes: "Kund klagar på felaktig faktura"
```

3. Aktivera pack i [`../manifest.yaml`](../manifest.yaml) (`enabled: true` för `sv_callcenter`).
4. Validera: `python -m src.evaluate audio validate`

## Köra tester

```bash
python -m src.evaluate audio list --pack sv_callcenter
python -m src.evaluate audio smoke --pack sv_callcenter --device cpu
python -m src.evaluate audio run --scenario pipeline --pack sv_callcenter --limit 2
```