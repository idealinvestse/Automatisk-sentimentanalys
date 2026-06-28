"""Generate balanced Swedish call center intent training/validation JSONL.

Creates:
    data/intent_train.jsonl  — stratified train split (default 80%)
    data/intent_val.jsonl    — held-out validation split (default 20%)

No duplicate (text, intent) pairs. Augmentation includes ASR fillers and overlapping phrases.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

# Base templates per intent (≥10 unique per class; augmented to ≥30)
INTENT_TEMPLATES: dict[str, list[str]] = {
    "account_update": [
        "Jag vill ändra min adress",
        "Kan ni uppdatera mitt telefonnummer?",
        "Jag har bytt email, kan ni ändra det?",
        "Mina kontaktuppgifter behöver uppdateras",
        "Hur ändrar jag min profil?",
        "Jag vill byta lösenord på mitt konto",
        "Kan ni hjälpa mig uppdatera mina uppgifter?",
        "Jag har flyttat och behöver ny adress",
        "Ändra min e-postadress tack",
        "Uppdatera mitt kundkonto är ni snälla",
        "Jag behöver ändra min adress i systemet",
        "Kan ni registrera min nya postadress?",
    ],
    "billing_inquiry": [
        "Jag har en fråga om min faktura",
        "Varför är fakturan så hög denna månad?",
        "Kan ni förklara den här avgiften?",
        "Jag har inte fått någon faktura än",
        "När ska jag betala?",
        "Kan jag få en specifikation på fakturan?",
        "Det står fel belopp på min faktura",
        "Hur mycket kostar det per månad?",
        "Jag vill ha fakturan via mail istället",
        "Kan ni skicka om fakturan?",
        "Vad är den här extra debiteringen på fakturan?",
        "Min faktura matchar inte mitt avtal",
    ],
    "technical_support": [
        "Produkten fungerar inte som den ska",
        "Jag får ett felmeddelande när jag loggar in",
        "Min enhet startar inte",
        "Det står bara och laddar",
        "Appen kraschar hela tiden",
        "Jag kan inte ansluta till internet",
        "Något är fel med min anslutning",
        "Hjälp, ingenting fungerar!",
        "Kan ni fjärrstyra och kolla vad som är fel?",
        "Jag behöver teknisk hjälp med installationen",
        "Wifi fungerar inte efter omstart",
        "Router blinkar rött hela tiden",
    ],
    "order_status": [
        "Var är min beställning?",
        "När kommer mitt paket?",
        "Kan ni spåra min leverans?",
        "Jag har inte fått någon orderbekräftelse",
        "Hur lång är leveranstiden?",
        "Är min order skickad än?",
        "Jag vill kolla status på min beställning",
        "När skickades paketet?",
        "Kan ni bekräfta att ni mottagit min order?",
        "Jag undrar om ni har skickat min vara",
        "Har paketet lämnat lagret?",
        "Spårningssidan visar inget",
    ],
    "cancellation": [
        "Jag vill avsluta mitt abonnemang",
        "Hur säger jag upp tjänsten?",
        "Jag vill avboka min prenumeration",
        "Kan ni avsluta mitt konto?",
        "Jag är inte intresserad längre, avsluta tack",
        "Säg upp allt, jag vill inte vara kund längre",
        "Avboka min beställning",
        "Jag ångrar mig och vill avbryta",
        "Hur lång uppsägningstid har ni?",
        "Avsluta mitt medlemskap omedelbart",
        "Jag vill säga upp mitt avtal",
        "Kan ni stänga mitt konto idag?",
    ],
    "complaint": [
        "Jag är mycket missnöjd med servicen",
        "Det här är helt oacceptabelt!",
        "Jag vill lämna ett klagomål",
        "Ni har behandlat mig dåligt",
        "Detta är under all kritik",
        "Jag känner mig lurad av er",
        "Sämsta servicen jag varit med om",
        "Jag vill eskalera mitt ärende",
        "Ingen har hjälpt mig trots flera samtal",
        "Det här är rena katastrofen",
        "Jag är arg och frustrerad över situationen",
        "Er support är helt värdelös",
    ],
    "information_request": [
        "Vilka öppettider har ni?",
        "Berätta mer om era tjänster",
        "Vad kostar det att bli kund?",
        "Har ni något erbjudande just nu?",
        "Hur fungerar er tjänst?",
        "Vilka betalningsalternativ finns?",
        "Jag undrar om ni har butik i Stockholm",
        "Kan ni berätta om era olika abonnemang?",
        "Vad ingår i grundpaketet?",
        "Har ni någon app man kan använda?",
        "Vilka kanaler kan jag kontakta er på?",
        "Finns det studentrabatt?",
    ],
    "refund_request": [
        "Jag vill ha pengarna tillbaka",
        "Kan ni återbetala min senaste faktura?",
        "Jag har blivit feldebiterad och vill ha återbetalning",
        "När får jag mina pengar tillbaka?",
        "Ni lovade återbetalning men inget har hänt",
        "Kreditera mitt konto tack",
        "Jag returnerade varan, var är återbetalningen?",
        "Ni drog för mycket, återbetala mellanskillnaden",
        "Jag vill ha kompensation för strulet",
        "Återbetala beloppet omgående",
        "Kan ni refunda den felaktiga avgiften?",
        "Jag kräver återbetalning för dubbeldebitering",
    ],
    "appointment_booking": [
        "Jag vill boka en tid",
        "Kan jag få en tid på torsdag?",
        "Jag behöver omboka mitt möte",
        "Finns det någon ledig tid imorgon?",
        "Boka in mig på förmiddagen tack",
        "Jag måste avboka min tid på fredag",
        "När har ni nästa lediga tid?",
        "Kan jag boka ett videosamtal?",
        "Jag vill ha ett möte med en handläggare",
        "Boka om min tid till nästa vecka",
        "Har ni drop-in imorgon?",
        "Jag behöver boka servicebesök",
    ],
    "other": [
        "Hej, jag har en allmän fråga",
        "Jag är osäker på vart jag ska vända mig",
        "Kan ni koppla mig till rätt avdelning?",
        "Jag vill bara testa er support",
        "Tack för hjälpen, hej då!",
        "Jag ringer för att kolla läget",
        "Ursäkta, jag kom fel",
        "Jag har blivit hänvisad hit",
        "Är detta kundtjänst?",
        "Jag söker någon som kan hjälpa mig",
        "God dag, jag hoppas ni kan hjälpa",
        "Jag vet inte riktigt vem jag ska prata med",
    ],
}

ASR_FILLERS = ["Eh, ", "Alltså, ", "Typ ", "Eh, alltså, "]
POLITE_PREFIXES = ["Ursäkta, ", "Hej, ", "God dag, "]

# Overlapping phrases for harder discrimination
OVERLAP_AUGMENTS: dict[str, list[str]] = {
    "complaint": [
        "Min faktura är fel och jag är arg på er",
        "Det fungerar inte och det är oacceptabelt",
    ],
    "billing_inquiry": [
        "Jag undrar varför fakturan är så hög, inte ett klagomål bara",
        "Kan ni förklara avgiften på min senaste faktura?",
    ],
    "refund_request": [
        "Fakturan är fel, jag vill ha pengarna tillbaka",
        "Ni debiterade fel, återbetala tack",
    ],
}


def _augment_text(text: str, intent: str) -> str:
    t = text
    if random.random() < 0.25:
        t = random.choice(POLITE_PREFIXES) + t[0].lower() + t[1:]
    if random.random() < 0.2:
        t = random.choice(ASR_FILLERS) + t
    return t.strip()


def build_examples(per_intent: int = 35) -> list[dict[str, str]]:
    """Build deduplicated examples with augmentation."""
    seen: set[tuple[str, str]] = set()
    examples: list[dict[str, str]] = []

    for intent, templates in INTENT_TEMPLATES.items():
        pool = list(templates) + OVERLAP_AUGMENTS.get(intent, [])
        while len([e for e in examples if e["intent"] == intent]) < per_intent:
            base = random.choice(pool)
            text = _augment_text(base, intent)
            key = (text.lower().strip(), intent)
            if key in seen:
                continue
            seen.add(key)
            examples.append({"text": text, "intent": intent})

    random.shuffle(examples)
    return examples


def stratified_split(
    examples: list[dict[str, str]], val_ratio: float = 0.2
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_intent: dict[str, list[dict[str, str]]] = defaultdict(list)
    for ex in examples:
        by_intent[ex["intent"]].append(ex)

    train, val = [], []
    for intent, items in sorted(by_intent.items()):
        random.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio)))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate balanced intent JSONL datasets")
    parser.add_argument("--per-intent", type=int, default=35, help="Examples per intent class")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--train-out", type=Path, default=Path("data/intent_train.jsonl"))
    parser.add_argument("--val-out", type=Path, default=Path("data/intent_val.jsonl"))
    args = parser.parse_args()

    examples = build_examples(per_intent=args.per_intent)
    train, val = stratified_split(examples, val_ratio=args.val_ratio)
    write_jsonl(args.train_out, train)
    write_jsonl(args.val_out, val)

    print(f"Wrote {len(train)} train + {len(val)} val examples")
    print(f"  train: {args.train_out}")
    print(f"  val:   {args.val_out}")


if __name__ == "__main__":
    main()
