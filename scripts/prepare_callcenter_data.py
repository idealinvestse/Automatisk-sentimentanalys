"""Generate and augment a Swedish call center training dataset.

Creates:
    data/callcenter_train.jsonl  – 3 000–5 000 labelled examples
    data/callcenter_val.jsonl    – 500 held-out examples

Augmentation strategies:
    - Negation insertion (inte, aldrig, ej)
    - Politeness phrase variation
    - Synonym swapping for common call center terms
    - Sentence length variation
"""

from __future__ import annotations

import csv
import json
import os
import random

random.seed(42)

# ---------------------------------------------------------------------------
# Base templates – (text, label) pairs covering typical call center scenarios
# ---------------------------------------------------------------------------
TEMPLATES: list[tuple[str, str]] = [
    # --- POSITIVA ---
    ("Tack för snabb hjälp med mitt ärende!", "positiv"),
    ("Kundtjänsten var mycket trevlig och professionell.", "positiv"),
    ("Ni löste mitt problem direkt, fantastiskt!", "positiv"),
    ("Jag är väldigt nöjd med servicen idag.", "positiv"),
    ("Äntligen fick jag svar på min fråga.", "positiv"),
    ("Operatören var otroligt hjälpsam och kunnig.", "positiv"),
    ("Tack vare er support fungerar allt perfekt nu.", "positiv"),
    ("Jag uppskattar verkligen ert tålamod.", "positiv"),
    ("Det var enkelt att komma fram och få hjälp.", "positiv"),
    ("Ni har verkligen överträffat mina förväntningar.", "positiv"),
    ("Bästa supporten jag varit med om på länge.", "positiv"),
    ("Jag är så glad att jag ringde er.", "positiv"),
    ("Tack för att ni tog er tid att förklara.", "positiv"),
    ("Smidigt och professionellt hanterat!", "positiv"),
    ("Jag känner mig trygg som kund hos er.", "positiv"),
    ("Vilken skillnad efter att ni hjälpte mig!", "positiv"),
    ("Allting fungerar klockrent nu, tack!", "positiv"),
    ("Jag rekommenderar er varmt till alla jag känner.", "positiv"),
    ("Ert bemötande var outstanding.", "positiv"),
    ("Ni gjorde en komplicerad process enkel.", "positiv"),
    ("Tack för snabb återkoppling på mitt mail.", "positiv"),
    ("Jag fick hjälp inom några minuter, imponerad!", "positiv"),
    ("Supporten förstod direkt vad jag menade.", "positiv"),
    ("Det känns som att ni verkligen bryr er.", "positiv"),
    ("Jag är helnöjd med lösningen ni gav mig.", "positiv"),
    # --- NEUTRALA ---
    ("Jag har en fråga om min faktura.", "neutral"),
    ("Kan ni bekräfta att ni tagit emot min beställning?", "neutral"),
    ("Jag vill bara kontrollera status på mitt ärende.", "neutral"),
    ("Har ni öppet på lördagar?", "neutral"),
    ("Jag behöver ändra min adress i systemet.", "neutral"),
    ("När kan jag förvänta mig en återkoppling?", "neutral"),
    ("Jag har inte hunnit testa lösningen ännu.", "neutral"),
    ("Kan ni skicka en bekräftelse på mail?", "neutral"),
    ("Jag undrar hur lång leveranstiden är.", "neutral"),
    ("Vilka betalningsalternativ erbjuder ni?", "neutral"),
    ("Jag fick ett meddelande om att ni försökt nå mig.", "neutral"),
    ("Kan ni uppdatera mina kontaktuppgifter?", "neutral"),
    ("Jag har precis bytt abonnemang och har en fråga.", "neutral"),
    ("Finns det någon manual för den här funktionen?", "neutral"),
    ("Jag ska precis skicka in dokumenten ni bad om.", "neutral"),
    ("Hur länge gäller erbjudandet?", "neutral"),
    ("Jag vill bara dubbelkolla en sak.", "neutral"),
    ("Kan jag få en specifikation på fakturan?", "neutral"),
    ("Jag har precis registrerat mig och loggar in för första gången.", "neutral"),
    ("Behöver jag boka tid för ett besök?", "neutral"),
    # --- NEGATIVA ---
    ("Jag har väntat i över en timme i telefonkö!", "negativ"),
    ("Det här är helt oacceptabelt.", "negativ"),
    ("Jag är mycket besviken på hur ni hanterat mitt ärende.", "negativ"),
    ("Ingen har hört av sig trots att ni lovade.", "negativ"),
    ("Detta är tredje gången jag ringer om samma sak.", "negativ"),
    ("Er support är helt värdelös.", "negativ"),
    ("Jag känner mig lurad av ert företag.", "negativ"),
    ("Fakturan stämmer inte alls och ingen hjälper mig.", "negativ"),
    ("Ni lovade återkoppling inom 24 timmar – det har gått en vecka.", "negativ"),
    ("Jag är arg och frustrerad över situationen.", "negativ"),
    ("Det här är rena katastrofen.", "negativ"),
    ("Produkten fungerar inte som utlovat.", "negativ"),
    ("Jag har blivit feldebiterad tre månader i rad.", "negativ"),
    ("Kommer aldrig mer anlita er igen.", "negativ"),
    ("Ni har tappat bort mina dokument.", "negativ"),
    ("Personalen var otrevlig och ointresserad.", "negativ"),
    ("Jag förstår inte varför detta ska vara så svårt.", "negativ"),
    ("Hemskt bemötande, kände mig helt ignorerad.", "negativ"),
    ("Jag har förlorat förtroendet för er helt.", "negativ"),
    ("Detta är sämsta servicen jag någonsin fått.", "negativ"),
]

# ---------------------------------------------------------------------------
# Augmentation patterns
# ---------------------------------------------------------------------------
POSITIVE_SYNONYMS = {
    "bra": ["utmärkt", "fantastisk", "toppen", "super", "kanon", "perfekt", "lysande"],
    "snabb": ["snabbt", "direkt", "omedelbar", "kvick", "snar", "effektiv"],
    "trevlig": ["vänlig", "hjälpsam", "tillmötesgående", "professionell", "positiv", "sympatisk"],
    "nöjd": ["belåten", "tillfreds", "tacksam", "glad", "imponerad", "entusiastisk"],
    "tack": ["tack så mycket", "tusen tack", "stort tack", "evigt tacksam"],
    "hjälp": ["support", "assistans", "vägledning", "service", "handräckning"],
    "löste": ["fixade", "åtgärdade", "hanterade", "klarerade", "redde ut"],
}

NEGATIVE_SYNONYMS = {
    "dålig": ["usel", "undermålig", "bedrövlig", "katastrofal", "fruktansvärd"],
    "besviken": ["missnöjd", "frustrerad", "arg", "irriterad", "upprörd", "förbannad"],
    "problem": ["bekymmer", "svårigheter", "krångel", "strul", "trubbel", "huvudvärk"],
    "väntat": ["suttit i kö", "hängt kvar", "legat på is", "stått still"],
}

POLITENESS_PHRASES = [
    "ursäkta att jag stör",
    "jag hoppas ni kan hjälpa mig",
    "jag skulle vara tacksam om",
    "jag undrar om ni möjligen",
    "ursäkta, men",
]

NEGATION_PREFIXES = ["inte", "aldrig", "ej", "knappast", "absolut inte"]


def augment_positive(text: str) -> str:
    """Add politeness, swap synonyms, vary wording."""
    t = text
    if random.random() < 0.3:
        t = f"{random.choice(POLITENESS_PHRASES)}, {t.lower()}"
        t = t[0].upper() + t[1:] if t else t
    for key, opts in POSITIVE_SYNONYMS.items():
        if key in t.lower() and random.random() < 0.4:
            t = t.lower().replace(key, random.choice(opts))
            t = t[0].upper() + t[1:] if t else t
    return t


def augment_negative(text: str) -> str:
    """Swap synonyms, intensify wording."""
    t = text
    for key, opts in NEGATIVE_SYNONYMS.items():
        if key in t.lower() and random.random() < 0.5:
            t = t.lower().replace(key, random.choice(opts))
            t = t[0].upper() + t[1:] if t else t
    if random.random() < 0.15:
        t += " Det är under all kritik."
    return t


def augment_neutral(text: str) -> str:
    """Add polite framing."""
    if random.random() < 0.3:
        return f"{random.choice(POLITENESS_PHRASES)}, {text.lower()}"
    return text


def apply_negation_flip(text: str, label: str) -> tuple[str, str]:
    """Create a negated version of a positive text (label flips to negativ)."""
    if label != "positiv":
        return text, label
    words = text.lower().split()
    if len(words) < 2:
        return text, label
    # Insert negation before a key sentiment word
    insert_pos = min(len(words) - 1, random.randint(1, 3))
    neg = random.choice(NEGATION_PREFIXES)
    words.insert(insert_pos, neg)
    new_text = " ".join(words)
    new_text = new_text[0].upper() + new_text[1:] if new_text else new_text
    return new_text, "negativ"


def generate_dataset(
    n_total: int = 3500, val_size: int = 500
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Generate training and validation datasets."""
    all_examples: list[dict[str, str]] = []

    # 1) Base templates (with augmentation)
    for text, label in TEMPLATES:
        all_examples.append({"text": text, "label": label})
        # Augment each template 2-4 times
        n_aug = random.randint(2, 4)
        for _ in range(n_aug):
            if label == "positiv":
                aug_text = augment_positive(text)
            elif label == "negativ":
                aug_text = augment_negative(text)
            else:
                aug_text = augment_neutral(text)
            all_examples.append({"text": aug_text, "label": label})

    # 2) Negation-flipped examples (positive → negative)
    n_negation = int(len(TEMPLATES) * 0.6)
    pos_templates = [(t, lbl) for t, lbl in TEMPLATES if lbl == "positiv"]
    for _ in range(n_negation):
        text, _ = random.choice(pos_templates)
        flipped_text, flipped_label = apply_negation_flip(text, "positiv")
        all_examples.append({"text": flipped_text, "label": flipped_label})

    # 3) Call center specific phrases (generate by combining patterns)
    call_center_actions = ["ringde", "mailade", "chattade", "kontaktade", "nådde"]
    call_center_subjects = [
        "kundtjänst",
        "supporten",
        "er avdelning",
        "tekniska supporten",
        "kundservice",
    ]
    call_center_outcomes_pos = [
        "löste problemet",
        "fick svar direkt",
        "blev hjälpt",
        "fick hjälp",
        "allt ordnade sig",
    ]
    call_center_outcomes_neg = [
        "fick inget svar",
        "blev runtkopplad",
        "lovades hjälp men inget hände",
        "ingen kunde hjälpa",
        "fick vänta länge",
    ]
    call_center_outcomes_neu = [
        "fick information",
        "skickade dokument",
        "bad om återkoppling",
        "ställde en fråga",
    ]

    for _ in range(400):
        action = random.choice(call_center_actions)
        subject = random.choice(call_center_subjects)
        if random.random() < 0.5:
            outcome = random.choice(call_center_outcomes_pos)
            label = "positiv"
        else:
            outcome = random.choice(call_center_outcomes_neg)
            label = "negativ"
        all_examples.append({"text": f"Jag {action} {subject} och {outcome}.", "label": label})

    for _ in range(200):
        action = random.choice(call_center_actions)
        subject = random.choice(call_center_subjects)
        outcome = random.choice(call_center_outcomes_neu)
        all_examples.append({"text": f"Jag {action} {subject} och {outcome}.", "label": "neutral"})

    # 4) Additional domain-specific short phrases
    domain_phrases_pos = [
        "Bra att ni lyssnade på min feedback.",
        "Jag gillar ert nya system.",
        "Tack för att ni följde upp mitt ärende.",
        "Uppskattar att ni ringde tillbaka.",
        "Mycket smidigare än jag trodde.",
        "Ni har verkligen förbättrat er sedan sist.",
    ]
    domain_phrases_neg = [
        "Jag är inte nöjd alls.",
        "Ni har inte löst någonting.",
        "Det här var inte vad jag förväntade mig.",
        "Jag kommer inte acceptera detta.",
        "Ingenting har förändrats sedan jag ringde förra veckan.",
    ]
    domain_phrases_neu = [
        "Jag avvaktar och ser hur det går.",
        "Återkommer om det behövs.",
        "Jag noterar vad ni sagt.",
        "Tack för informationen.",
        "Jag ska läsa igenom villkoren först.",
    ]

    for phrase in domain_phrases_pos * 30:
        all_examples.append({"text": phrase, "label": "positiv"})
    for phrase in domain_phrases_neg * 30:
        all_examples.append({"text": phrase, "label": "negativ"})
    for phrase in domain_phrases_neu * 30:
        all_examples.append({"text": phrase, "label": "neutral"})

    # Shuffle and split
    random.shuffle(all_examples)

    # Ensure we have at least n_total + val_size examples
    target = n_total + val_size
    while len(all_examples) < target:
        base = random.choice(all_examples)
        all_examples.append(dict(base))

    all_examples = all_examples[:target]
    train = all_examples[:n_total]
    val = all_examples[n_total:target]

    return train, val


def write_jsonl(path: str, data: list[dict[str, str]]) -> None:
    """Write examples as JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_csv(path: str, data: list[dict[str, str]]) -> None:
    """Write examples as CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(data)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate call center training dataset")
    parser.add_argument(
        "--target-size",
        type=int,
        default=10000,
        help="Total examples to generate (default: 10000)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    args = parser.parse_args()
    val_size = max(100, int(args.target_size * args.val_ratio))
    n_total = args.target_size

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(repo_root, "data")

    print(f"Generating Swedish call center training dataset (n={n_total}, val={val_size})...")
    train, val = generate_dataset(n_total=n_total, val_size=val_size)

    # Count distribution
    dist_train: dict[str, int] = {}
    dist_val: dict[str, int] = {}
    for ex in train:
        dist_train[ex["label"]] = dist_train.get(ex["label"], 0) + 1
    for ex in val:
        dist_val[ex["label"]] = dist_val.get(ex["label"], 0) + 1

    print(f"Train: {len(train)} examples, distribution: {dist_train}")
    print(f"Val:   {len(val)} examples, distribution: {dist_val}")

    # Write both JSONL and CSV formats
    write_jsonl(os.path.join(data_dir, "callcenter_train.jsonl"), train)
    write_jsonl(os.path.join(data_dir, "callcenter_val.jsonl"), val)
    write_csv(os.path.join(data_dir, "callcenter_train.csv"), train)
    write_csv(os.path.join(data_dir, "callcenter_val.csv"), val)

    print(f"Files written to {data_dir}/")


if __name__ == "__main__":
    main()
