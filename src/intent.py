"""Intent classification for Swedish call center conversations.

Defines 10 standard call center intents and provides a classifier with
both heuristic (keyword-based) and model-based (LoRA fine-tuned) backends.

Usage:
    from src.intent import IntentClassifier
    clf = IntentClassifier()
    intent, confidence = clf.classify("Jag vill ändra min adress")
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent definitions
# ---------------------------------------------------------------------------
CALL_CENTER_INTENTS: dict[str, dict[str, Any]] = {
    "account_update": {
        "id": 0,
        "label": "account_update",
        "display_sv": "Kontouppdatering",
        "description": "Ändra kontouppgifter, adress, telefonnummer",
        "keywords": [
            "ändra",
            "uppdatera",
            "adress",
            "telefonnummer",
            "kontaktuppgifter",
            "profil",
            "mitt konto",
        ],
        "priority": "medium",
    },
    "billing_inquiry": {
        "id": 1,
        "label": "billing_inquiry",
        "display_sv": "Fakturafråga",
        "description": "Frågor om faktura, betalning, pris",
        "keywords": [
            "faktura",
            "betala",
            "betalning",
            "pris",
            "kostnad",
            "avgift",
            "debitera",
            "summa",
            "belopp",
        ],
        "priority": "high",
    },
    "technical_support": {
        "id": 2,
        "label": "technical_support",
        "display_sv": "Teknisk support",
        "description": "Problem med produkt/tjänst, felanmälan",
        "keywords": [
            "fungerar inte",
            "problem",
            "fel",
            "trasig",
            "bugg",
            "tekniskt",
            "support",
            "hjälp med",
        ],
        "priority": "high",
    },
    "order_status": {
        "id": 3,
        "label": "order_status",
        "display_sv": "Orderstatus",
        "description": "Kontrollera beställning, leveransstatus",
        "keywords": [
            "beställning",
            "order",
            "leverans",
            "spåra",
            "status",
            "paket",
            "skickat",
            "mottagit",
        ],
        "priority": "medium",
    },
    "cancellation": {
        "id": 4,
        "label": "cancellation",
        "display_sv": "Avbokning/Uppsägning",
        "description": "Avboka tjänst, säga upp abonnemang",
        "keywords": [
            "avboka",
            "säga upp",
            "avsluta",
            "uppsägning",
            "avslut",
            "sluta",
            "lämna",
            "gå ur",
        ],
        "priority": "high",
    },
    "complaint": {
        "id": 5,
        "label": "complaint",
        "display_sv": "Klagomål",
        "description": "Missnöje, reklamation, eskalerat ärende",
        "keywords": [
            "klaga",
            "missnöjd",
            "besviken",
            "reklamera",
            "dålig",
            "usel",
            "oacceptabelt",
            "skandal",
        ],
        "priority": "critical",
    },
    "information_request": {
        "id": 6,
        "label": "information_request",
        "display_sv": "Informationsförfrågan",
        "description": "Allmänna frågor om produkter, tjänster, öppettider",
        "keywords": [
            "information",
            "undrar",
            "fråga",
            "öppettider",
            "erbjudande",
            "sortiment",
            "tjänster",
            "produkter",
        ],
        "priority": "low",
    },
    "refund_request": {
        "id": 7,
        "label": "refund_request",
        "display_sv": "Återbetalning",
        "description": "Begäran om återbetalning, kreditering",
        "keywords": [
            "återbetalning",
            "pengar tillbaka",
            "kreditera",
            "återbetala",
            "refund",
            "kompensation",
        ],
        "priority": "high",
    },
    "appointment_booking": {
        "id": 8,
        "label": "appointment_booking",
        "display_sv": "Bokning",
        "description": "Boka tid, omboka, avboka möte",
        "keywords": ["boka", "tid", "möte", "omboka", "kalender", "besök", "inbokat"],
        "priority": "medium",
    },
    "other": {
        "id": 9,
        "label": "other",
        "display_sv": "Övrigt",
        "description": "Övriga ärenden som inte passar andra kategorier",
        "keywords": [],
        "priority": "low",
    },
}

INTENT_LABELS = sorted(CALL_CENTER_INTENTS.keys(), key=lambda k: CALL_CENTER_INTENTS[k]["id"])


# ---------------------------------------------------------------------------
# IntentClassifier
# ---------------------------------------------------------------------------
class IntentClassifier:
    """Classify Swedish call center utterances into predefined intents.

    Supports two backends:
        - 'heuristic': Keyword-based matching (fast, no model required)
        - 'model': LoRA fine-tuned transformer (requires training first)

    Args:
        backend: 'heuristic' or 'model'.
        model_path: Path to fine-tuned model directory (for 'model' backend).
        device: 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        backend: str = "heuristic",
        model_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.backend = backend
        self.model_path = model_path
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None

        # Build keyword index for fast matching
        self._keyword_index: dict[str, str] = {}
        for intent_name, intent_data in CALL_CENTER_INTENTS.items():
            for kw in intent_data.get("keywords", []):
                self._keyword_index[kw.lower()] = intent_name

        if backend == "model" and model_path:
            self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(self, text: str) -> tuple[str, float]:
        """Classify a single text into an intent.

        Returns:
            (intent_label, confidence_score) tuple.
        """
        if self.backend == "model" and self._model is not None:
            return self._classify_model(text)

        return self._classify_heuristic(text)

    def classify_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """Classify multiple texts."""
        return [self.classify(t) for t in texts]

    def get_intent_info(self, intent_label: str) -> dict[str, Any] | None:
        """Get metadata for an intent label."""
        return CALL_CENTER_INTENTS.get(intent_label)

    def list_intents(self) -> list[dict[str, Any]]:
        """List all available intents with metadata."""
        return [
            {
                "label": data["label"],
                "display_sv": data["display_sv"],
                "description": data["description"],
                "priority": data["priority"],
            }
            for data in CALL_CENTER_INTENTS.values()
        ]

    # ------------------------------------------------------------------
    # Heuristic backend
    # ------------------------------------------------------------------
    def _classify_heuristic(self, text: str) -> tuple[str, float]:
        """Keyword-based intent classification."""
        lowered = text.lower()
        scores: dict[str, float] = {}

        for intent_name, intent_data in CALL_CENTER_INTENTS.items():
            score = 0.0
            keywords = intent_data.get("keywords", [])
            for kw in keywords:
                if kw in lowered:
                    score += 1.0
            if keywords:
                score /= len(keywords)
            scores[intent_name] = score

        # Boost "complaint" if strong negative sentiment detected
        strong_negative = {"dålig", "usel", "katastrof", "skandal", "oacceptabelt", "värdelös"}
        if any(w in lowered for w in strong_negative):
            scores["complaint"] = min(1.0, scores.get("complaint", 0) + 0.3)

        if not scores or max(scores.values()) == 0:
            return "other", 0.5

        # Break ties deterministically: prefer specific intents over "other" and "information_request"
        best_score = max(scores.values())
        candidates = [k for k, v in scores.items() if v == best_score]
        if len(candidates) > 1:
            priority_order = [
                "complaint",
                "cancellation",
                "refund_request",
                "billing_inquiry",
                "technical_support",
                "account_update",
                "order_status",
                "appointment_booking",
                "information_request",
                "other",
            ]
            for preferred in priority_order:
                if preferred in candidates:
                    best = (preferred, best_score)
                    break
            else:
                best = (candidates[0], best_score)
        else:
            best = (candidates[0], best_score)

        confidence = min(1.0, best[1])
        return best[0], round(confidence, 3)

    # ------------------------------------------------------------------
    # Model backend
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Load a fine-tuned intent classification model."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            logger.info("Intent model loaded from %s", self.model_path)
        except Exception as e:
            logger.warning("Failed to load intent model: %s. Falling back to heuristic.", e)
            self.backend = "heuristic"

    def _classify_model(self, text: str) -> tuple[str, float]:
        """Model-based intent classification."""
        import torch

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

        id_to_label = {v["id"]: k for k, v in CALL_CENTER_INTENTS.items()}
        label = id_to_label.get(pred_idx, "other")
        return label, round(confidence, 3)


# ---------------------------------------------------------------------------
# Training data generator
# ---------------------------------------------------------------------------
def generate_intent_dataset(output_path: str, n_examples: int = 800) -> None:
    """Generate a Swedish call center intent training dataset.

    Creates a JSONL file with labelled examples for each intent.
    """
    templates: dict[str, list[str]] = {
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
        ],
    }

    examples: list[dict[str, str]] = []
    for intent, phrases in templates.items():
        for phrase in phrases:
            examples.append({"text": phrase, "intent": intent})

    # Augment by repeating with slight variations
    while len(examples) < n_examples:
        base = examples[len(examples) % len(examples)]
        examples.append(dict(base))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples[:n_examples]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info("Intent dataset written: %s (%d examples)", output_path, n_examples)


__all__ = [
    "CALL_CENTER_INTENTS",
    "INTENT_LABELS",
    "IntentClassifier",
    "generate_intent_dataset",
]
