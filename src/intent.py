"""Intent classification for Swedish call center conversations.

Defines 10 standard call center intents and provides a classifier with
both heuristic (keyword-based) and model-based (LoRA fine-tuned) backends.

Usage:
    from src.intent import IntentClassifier
    clf = IntentClassifier()
    intent, confidence = clf.classify("Jag vill ändra min adress")
"""

from __future__ import annotations

import logging
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
        if backend not in {"heuristic", "model", "auto"}:
            raise ValueError(f"Unsupported intent backend: {backend}")
        self.backend = backend
        self.resolved_backend = "heuristic" if backend == "auto" else backend
        self.model_path = model_path
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None

        # Build keyword index for fast matching
        self._keyword_index: dict[str, str] = {}
        for intent_name, intent_data in CALL_CENTER_INTENTS.items():
            for kw in intent_data.get("keywords", []):
                self._keyword_index[kw.lower()] = intent_name

        if backend in {"model", "auto"} and model_path:
            self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(self, text: str) -> tuple[str, float]:
        """Classify a single text into an intent.

        Returns:
            (intent_label, confidence_score) tuple.
        """
        if self.resolved_backend == "model" and self._model is not None:
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
        """Keyword-based intent classification with phrase boosts."""
        lowered = text.lower()
        scores: dict[str, float] = {}

        # Strong multi-word phrase boosts (checked first)
        phrase_boosts: dict[str, list[str]] = {
            "refund_request": [
                "pengarna tillbaka",
                "återbetala",
                "återbetalning",
                "kreditera",
                "feldebiterad",
                "kompensation",
            ],
            "cancellation": [
                "säga upp",
                "avsluta mitt abonnemang",
                "avsluta mitt konto",
                "uppsägning",
                "inte intresserad längre",
                "gå ur",
            ],
            "complaint": [
                "klagomål",
                "missnöjd",
                "oacceptabelt",
                "under all kritik",
                "eskalera",
                "katastrof",
                "värdelös",
            ],
            "appointment_booking": [
                "boka en tid",
                "boka om",
                "omboka",
                "ledig tid",
                "videosamtal",
                "servicebesök",
            ],
            "order_status": [
                "var är min beställning",
                "orderstatus",
                "leveransstatus",
                "spåra min leverans",
                "orderbekräftelse",
            ],
            "technical_support": [
                "fungerar inte",
                "felmeddelande",
                "kraschar",
                "teknisk hjälp",
                "wifi",
                "router",
                "internet",
            ],
            "billing_inquiry": [
                "min faktura",
                "fakturan",
                "debitering",
                "avgift",
                "belopp",
            ],
            "account_update": [
                "ändra min adress",
                "uppdatera mitt telefonnummer",
                "kontaktuppgifter",
                "byta lösenord",
                "ny adress",
            ],
            "information_request": [
                "öppettider",
                "erbjudande",
                "betalningsalternativ",
                "grundpaketet",
                "studentrabatt",
            ],
        }
        for intent_name, phrases in phrase_boosts.items():
            hits = sum(1 for p in phrases if p in lowered)
            if hits:
                scores[intent_name] = scores.get(intent_name, 0.0) + hits * 0.35

        for intent_name, intent_data in CALL_CENTER_INTENTS.items():
            score = scores.get(intent_name, 0.0)
            keywords = intent_data.get("keywords", [])
            kw_hits = sum(1 for kw in keywords if kw in lowered)
            if kw_hits:
                score += kw_hits * 0.15
            scores[intent_name] = score

        # Boost complaint on strong negative words
        strong_negative = {
            "dålig",
            "usel",
            "katastrof",
            "skandal",
            "oacceptabelt",
            "värdelös",
            "arg",
        }
        if any(w in lowered for w in strong_negative):
            scores["complaint"] = scores.get("complaint", 0) + 0.25

        # Penalize generic "other" unless nothing else matches
        if (
            scores.get("other", 0) > 0
            and max((v for k, v in scores.items() if k != "other"), default=0) > 0.15
        ):
            scores["other"] *= 0.3

        if not scores or max(scores.values()) == 0:
            return "other", 0.5

        # Disambiguation rules
        if "boka" in lowered and "avboka" not in lowered and "avsluta" not in lowered:
            scores["appointment_booking"] = scores.get("appointment_booking", 0) + 0.2
            scores["cancellation"] = scores.get("cancellation", 0) * 0.5
        if any(w in lowered for w in ("avsluta", "säga upp", "uppsägning")):
            scores["cancellation"] = scores.get("cancellation", 0) + 0.3
            scores["appointment_booking"] = scores.get("appointment_booking", 0) * 0.4
        if (
            any(w in lowered for w in ("faktura", "debiter", "avgift"))
            and "återbetala" not in lowered
        ):
            scores["billing_inquiry"] = scores.get("billing_inquiry", 0) + 0.2
        if "återbetala" in lowered or "pengarna tillbaka" in lowered:
            scores["refund_request"] = scores.get("refund_request", 0) + 0.4

        best_score = max(scores.values())
        candidates = [k for k, v in scores.items() if v == best_score]
        if len(candidates) > 1:
            priority_order = [
                "complaint",
                "refund_request",
                "cancellation",
                "billing_inquiry",
                "technical_support",
                "account_update",
                "order_status",
                "appointment_booking",
                "information_request",
                "other",
            ]
            best_intent = candidates[0]
            for preferred in priority_order:
                if preferred in candidates:
                    best_intent = preferred
                    break
        else:
            best_intent = candidates[0]

        confidence = min(1.0, best_score / 1.5)
        return best_intent, round(max(confidence, 0.35), 3)

    # ------------------------------------------------------------------
    # Model backend
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Load a fine-tuned intent model (sklearn smoke artifact or transformers)."""
        from pathlib import Path

        root = Path(self.model_path) if self.model_path else None
        if root is not None:
            joblib_path = root / "model.joblib"
            backend_marker = root / "backend.txt"
            prefer_sklearn = False
            if backend_marker.is_file():
                prefer_sklearn = "sklearn" in backend_marker.read_text(encoding="utf-8").lower()
            if prefer_sklearn or joblib_path.is_file():
                try:
                    import joblib

                    self._model = joblib.load(joblib_path)
                    self._tokenizer = "sklearn"
                    self.resolved_backend = "model"
                    logger.info("Intent sklearn model loaded from %s", joblib_path)
                    return
                except Exception as e:
                    logger.warning("Failed to load sklearn intent model: %s", e)

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if not self.model_path:
                raise ValueError("model_path is required for transformers intent backend")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.resolved_backend = "model"
            logger.info("Intent model loaded from %s", self.model_path)
        except Exception as e:
            logger.warning("Failed to load intent model: %s. Falling back to heuristic.", e)
            self.backend = "heuristic"
            self.resolved_backend = "heuristic"

    def _classify_model(self, text: str) -> tuple[str, float]:
        """Model-based intent classification."""
        if self._tokenizer == "sklearn":
            pred = self._model.predict([text])[0]
            proba = None
            if hasattr(self._model, "predict_proba"):
                try:
                    probs = self._model.predict_proba([text])[0]
                    classes = list(self._model.classes_)
                    idx = classes.index(pred)
                    proba = float(probs[idx])
                except Exception:
                    proba = None
            return str(pred), round(proba if proba is not None else 0.7, 3)

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


def generate_intent_dataset(
    n_samples_per_intent: int = 50, seed: int = 42
) -> list[dict[str, Any]]:
    """Generate synthetic intent dataset for testing and baseline training (deprecated).

    .. deprecated:: 0.5.1
        Use ``data/intent_train.jsonl`` or domain training scripts instead.
    """
    import random
    import warnings

    warnings.warn(
        "generate_intent_dataset is deprecated and will be removed in v0.6.0; "
        "use verified datasets in data/ or scripts/validate_intent_corpus.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    rng = random.Random(seed)
    templates = [
        "Jag vill {kw}",
        "Kan ni hjälpa mig med {kw}?",
        "Det gäller {kw}",
        "Hej, jag har problem med {kw}",
        "Angående {kw}",
        "Hur gör jag med {kw}?",
        "Vi behöver ordna {kw}",
        "Min {kw} fungerar inte",
    ]

    dataset: list[dict[str, Any]] = []
    for label, info in CALL_CENTER_INTENTS.items():
        keywords = info.get("keywords", [label])
        for _ in range(n_samples_per_intent):
            kw = rng.choice(keywords)
            tmpl = rng.choice(templates)
            text = tmpl.format(kw=kw)
            dataset.append(
                {
                    "text": text,
                    "label": label,
                    "intent_id": info.get("id", 0),
                }
            )
    rng.shuffle(dataset)
    return dataset


__all__ = [
    "CALL_CENTER_INTENTS",
    "INTENT_LABELS",
    "IntentClassifier",
    "generate_intent_dataset",
]
