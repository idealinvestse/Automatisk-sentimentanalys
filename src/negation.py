"""Swedish negation detection and handling for sentiment analysis.

Handles:
    - Simple negation terms (inte, ej, aldrig, icke, knappast)
    - Multi-word negations (absolut inte, inte alls, inte längre)
    - Negation scope (window-based)
    - Polarity flipping for lexicon scores

Usage:
    from src.negation import detect_negation, flip_polarity, apply_negation_heuristic
"""

from __future__ import annotations

# Core negation terms
NEGATION_TERMS: set[str] = {"inte", "ej", "aldrig", "icke", "knappast"}

# Multi-word negation patterns (word1, word2)
MULTI_WORD_NEGATIONS: list[tuple[str, str]] = [
    ("absolut", "inte"),
    ("inte", "alls"),
    ("inte", "längre"),
    ("aldrig", "mer"),
    ("aldrig", "någonsin"),
    ("inte", "heller"),
    ("varken", "eller"),
]

# Words that can reinforce or weaken negation
NEGATION_REINFORCERS: set[str] = {"alls", "alls.", "heller", "någonsin"}
NEGATION_WEAKENERS: set[str] = {"kanske", "möjligen", "eventuellt"}


def tokenize_lower(text: str) -> list[str]:
    """Tokenize text into lowercase words, stripping punctuation."""
    return [tok.strip(".,!?;:()[]\"'").lower() for tok in str(text).split()]


def detect_negation(text: str, window: int = 3) -> bool:
    """Detect whether a Swedish negation appears in the text.

    Checks both single-term and multi-word negations.

    Args:
        text: Input Swedish text.
        window: Number of words after negation to check for sentiment terms.

    Returns:
        True if a negation is detected with sentiment-bearing words in scope.
    """
    tokens = tokenize_lower(text)
    n = len(tokens)

    for i, token in enumerate(tokens):
        # Single-word negation
        if token in NEGATION_TERMS:
            # Check if there are sentiment-bearing words within the window
            scope = tokens[i + 1 : i + 1 + window]
            if scope:
                return True

        # Multi-word negation
        if i + 1 < n:
            pair = (token, tokens[i + 1])
            if pair in MULTI_WORD_NEGATIONS:
                scope = tokens[i + 2 : i + 2 + window]
                if scope:
                    return True

    return False


def detect_negation_with_position(text: str, window: int = 3) -> list[tuple[int, str]]:
    """Detect negations and return their positions and types.

    Returns:
        List of (token_index, negation_type) tuples.
    """
    tokens = tokenize_lower(text)
    hits: list[tuple[int, str]] = []
    n = len(tokens)

    for i, token in enumerate(tokens):
        if token in NEGATION_TERMS:
            scope = tokens[i + 1 : i + 1 + window]
            if scope:
                hits.append((i, "single"))
        if i + 1 < n:
            pair = (token, tokens[i + 1])
            if pair in MULTI_WORD_NEGATIONS:
                scope = tokens[i + 2 : i + 2 + window]
                if scope:
                    hits.append((i, "multi"))

    return hits


def flip_polarity(score: float, strength: float = 0.8) -> float:
    """Flip a sentiment polarity score for negation.

    Args:
        score: Original sentiment score in [-1, 1].
        strength: How strongly to flip (0 = no flip, 1 = full flip).

    Returns:
        Flipped score.
    """
    return -score * strength


def apply_negation_heuristic(
    text: str,
    dist: dict[str, float],
    window: int = 3,
    flip_strength: float = 0.7,
) -> dict[str, float]:
    """Apply rule-based negation handling to a sentiment distribution.

    Swaps positive and negative probabilities when negation is detected
    near sentiment-bearing phrases.

    Args:
        text: Input Swedish text.
        dist: Sentiment distribution {'negativ': p, 'neutral': p, 'positiv': p}.
        window: Negation scope window in tokens.
        flip_strength: How strongly to flip polarity [0, 1].

    Returns:
        Adjusted distribution (normalized to sum=1).
    """
    if not detect_negation(text, window):
        return dict(dist)

    # Swap positive and negative with blending
    neg = dist.get("negativ", 0.0)
    pos = dist.get("positiv", 0.0)
    neu = dist.get("neutral", 0.0)

    adjusted = {
        "negativ": (1.0 - flip_strength) * neg + flip_strength * pos,
        "neutral": neu,
        "positiv": (1.0 - flip_strength) * pos + flip_strength * neg,
    }

    total = sum(adjusted.values())
    return {k: v / total for k, v in adjusted.items()} if total > 0 else adjusted


def is_negated_example(text: str) -> bool:
    """Check if a text contains an explicit negation (for evaluation purposes)."""
    tokens = tokenize_lower(text)
    for tok in tokens:
        if tok in NEGATION_TERMS:
            return True
    return any((tokens[i], tokens[i + 1]) in MULTI_WORD_NEGATIONS for i in range(len(tokens) - 1))


__all__ = [
    "NEGATION_TERMS",
    "MULTI_WORD_NEGATIONS",
    "detect_negation",
    "detect_negation_with_position",
    "flip_polarity",
    "apply_negation_heuristic",
    "is_negated_example",
    "tokenize_lower",
]
