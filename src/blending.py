"""Learned blending for model + lexicon sentiment distributions.

Implements a lightweight logistic regression that learns optimal blending weights
from labelled data. Falls back to a heuristic blend when training data is unavailable.

Usage:
    from src.blending import LearnedBlender
    blender = LearnedBlender()
    blender.fit(model_dists, lexicon_dists, labels)   # optional training
    blended = blender.blend(model_dist, lexicon_tuple)
"""

from __future__ import annotations

import json
import os
from typing import Any


class LearnedBlender:
    """Blend model and lexicon distributions using learned or heuristic weights.

    The blender stores a weight vector [w_neg, w_neu, w_pos] where each w_i
    represents how much to trust the lexicon for class i vs the model.
    When untrained, uses a sensible default based on Swedish call center heuristics.
    """

    def __init__(
        self,
        default_lexicon_weight: float = 0.25,
        weight_path: str | None = None,
    ) -> None:
        """Initialize the blender.

        Args:
            default_lexicon_weight: Heuristic fallback weight [0, 1] when untrained.
            weight_path: Optional path to saved weights (JSON or pickle).
        """
        self.default_lexicon_weight = max(0.0, min(1.0, default_lexicon_weight))
        self.weights: dict[str, float] = {
            "negativ": self.default_lexicon_weight,
            "neutral": self.default_lexicon_weight,
            "positiv": self.default_lexicon_weight,
        }
        self._fitted = False

        if weight_path and os.path.isfile(weight_path):
            self.load(weight_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def blend(
        self,
        model_dist: dict[str, float],
        lexicon_tuple: tuple[float, float, float],
    ) -> dict[str, float]:
        """Blend model and lexicon distributions.

        Args:
            model_dist: {'negativ': p_neg, 'neutral': p_neu, 'positiv': p_pos}
            lexicon_tuple: (p_neg, p_neu, p_pos) from lexicon scalar_to_dist.

        Returns:
            Blended distribution dict with the same keys, normalized to sum=1.
        """
        labels = ["negativ", "neutral", "positiv"]
        ln, le, lp = lexicon_tuple

        out: dict[str, float] = {}
        for i, label in enumerate(labels):
            w = self.weights.get(label, self.default_lexicon_weight)
            lex_val = (ln, le, lp)[i]
            model_val = model_dist.get(label, 0.0)
            out[label] = (1.0 - w) * model_val + w * lex_val

        # Normalize
        total = sum(out.values())
        if total > 0:
            out = {k: v / total for k, v in out.items()}
        return out

    def fit(
        self,
        model_dists: list[dict[str, float]],
        lexicon_dists: list[tuple[float, float, float]],
        labels: list[str],
        learning_rate: float = 0.01,
        epochs: int = 100,
    ) -> dict[str, float]:
        """Learn per-class blending weights using gradient-free grid search.

        Robust fallback that works without sklearn. For a proper
        logistic-regression fit, use `fit_sklearn` when available.

        Args:
            model_dists: List of model output distributions.
            lexicon_dists: List of lexicon (neg, neu, pos) tuples.
            labels: True labels (negativ/neutral/positiv).
            learning_rate: Step size for weight updates.
            epochs: Number of optimization passes.

        Returns:
            The learned weight dict.
        """
        if len(model_dists) == 0 or len(model_dists) != len(lexicon_dists) != len(labels):
            return self.weights  # not enough data

        label_set = ["negativ", "neutral", "positiv"]

        # Initialize weights
        weights = {lbl: self.default_lexicon_weight for lbl in label_set}

        best_weights = dict(weights)
        best_acc = 0.0

        for _ in range(epochs):
            correct = 0
            for md, ld, true_label in zip(model_dists, lexicon_dists, labels, strict=False):
                # Blend with current weights
                blended: dict[str, float] = {}
                for i, lbl in enumerate(label_set):
                    w = weights[lbl]
                    blended[lbl] = (1.0 - w) * md.get(lbl, 0.0) + w * ld[i]
                total = sum(blended.values())
                if total > 0:
                    blended = {k: v / total for k, v in blended.items()}
                pred = max(blended.items(), key=lambda kv: kv[1])[0]
                if pred == true_label:
                    correct += 1

            acc = correct / len(labels)
            if acc > best_acc:
                best_acc = acc
                best_weights = dict(weights)

            # Perturb weights randomly and keep if accuracy improves
            for lbl in label_set:
                delta = (random() * 2 - 1) * learning_rate
                new_w = max(0.0, min(1.0, weights[lbl] + delta))

                # Evaluate new weight
                old_w = weights[lbl]
                weights[lbl] = new_w
                new_correct = 0
                for md, ld, true_label in zip(model_dists, lexicon_dists, labels, strict=False):
                    blended = {}
                    for i, lbl2 in enumerate(label_set):
                        w2 = weights[lbl2]
                        blended[lbl2] = (1.0 - w2) * md.get(lbl2, 0.0) + w2 * ld[i]
                    total = sum(blended.values())
                    if total > 0:
                        blended = {k: v / total for k, v in blended.items()}
                    if max(blended.items(), key=lambda kv: kv[1])[0] == true_label:
                        new_correct += 1

                if new_correct / len(labels) > acc:
                    acc = new_correct / len(labels)
                else:
                    weights[lbl] = old_w  # revert

            if acc > best_acc:
                best_acc = acc
                best_weights = dict(weights)

        self.weights = best_weights
        self._fitted = True
        return self.weights

    def fit_sklearn(
        self,
        model_dists: list[dict[str, float]],
        lexicon_dists: list[tuple[float, float, float]],
        labels: list[str],
        max_iter: int = 1000,
    ) -> dict[str, float]:
        """Learn blending weights using sklearn LogisticRegression.

        Fits a multinomial logistic regression over concatenated
        [model_probs, lexicon_probs] features to predict the label.
        The per-class weights are derived from the learned coefficients
        so that classes where the lexicon is more informative get a
        higher lexicon weight.

        Args:
            model_dists: List of model output distributions.
            lexicon_dists: List of lexicon (neg, neu, pos) tuples.
            labels: True labels (negativ/neutral/positiv).
            max_iter: Maximum iterations for LogisticRegression.

        Returns:
            The learned weight dict.
        """
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import LabelEncoder
        except ImportError as exc:
            raise ImportError(
                "sklearn is required for fit_sklearn. " "Install it with: pip install scikit-learn"
            ) from exc

        if len(model_dists) == 0 or len(model_dists) != len(lexicon_dists) != len(labels):
            return self.weights

        label_set = ["negativ", "neutral", "positiv"]

        # Build feature matrix: [model_neg, model_neu, model_pos, lex_neg, lex_neu, lex_pos]
        features_matrix: list[list[float]] = []
        for md, ld in zip(model_dists, lexicon_dists, strict=True):
            features = [md.get(lbl, 0.0) for lbl in label_set] + list(ld)
            features_matrix.append(features)

        x_arr = np.array(features_matrix)
        le = LabelEncoder()
        y = le.fit_transform(labels)

        clf = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=max_iter,
            random_state=42,
        )
        clf.fit(x_arr, y)

        # Coefficients shape: (n_classes, n_features=6)
        # For each class, compute how much the lexicon features (indices 3-5)
        # contribute relative to the model features (indices 0-2).
        coef = clf.coef_  # type: ignore[has-type]
        learned: dict[str, float] = {}
        for i, lbl in enumerate(label_set):
            model_sum = np.sum(np.abs(coef[i, :3]))
            lex_sum = np.sum(np.abs(coef[i, 3:]))
            total = model_sum + lex_sum
            if total > 0:
                learned[lbl] = float(np.clip(lex_sum / total, 0.0, 1.0))
            else:
                learned[lbl] = self.default_lexicon_weight

        self.weights = learned
        self._fitted = True
        return self.weights

    def save(self, path: str) -> None:
        """Persist learned weights to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data: dict[str, Any] = {
            "weights": self.weights,
            "default_lexicon_weight": self.default_lexicon_weight,
            "fitted": self._fitted,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load weights from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.weights = data.get("weights", self.weights)
        self.default_lexicon_weight = data.get(
            "default_lexicon_weight", self.default_lexicon_weight
        )
        self._fitted = data.get("fitted", False)

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# Module-level convenience
_DEFAULT_BLENDER: LearnedBlender | None = None


def get_blender(weight_path: str | None = None) -> LearnedBlender:
    """Return a module-level cached blender instance."""
    global _DEFAULT_BLENDER
    if _DEFAULT_BLENDER is None:
        _DEFAULT_BLENDER = LearnedBlender(weight_path=weight_path)
    return _DEFAULT_BLENDER


# Need random for the fit method
from random import random  # noqa: E402

__all__ = ["LearnedBlender", "get_blender"]
