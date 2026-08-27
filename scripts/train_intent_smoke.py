#!/usr/bin/env python3
"""Smoke-train a tiny sklearn intent classifier for local promotion checks.

This is intentionally lighter than ``train_intent.py`` (BERT/PEFT) so CI and
laptops can produce ``models/intent_classifier_smoke/`` and run
``benchmark_intent.py --backend model`` without a GPU.

Usage:
    python scripts/train_intent_smoke.py
    python scripts/benchmark_intent.py --backend model \\
        --model-path models/intent_classifier_smoke --min-macro-f1 0.70
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-train sklearn intent classifier")
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "intent_train.jsonl")
    parser.add_argument("--val", type=Path, default=ROOT / "data" / "intent_val.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "intent_classifier_smoke",
    )
    args = parser.parse_args()

    if not args.train.is_file():
        print(f"FAIL: missing train file {args.train}", file=sys.stderr)
        print("Run: python scripts/prepare_intent_data.py --per-intent 120 --val-ratio 0.25")
        return 1

    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import Pipeline

    def load(path: Path) -> tuple[list[str], list[str]]:
        texts, labels = [], []
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(row["intent"])
        return texts, labels

    x_train, y_train = load(args.train)
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            ),
        ]
    )
    pipe.fit(x_train, y_train)

    metrics: dict = {"n_train": len(x_train)}
    if args.val.is_file():
        x_val, y_val = load(args.val)
        preds = pipe.predict(x_val)
        metrics["n_val"] = len(x_val)
        metrics["macro_f1"] = round(float(f1_score(y_val, preds, average="macro")), 4)

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "model.joblib"
    joblib.dump(pipe, model_path)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    # Marker so IntentClassifier can discover a sklearn backend if supported
    (args.output / "backend.txt").write_text("sklearn_tfidf\n", encoding="utf-8")
    print(f"OK: wrote {model_path}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
