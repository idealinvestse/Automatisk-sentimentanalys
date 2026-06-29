"""Benchmark intent classifier backends (heuristic vs fine-tuned model).

Usage:
    python scripts/benchmark_intent.py --backend heuristic --val-file data/intent_val.jsonl
    python scripts/benchmark_intent.py --backend model --model-path models/intent_classifier
    python scripts/benchmark_intent.py --backend both --output reports/intent_baseline.json
    python scripts/benchmark_intent.py --val-file data/intent_val.jsonl --min-macro-f1 0.75
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from src.intent import CALL_CENTER_INTENTS, IntentClassifier

logger = logging.getLogger(__name__)

DEFAULT_DATA = Path("data/intent_train.jsonl")
DEFAULT_VAL = Path("data/intent_val.jsonl")
DEFAULT_OUTPUT = Path("reports/intent_baseline.json")
LABEL_ORDER = sorted(CALL_CENTER_INTENTS.keys(), key=lambda k: CALL_CENTER_INTENTS[k]["id"])


def load_intent_jsonl(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            texts.append(str(item["text"]))
            labels.append(str(item["intent"]))
    return texts, labels


def per_class_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    report = classification_report(
        y_true, y_pred, labels=LABEL_ORDER, output_dict=True, zero_division=0
    )
    per_class = {
        label: {
            "precision": round(float(report[label]["precision"]), 4),
            "recall": round(float(report[label]["recall"]), 4),
            "f1": round(float(report[label]["f1-score"]), 4),
            "support": int(report[label]["support"]),
        }
        for label in LABEL_ORDER
        if label in report
    }
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    return {
        "per_class": per_class,
        "confusion_matrix": {
            "labels": LABEL_ORDER,
            "matrix": cm.tolist(),
        },
    }


def benchmark_backend(
    texts: list[str],
    labels: list[str],
    *,
    backend: str,
    model_path: str | None = None,
    device: str = "cpu",
) -> dict:
    clf = IntentClassifier(backend=backend, model_path=model_path, device=device)
    t0 = time.perf_counter()
    preds = [clf.classify(t)[0] for t in texts]
    elapsed = time.perf_counter() - t0
    per_item_ms = (elapsed / max(len(texts), 1)) * 1000.0

    result = {
        "backend": backend,
        "model_path": model_path,
        "n_samples": len(texts),
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "f1_macro": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
        "latency_ms_p50": round(per_item_ms, 3),
        "latency_total_s": round(elapsed, 3),
    }
    result.update(per_class_metrics(labels, preds))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark intent classification backends")
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA, help="Train file for random split"
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=None,
        help="Fixed validation JSONL (preferred for CI)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backend", choices=["heuristic", "model", "both"], default="heuristic")
    parser.add_argument(
        "--model-path", default=os.getenv("INTENT_MODEL_PATH", "models/intent_classifier")
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--min-macro-f1",
        type=float,
        default=None,
        help="Exit 1 if heuristic macro F1 below threshold",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.val_file:
        if not args.val_file.is_file():
            raise SystemExit(f"Validation file not found: {args.val_file}")
        x_test, y_test = load_intent_jsonl(args.val_file)
        testset_ref = str(args.val_file)
        holdout_note = "fixed_val_file"
    else:
        if not args.data.is_file():
            raise SystemExit(f"Dataset not found: {args.data}")
        texts, labels = load_intent_jsonl(args.data)
        _, x_test, _, y_test = train_test_split(
            texts,
            labels,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=labels,
        )
        testset_ref = str(args.data)
        holdout_note = f"random_split_{args.test_size}"

    backends = ["heuristic", "model"] if args.backend == "both" else [args.backend]
    results: dict = {
        "testset": testset_ref,
        "holdout": holdout_note,
        "seed": args.seed,
        "n_test": len(x_test),
        "backends": {},
        "note": "Macro F1 is primary metric; fixed val file preferred for CI.",
    }

    exit_code = 0
    for backend in backends:
        model_path = args.model_path if backend == "model" else None
        if backend == "model" and not Path(model_path or "").exists():
            logger.warning("Model path missing (%s); skipping model backend", model_path)
            results["backends"][backend] = {"skipped": True, "reason": "model_path_missing"}
            continue
        logger.info("Benchmarking %s on %d samples", backend, len(x_test))
        metrics = benchmark_backend(
            x_test,
            y_test,
            backend=backend,
            model_path=model_path,
            device=args.device,
        )
        results["backends"][backend] = metrics
        if (
            args.min_macro_f1 is not None
            and backend == "heuristic"
            and metrics["f1_macro"] < args.min_macro_f1
        ):
            logger.error(
                "Heuristic macro F1 %.4f below threshold %.4f",
                metrics["f1_macro"],
                args.min_macro_f1,
            )
            exit_code = 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
