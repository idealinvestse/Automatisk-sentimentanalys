"""Benchmark intent classifier backends (heuristic vs fine-tuned model).

Usage:
    python scripts/benchmark_intent.py --backend heuristic
    python scripts/benchmark_intent.py --backend model --model-path models/intent_classifier
    python scripts/benchmark_intent.py --backend both --output reports/intent_baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.intent import IntentClassifier

logger = logging.getLogger(__name__)

DEFAULT_DATA = Path("data/intent_train.jsonl")
DEFAULT_OUTPUT = Path("reports/intent_baseline.json")


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

    return {
        "backend": backend,
        "model_path": model_path,
        "n_samples": len(texts),
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "f1_macro": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
        "latency_ms_p50": round(per_item_ms, 3),
        "latency_total_s": round(elapsed, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark intent classification backends")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backend", choices=["heuristic", "model", "both"], default="heuristic")
    parser.add_argument("--model-path", default=os.getenv("INTENT_MODEL_PATH", "models/intent_classifier"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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

    backends = ["heuristic", "model"] if args.backend == "both" else [args.backend]
    results: dict = {
        "testset": str(args.data),
        "holdout_ratio": args.test_size,
        "seed": args.seed,
        "n_test": len(x_test),
        "backends": {},
        "note": "Holdout benchmark; does not change pipeline default backend.",
    }

    for backend in backends:
        model_path = args.model_path if backend == "model" else None
        if backend == "model" and not Path(model_path or "").exists():
            logger.warning("Model path missing (%s); skipping model backend", model_path)
            results["backends"][backend] = {"skipped": True, "reason": "model_path_missing"}
            continue
        logger.info("Benchmarking %s on %d samples", backend, len(x_test))
        results["backends"][backend] = benchmark_backend(
            x_test,
            y_test,
            backend=backend,
            model_path=model_path,
            device=args.device,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
