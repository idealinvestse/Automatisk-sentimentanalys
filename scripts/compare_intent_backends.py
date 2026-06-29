"""Compare intent heuristic vs model backend on fixed validation set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_intent import benchmark_backend, load_intent_jsonl

DEFAULT_CONFIG = Path("configs/analyzer_eval.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B intent backends on val set")
    parser.add_argument("--val-file", type=Path, default=Path("data/intent_val.jsonl"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model-path", default=os.getenv("INTENT_MODEL_PATH", "models/intent_classifier")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/intent_backend_comparison.json")
    )
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    intent_cfg = cfg.get("intent", {})
    min_gain = float(intent_cfg.get("model_switch_min_f1_gain", 0.05))

    texts, labels = load_intent_jsonl(args.val_file)
    heur = benchmark_backend(texts, labels, backend="heuristic")
    report: dict = {"heuristic": heur, "model": None, "recommendation": "heuristic"}

    if Path(args.model_path).is_dir():
        model = benchmark_backend(texts, labels, backend="model", model_path=args.model_path)
        report["model"] = model
        gain = model["f1_macro"] - heur["f1_macro"]
        report["f1_macro_gain"] = round(gain, 4)
        if gain >= min_gain:
            report["recommendation"] = "model"
            print(f"Model beats heuristic by {gain:.4f} F1 — consider switching default")
        else:
            print(f"Keep heuristic default (gain {gain:.4f} < {min_gain})")
    else:
        print(f"Model path missing ({args.model_path}); heuristic-only comparison")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
