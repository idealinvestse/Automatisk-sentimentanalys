"""Fine-tune a Swedish BERT model for call center intent classification.

Usage:
    python scripts/train_intent.py \
        --train data/intent_train.jsonl \
        --output models/intent_classifier \
        --base-model KBLab/bert-base-swedish-cased \
        --epochs 5

Requires: transformers, datasets, peft, accelerate, scikit-learn
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add project root for imports (before local imports)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intent import INTENT_LABELS  # noqa: E402

logger = logging.getLogger(__name__)

LABEL2ID = {label: idx for idx, label in enumerate(INTENT_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML training configuration without loading model dependencies."""
    import yaml

    with Path(path).open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a mapping: {path}")
    return config


def corpus_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a training corpus."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path: str) -> tuple[list[str], list[int]]:
    """Load JSONL intent training data."""
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(LABEL2ID.get(item["intent"], LABEL2ID["other"]))
    return texts, labels


def compute_metrics(eval_pred):
    """Compute classification metrics for Trainer."""
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Swedish BERT for intent classification")
    parser.add_argument("--config", help="Optional YAML config; CLI arguments override it")
    parser.add_argument("--train", help="Path to training JSONL")
    parser.add_argument("--output", help="Output directory for model")
    parser.add_argument(
        "--base-model",
        default=None,
        help="HuggingFace model name",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=None, help="Max token length")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--eval-split", type=float, default=None, help="Validation split ratio")
    parser.add_argument("--early-stopping", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--device", default=None, help="Device (cpu/cuda/auto)")
    parser.add_argument(
        "--val-file", default=None, help="Fixed validation JSONL for final evaluation"
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else {}

    def setting(name: str, default: Any = None) -> Any:
        value = getattr(args, name.replace("-", "_"), None)
        return value if value is not None else config.get(name, default)

    train_path = setting("train", "data/intent_train.jsonl")
    output_path = setting("output", "models/intent_classifier")
    base_model = setting("base_model", "KBLab/bert-base-swedish-cased")
    epochs = int(setting("epochs", 5))
    batch_size = int(setting("batch_size", 16))
    learning_rate = float(setting("lr", 5e-5))
    max_length = int(setting("max_length", 128))
    seed = int(setting("seed", 42))
    eval_split = float(setting("eval_split", 0.1))
    early_stopping = int(setting("early_stopping", 3))
    device_arg = setting("device", "auto")
    val_file = setting("val_file")
    max_train_samples = setting("max_train_samples")
    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)

    import torch
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    logger.info("Using device: %s", device)

    # Load data
    logger.info("Loading data from %s", train_path)
    texts, labels = load_data(train_path)
    if max_train_samples is not None:
        if max_train_samples <= 0:
            raise ValueError("--max-train-samples must be positive")
        texts, labels = texts[:max_train_samples], labels[:max_train_samples]
    logger.info("Loaded %d examples", len(texts))

    # Label distribution
    from collections import Counter

    dist = Counter(labels)
    logger.info("Label distribution: %s", {ID2LABEL[k]: v for k, v in sorted(dist.items())})

    # Train/val split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=eval_split, random_state=seed, stratify=labels
    )

    # Tokenizer & model
    logger.info("Loading model %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(INTENT_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Training args
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "logging_dir": str(output_dir / "logs"),
        "logging_steps": 50,
        "seed": seed,
        "remove_unused_columns": False,
    }
    if "eval_strategy" in TrainingArguments.__dataclass_fields__:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping)],
    )

    logger.info("Starting training...")
    trainer.train()

    # Save
    logger.info("Saving model to %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Final eval
    metrics = trainer.evaluate()
    logger.info("Final metrics: %s", metrics)

    # Save metrics
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fixed_val_metrics = None
    if val_file:
        from scripts.benchmark_intent import benchmark_backend, load_intent_jsonl

        val_texts, val_labels = load_intent_jsonl(Path(val_file))
        fixed_val_metrics = benchmark_backend(
            val_texts,
            val_labels,
            backend="model",
            model_path=str(output_dir),
            device=device,
        )
        report_path = Path("reports/intent_model_eval.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(fixed_val_metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    metadata = {
        "base_model": base_model,
        "seed": seed,
        "train_file": str(train_path),
        "train_sha256": corpus_sha256(train_path),
        "val_file": str(val_file) if val_file else None,
        "fixed_val_metrics": fixed_val_metrics,
        "config": config,
        "trained_at_utc": datetime.now(UTC).isoformat(),
    }
    (output_dir / "training_meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Done! Model saved to %s", output_dir)


if __name__ == "__main__":
    main()
