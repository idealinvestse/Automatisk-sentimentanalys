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
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Add project root for imports (before local imports)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intent import INTENT_LABELS  # noqa: E402

logger = logging.getLogger(__name__)

LABEL2ID = {label: idx for idx, label in enumerate(INTENT_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


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
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Swedish BERT for intent classification"
    )
    parser.add_argument("--train", required=True, help="Path to training JSONL")
    parser.add_argument("--output", required=True, help="Output directory for model")
    parser.add_argument(
        "--base-model",
        default="KBLab/bert-base-swedish-cased",
        help="HuggingFace model name",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=128, help="Max token length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--early-stopping", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--device", default="auto", help="Device (cpu/cuda/auto)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    # Load data
    logger.info("Loading data from %s", args.train)
    texts, labels = load_data(args.train)
    logger.info("Loaded %d examples", len(texts))

    # Label distribution
    from collections import Counter

    dist = Counter(labels)
    logger.info("Label distribution: %s", {ID2LABEL[k]: v for k, v in sorted(dist.items())})

    # Train/val split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=args.eval_split, random_state=args.seed, stratify=labels
    )

    # Tokenizer & model
    logger.info("Loading model %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(INTENT_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Training args
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping)],
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

    logger.info("Done! Model saved to %s", output_dir)


if __name__ == "__main__":
    main()
