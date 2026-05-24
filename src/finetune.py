"""LoRA/PEFT fine-tuning pipeline for Swedish call center sentiment.

Usage:
    python -m src.finetune --config configs/finetune.yaml

The implementation is intentionally lightweight: imports for training-only
dependencies are lazy so the normal CLI/API can run without PEFT installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
import typer
from rich.console import Console

LABEL_TO_ID = {"negativ": 0, "neutral": 1, "positiv": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

app = typer.Typer(help="Fine-tune Swedish sentiment models with LoRA/PEFT")
console = Console()


@dataclass
class FinetuneConfig:
    """Training configuration loaded from YAML."""

    model_name: str
    train_file: str
    eval_file: str
    output_dir: str
    text_column: str = "text"
    label_column: str = "label"
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    max_length: int = 256
    early_stopping_patience: int = 2
    lora: dict[str, Any] | None = None


def load_config(path: str) -> FinetuneConfig:
    """Load and validate a fine-tuning config."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency optional in runtime env
        raise RuntimeError(
            "Install PyYAML to load fine-tuning configs: pip install pyyaml"
        ) from exc

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return FinetuneConfig(**raw)


def load_labelled_csv(
    path: str, text_column: str = "text", label_column: str = "label"
) -> pd.DataFrame:
    """Load a labelled CSV and normalize Swedish sentiment labels."""
    df = pd.read_csv(path)
    missing = {text_column, label_column} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    df = df[[text_column, label_column]].dropna()
    df[label_column] = df[label_column].astype(str).str.strip().str.lower()
    unknown = set(df[label_column]) - set(LABEL_TO_ID)
    if unknown:
        raise ValueError(f"Unknown labels in {path}: {sorted(unknown)}")
    return df


def build_dataset(cfg: FinetuneConfig):
    """Build Hugging Face datasets from configured CSV files."""
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:  # pragma: no cover - dependency optional in runtime env
        raise RuntimeError("Install datasets to run fine-tuning: pip install datasets") from exc

    train_df = load_labelled_csv(cfg.train_file, cfg.text_column, cfg.label_column)
    eval_df = load_labelled_csv(cfg.eval_file, cfg.text_column, cfg.label_column)
    train_df["label"] = train_df[cfg.label_column].map(LABEL_TO_ID)
    eval_df["label"] = eval_df[cfg.label_column].map(LABEL_TO_ID)
    train_df = train_df.rename(columns={cfg.text_column: "text"})
    eval_df = eval_df.rename(columns={cfg.text_column: "text"})
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False),
            "validation": Dataset.from_pandas(eval_df[["text", "label"]], preserve_index=False),
        }
    )


def train(cfg: FinetuneConfig) -> str:
    """Run LoRA fine-tuning and return the output directory."""
    try:
        import numpy as np
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - dependency optional in runtime env
        raise RuntimeError(
            "Install training dependencies: pip install transformers peft datasets accelerate scikit-learn"
        ) from exc

    dataset = build_dataset(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"], truncation=True, max_length=cfg.max_length, padding="max_length"
        )

    tokenized = dataset.map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    lora_cfg = cfg.lora or {}
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=lora_cfg.get("target_modules"),
    )
    model = get_peft_model(model, peft_cfg)

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=20,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )
    trainer.train()
    os.makedirs(cfg.output_dir, exist_ok=True)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    return cfg.output_dir


@app.command()
def main(config: str = typer.Option("configs/finetune.yaml", "--config")) -> None:
    """Run PEFT/LoRA fine-tuning from a YAML config."""
    cfg = load_config(config)
    console.print(f"[cyan]Fine-tuning model:[/cyan] {cfg.model_name}")
    out = train(cfg)
    console.print(f"[green]Saved fine-tuned adapter/model:[/green] {out}")


if __name__ == "__main__":
    app()
