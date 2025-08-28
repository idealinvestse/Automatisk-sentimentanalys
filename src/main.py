from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import List, Optional

import typer
import pandas as pd
from transformers import pipeline
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()

DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


def load_pipeline(model_name: str):
    try:
        nlp = pipeline(
            task="sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
        )
        return nlp
    except Exception as e:
        console.print(f"[red]Kunde inte ladda modellen '{model_name}': {e}[/red]")
        raise typer.Exit(code=2)


def predict(nlp, texts: List[str], batch_size: int = 16):
    # transformers pipeline supports batching internally
    return nlp(texts, batch_size=batch_size, truncation=True)


def ensure_dir(path: str):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def normalize_label(label: str) -> str:
    l = str(label).strip().lower()
    if l in {"label_0", "negative", "neg"}:
        return "negativ"
    if l in {"label_1", "neutral"}:
        return "neutral"
    if l in {"label_2", "positive", "pos"}:
        return "positiv"
    return label


@app.command()
def run(
    text: Optional[str] = typer.Option(
        None, help="Analysera en enskild text"
    ),
    txt_file: Optional[str] = typer.Option(
        None, "--txt-file", help="Sökväg till .txt (en text per rad)"
    ),
    csv_file: Optional[str] = typer.Option(
        None, "--csv-file", help="Sökväg till .csv med texter"
    ),
    text_column: str = typer.Option(
        "text", help="Kolumnnamn i CSV som innehåller text"
    ),
    model: str = typer.Option(
        DEFAULT_MODEL, "--model", help="Hugging Face-modell att använda"
    ),
    batch_size: int = typer.Option(16, help="Batch-storlek för inferens"),
    max_rows: Optional[int] = typer.Option(
        None, help="Analysera högst N rader (debug/snabbtest)"
    ),
    output: Optional[str] = typer.Option(
        None, help="Spara resultat till CSV (t.ex. outputs/predictions.csv)"
    ),
):
    """Kör svensk sentimentanalys från text, .txt eller .csv"""

    sources = sum([
        1 if text is not None else 0,
        1 if txt_file is not None else 0,
        1 if csv_file is not None else 0,
    ])
    if sources == 0:
        console.print("[yellow]Ange en källa: --text, --txt-file eller --csv-file[/yellow]")
        raise typer.Exit(code=1)
    if sources > 1:
        console.print("[red]Ange endast EN av --text, --txt-file eller --csv-file[/red]")
        raise typer.Exit(code=1)

    # 1) Läs in texter
    texts: List[str] = []
    if text is not None:
        texts = [text.strip()]
    elif txt_file is not None:
        if not os.path.isfile(txt_file):
            console.print(f"[red]Hittar inte txt-fil: {txt_file}[/red]")
            raise typer.Exit(code=1)
        with open(txt_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    elif csv_file is not None:
        if not os.path.isfile(csv_file):
            console.print(f"[red]Hittar inte csv-fil: {csv_file}[/red]")
            raise typer.Exit(code=1)
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            console.print(f"[red]Kunde inte läsa CSV: {e}[/red]")
            raise typer.Exit(code=1)
        if text_column not in df.columns:
            console.print(
                f"[red]Kolumn '{text_column}' finns inte i CSV. Tillgängliga kolumner: {list(df.columns)}[/red]"
            )
            raise typer.Exit(code=1)
        if max_rows is not None:
            df = df.head(max_rows)
        texts = df[text_column].astype(str).fillna("").str.strip().tolist()

    if max_rows is not None and text is None and txt_file is not None:
        texts = texts[:max_rows]

    if not texts:
        console.print("[red]Inga texter att analysera.[/red]")
        raise typer.Exit(code=1)

    # 2) Ladda modell
    console.print(f"[green]Laddar modell:[/green] {model}")
    nlp = load_pipeline(model)

    # 3) Kör inferens
    console.print(f"[green]Analyserar {len(texts)} texter...[/green]")
    try:
        results = predict(nlp, texts, batch_size=batch_size)
    except Exception as e:
        console.print(f"[red]Fel under inferens: {e}[/red]")
        raise typer.Exit(code=2)

    # 4) Paketera resultat
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    rows = []
    for t, r in zip(texts, results):
        rows.append({
            "text": t,
            "label": normalize_label(r.get("label")),
            "score": float(r.get("score", 0.0)),
            "model": model,
            "timestamp": now_iso,
        })
    out_df = pd.DataFrame(rows)

    # 5) Output
    if output:
        if not output.lower().endswith(".csv"):
            output = output + ".csv"
        ensure_dir(output)
        try:
            out_df.to_csv(output, index=False)
            console.print(f"[green]Sparat:[/green] {output}")
        except Exception as e:
            console.print(f"[red]Kunde inte spara CSV: {e}[/red]")
            raise typer.Exit(code=1)
    else:
        # visa några rader snyggt i terminal
        head_df = out_df.head(20)
        table = Table(show_header=True, header_style="bold magenta")
        for col in head_df.columns:
            table.add_column(col)
        for _, row in head_df.iterrows():
            table.add_row(*(str(row[c]) for c in head_df.columns))
        console.print(table)
        if len(out_df) > len(head_df):
            console.print(f"... visade 20 av {len(out_df)} rader. Använd --output för att spara allt.")


if __name__ == "__main__":
    app()
