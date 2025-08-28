from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

import typer
import pandas as pd
from rich.console import Console
from rich.table import Table
from .sentiment import load as load_sentiment
from .profiles import resolve_profile
from .clean import clean_texts
from .lexicon import load_lexicon, score_text, scalar_to_dist, blend_distributions

console = Console()

DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


# load_pipeline removed; using src.sentiment.load


# predict removed; handled inside SentimentPipeline.analyze


def ensure_dir(path: str):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


# normalize_label removed; normalization done in src.sentiment


def main(
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
    model: Optional[str] = typer.Option(
        None, "--model", help="Hugging Face-modell att använda (standard väljs via profil)"
    ),
    batch_size: int = typer.Option(16, help="Batch-storlek för inferens"),
    max_rows: Optional[int] = typer.Option(
        None, help="Analysera högst N rader (debug/snabbtest)"
    ),
    output: Optional[str] = typer.Option(
        None, help="Spara resultat till CSV (t.ex. outputs/predictions.csv)"
    ),
    device: Optional[str] = typer.Option(
        "auto", help="Enhet: 'auto' (default), 'cpu', 'cuda', 'cuda:0', 'mps'"
    ),
    return_all_scores: bool = typer.Option(
        False, "--return-all-scores", help="Returnera sannolikheter för alla klasser"
    ),
    max_length: Optional[int] = typer.Option(
        None, help="Max token-längd vid inferens (om ej satt används profilens)"
    ),
    datatype: Optional[str] = typer.Option(
        None, "--datatype", help="Datatyp: t.ex. 'post', 'comment', 'article', 'review'"
    ),
    source: Optional[str] = typer.Option(
        None, "--source", help="Källa: t.ex. 'forum', 'magazine', 'news', 'social'"
    ),
    profile: Optional[str] = typer.Option(
        None, "--profile", help="Profil att använda (åsidolägger datatype/source). T.ex. 'forum', 'magazine'"
    ),
    lexicon_file: Optional[str] = typer.Option(
        None, "--lexicon-file", help="Sökväg till svenskt lexikon (CSV/TSV) med kolumner term|word och polarity|score|sentiment"
    ),
    lexicon_weight: float = typer.Option(
        0.0, "--lexicon-weight", min=0.0, max=1.0, help="Vikt för lexikon-blandning [0..1]. 0=inaktiverad"
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

    # 2) Välj profil och förbered
    profile_name, spec = resolve_profile(datatype=datatype, source=source, profile=profile)
    chosen_model = model or spec.get("model", DEFAULT_MODEL)
    resolved_max_length = max_length or spec.get("max_length", 256)

    # Rengör texter enligt profil
    texts = clean_texts(texts, spec.get("cleaning", {}))

    # 3) Ladda modell
    console.print(f"[green]Profil:[/green] {profile_name}")
    console.print(f"[green]Laddar modell:[/green] {chosen_model}")
    try:
        sp = load_sentiment(
            chosen_model,
            device=device,
            return_all_scores=return_all_scores,
            max_length=resolved_max_length,
        )
    except Exception as e:
        console.print(f"[red]Kunde inte ladda modellen '{chosen_model}': {e}[/red]")
        raise typer.Exit(code=2)

    # 4) Kör inferens
    console.print(f"[green]Analyserar {len(texts)} texter...[/green]")
    try:
        results = sp.analyze(
            texts,
            batch_size=batch_size,
            return_all_scores=return_all_scores,
            max_length=resolved_max_length,
        )
    except Exception as e:
        console.print(f"[red]Fel under inferens: {e}[/red]")
        raise typer.Exit(code=2)

    # 5) Lexikon (valfritt)
    lex = None
    use_lex = lexicon_file is not None and lexicon_weight > 0.0
    if use_lex:
        try:
            lex = load_lexicon(lexicon_file)
            console.print(f"[green]Lexikon laddat:[/green] {lexicon_file} ({len(lex)} termer)")
        except Exception as e:
            console.print(f"[yellow]Varning: kunde inte ladda lexikon '{lexicon_file}': {e}. Fortsätter utan lexikon.[/yellow]")
            use_lex = False

    # 6) Paketera resultat
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    rows = []
    if results and isinstance(results[0], list):
        # Expand to per-class columns
        for t, inner in zip(texts, results):
            # inner: List[{"label": 'negativ|neutral|positiv', 'score': float}]
            scores = {e.get("label"): float(e.get("score", 0.0)) for e in inner}
            # Ensure keys
            for k in ["negativ", "neutral", "positiv"]:
                scores.setdefault(k, 0.0)
            # Optional blending with lexicon
            if use_lex and lex is not None:
                s_scalar = score_text(t, lex)
                ln, le, lp = scalar_to_dist(s_scalar)
                scores = blend_distributions(scores, (ln, le, lp), lexicon_weight)
            # Top prediction
            top_label = max(scores.items(), key=lambda kv: kv[1])[0]
            top_score = scores[top_label]
            rows.append({
                "text": t,
                "label": top_label,
                "score": float(top_score),
                "negativ": scores["negativ"],
                "neutral": scores["neutral"],
                "positiv": scores["positiv"],
                "model": chosen_model,
                "profile": profile_name,
                "timestamp": now_iso,
            })
    else:
        for t, r in zip(texts, results):
            label = r.get("label")
            score = float(r.get("score", 0.0))
            # If lexicon blending is requested but we don't have a full distribution,
            # approximate a distribution from the top-1 by placing mass on the label.
            neg = neu = pos = 0.0
            if label == "negativ":
                neg = score
            elif label == "neutral":
                neu = score
            else:
                pos = score
            # Normalize to sum=1 if score<1 (heuristic)
            ssum = neg + neu + pos
            if ssum <= 0:
                neu = 1.0
                ssum = 1.0
            model_dist = {"negativ": neg/ssum, "neutral": neu/ssum, "positiv": pos/ssum}
            if use_lex and lex is not None:
                s_scalar = score_text(t, lex)
                ln, le, lp = scalar_to_dist(s_scalar)
                blended = blend_distributions(model_dist, (ln, le, lp), lexicon_weight)
                # update label/score by top of blended
                label = max(blended.items(), key=lambda kv: kv[1])[0]
                score = float(blended[label])
            rows.append({
                "text": t,
                "label": label,
                "score": score,
                "model": chosen_model,
                "profile": profile_name,
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
    typer.run(main)
