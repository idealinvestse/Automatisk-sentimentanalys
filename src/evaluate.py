"""Utvärderingsramverk för svensk sentimentanalys.

Användning:
    python -m src.evaluate --testset data/test_swedish.csv --output reports/baseline.json
    python -m src.evaluate --testset data/test_swedish.csv --profile call --lexicon-weight 0.3
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .lexicon import load_lexicon, scalar_to_dist, score_text
from .profiles import AVAILABLE_PROFILES
from .sentiment import analyze_smart

SCENARIOS = {
    "forum": {"profile": "forum", "description": "Forum/sociala kommentarer"},
    "call": {"profile": "call", "description": "ASR-transkriberade kundtjänstsamtal"},
    "news": {"profile": "news", "description": "Nyhets-/artikeltext"},
}
ASR_BASELINES = [
    {
        "name": "OpenAI Whisper large-v3",
        "model": "openai/whisper-large-v3",
        "revision": None,
        "recommended_for": "Allmän flerspråkig ASR-baseline",
        "relative_swedish_wer": 1.0,
    },
    {
        "name": "KB-Whisper large strict",
        "model": "KBLab/kb-whisper-large",
        "revision": "strict",
        "recommended_for": "Svenska call center-samtal (verbatim)",
        "relative_swedish_wer": 0.53,
    },
]

app = typer.Typer(help="Utvärderingsramverk: kör sentimentanalys mot testset och beräkna metrics")
console = Console()

LABELS = ["negativ", "neutral", "positiv"]


def load_testset(path: str) -> pd.DataFrame:
    """Ladda testset från CSV med kolumner 'text', 'label'."""
    df = pd.read_csv(path, encoding="utf-8")
    for col in ("text", "label"):
        if col not in df.columns:
            raise ValueError(f"Testset måste ha kolumn '{col}'. Hittade: {list(df.columns)}")
    df["label"] = df["label"].str.strip().str.lower()
    unknown = set(df["label"].unique()) - set(LABELS)
    if unknown:
        console.print(
            f"[yellow]Varning: okända labels i testset: {unknown}. Endast {LABELS} stöds.[/yellow]"
        )
    return df


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    scores: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Beräkna accuracy, per-klass F1, macro-F1, och confusion matrix."""
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == p)
    accuracy = correct / n if n > 0 else 0.0

    # Per-class metrics
    per_class: dict[str, dict[str, float]] = {}
    for label in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for t in y_true if t == label),
        }

    macro_f1 = sum(per_class[lbl]["f1"] for lbl in LABELS) / 3.0

    # Confusion matrix
    cm: dict[str, dict[str, int]] = {t: {p: 0 for p in LABELS} for t in LABELS}
    for t, p in zip(y_true, y_pred, strict=False):
        if t in cm and p in cm[t]:
            cm[t][p] += 1

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "n_samples": n,
    }


def _heuristic_sentiment(
    texts: list[str],
) -> tuple[list[list[dict[str, float | str]]], dict[str, str | int]]:
    """Offline deterministic fallback using lexicon+clean+negation (improved baseline)."""
    from .clean import clean_texts
    from .profiles import resolve_profile
    try:
        lex = load_lexicon("data/sensaldo_lexicon.csv")
    except Exception:
        lex = {}
    _, spec = resolve_profile(profile="callcenter")
    proc = clean_texts(texts, spec.get("cleaning", {}))
    results: list[list[dict[str, float | str]]] = []
    for t in proc:
        s = score_text(t, lex)
        ln, le, lp = scalar_to_dist(s)
        dist = {"negativ": ln, "neutral": le, "positiv": lp}
        results.append([{"label": label, "score": score} for label, score in dist.items()])
    return results, {"profile": "heuristic", "model": "lexicon-heuristic-via-score", "max_length": 0}


def run_evaluation(
    df: pd.DataFrame,
    profile: str = "default",
    model: str | None = None,
    device: str = "auto",
    batch_size: int = 16,
    lexicon_file: str | None = None,
    lexicon_weight: float = 0.0,
    max_length: int | None = None,
    backend: str = "model",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Kör utvärdering på hela testsetet.

    Returns:
        (metrics_dict, detailed_results_list)
    """
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].tolist()

    console.print(f"[cyan]Kör sentimentanalys med profil:[/cyan] {profile}")
    t0 = time.time()

    if backend == "heuristic":
        results, meta = _heuristic_sentiment(texts)
        meta["profile"] = profile
    else:
        results, meta = analyze_smart(
            texts,
            profile=profile,
            model_name=model,
            device=device,
            batch_size=batch_size,
            return_all_scores=True,
            max_length=max_length,
            clean=True,
            lexicon_file=lexicon_file,
            lexicon_weight=lexicon_weight,
        )

    # Extrahera predictioner och scores (redan blended om lexicon angavs via analyze_smart)
    y_pred: list[str] = []
    scores_list: list[dict[str, float]] = []
    for inner in results:
        if isinstance(inner, list):
            dist = {e.get("label", ""): float(e.get("score", 0.0)) for e in inner}
            for k in LABELS:
                dist.setdefault(k, 0.0)
        elif isinstance(inner, dict):
            label = inner.get("label", "neutral")
            score = float(inner.get("score", 0.0))
            dist = {k: 0.0 for k in LABELS}
            if label in dist:
                dist[label] = score
        else:
            dist = {k: 0.0 for k in LABELS}
            dist["neutral"] = 1.0

        scores_list.append(dist)
        y_pred.append(max(dist.items(), key=lambda kv: kv[1])[0])

    proc_time = time.time() - t0

    metrics = compute_metrics(y_true, y_pred, scores_list)
    metrics["processing_time_s"] = round(proc_time, 2)
    metrics["profile"] = profile
    metrics["model"] = meta.get("model", "unknown")
    metrics["lexicon_weight"] = lexicon_weight
    metrics["lexicon_file"] = lexicon_file
    metrics["backend"] = backend

    # Detaljerade resultat per text
    details = []
    for i, (text, true_label, pred_label, scores) in enumerate(
        zip(texts, y_true, y_pred, scores_list, strict=False)
    ):
        details.append(
            {
                "index": i,
                "text": text[:200],
                "true_label": true_label,
                "pred_label": pred_label,
                "correct": true_label == pred_label,
                "scores": scores,
            }
        )

    return metrics, details


def print_results(metrics: dict[str, Any]) -> None:
    """Skriv ut resultaten i en fin tabell."""
    console.print()
    console.print(Panel.fit("[bold]Utvärderingsresultat[/bold]", border_style="cyan"))

    # Översikt
    overview = Table(title="Översikt")
    overview.add_column("Metrik", style="cyan")
    overview.add_column("Värde", style="green")
    overview.add_row("Accuracy", f"{metrics['accuracy']:.2%}")
    overview.add_row("Macro F1", f"{metrics['macro_f1']:.2%}")
    overview.add_row("Modell", str(metrics.get("model", "N/A")))
    overview.add_row("Profil", str(metrics.get("profile", "N/A")))
    overview.add_row("Lexikonvikt", str(metrics.get("lexicon_weight", 0.0)))
    overview.add_row("Antal sampel", str(metrics.get("n_samples", 0)))
    overview.add_row("Processtid", f"{metrics.get('processing_time_s', 0):.2f}s")
    console.print(overview)

    # Per-klass
    per_class = Table(title="Per-klass Metrics")
    per_class.add_column("Klass")
    per_class.add_column("Precision")
    per_class.add_column("Recall")
    per_class.add_column("F1")
    per_class.add_column("Support")
    for label in LABELS:
        pc = metrics.get("per_class", {}).get(label, {})
        per_class.add_row(
            label,
            f"{pc.get('precision', 0):.2%}",
            f"{pc.get('recall', 0):.2%}",
            f"{pc.get('f1', 0):.2%}",
            str(pc.get("support", 0)),
        )
    console.print(per_class)

    # Confusion matrix
    cm = metrics.get("confusion_matrix", {})
    if cm:
        cm_table = Table(title="Confusion Matrix (rad=true, kol=pred)")
        cm_table.add_column("", style="bold")
        for label in LABELS:
            cm_table.add_column(f"pred {label}", style="yellow")
        for true_label in LABELS:
            row = [f"true {true_label}"]
            for pred_label in LABELS:
                row.append(str(cm.get(true_label, {}).get(pred_label, 0)))
            cm_table.add_row(*row)
        console.print(cm_table)


@app.command()
def evaluate(
    testset: str = typer.Option(
        "data/test_swedish.csv",
        "--testset",
        help="Sökväg till testset CSV (kolumner: text, label)",
    ),
    profile: str = typer.Option(
        "default",
        "--profile",
        help=f"Profil att använda. Tillgängliga: {', '.join(AVAILABLE_PROFILES)}",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Hugging Face-modell (default: profilens standardmodell)",
    ),
    device: str = typer.Option("auto", "--device", help="Enhet: auto, cpu, cuda, mps"),
    batch_size: int = typer.Option(16, "--batch-size", help="Batch-storlek"),
    max_length: int | None = typer.Option(None, "--max-length", help="Max token-längd"),
    lexicon_file: str | None = typer.Option(
        None,
        "--lexicon-file",
        help="Sökväg till svenskt lexikon för blending",
    ),
    lexicon_weight: float = typer.Option(
        0.0,
        "--lexicon-weight",
        min=0.0,
        max=1.0,
        help="Vikt för lexikon-blending [0..1]",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        help="Spara resultat som JSON (t.ex. reports/baseline.json)",
    ),
    output_csv: str | None = typer.Option(
        None,
        "--output-csv",
        help="Spara detaljerade resultat som CSV",
    ),
    backend: str = typer.Option(
        "heuristic",
        "--backend",
        help="model för Hugging Face-inferens eller heuristic för snabb/offline baseline",
    ),
):
    """Utvärdera sentimentanalys mot ett testset."""
    if not os.path.isfile(testset):
        console.print(f"[red]Testset hittades inte: {testset}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Laddar testset:[/cyan] {testset}")
    df = load_testset(testset)
    console.print(f"  {len(df)} sampel, fördelning: {dict(Counter(df['label']))}")

    metrics, details = run_evaluation(
        df,
        profile=profile,
        model=model,
        device=device,
        batch_size=batch_size,
        lexicon_file=lexicon_file,
        lexicon_weight=lexicon_weight,
        max_length=max_length,
        backend=backend,
    )

    print_results(metrics)

    # Spara resultat
    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "testset": testset,
            "metrics": metrics,
            "scenarios": SCENARIOS,
            "asr_comparison": ASR_BASELINES,
        }
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        console.print(f"\n[green]Rapport sparad:[/green] {output}")

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df_out = pd.DataFrame(details)
        df_out.to_csv(output_csv, index=False, encoding="utf-8")
        console.print(f"[green]Detaljer sparade:[/green] {output_csv}")


@app.command("scenarios")
def evaluate_scenarios(
    testset: str = typer.Option("data/test_swedish.csv", "--testset"),
    output: str = typer.Option("reports/baseline_results.json", "--output"),
    backend: str = typer.Option("heuristic", "--backend"),
):
    """Run the three required benchmark scenarios: forum, call and news."""
    df = load_testset(testset)
    scenario_results: dict[str, Any] = {}
    for name, cfg in SCENARIOS.items():
        metrics, _ = run_evaluation(df, profile=cfg["profile"], backend=backend)
        metrics["description"] = cfg["description"]
        scenario_results[name] = metrics
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "testset": testset,
        "scenarios": scenario_results,
        "asr_comparison": ASR_BASELINES,
        "note": "ASR comparison uses published relative Swedish WER; sentiment metrics use the selected backend.",
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    console.print(f"[green]Baseline scenarios saved:[/green] {output}")


@app.command("asr-compare")
def asr_compare(output: str = typer.Option("reports/asr_model_comparison.json", "--output")):
    """Save OpenAI Whisper vs KB-Whisper model recommendation metadata."""
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    report = {"timestamp": datetime.now(UTC).isoformat(), "models": ASR_BASELINES}
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    console.print(f"[green]ASR comparison saved:[/green] {output}")


@app.command("negation")
def evaluate_negation(
    testset: str = typer.Option("data/test_swedish.csv", "--testset"),
    output: str | None = typer.Option(None, "--output"),
    backend: str = typer.Option("heuristic", "--backend"),
):
    """Evaluate performance specifically on negation-containing examples."""
    from .negation import is_negated_example

    df = load_testset(testset)
    mask = df["text"].apply(is_negated_example)
    df_neg = df[mask].copy()
    df_no_neg = df[~mask].copy()
    console.print(f"[cyan]Negation examples:[/cyan] {len(df_neg)}")
    console.print(f"[cyan]Non-negation examples:[/cyan] {len(df_no_neg)}")
    results: dict[str, Any] = {}
    for name, subset in [("negation", df_neg), ("non_negation", df_no_neg)]:
        if len(subset) == 0:
            results[name] = {"n_samples": 0, "note": "no examples in subset"}
            continue
        metrics, _ = run_evaluation(subset, backend=backend)
        results[name] = metrics
    metrics_all, _ = run_evaluation(df, backend=backend)
    results["overall"] = metrics_all
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "testset": testset,
        "negation_evaluation": results,
    }
    print_results(metrics_all)
    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        console.print(f"\n[green]Negation evaluation saved:[/green] {output}")


@app.command()
def list_profiles():
    """Lista tillgängliga profiler."""
    table = Table(title="Tillgängliga profiler")
    table.add_column("Profil")
    table.add_column("Modell")
    table.add_column("Max längd")
    from .profiles import PROFILE_SPECS

    for name, spec in PROFILE_SPECS.items():
        table.add_row(
            name,
            spec.get("model", "N/A"),
            str(spec.get("max_length", "N/A")),
        )
    console.print(table)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["scenarios", "--output", "reports/baseline_results.json"])
    app()
