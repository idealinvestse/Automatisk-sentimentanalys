from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .clean import clean_texts
from .cli_audio import register_audio_commands
from .cli_support import ensure_dir, setup_logging
from .core.config import DEFAULT_SENTIMENT_MODEL
from .core.logging_config import configure_logging
from .core.serialization import score_dict, utc_now_iso
from .core.serialization import top_label as top_label_pair
from .lexicon import load_lexicon
from .profiles import resolve_profile
from .sentiment import analyze_smart

app = typer.Typer(help="Svenskt sentiment- och samtalsanalyssystem")
console = Console()
register_audio_commands(app)


@app.callback()
def _cli_global_options(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Aktivera DEBUG-loggning"),
    log_level: str | None = typer.Option(
        None, "--log-level", help="Loggnivå: DEBUG|INFO|WARNING|ERROR"
    ),
) -> None:
    """Global CLI options."""
    configure_logging()
    root = logging.getLogger()
    if verbose:
        root.setLevel(logging.DEBUG)
    elif log_level:
        root.setLevel(getattr(logging, log_level.upper(), logging.INFO))


@app.command("new-analyzer")
def new_analyzer_cmd(
    name: str = typer.Argument(..., help="Analyzer name (snake_case)"),
    force: bool = typer.Option(False, "--force", help="Overwrite if file exists"),
) -> None:
    """Scaffold a new analyzer from the project template (EXT-01)."""
    import re
    from pathlib import Path

    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        raise typer.BadParameter("Name must be snake_case (e.g. my_insight)")

    template = (
        Path(__file__).resolve().parent / "analysis" / "templates" / "new_analyzer_template.py"
    )
    target = Path(__file__).resolve().parent / "analysis" / f"{name}.py"
    if target.exists() and not force:
        raise typer.Exit(code=1)
    content = template.read_text(encoding="utf-8")
    content = content.replace("your_analyzer", name).replace(
        "YourAnalyzer", "".join(p.capitalize() for p in name.split("_")) + "Analyzer"
    )
    target.write_text(content, encoding="utf-8")
    console.print(f"[green]Created[/green] {target}")
    console.print(
        "Autodiscovery will register it on next import. Run: python -m src.cli analyzers-graph"
    )


@app.command("analyzers-graph")
def analyzers_graph_cmd(
    profile: str = typer.Option(
        "callcenter",
        "--profile",
        help="Highlight analyzers active for this profile",
    ),
    fmt: str = typer.Option(
        "text",
        "--format",
        help="Output format: text | mermaid | json",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional file path to write graph output",
    ),
) -> None:
    """Show registered analyzer dependency graph (debugging / documentation)."""
    from .analysis.graph import build_dependency_graph, to_json, to_mermaid, to_text_summary
    from .analysis.registry import (
        ensure_analyzers_loaded,
        get_analyzer_registry,
        resolve_analyzers_for_profile,
    )

    ensure_analyzers_loaded()
    registry = get_analyzer_registry()
    selected = resolve_analyzers_for_profile(profile)
    highlight = set(selected) if selected else set(registry.keys())
    graph = build_dependency_graph(registry, selected=highlight)

    if fmt == "mermaid":
        content = to_mermaid(graph, highlight=highlight)
    elif fmt == "json":
        content = to_json(graph)
    else:
        content = to_text_summary(graph)
        if selected:
            content = f"Profile: {profile}\nSelected: {', '.join(selected)}\n\n" + content

    if output:
        Path(output).write_text(content, encoding="utf-8")
        console.print(f"[green]Wrote graph to {output}[/green]")
    else:
        console.print(content)


@app.command("sentiment")
def sentiment_cmd(
    text: str | None = typer.Option(None, help="Analysera en enskild text"),
    txt_file: str | None = typer.Option(
        None, "--txt-file", help="Sökväg till .txt (en text per rad)"
    ),
    csv_file: str | None = typer.Option(None, "--csv-file", help="Sökväg till .csv med texter"),
    text_column: str = typer.Option("text", help="Kolumnnamn i CSV som innehåller text"),
    model: str | None = typer.Option(
        None, "--model", help="Hugging Face-modell att använda (standard väljs via profil)"
    ),
    batch_size: int = typer.Option(16, help="Batch-storlek för inferens"),
    max_rows: int | None = typer.Option(None, help="Analysera högst N rader (debug/snabbtest)"),
    output: str | None = typer.Option(
        None, help="Spara resultat till CSV (t.ex. outputs/predictions.csv)"
    ),
    device: str | None = typer.Option(
        "auto", help="Enhet: 'auto' (default), 'cpu', 'cuda', 'cuda:0', 'mps'"
    ),
    return_all_scores: bool = typer.Option(
        False, "--return-all-scores", help="Returnera sannolikheter för alla klasser"
    ),
    max_length: int | None = typer.Option(
        None, help="Max token-längd vid inferens (om ej satt används profilens)"
    ),
    datatype: str | None = typer.Option(
        None, "--datatype", help="Datatyp: t.ex. 'post', 'comment', 'article', 'review'"
    ),
    source: str | None = typer.Option(
        None, "--source", help="Källa: t.ex. 'forum', 'magazine', 'news', 'social'"
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Profil att använda (åsidolägger datatype/source). T.ex. 'forum', 'magazine'",
    ),
    lexicon_file: str | None = typer.Option(
        None,
        "--lexicon-file",
        help="Sökväg till svenskt lexikon (CSV/TSV) med kolumner term|word och polarity|score|sentiment",
    ),
    lexicon_weight: float = typer.Option(
        0.0,
        "--lexicon-weight",
        min=0.0,
        max=1.0,
        help="Vikt för lexikon-blandning [0..1]. 0=inaktiverad",
    ),
    log_level: str = typer.Option("INFO", help="Logging level: DEBUG|INFO|WARNING|ERROR"),
):
    """Kör svensk sentimentanalys på text, .txt eller .csv."""
    setup_logging(log_level)

    sources = sum(
        [
            1 if text is not None else 0,
            1 if txt_file is not None else 0,
            1 if csv_file is not None else 0,
        ]
    )
    if sources == 0:
        console.print("[yellow]Ange en källa: --text, --txt-file eller --csv-file[/yellow]")
        raise typer.Exit(code=1)
    if sources > 1:
        console.print("[red]Ange endast EN av --text, --txt-file eller --csv-file[/red]")
        raise typer.Exit(code=1)

    # 1) Läs in texter
    texts: list[str] = []
    if text is not None:
        texts = [text.strip()]
    elif txt_file is not None:
        if not os.path.isfile(txt_file):
            console.print(f"[red]Hittar inte txt-fil: {txt_file}[/red]")
            raise typer.Exit(code=1)
        with open(txt_file, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    elif csv_file is not None:
        if not os.path.isfile(csv_file):
            console.print(f"[red]Hittar inte csv-fil: {csv_file}[/red]")
            raise typer.Exit(code=1)
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            console.print(f"[red]Kunde inte läsa CSV: {e}[/red]")
            raise typer.Exit(code=1) from e
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
    chosen_model = model or spec.get("model", DEFAULT_SENTIMENT_MODEL)
    resolved_max_length = max_length or spec.get("max_length", 256)

    # Rengör texter enligt profil (för display + skicka till analyze_smart)
    texts = clean_texts(texts, spec.get("cleaning", {}))

    # 3) Kör via analyze_smart (hanterar clean + modell + ev. lexikon-default från profil)
    console.print(f"[green]Profil:[/green] {profile_name}")
    console.print(f"[green]Laddar modell:[/green] {chosen_model}")
    try:
        results, meta = analyze_smart(
            texts=texts,
            profile=profile_name,
            model_name=chosen_model,
            device=device,
            batch_size=batch_size,
            normalize=True,
            return_all_scores=return_all_scores,
            max_length=resolved_max_length,
            clean=True,
            lexicon_file=lexicon_file,
            lexicon_weight=lexicon_weight,
        )
    except Exception as e:
        console.print(f"[red]Fel under analys: {e}[/red]")
        raise typer.Exit(code=2) from e

    # Lexikon-info (om auto från profil eller explicit)
    use_lex = bool(meta.get("lexicon_file")) or (lexicon_file is not None and lexicon_weight > 0.0)
    if meta.get("lexicon_file"):
        console.print(
            f"[green]Lexikon (auto från profil eller explicit):[/green] {meta['lexicon_file']} (vikt={meta.get('lexicon_weight', 0)})"
        )
    elif lexicon_file and use_lex:
        try:
            lex = load_lexicon(lexicon_file)
            console.print(f"[green]Lexikon laddat:[/green] {lexicon_file} ({len(lex)} termer)")
        except Exception as e:
            console.print(
                f"[yellow]Varning: kunde inte ladda lexikon '{lexicon_file}': {e}. Fortsätter utan.[/yellow]"
            )
            use_lex = False
    now_iso = utc_now_iso()
    rows: list[dict[str, str | float]] = []
    for t, result in zip(texts, results, strict=False):
        scores = score_dict(result)
        lbl, top_score = top_label_pair(scores)
        rows.append(
            {
                "text": t,
                "label": lbl,
                "score": float(top_score),
                "negativ": scores["negativ"],
                "neutral": scores["neutral"],
                "positiv": scores["positiv"],
                "model": chosen_model,
                "profile": profile_name,
                "timestamp": now_iso,
            }
        )

    # 7) Visa
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Text")
    table.add_column("Klass")
    table.add_column("Konfidens")
    if return_all_scores or use_lex:
        table.add_column("Negativ")
        table.add_column("Neutral")
        table.add_column("Positiv")

    for r in rows[:20]:
        text = str(r["text"])
        label = str(r["label"])
        score = float(r["score"])
        t_trunc = text[:60] + "..." if len(text) > 60 else text
        if return_all_scores or use_lex:
            table.add_row(
                t_trunc,
                label,
                f"{score:.3f}",
                f"{float(r['negativ']):.3f}",
                f"{float(r['neutral']):.3f}",
                f"{float(r['positiv']):.3f}",
            )
        else:
            table.add_row(t_trunc, label, f"{score:.3f}")

    console.print(table)
    if len(rows) > 20:
        console.print(f"[yellow]Visar 20 av {len(rows)} rader.[/yellow]")

    # 8) Spara
    if output:
        try:
            ensure_dir(output)
            pd.DataFrame(rows).to_csv(output, index=False)
            console.print(f"[green]Resultat sparade till CSV:[/green] {output}")
        except Exception as e:
            console.print(f"[red]Kunde inte spara till CSV: {e}[/red]")

