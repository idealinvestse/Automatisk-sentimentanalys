from __future__ import annotations

import json
import os
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table

from .asr import transcribe as asr_transcribe
from .lexicon import load_lexicon, score_text, scalar_to_dist, blend_distributions
from .sentiment import analyze_smart

app = typer.Typer(help="ASR tools: transcribe audio and analyze call sentiment")
console = Console()


def ensure_dir(path: str):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


@app.command()
def transcribe(
    audio: str = typer.Argument(..., help="Path to audio file"),
    model: str = typer.Option("kb-whisper-large", help="ASR model: kb-whisper-large | openai/whisper-large-v3"),
    backend: str = typer.Option("faster", help="Backend: faster | transformers"),
    device: str = typer.Option("auto", help="Device: auto|cpu|cuda|cuda:0|mps"),
    language: str = typer.Option("sv", help="ASR language code (sv)"),
    beam_size: int = typer.Option(5, min=1, max=10),
    vad: bool = typer.Option(True, help="Enable VAD filter (faster-whisper)"),
    word_timestamps: bool = typer.Option(True, help="Return word timestamps if supported"),
    chunk_length_s: int = typer.Option(30, min=5, max=60, help="Chunk length (transformers)"),
    output_json: Optional[str] = typer.Option(None, help="Optional path to save transcript JSON"),
):
    """Transcribe an audio file and print a summary."""
    try:
        tr = asr_transcribe(
            audio_path=audio,
            model=model,
            backend=backend,
            device=device,
            language=language,
            beam_size=beam_size,
            vad=vad,
            word_timestamps=word_timestamps,
            chunk_length_s=chunk_length_s,
        )
    except Exception as e:
        console.print(f"[red]ASR failed: {e}[/red]")
        raise typer.Exit(code=1)

    segs = tr.get("segments", []) or []
    console.print(f"[green]Model:[/green] {tr.get('model')} | [green]Backend:[/green] {tr.get('backend')} | [green]Lang:[/green] {tr.get('language')}")
    console.print(f"[green]Segments:[/green] {len(segs)} | [green]Processing time:[/green] {tr.get('processing_time'):.2f}s")

    # Show first 10 segments
    head = segs[:10]
    if head:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#")
        table.add_column("start")
        table.add_column("end")
        table.add_column("text")
        for i, s in enumerate(head):
            table.add_row(str(i), f"{s.get('start', '')}", f"{s.get('end', '')}", s.get("text", "")[:100])
        console.print(table)
        if len(segs) > len(head):
            console.print(f"... showing 10 of {len(segs)} segments")

    if output_json:
        try:
            ensure_dir(output_json)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(tr, f, ensure_ascii=False, indent=2)
            console.print(f"[green]Saved transcript:[/green] {output_json}")
        except Exception as e:
            console.print(f"[red]Failed to save JSON: {e}[/red]")
            raise typer.Exit(code=1)


@app.command("analyze-call")
def analyze_call(
    audio: str = typer.Argument(..., help="Path to audio file"),
    # ASR
    model: str = typer.Option("kb-whisper-large", help="ASR model"),
    backend: str = typer.Option("faster", help="ASR backend: faster | transformers"),
    device: str = typer.Option("auto", help="Device: auto|cpu|cuda|cuda:0|mps"),
    language: str = typer.Option("sv", help="ASR language code"),
    beam_size: int = typer.Option(5, min=1, max=10),
    vad: bool = typer.Option(True),
    word_timestamps: bool = typer.Option(False),
    chunk_length_s: int = typer.Option(30, min=5, max=60),
    # Sentiment
    sentiment_model: Optional[str] = typer.Option(None, help="Optional override for sentiment model"),
    lexicon_file: Optional[str] = typer.Option(None, help="Optional Swedish lexicon CSV/TSV"),
    lexicon_weight: float = typer.Option(0.0, min=0.0, max=1.0, help="Blend weight [0..1]"),
    output_csv: Optional[str] = typer.Option(None, help="Save segment sentiments to CSV"),
):
    """Transcribe the call and run per-segment sentiment using the 'call' profile."""
    try:
        tr = asr_transcribe(
            audio_path=audio,
            model=model,
            backend=backend,
            device=device,
            language=language,
            beam_size=beam_size,
            vad=vad,
            word_timestamps=word_timestamps,
            chunk_length_s=chunk_length_s,
        )
    except Exception as e:
        console.print(f"[red]ASR failed: {e}[/red]")
        raise typer.Exit(code=1)

    segments = tr.get("segments", []) or []
    texts: List[str] = [s.get("text", "").strip() for s in segments]
    if not texts or all(not t for t in texts):
        texts = [" ".join([s.get("text", "").strip() for s in segments if s.get("text")]).strip()] if segments else []
        if not texts:
            console.print("[yellow]No transcript text produced.[/yellow]")
            raise typer.Exit(code=1)

    results, meta = analyze_smart(
        texts,
        profile="call",
        model_name=sentiment_model,
        device="auto",
        batch_size=16,
        normalize=True,
        return_all_scores=True,
        max_length=None,
        clean=True,
    )

    # Optional lexicon blending
    use_lex = lexicon_file is not None and lexicon_weight > 0.0
    lex = None
    if use_lex:
        try:
            lex = load_lexicon(lexicon_file)
            console.print(f"[green]Lexicon loaded:[/green] {lexicon_file} ({len(lex)} terms)")
        except Exception as e:
            console.print(f"[yellow]Warning: failed to load lexicon '{lexicon_file}': {e}. Continuing without lexicon.[/yellow]")
            use_lex = False

    rows = []
    for idx, (t, inner) in enumerate(zip(texts, results)):
        scores = {e.get("label"): float(e.get("score", 0.0)) for e in inner}
        for k in ["negativ", "neutral", "positiv"]:
            scores.setdefault(k, 0.0)
        if use_lex and lex is not None:
            s_scalar = score_text(t, lex)
            ln, le, lp = scalar_to_dist(s_scalar)
            scores = blend_distributions(scores, (ln, le, lp), lexicon_weight)
        top_label = max(scores.items(), key=lambda kv: kv[1])[0]
        top_score = float(scores[top_label])
        start = float(segments[idx].get("start", 0.0) or 0.0) if idx < len(segments) else None
        end = float(segments[idx].get("end", 0.0) or 0.0) if idx < len(segments) else None
        rows.append({
            "index": idx,
            "start": start,
            "end": end,
            "text": t,
            "label": top_label,
            "score": top_score,
            "negativ": scores.get("negativ"),
            "neutral": scores.get("neutral"),
            "positiv": scores.get("positiv"),
            "model": meta.get("model"),
            "profile": meta.get("profile"),
        })

    # Show table
    table = Table(show_header=True, header_style="bold magenta")
    for col in ["index", "start", "end", "label", "score", "text"]:
        table.add_column(col)
    for r in rows[:20]:
        table.add_row(str(r["index"]), str(r["start"]), str(r["end"]), r["label"], f"{r['score']:.3f}", r["text"][:100])
    console.print(table)
    if len(rows) > 20:
        console.print(f"... showing 20 of {len(rows)} segments. Use --output-csv to save all.")

    if output_csv:
        try:
            ensure_dir(output_csv)
            import pandas as pd
            pd.DataFrame(rows).to_csv(output_csv, index=False)
            console.print(f"[green]Saved CSV:[/green] {output_csv}")
        except Exception as e:
            console.print(f"[red]Failed to save CSV: {e}[/red]")
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
