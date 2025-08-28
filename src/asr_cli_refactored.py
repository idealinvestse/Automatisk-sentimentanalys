from __future__ import annotations

import json
import os
import glob
import time
import logging
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, TextColumn
import pandas as pd

from .asr import transcribe as asr_transcribe
from .lexicon import load_lexicon, score_text, scalar_to_dist, blend_distributions
from .sentiment import analyze_smart

app = typer.Typer(help="ASR tools: transcribe audio and analyze call sentiment")
console = Console()


def ensure_dir(path: str):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".wma", ".aac"}


def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_audio_paths(inputs: List[str]) -> List[str]:
    files: List[str] = []
    for inp in inputs:
        # Expand globs
        if any(ch in inp for ch in ["*", "?", "["]):
            for p in glob.glob(inp, recursive=True):
                if os.path.isfile(p) and os.path.splitext(p)[1].lower() in AUDIO_EXTS:
                    files.append(os.path.abspath(p))
            continue
        # Directory -> scan
        if os.path.isdir(inp):
            for root, _, fnames in os.walk(inp):
                for fn in fnames:
                    if os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                        files.append(os.path.abspath(os.path.join(root, fn)))
            continue
        # File
        if os.path.isfile(inp) and os.path.splitext(inp)[1].lower() in AUDIO_EXTS:
            files.append(os.path.abspath(inp))
    # De-dup and stable sort
    files = sorted(dict.fromkeys(files))
    return files


@app.command()
def transcribe(
    inputs: List[str] = typer.Argument(..., help="Audio files, directories or globs"),
    model: str = typer.Option("kb-whisper-large", help="ASR model: kb-whisper-large | openai/whisper-large-v3"),
    backend: str = typer.Option("faster", help="Backend: faster | transformers"),
    device: str = typer.Option("auto", help="Device: auto|cpu|cuda|cuda:0|mps"),
    language: str = typer.Option("sv", help="ASR language code (sv)"),
    beam_size: int = typer.Option(5, min=1, max=10),
    vad: bool = typer.Option(True, help="Enable VAD filter (faster-whisper)"),
    word_timestamps: bool = typer.Option(True, help="Return word timestamps if supported"),
    chunk_length_s: int = typer.Option(30, min=5, max=60, help="Chunk length (transformers)"),
    output_json: Optional[str] = typer.Option(None, help="Optional path to save transcript JSON (single input)"),
    output_dir: Optional[str] = typer.Option(None, help="Directory to save per-file JSON (multiple inputs)"),
    log_level: str = typer.Option("INFO", help="Logging level: DEBUG|INFO|WARNING|ERROR"),
    mode: str = typer.Option("full", help="Operation mode: full (transcribe + sentiment) | transcribe-only (transcribe only)"),
    sentiment_model: Optional[str] = typer.Option(None, help="Optional override for sentiment model"),
    lexicon_file: Optional[str] = typer.Option(None, help="Optional Swedish lexicon CSV/TSV"),
    lexicon_weight: float = typer.Option(0.0, min=0.0, max=1.0, help="Blend weight [0..1]"),
    output_csv: Optional[str] = typer.Option(None, help="Save segment sentiments to CSV (aggregate for multiple inputs)"),
):
    """Transcribe one or many audio files and print/save summaries."""
    setup_logging(log_level)
    files = resolve_audio_paths(inputs)
    if not files:
        console.print("[red]No audio files found. Provide files, directories or globs.[/red]")
        raise typer.Exit(code=1)

    if len(files) > 1 and not output_dir and output_json:
        console.print("[yellow]Multiple inputs detected; ignoring --output-json and using --output-dir=outputs/transcripts[/yellow]")
        output_dir = os.path.join("outputs", "transcripts")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Optional lexicon (only used in full mode)
    use_lex = lexicon_file is not None and lexicon_weight > 0.0
    lex = None
    if use_lex and mode == "full":
        try:
            lex = load_lexicon(lexicon_file)
            console.print(f"[green]Lexicon loaded:[/green] {lexicon_file} ({len(lex)} terms)")
        except Exception as e:
            console.print(f"[yellow]Warning: failed to load lexicon '{lexicon_file}': {e}. Continuing without lexicon.[/yellow]")
            use_lex = False

    ok, fail = 0, 0
    start_all = time.time()
    console.print(f"[cyan]Found {len(files)} audio file(s). Starting transcription...[/cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Transcribing", total=len(files))
        for idx, path in enumerate(files, start=1):
            progress.update(task, description=f"[{idx}/{len(files)}] {os.path.basename(path)}")
            t0 = time.time()
            try:
                tr = asr_transcribe(
                    audio_path=path,
                    model=model,
                    backend=backend,
                    device=device,
                    language=language,
                    beam_size=beam_size,
                    vad=vad,
                    word_timestamps=word_timestamps,
                    chunk_length_s=chunk_length_s,
                )
                ok += 1
            except Exception as e:
                fail += 1
                console.print(f"[red]ASR failed for {path}: {e}[/red]")
                progress.advance(task, 1)
                continue

            dur = tr.get("processing_time")
            segs = tr.get("segments", []) or []
            console.print(
                f"[green]Done:[/green] {os.path.basename(path)} | segs={len(segs)} | time={time.time()-t0:.2f}s (ASR={dur:.2f}s)"
            )

            # Show head
            head = segs[:5]
            if head:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("#")
                table.add_column("start")
                table.add_column("end")
                table.add_column("text")
                for i, s in enumerate(head):
                    table.add_row(str(i), f"{s.get('start', '')}", f"{s.get('end', '')}", s.get("text", "")[:100])
                console.print(table)

            # Save transcript JSON
            try:
                if len(files) == 1 and output_json:
                    ensure_dir(output_json)
                    with open(output_json, "w", encoding="utf-8") as f:
                        json.dump(tr, f, ensure_ascii=False, indent=2)
                    console.print(f"[green]Saved transcript:[/green] {output_json}")
                elif output_dir:
                    base = os.path.splitext(os.path.basename(path))[0]
                    out_path = os.path.join(output_dir, f"{base}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(tr, f, ensure_ascii=False, indent=2)
                    console.print(f"[green]Saved transcript:[/green] {out_path}")
            except Exception as e:
                console.print(f"[red]Failed to save transcript for {path}: {e}[/red]")

            # Run sentiment analysis only if mode is "full"
            if mode == "full":
                texts: List[str] = [s.get("text", "").strip() for s in segs]
                if not texts or all(not t for t in texts):
                    texts = [" ".join([s.get("text", "").strip() for s in segs if s.get("text")]).strip()] if segs else []
                    if not texts:
                        console.print(f"[yellow]No transcript text produced for {path}.[/yellow]")
                        fail += 1
                        progress.advance(task, 1)
                        continue

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

                rows = []
                for idx_seg, (t, inner) in enumerate(zip(texts, results)):
                    scores = {e.get("label"): float(e.get("score", 0.0)) for e in inner}
                    for k in ["negativ", "neutral", "positiv"]:
                        scores.setdefault(k, 0.0)
                    if use_lex and lex is not None:
                        s_scalar = score_text(t, lex)
                        ln, le, lp = scalar_to_dist(s_scalar)
                        scores = blend_distributions(scores, (ln, le, lp), lexicon_weight)
                    top_label = max(scores.items(), key=lambda kv: kv[1])[0]
                    top_score = float(scores[top_label])
                    start = float(segs[idx_seg].get("start", 0.0) or 0.0) if idx_seg < len(segs) else None
                    end = float(segs[idx_seg].get("end", 0.0) or 0.0) if idx_seg < len(segs) else None
                    row = {
                        "file": path,
                        "index": idx_seg,
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
                    }
                    rows.append(row)

                # Show sentiment analysis results
                table = Table(show_header=True, header_style="bold magenta")
                for col in ["index", "start", "end", "label", "score", "text"]:
                    table.add_column(col)
                for r in rows[:10]:
                    table.add_row(str(r["index"]), str(r["start"]), str(r["end"]), r["label"], f"{r['score']:.3f}", r["text"][:100])
                console.print(table)
                if len(rows) > 10:
                    console.print(f"... showing 10 of {len(rows)} segments for {os.path.basename(path)}")

                # Save sentiment analysis results to CSV if requested
                if output_csv and rows:
                    try:
                        ensure_dir(output_csv)
                        pd.DataFrame(rows).to_csv(output_csv, index=False)
                        console.print(f"[green]Saved sentiment analysis:[/green] {output_csv}")
                    except Exception as e:
                        console.print(f"[red]Failed to save CSV: {e}[/red]")

            progress.advance(task, 1)

    console.print(f"[bold]Completed[/bold]: ok={ok}, failed={fail}, total={len(files)} | elapsed={time.time()-start_all:.2f}s")


@app.command("analyze-call")
def analyze_call(
    inputs: List[str] = typer.Argument(..., help="Audio files, directories or globs"),
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
    output_csv: Optional[str] = typer.Option(None, help="Save segment sentiments to CSV (aggregate for multiple inputs)"),
    log_level: str = typer.Option("INFO", help="Logging level: DEBUG|INFO|WARNING|ERROR"),
):
    """Transcribe the call(s) and run per-segment sentiment using the 'call' profile."""
    # This function is kept for backward compatibility but now just calls transcribe with mode="full"
    transcribe(
        inputs=inputs,
        model=model,
        backend=backend,
        device=device,
        language=language,
        beam_size=beam_size,
        vad=vad,
        word_timestamps=word_timestamps,
        chunk_length_s=chunk_length_s,
        output_json=None,
        output_dir=None,
        log_level=log_level,
        mode="full",
        sentiment_model=sentiment_model,
        lexicon_file=lexicon_file,
        lexicon_weight=lexicon_weight,
        output_csv=output_csv,
    )


if __name__ == "__main__":
    app()
