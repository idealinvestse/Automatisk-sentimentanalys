"""Unified CLI for Swedish sentiment and call analysis.

Provides commands for text sentiment analysis, audio transcription, and full call analysis.
"""

from __future__ import annotations

import json
import logging
import os
import time

import pandas as pd
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Core & Transcription imports
from .clean import clean_texts
from .core.audio import resolve_audio_paths as _core_resolve_audio
from .core.config import DEFAULT_ASR_MODEL, DEFAULT_SENTIMENT_MODEL
from .core.serialization import score_dict, utc_now_iso
from .core.serialization import top_label as top_label_pair
from .lexicon import blend_results_with_lexicon, load_lexicon
from .pipeline import CallAnalysisPipeline
from .profiles import resolve_profile
from .sentiment import analyze_smart
from .transcription import get_transcriber

app = typer.Typer(help="Svenskt sentiment- och samtalsanalyssystem")
console = Console()


def ensure_dir(path: str):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_audio_paths(inputs: list[str]) -> list[str]:
    """Resolve file paths, directories, and glob patterns.

    Thin wrapper around :func:`src.core.audio.resolve_audio_paths` with the
    flat ``inputs`` signature expected by the CLI commands.
    """
    return _core_resolve_audio(audio_paths=inputs)


def _parse_asr_hotwords(
    hotwords: str | None,
    language: str,
    *,
    auto_load: bool = True,
) -> list[str] | None:
    """Parse CLI hotwords and optionally auto-load Swedish callcenter terms."""
    parsed: list[str] | None = None
    if hotwords:
        parsed = [w.strip() for w in hotwords.replace(",", " ").split() if w.strip()]

    if parsed or not auto_load:
        return parsed

    if not language.lower().startswith("sv"):
        return None

    default_hw_path = os.path.join("configs", "callcenter_hotwords.txt")
    if not os.path.exists(default_hw_path):
        return None

    try:
        with open(default_hw_path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        if lines:
            console.print(f"[cyan]Auto-loaded {len(lines)} hotwords from {default_hw_path}[/cyan]")
            return lines
    except Exception:
        pass
    return None


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
        console.print(f"[green]Lexikon (auto från profil eller explicit):[/green] {meta['lexicon_file']} (vikt={meta.get('lexicon_weight', 0)})")
    elif lexicon_file and use_lex:
        try:
            lex = load_lexicon(lexicon_file)
            console.print(f"[green]Lexikon laddat:[/green] {lexicon_file} ({len(lex)} termer)")
        except Exception as e:
            console.print(f"[yellow]Varning: kunde inte ladda lexikon '{lexicon_file}': {e}. Fortsätter utan.[/yellow]")
            use_lex = False
    now_iso = utc_now_iso()
    rows = []
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
        t_trunc = r["text"][:60] + "..." if len(r["text"]) > 60 else r["text"]
        if return_all_scores or use_lex:
            table.add_row(
                t_trunc,
                r["label"],
                f"{r['score']:.3f}",
                f"{r['negativ']:.3f}",
                f"{r['neutral']:.3f}",
                f"{r['positiv']:.3f}",
            )
        else:
            table.add_row(t_trunc, r["label"], f"{r['score']:.3f}")

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


@app.command("transcribe")
def transcribe_cmd(
    inputs: list[str] = typer.Argument(..., help="Audio files, directories or globs"),
    model: str = typer.Option(DEFAULT_ASR_MODEL, help="ASR model name or HuggingFace ID"),
    backend: str = typer.Option(
        "faster",
        help="ASR backend: faster (default, best Swedish WER via KB-Whisper) | transformers | whisperx (better alignment + integrated diarization)",
    ),
    device: str = typer.Option("auto", help="Device: auto|cpu|cuda|cuda:0|mps"),
    language: str = typer.Option("sv", help="ASR language code (sv)"),
    beam_size: int = typer.Option(5, min=1, max=10),
    vad: bool = typer.Option(True, help="Enable VAD filter (faster-whisper)"),
    word_timestamps: bool = typer.Option(True, help="Return word timestamps if supported"),
    chunk_length_s: int = typer.Option(30, min=5, max=60, help="Chunk length (transformers)"),
    revision: str | None = typer.Option(
        None,
        help="KB-Whisper revision: standard|strict|subtitle (strict recommended for call center)",
    ),
    diarize: bool = typer.Option(False, "--diarize", help="Run speaker diarization"),
    num_speakers: int | None = typer.Option(
        None, "--num-speakers", help="Expected number of speakers"
    ),
    hotwords: str | None = typer.Option(
        None,
        "--hotwords",
        help="Comma or space separated list of domain words to boost (e.g. 'fakturering,återbetalning,acme'). Auto-loaded from configs/callcenter_hotwords.txt for Swedish (sv) only.",
    ),
    no_hotwords: bool = typer.Option(
        False,
        "--no-hotwords",
        help="Disable automatic Swedish callcenter hotword loading.",
    ),
    initial_prompt: str | None = typer.Option(
        None, "--initial-prompt", help="Text prompt to condition the ASR decoder (e.g. expected names or style at start of call)."
    ),
    preprocess: bool = typer.Option(
        False, "--preprocess", help="Enable audio preprocessing (high-pass filter + optional noise reduction) before ASR. Useful for noisy recordings."
    ),
    output_json: str | None = typer.Option(
        None, help="Optional path to save transcript JSON (single input)"
    ),
    output_dir: str | None = typer.Option(
        None, help="Directory to save per-file JSON (multiple inputs)"
    ),
    log_level: str = typer.Option("INFO", help="Logging level: DEBUG|INFO|WARNING|ERROR"),
):
    """Transcribe one or many audio files using Faster-Whisper, Transformers or WhisperX ASR.

    WhisperX backend gives superior word-level timestamps (valuable for aspect
    evidence spans) and can perform diarization in the same pass.
    """
    setup_logging(log_level)
    files = resolve_audio_paths(inputs)
    if not files:
        console.print("[red]No audio files found. Provide files, directories or globs.[/red]")
        raise typer.Exit(code=1)

    if len(files) > 1 and not output_dir and output_json:
        console.print(
            "[yellow]Multiple inputs detected; ignoring --output-json and using --output-dir=outputs/transcripts[/yellow]"
        )
        output_dir = os.path.join("outputs", "transcripts")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ok, fail = 0, 0
    start_all = time.time()
    console.print(f"[cyan]Found {len(files)} audio file(s). Starting transcription...[/cyan]")

    # Initialize transcriber
    try:
        transcriber = get_transcriber(backend=backend, model_name=model, device=device)
    except Exception as e:
        console.print(f"[red]Failed to initialize ASR transcriber: {e}[/red]")
        raise typer.Exit(code=2) from e

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
                parsed_hotwords = _parse_asr_hotwords(
                    hotwords,
                    language,
                    auto_load=not no_hotwords,
                )

                tr_obj = transcriber.transcribe(
                    audio_path=path,
                    language=language,
                    beam_size=beam_size,
                    vad=vad,
                    word_timestamps=word_timestamps,
                    chunk_length_s=chunk_length_s,
                    revision=revision,
                    diarize=diarize,
                    num_speakers=num_speakers,
                    hotwords=parsed_hotwords,
                    initial_prompt=initial_prompt,
                    preprocess=preprocess,
                )
                tr = tr_obj.to_dict()
                ok += 1
            except Exception as e:
                fail += 1
                console.print(f"[red]ASR failed for {path}: {e}[/red]")
                progress.advance(task, 1)
                continue

            dur = tr.get("processing_time")
            segs = tr.get("segments", []) or []
            console.print(
                f"[green]Done:[/green] {os.path.basename(path)} | segs={len(segs)} | time={time.time() - t0:.2f}s (ASR={dur:.2f}s)"
            )

            # Show head segments
            head = segs[:5]
            if head:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("#")
                table.add_column("start")
                table.add_column("end")
                table.add_column("text")
                for i, s in enumerate(head):
                    table.add_row(
                        str(i),
                        f"{s.get('start', ''):.2f}" if s.get("start") is not None else "",
                        f"{s.get('end', ''):.2f}" if s.get("end") is not None else "",
                        s.get("text", "")[:100],
                    )
                console.print(table)

            # Save results
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

            progress.advance(task, 1)

    console.print(
        f"[bold]Completed[/bold]: ok={ok}, failed={fail}, total={len(files)} | elapsed={time.time() - start_all:.2f}s"
    )


@app.command("analyze-call")
def analyze_call_cmd(
    inputs: list[str] = typer.Argument(..., help="Audio files, directories or globs"),
    # ASR settings
    model: str = typer.Option(DEFAULT_ASR_MODEL, help="ASR model name"),
    backend: str = typer.Option(
        "faster",
        help="ASR backend: faster (default, best Swedish WER via KB-Whisper) | transformers | whisperx (better alignment + integrated diarization)",
    ),
    device: str = typer.Option("auto", help="Device: auto|cpu|cuda|cuda:0|mps"),
    language: str = typer.Option("sv", help="ASR language code"),
    beam_size: int = typer.Option(5, min=1, max=10),
    vad: bool = typer.Option(True),
    word_timestamps: bool = typer.Option(False),
    chunk_length_s: int = typer.Option(30, min=5, max=60),
    revision: str | None = typer.Option(None, help="KB-Whisper revision: standard|strict|subtitle"),
    diarize: bool = typer.Option(False, "--diarize", help="Run speaker diarization"),
    num_speakers: int | None = typer.Option(
        None, "--num-speakers", help="Expected number of speakers"
    ),
    hotwords: str | None = typer.Option(
        None,
        "--hotwords",
        help="Comma/space separated domain words to boost in ASR (e.g. fakturering,återbetalning). Auto-loaded for Swedish (sv) only.",
    ),
    no_hotwords: bool = typer.Option(
        False,
        "--no-hotwords",
        help="Disable automatic Swedish callcenter hotword loading.",
    ),
    initial_prompt: str | None = typer.Option(
        None, "--initial-prompt", help="Conditioning prompt for ASR decoder."
    ),
    preprocess: bool = typer.Option(
        False, "--preprocess", help="Enable audio preprocessing (high-pass + noise reduction) before ASR."
    ),
    # LLM / Mistral holistic (Fas 3.2+)
    use_mistral_llm: bool = typer.Option(
        False,
        "--use-mistral-llm",
        help="Enable Mistral via OpenRouter for full-conversation holistic analysis (trajectory, root cause, actionable QA recommendations, agent assessment). Requires OPENROUTER_API_KEY env var. European/GDPR-preferred models.",
    ),
    llm_model: str | None = typer.Option(
        None,
        "--llm-model",
        help="Mistral model slug on OpenRouter (default from profile or mistralai/mistral-medium-3.5). Example: mistralai/mistral-large-3",
    ),
    deep_analysis: bool = typer.Option(
        False, "--deep-analysis", help="Force the deep LLM path (equivalent to --use-mistral-llm for callcenter use)."
    ),
    # Sentiment settings
    sentiment_model: str = typer.Option(DEFAULT_SENTIMENT_MODEL, help="Sentiment model name"),
    lexicon_file: str | None = typer.Option(None, help="Optional Swedish lexicon CSV/TSV"),
    lexicon_weight: float = typer.Option(0.0, min=0.0, max=1.0, help="Blend weight [0..1]"),
    output_csv: str | None = typer.Option(
        None, help="Save segment sentiments to CSV (aggregate for multiple inputs)"
    ),
    log_level: str = typer.Option("INFO", help="Logging level: DEBUG|INFO|WARNING|ERROR"),
):
    """Transcribe Swedish call(s) and perform end-to-end sentiment, intent, and summary analysis.

    Supports --backend whisperx for improved alignment and built-in diarization
    (recommended when you need accurate per-speaker timestamps for trajectory / ABSA).

    --use-mistral-llm / --deep-analysis activates the hybrid Mistral layer (Fas 3) for
    full-conversation reasoning (trajectory, root cause, actionable QA insights, agent assessment).
    Requires OPENROUTER_API_KEY. Uses European-preferred models by default.
    """
    setup_logging(log_level)
    files = resolve_audio_paths(inputs)
    if not files:
        console.print("[red]No audio files found. Provide files, directories or globs.[/red]")
        raise typer.Exit(code=1)

    # Optional lexicon
    use_lex = lexicon_file is not None and lexicon_weight > 0.0
    lex = None
    if use_lex:
        try:
            lex = load_lexicon(lexicon_file)
            console.print(f"[green]Lexicon loaded:[/green] {lexicon_file} ({len(lex)} terms)")
        except Exception as e:
            console.print(
                f"[yellow]Warning: failed to load lexicon '{lexicon_file}': {e}. Continuing without lexicon.[/yellow]"
            )
            use_lex = False

    all_rows = []
    ok, fail = 0, 0
    start_all = time.time()
    console.print(f"[cyan]Found {len(files)} audio file(s). Starting analyze-call...[/cyan]")

    # Initialize pipeline
    # NOTE: asr_backend + asr_model are forwarded so that --backend whisperx (and --model)
    # actually affect the transcription step inside analyze-call. Previously these
    # CLI flags were accepted but ignored for the full pipeline command.
    pipeline = CallAnalysisPipeline(
        sentiment_model=sentiment_model,
        device=device,
        asr_backend=backend,
        asr_model=model,
        use_mistral_llm=use_mistral_llm,
        llm_model=llm_model,
        deep_analysis=deep_analysis,
    )

    if use_mistral_llm or deep_analysis:
        console.print(
            "[yellow]Mistral/OpenRouter LLM deep analysis ENABLED for this run. "
            "Full conversation (with roles) will be sent to external service. "
            "See INFO logs for GDPR/egress notice. Cost tracked in meta.[/yellow]"
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Analyzing", total=len(files))
        for idx_file, path in enumerate(files, start=1):
            progress.update(task, description=f"[{idx_file}/{len(files)}] {os.path.basename(path)}")
            try:
                parsed_hotwords = _parse_asr_hotwords(
                    hotwords,
                    language,
                    auto_load=not no_hotwords,
                )

                report = pipeline.analyze_audio(
                    audio_path=path,
                    num_speakers=num_speakers,
                    language=language,
                    run_diarization=diarize,
                    hotwords=parsed_hotwords,
                    initial_prompt=initial_prompt,
                    preprocess=preprocess,
                )
                report_dict = report.to_dict()
            except Exception as e:
                fail += 1
                console.print(f"[red]Pipeline analysis failed for {path}: {e}[/red]")
                progress.advance(task, 1)
                continue

            segments = report_dict.get("segments", []) or []
            sentiment_results = report_dict.get("sentiment_results", []) or []
            intent_results = report_dict.get("intent_results", []) or []

            # Blend lexicon over all segments at once (no-op when use_lex=False)
            seg_texts = [s.get("text", "").strip() for s in segments]
            # Extract per-segment confidences so that low-confidence ASR segments
            # automatically receive a boosted lexicon_weight (Task 1.2).
            seg_confs = [s.get("confidence") or s.get("avg_confidence") for s in segments]
            sentiment_results = blend_results_with_lexicon(
                seg_texts,
                sentiment_results,
                lexicon_file if use_lex else None,
                lexicon_weight,
                segment_confidences=seg_confs,
            )

            rows = []
            for idx, s in enumerate(segments):
                text_val = seg_texts[idx]
                sent_val = sentiment_results[idx] if idx < len(sentiment_results) else {}
                sent_score = score_dict(sent_val)
                lbl, top_score = top_label_pair(sent_score)

                # Get intent
                intent_val = "other"
                intent_conf = 0.0
                if idx < len(intent_results):
                    res_entry = intent_results[idx]
                    intent_val = res_entry.get("intent", "other")
                    intent_conf = float(res_entry.get("confidence", 0.0))

                row = {
                    "file": path,
                    "index": idx,
                    "start": s.get("start"),
                    "end": s.get("end"),
                    "speaker": s.get("speaker"),
                    "text": text_val,
                    "label": lbl,
                    "score": top_score,
                    "negativ": sent_score.get("negativ"),
                    "neutral": sent_score.get("neutral"),
                    "positiv": sent_score.get("positiv"),
                    "intent": intent_val,
                    "intent_confidence": intent_conf,
                }
                rows.append(row)
                all_rows.append(row)

            # Show preview table per file
            table = Table(show_header=True, header_style="bold magenta")
            for col in ["index", "start", "end", "speaker", "intent", "label", "score", "text"]:
                table.add_column(col)
            for r in rows[:10]:
                table.add_row(
                    str(r["index"]),
                    f"{r['start']:.2f}" if r["start"] is not None else "",
                    f"{r['end']:.2f}" if r["end"] is not None else "",
                    str(r["speaker"]) if r["speaker"] is not None else "",
                    r["intent"],
                    r["label"],
                    f"{r['score']:.3f}",
                    r["text"][:80],
                )
            console.print(table)
            if len(rows) > 10:
                console.print(
                    f"... showing 10 of {len(rows)} segments for {os.path.basename(path)}"
                )

            # Print summary if available
            summary = report_dict.get("summary", {})
            if summary and summary.get("overall_summary"):
                console.print(
                    f"\n[bold cyan]Overall Summary for {os.path.basename(path)}:[/bold cyan]"
                )
                console.print(summary.get("overall_summary"))
                action_items = summary.get("action_items", [])
                if action_items:
                    console.print("[bold yellow]Action Items:[/bold yellow]")
                    for item in action_items:
                        console.print(
                            f"- [{item.get('assignee', 'Unassigned')}] {item.get('text')}"
                        )
                console.print("\n")

            # Fas 4 call-center highlights (agent performance + QA) if present in results
            # Makes the new actionable features visible in CLI (addresses plan "syns i CLI").
            res = report_dict.get("results", {}) or {}
            ap = res.get("agent_performance") or {}
            if ap and isinstance(ap, dict):
                a = ap.get("agent", {}) or {}
                console.print(
                    f"[bold green]Agent Performance (Fas4):[/bold green] "
                    f"empathy={a.get('empathy_score', 0):.2f} "
                    f"talk_ratio={a.get('talk_ratio', 0):.2f} "
                    f"flags={a.get('compliance_flags', [])}"
                )
                hints = ap.get("local_coaching_hints", []) or []
                if hints:
                    console.print("[yellow]Local coaching hints:[/yellow]")
                    for h in hints[:2]:
                        console.print(f"  - {h}")
            qa = res.get("qa") or res.get("compliance_qa") or {}
            if qa and isinstance(qa, dict) and "overall_qa_score" in qa:
                console.print(
                    f"[bold green]QA / Compliance (Fas4):[/bold green] "
                    f"{qa.get('overall_qa_score', 0):.1f}/100 "
                    f"passed={qa.get('passed')} risk={qa.get('risk_level')} "
                    f"flags={len(qa.get('compliance_flags', []))}"
                )

            ok += 1
            progress.advance(task, 1)

    # Save aggregate CSV if requested
    if output_csv and all_rows:
        try:
            ensure_dir(output_csv)
            pd.DataFrame(all_rows).to_csv(output_csv, index=False)
            console.print(f"[green]Saved CSV:[/green] {output_csv}")
        except Exception as e:
            console.print(f"[red]Failed to save CSV: {e}[/red]")
            raise typer.Exit(code=1) from e

    console.print(
        f"[bold]Completed[/bold]: ok={ok}, failed={fail}, total={len(files)} | elapsed={time.time() - start_all:.2f}s"
    )


if __name__ == "__main__":
    app()
