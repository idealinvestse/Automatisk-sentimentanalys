"""End-to-end call analysis pipeline for Swedish call center conversations.

Orchestrates ASR and text analysis through modular components.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .analysis.intent_utils import intents_as_tuples
from .core.status import get_status_reporter
from .core.tracing import span
from .core.models import CallAnalysisReport, Segment
from .pipeline_steps import (
    PipelineLLMContext,
    apply_early_pii_redaction,
    run_fas4_enrichment,
    run_registry_analyzers,
    should_use_any_llm,
)
from .transcription import get_transcriber
from .transcription.factory import resolve_preprocess_mode

logger = logging.getLogger(__name__)


class CallAnalysisPipeline:
    """End-to-end pipeline for analyzing Swedish call center conversations.

    Orchestrates transcription and modular text analysis.

    Args:
        sentiment_model: HuggingFace model for sentiment analysis.
        intent_backend: 'heuristic' or 'model' for intent classification.
        diarization_backend: 'heuristic' or 'pyannote' for speaker diarization.
        hf_token: HuggingFace token (required for pyannote diarization).
        device: 'cpu', 'cuda', or 'auto'.
        profile: Sentiment profile (e.g., 'callcenter', 'default').
        asr_backend: ASR backend to use ('faster' (default), 'transformers', 'whisperx').
        asr_model: ASR model name or alias (e.g., 'kb-whisper-large').
        Note: 'whisperx' provides word-level alignment and optional integrated diarization.
        use_mistral_llm: Force use of Mistral via OpenRouter for holistic analysis (Task 3.2.2).
        llm_model: Specific LLM model slug (OpenRouter or Groq, depending on provider).
        deep_analysis: Alias / stronger signal to enable the LLM deep path (profile + length + low conf heuristics also apply).
        llm_api_key: Explicit API key override (e.g. from dashboard UI).
        provider: LLM provider: 'openrouter' (default) | 'groq'.
        groq_eu_residency: GDPR gate for Groq (US/Saudi data centers).
    """

    def __init__(
        self,
        sentiment_model: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        intent_backend: str = "heuristic",
        diarization_backend: str = "heuristic",  # Kept for backward compatibility
        hf_token: str | None = None,
        device: str = "cpu",
        profile: str = "default",
        asr_backend: str = "faster",
        asr_model: str = "kb-whisper-large",
        # --- LLM / Mistral (Fas 3.2) ---
        use_mistral_llm: bool = False,
        llm_model: str | None = None,
        deep_analysis: bool = False,
        # explicit override (e.g. from dashboard UI or API). Highest priority in get_openrouter_api_key.
        llm_api_key: str | None = None,
        provider: str = "openrouter",
        groq_eu_residency: bool = False,
        cache: Any | None = None,
        async_analyzers: bool = False,
    ) -> None:
        self.sentiment_model = sentiment_model
        self.intent_backend = intent_backend
        self.diarization_backend = diarization_backend
        self.hf_token = hf_token
        self.device = device
        self.profile = profile
        self.asr_backend = asr_backend
        self.asr_model = asr_model
        self.use_mistral_llm = use_mistral_llm
        self.llm_model = llm_model
        self.deep_analysis = deep_analysis
        self.llm_api_key = llm_api_key
        self.provider = provider
        self.groq_eu_residency = groq_eu_residency
        self.async_analyzers = async_analyzers

        # Task 3.2.3: profile-driven LLM defaults (callcenter enables by default)
        try:
            from .profiles import resolve_profile
            _, spec = resolve_profile(profile=profile)
            llm_spec = (spec or {}).get("llm", {}) or {}
            if not self.use_mistral_llm and not self.deep_analysis:
                self.use_mistral_llm = bool(llm_spec.get("enabled", False))
            if self.llm_model is None:
                self.llm_model = llm_spec.get("default_model")
            # Auto-detect provider from profile config
            if self.provider == "openrouter" and llm_spec.get("provider"):
                self.provider = str(llm_spec.get("provider", "openrouter"))
            # Groq GDPR gate from profile
            if self.provider == "groq" and not self.groq_eu_residency:
                groq_cfg = (spec or {}).get("groq", {}) or {}
                self.groq_eu_residency = bool(groq_cfg.get("groq_eu_residency", False))
        except Exception as exc:
            logger.debug("Profile resolution skipped (optional): %s", exc)

        # Fas 4.5.1: Advanced caching / pre-computation (file by default, Redis optional)
        from .caching import AggregateCache

        if cache is not None:
            self.cache = cache
        else:
            self.cache = AggregateCache(use_redis=False)  # set True + redis_url for prod

    def _build_analyzer_configs(self) -> dict[str, dict[str, Any]]:
        """Build per-analyzer configuration from pipeline settings.

        Returns:
            Mapping of analyzer name → constructor kwargs.
        """
        return {
            "sentiment": {
                "model_name": self.sentiment_model,
                "device": self.device,
            },
            "intent": {
                "backend": self.intent_backend,
            },
            "llm_judge": {
                "min_confidence": 0.6,
                "max_segments_per_call": 5,
                "max_cost_usd": 0.10,
                "provider": self.provider,
                "model": self.llm_model or "llama-3.1-8b-instant",
                "api_key": self.llm_api_key,
            },
        }

    def _llm_context(self) -> PipelineLLMContext:
        """Build LLM routing context for Fas-4 pipeline steps."""
        return PipelineLLMContext(
            profile=self.profile,
            provider=self.provider,
            use_mistral_llm=self.use_mistral_llm,
            deep_analysis=self.deep_analysis,
            llm_model=self.llm_model,
            llm_api_key=self.llm_api_key,
            groq_eu_residency=self.groq_eu_residency,
        )

    def _run_fas4_enrichment(
        self,
        segments: list[Segment],
        results: dict[str, Any],
    ) -> dict[str, Any]:
        """Run FAS 4 enrichment steps shared by audio and segment analysis."""
        return run_fas4_enrichment(segments, results, self._llm_context())

    def _build_report(
        self,
        *,
        segments: list[Segment],
        results: dict[str, Any],
        llm_result: dict[str, Any],
        proc_time: float,
        diarization: dict[str, Any] | None,
    ) -> CallAnalysisReport:
        """Build the backwards-compatible report object."""
        return CallAnalysisReport(
            segments=[s.to_dict() for s in segments],
            sentiment_results=results.get("sentiment", []),
            intent_results=intents_as_tuples(results.get("intent", [])),
            diarization=diarization,
            summary=results.get("summary", {}),
            topics=results.get("topics", {}),
            insights=results.get("insights", {}),
            risks=results.get("predictive", {}),
            processing_time_s=proc_time,
            results=results,
            llm=llm_result,
        )

    def _run_local_analysis(
        self,
        segments: list[Segment],
        *,
        selected_analyzers: list[str] | None,
        transcript: Any | None = None,
    ) -> tuple[list[Segment], dict[str, Any], Any | None]:
        """Run early redaction and registry analyzers on a segment list."""
        status = get_status_reporter()
        with span("pii_redact", segment_count=len(segments)):
            status.phase("pipeline", "pii_redact", "PII-redigering", segment_count=len(segments))
            redacted_segments, pii_log = apply_early_pii_redaction(
                segments,
                profile_name=self.profile,
            )

        with span("run_analyzers", profile=self.profile):
            status.phase(
                "pipeline",
                "run_analyzers",
                "Kör analyzer-registry",
                segment_count=len(redacted_segments),
            )
            results = run_registry_analyzers(
                redacted_segments,
                profile=self.profile,
                selected_analyzers=selected_analyzers,
                analyzer_configs=self._build_analyzer_configs(),
                async_mode=self.async_analyzers,
                transcript=transcript,
                skip_llm_superseded=should_use_any_llm(
                    redacted_segments or [], self._llm_context()
                ),
            )

        if pii_log is not None and pii_log.total_redacted > 0:
            results["pii_redaction"] = pii_log.model_dump()

        return redacted_segments, results, pii_log

    def analyze_audio(
        self,
        audio_path: str,
        num_speakers: int | None = 2,
        language: str = "sv",
        run_diarization: bool = True,
        selected_analyzers: list[str] | None = None,
        hotwords: list[str] | None = None,
        initial_prompt: str | None = None,
        preprocess: bool = False,
        preprocess_mode: str | None = None,
    ) -> CallAnalysisReport:
        """Analyze a call from an audio file.

        Args:
            audio_path: Path to audio file.
            num_speakers: Expected number of speakers.
            language: Language code for ASR.
            run_diarization: Whether to run speaker diarization.
            selected_analyzers: Optional list of analyzer names to run. Runs all by default.

        Returns:
            CallAnalysisReport with full analysis.
        """
        t0 = time.time()
        status = get_status_reporter()
        status.phase("pipeline", "load_audio", f"Laddar ljud: {audio_path}")

        # 1. Transcribe and optionally diarize
        with span("transcribe", backend=self.asr_backend, model=self.asr_model):
            status.phase(
                "pipeline",
                "transcribe",
                "Transkriberar ljud",
                backend=self.asr_backend,
                model=self.asr_model,
            )
            try:
                transcriber = get_transcriber(
                    backend=self.asr_backend,
                    model_name=self.asr_model,
                    device=self.device,
                )
                resolved_preprocess_mode = resolve_preprocess_mode(
                    preprocess=preprocess,
                    preprocess_mode=preprocess_mode,
                    profile=self.profile,
                )
                transcript = transcriber.transcribe(
                    audio_path=audio_path,
                    language=language,
                    diarize=run_diarization,
                    num_speakers=num_speakers,
                    hotwords=hotwords,
                    initial_prompt=initial_prompt,
                    preprocess=preprocess,
                    preprocess_mode=resolved_preprocess_mode,
                )
                status.info(
                    "pipeline",
                    "transcribe",
                    "Transkribering klar",
                    segment_count=len(transcript.segments),
                )
            except Exception as e:
                logger.error(
                    "Transcription or ASR initialization failed for %s: %s",
                    audio_path,
                    e,
                    exc_info=True,
                )
                status.error("pipeline", "transcribe", f"Transkribering misslyckades: {e}")
                from .core.models import Transcript

                transcript = Transcript(
                    model=self.asr_model,
                    backend=self.asr_backend,
                    language=language,
                    duration=0.0,
                    processing_time=0.0,
                    segments=[],
                    diarization={"segments": [], "backend": "failed", "error": str(e)},
                )

        # --- Fas 4.4.1: Early PII Redaction (before ANY local analyzers or LLM) ---
        transcript.segments, results, _pii_log = self._run_local_analysis(
            transcript.segments,
            selected_analyzers=selected_analyzers,
            transcript=transcript,
        )

        proc_time = round(time.time() - t0, 2)
        with span("fas4_enrichment", profile=self.profile):
            status.phase("pipeline", "fas4_enrichment", "Fas 4-enrichment")
            llm_result = self._run_fas4_enrichment(transcript.segments or [], results)
        status.phase("pipeline", "complete", "Analys klar", processing_time_s=proc_time)
        return self._build_report(
            segments=transcript.segments,
            results=results,
            llm_result=llm_result,
            proc_time=proc_time,
            diarization=transcript.diarization,
        )

    def analyze_segments(
        self,
        segments: list[dict[str, Any]],
        selected_analyzers: list[str] | None = None,
    ) -> CallAnalysisReport:
        """Analyze pre-existing transcript segments.

        Args:
            segments: List of dicts with 'text' key and optionally 'speaker'.
            selected_analyzers: Optional list of analyzer names to run. Runs all by default.

        Returns:
            CallAnalysisReport with full analysis.
        """
        t0 = time.time()
        status = get_status_reporter()
        status.phase("pipeline", "analyze_segments", "Analyserar segment", count=len(segments))

        # Convert segment dicts to Segment dataclasses
        typed_segments = []
        for s in segments:
            try:
                start = float(s.get("start", 0.0) or 0.0)
                end = float(s.get("end", 0.0) or 0.0)
            except (TypeError, ValueError):
                start = 0.0
                end = 0.0
            typed_segments.append(
                Segment(
                    start=start,
                    end=end,
                    text=str(s.get("text", "")),
                    speaker=s.get("speaker"),
                    avg_confidence=s.get("avg_confidence"),
                    confidence=s.get("confidence") or s.get("avg_confidence"),
                    low_confidence=bool(s.get("low_confidence", False)),
                )
            )

        # --- Fas 4.4.1: Early PII Redaction (segments path, before analyzers/LLM) ---
        typed_segments, results, _pii_log = self._run_local_analysis(
            typed_segments,
            selected_analyzers=selected_analyzers,
        )

        proc_time = round(time.time() - t0, 2)
        with span("fas4_enrichment", profile=self.profile):
            status.phase("pipeline", "fas4_enrichment", "Fas 4-enrichment")
            llm_result = self._run_fas4_enrichment(typed_segments, results)
        status.phase("pipeline", "complete", "Analys klar", processing_time_s=proc_time)
        return self._build_report(
            segments=typed_segments,
            results=results,
            llm_result=llm_result,
            proc_time=proc_time,
            diarization=None,
        )

    # ------------------------------------------------------------------
    # Fas 4.3: Explicit batch aggregation integration (Insights Aggregator)
    # ------------------------------------------------------------------
    def aggregate_insights(
        self,
        reports: list[CallAnalysisReport],
    ) -> dict[str, Any]:
        """Aggregate multiple per-call reports into cross-call insights (Fas 4.3.1).

        Explicit integration point as required by the plan:
            - Called from pipeline after collecting several CallAnalysisReport (batch jobs, dashboard backend, etc.).
            - Uses the new Pydantic AggregatedInsights (hot_topics with evidence_spans, trends, clusters, agent issues).
            - If self.use_mistral_llm / deep_analysis, passes a Mistral analyzer so that hot topic / cluster descriptions can be LLM-enriched (documented inside aggregator).
            - Output is directly mergable / storable (e.g. attach to a "batch" result or API response).

        Example usage (outside this file):
            pipe = CallAnalysisPipeline(profile="callcenter", use_mistral_llm=True)
            reports = [pipe.analyze_audio(p) for p in call_files]
            agg = pipe.aggregate_insights(reports)
            # agg["hot_topics"] contains volume, sentiment, trend, evidence_spans + optional llm_summary

        See src/insights_aggregator.py for full details, fallbacks, caching, and privacy notes.
        """
        try:
            from .insights_aggregator import aggregate_call_reports

            mistral = None
            if self.use_mistral_llm or self.deep_analysis:
                if self.provider == "groq":
                    from .llm.groq_analyzer import GroqAnalyzer
                    mistral = GroqAnalyzer(
                        model=self.llm_model,
                        api_key=self.llm_api_key,
                        groq_eu_residency=self.groq_eu_residency,
                    )
                    logger.info("Fas 4.3 aggregator using Groq for cluster/topic descriptions (selective)")
                else:
                    from .llm.mistral_analyzer import ConversationMistralAnalyzer
                    mistral = ConversationMistralAnalyzer(
                        model=self.llm_model,
                        api_key=self.llm_api_key,
                    )
                    logger.info("Fas 4.3 aggregator using Mistral for cluster/topic descriptions (selective)")

            agg_dict = aggregate_call_reports(reports, mistral_analyzer=mistral)
            logger.info(
                "Fas 4.3 aggregate_insights complete | calls=%d hot_topics=%d",
                len(reports),
                len(agg_dict.get("hot_topics", [])),
            )

            # Also run alerting on aggregator output (per plan 4.4.2: trend-based alerts)
            try:
                from .alerting import AlertEngine
                eng = AlertEngine()
                agg_alerts = eng.check_from_aggregate(agg_dict)
                if agg_alerts:
                    agg_dict["alerts_from_trends"] = [a.model_dump() if hasattr(a, "model_dump") else a for a in agg_alerts]
            except Exception as exc:
                logger.debug("Trend alerts skipped (non-fatal): %s", exc)

            return agg_dict
        except Exception as e:
            logger.warning("Fas 4.3 aggregate_insights failed (non-fatal): %s", e)
            return {"error": str(e), "hot_topics": [], "meta": {"num_calls": len(reports)}}

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        corpus: list[CallAnalysisReport] | list[dict] | None = None,
    ) -> dict[str, Any]:
        """Fas 4.3.2: Hybrid semantic + keyword search over call data.

        Explicit integration:
            hits = pipe.semantic_search("faktura AND låg empati", top_k=5, corpus=reports)
            # Returns Pydantic-like dict with hits containing score, highlights, evidence_spans, metadata.

        If corpus is None, builds a trivial index (not recommended for large use; pass pre-built reports).
        Uses optional embeddings/FAISS for vector part + keyword boost.
        """
        try:
            from .semantic_search import SemanticSearchEngine, build_semantic_index_from_reports

            if corpus:
                engine = build_semantic_index_from_reports(corpus)
            else:
                engine = SemanticSearchEngine()
                # minimal empty
            res = engine.search(query, top_k=top_k, filters=filters or {})
            logger.info("Fas 4.3.2 semantic_search | q=%s hits=%d", query[:50], len(res.hits))
            return res.model_dump()
        except Exception as e:
            logger.warning("Fas 4.3.2 semantic_search failed: %s", e)
            return {"query": query, "hits": [], "meta": {"error": str(e)}}

    # ------------------------------------------------------------------
    # Fas 4.5.1: Pre-computation & Advanced Caching (explicit integration)
    # ------------------------------------------------------------------
    def get_cached_agent_performance(
        self,
        agent_id: str,
        reports: list[CallAnalysisReport],
        window: str = "7d",
    ) -> dict[str, Any]:
        """Pre-computed + cached agent metrics (Fas 4.5.1).

        Uses general AggregateCache (file/Redis). Invalidation: new calls for this agent
        should call self.cache.invalidate(f"agent:{agent_id}") or use time-bucketed keys.

        Example (dashboard):
            metrics = pipe.get_cached_agent_performance("Agent-5", recent_reports)
            # fast even if 10000 calls in reports
        """
        from .caching import precompute_agent_aggregates
        return precompute_agent_aggregates(
            reports, cache=self.cache, agent_id=agent_id, window=window
        )

    def get_cached_hot_topics(
        self,
        reports: list[CallAnalysisReport],
        window: str = "7d",
    ) -> dict[str, Any]:
        """Pre-computed + cached hot topics/trends (Fas 4.5.1).

        Smart invalidation strategy documented in src/caching.py.
        """
        from .caching import precompute_hot_topics
        return precompute_hot_topics(reports, cache=self.cache, window=window)

    def invalidate_aggregate_cache(self, prefix: str):
        """Explicit invalidation hook (call on new call data for agent or scheduled)."""
        self.cache.invalidate(prefix)
