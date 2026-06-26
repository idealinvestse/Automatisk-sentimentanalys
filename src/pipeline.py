"""End-to-end call analysis pipeline for Swedish call center conversations.

Orchestrates ASR and text analysis through modular components.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .analysis import run_analyzers
from .core.models import AnalysisContext, CallAnalysisReport, Segment
from .llm.mistral_analyzer import ConversationMistralAnalyzer
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
        except Exception:
            # Profile system optional for pure LLM usage; non-fatal
            pass

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

    def _should_use_mistral_llm(self, segments: list) -> bool:
        """Decision logic for hybrid path (profile + length + confidence + explicit flags)."""
        if self.deep_analysis or self.use_mistral_llm:
            return True
        # Heuristic for callcenter profile: longer calls or many segments benefit from holistisk view
        if self.profile in {"callcenter", "call", "customer_service"} and len(segments) >= 6:
            return True
        return False

    def _run_mistral_holistic(
        self,
        segments: list[Segment] | list[dict[str, Any]],
        results: dict[str, Any],
    ) -> dict[str, Any]:
        """Call the Mistral analyzer (if available) and return enriched result or fallback dict."""
        try:
            role_map = results.get("role") or {}
            # Convert Segment objects if needed for the analyzer
            seg_dicts: list[dict[str, Any]] = []
            for s in segments:
                if isinstance(s, dict):
                    seg_dicts.append(s)
                else:
                    seg_dicts.append(s.to_dict())

            mistral = ConversationMistralAnalyzer(
                model=self.llm_model,
                api_key=self.llm_api_key,
            )
            llm_out = mistral.analyze_full_conversation(
                segments=seg_dicts,
                role_map=role_map if isinstance(role_map, dict) else {},
                local_results=results,
                profile_name=self.profile,
            )
            if llm_out.get("fallback"):
                llm_out["meta"] = llm_out.get("meta", {})
                llm_out["meta"]["llm_used"] = False
                llm_out["meta"]["llm_fallback_reason"] = llm_out.get("meta", {}).get("fallback_reason", "llm_error_or_disabled")
            else:
                llm_out.setdefault("meta", {})
                llm_out["meta"]["llm_used"] = True
            return llm_out
        except Exception as e:
            logger.warning("Mistral holistic step failed (will use local only): %s", e)
            return {"llm_used": False, "llm_fallback_reason": str(e), "error": str(e)}

    def _should_use_groq_llm(self, segments: list) -> bool:
        """Decision logic for Groq hybrid path."""
        if self.provider != "groq":
            return False
        if self.deep_analysis or self.use_mistral_llm:
            return True
        if self.profile in {"callcenter", "call", "customer_service"} and len(segments) >= 6:
            return True
        return False

    def _run_groq_holistic(
        self,
        segments: list[Segment] | list[dict[str, Any]],
        results: dict[str, Any],
    ) -> dict[str, Any]:
        """Call the Groq analyzer (if available) and return enriched result or fallback dict.

        GDPR gate: groq_eu_residency must be True or anonymize_before_llm must be active.
        """
        try:
            from .llm.groq_analyzer import GroqAnalyzer

            role_map = results.get("role") or {}
            seg_dicts: list[dict[str, Any]] = []
            for s in segments:
                if isinstance(s, dict):
                    seg_dicts.append(s)
                else:
                    seg_dicts.append(s.to_dict())

            # GDPR gate: check if PII redaction was applied or eu_residency flag is set
            pii_redacted = bool(
                results.get("pii_redaction", {}).get("total_redacted", 0) > 0
                if isinstance(results.get("pii_redaction"), dict)
                else False
            )

            if not self.groq_eu_residency and not pii_redacted:
                logger.warning(
                    "GROQ GDPR GATE: groq_eu_residency=OFF and no PII redaction detected. "
                    "Groq data centers are US/Saudi Arabia (no EU hosting). "
                    "Falling back to local analysis only."
                )
                return {
                    "llm_used": False,
                    "meta": {
                        "llm_used": False,
                        "llm_fallback_reason": "groq_gdpr_gate",
                        "provider": "groq",
                    },
                }

            groq_analyzer = GroqAnalyzer(
                model=self.llm_model,
                api_key=self.llm_api_key,
                groq_eu_residency=self.groq_eu_residency,
            )
            llm_out = groq_analyzer.analyze_full_conversation(
                segments=seg_dicts,
                role_map=role_map if isinstance(role_map, dict) else {},
                local_results=results,
                profile_name=self.profile,
                anonymize_before_llm=pii_redacted,
            )
            if llm_out.get("fallback"):
                llm_out["meta"] = llm_out.get("meta", {})
                llm_out["meta"]["llm_used"] = False
                llm_out["meta"]["llm_fallback_reason"] = llm_out.get("meta", {}).get(
                    "fallback_reason", "groq_llm_error_or_disabled"
                )
            else:
                llm_out.setdefault("meta", {})
                llm_out["meta"]["llm_used"] = True
                llm_out["meta"]["provider"] = "groq"
            return llm_out
        except Exception as e:
            logger.warning("Groq holistic step failed (will use local only): %s", e)
            return {
                "llm_used": False,
                "llm_fallback_reason": str(e),
                "error": str(e),
                "meta": {"provider": "groq"},
            }

    def _should_use_any_llm(self, segments: list) -> bool:
        """Unified decision: should we use ANY LLM path based on provider?"""
        if self.provider == "groq":
            return self._should_use_groq_llm(segments)
        return self._should_use_mistral_llm(segments)

    def _run_llm_holistic(
        self,
        segments: list[Segment] | list[dict[str, Any]],
        results: dict[str, Any],
    ) -> dict[str, Any]:
        """Route to the correct LLM analyzer based on provider."""
        if self.provider == "groq":
            return self._run_groq_holistic(segments, results)
        return self._run_mistral_holistic(segments, results)

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

        # 1. Transcribe and optionally diarize
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
        except Exception as e:
            logger.error("Transcription or ASR initialization failed for %s: %s", audio_path, e)
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
        # Privacy by design: if profile enables anonymize_before_llm, redact here so that
        # local analysis (sentiment, role, intent, etc.) + LLM + final report all see masked data.
        # Detailed log (what/where/type) is attached to results["pii_redaction"] for audit.
        pii_log = None
        try:
            from .llm.pii_redactor import redact_segments

            seg_dicts = [s.to_dict() for s in transcript.segments]
            redacted_dicts, pii_log = redact_segments(
                seg_dicts, profile_name=self.profile, return_log=True
            )
            if pii_log and pii_log.total_redacted > 0:
                logger.info(
                    "Fas 4.4.1 PII redaction (early): %d events, types=%s. Redacted text used for local analysis + LLM + report.",
                    pii_log.total_redacted, pii_log.types_redacted
                )
            # Rebuild segments with redacted text (if any redactions happened)
            if pii_log and pii_log.total_redacted > 0:
                transcript.segments = [Segment.from_dict(d) for d in redacted_dicts]
        except Exception as e:
            logger.debug("Early PII redaction skipped or failed (non-fatal): %s", e)
            pii_log = None

        # 2. Create context and run text analysis
        ctx = AnalysisContext(
            transcript=transcript,
            segments=transcript.segments,
        )

        results = run_analyzers(
            ctx,
            selected=selected_analyzers,
            analyzer_configs=self._build_analyzer_configs(),
        )

        # Attach early PII redaction log (if any) to results for audit / downstream (Fas 4.4.1)
        if pii_log is not None and pii_log.total_redacted > 0:
            results["pii_redaction"] = pii_log.model_dump()

        # Pre-extract role for Fas 4.1/4.2 (supports richer output from extended role_classifier)
        role_res = results.get("role") or {}
        role_map = role_res.get("roles", role_res) if isinstance(role_res, dict) else {}

        # --- Fas 4.1: Agent Performance & Customer Metrics (explicit integration) ---
        # MUST run BEFORE any LLM call so that local metrics are available in results
        # for _run_mistral_holistic -> local_ctx["agent_performance_local"] (required for 4.1.2 hybrid).
        # Always run (rule-based, zero cost). Produces Pydantic CallAgentPerformance.
        # Output merged into results["agent_performance"] (and local assessment snapshot).
        # Later (4.1.2) LLM agent_assessment is merged on top for coaching recs + evidence.
        # See UTVECKLINGSPLAN_Fas4... Task 4.1.1 and src/agent_performance.py .
        try:
            from .agent_performance import compute_call_agent_performance

            sent_res = results.get("sentiment") or []
            agent_perf = compute_call_agent_performance(
                segments=transcript.segments or [],
                role_map=role_map if isinstance(role_map, dict) else {},
                sentiment_results=sent_res,
                profile_name=self.profile,
            )
            results["agent_performance"] = agent_perf.model_dump()
            # Local baseline for agent_assessment (enriched by LLM path in 4.1.2)
            local_assess = {
                "empathy_score": agent_perf.agent.empathy_score,
                "compliance_flags": agent_perf.agent.compliance_flags,
                "strengths": [],
                "weaknesses": [],
                "specific_coaching_recommendations": [],
                "overall_assessment": None,
                "source": "local_rules_fas4.1",
                "talk_listen_ratio": agent_perf.agent.talk_listen_ratio,
                "intervention_count": agent_perf.agent.intervention_count,
                "evidence_spans": [],  # local rules provide flags; spans come from LLM in 4.1.2
            }
            results["agent_assessment_local"] = local_assess
            # Make report always contain agent_assessment + customer_metrics (per Task 4.1.1 acceptance)
            results["agent_assessment"] = local_assess
            results["customer_metrics"] = agent_perf.customer.model_dump()
            logger.info(
                "Fas 4.1 agent_performance computed | empathy=%.2f flags=%s talk_ratio=%.2f",
                agent_perf.agent.empathy_score,
                agent_perf.agent.compliance_flags,
                agent_perf.agent.talk_ratio,
            )
        except Exception as e:
            logger.warning("Fas 4.1 agent_performance step failed (non-fatal): %s", e)
            results["agent_performance"] = {"error": str(e), "fallback": True}
            local_assess = None

        # --- LLM holistic layer (Fas 3.2) ---
        # Now runs *after* agent_performance so local_ctx receives
        # the metrics for detailed assessment + hybrid coaching recommendations (4.1.2).
        # Provider: Mistral/OpenRouter (default) or Groq (fast/cheap, but US-hosted).
        llm_result: dict[str, Any] = {}
        if self._should_use_any_llm(transcript.segments or []):
            llm_result = self._run_llm_holistic(transcript.segments or [], results)
            results["llm"] = llm_result
            if llm_result.get("meta", {}).get("llm_used"):
                logger.info("Pipeline used %s LLM for holistic analysis (model=%s, cached=%s)",
                            self.provider,
                            llm_result.get("meta", {}).get("model"),
                            llm_result.get("meta", {}).get("cached"))

        # Merge LLM agent_assessment (if present from 4.1.2+ or Fas3) over local for results["agent_assessment"]
        # This ensures report always has rich agent_assessment when deep analysis used.
        llm_assess = (results.get("llm") or {}).get("agent_assessment")
        if llm_assess and isinstance(llm_assess, dict) and llm_assess.get("empathy_score") is not None:
            # Prefer LLM version (has evidence_spans + coaching recs); keep local metrics in agent_performance
            results["agent_assessment"] = llm_assess
            logger.debug("Merged LLM agent_assessment into results (evidence-based coaching available)")

        # --- Fas 4.2: Compliance & QA Auto-Scoring Engine (explicit integration, 4.2.1+4.2.2) ---
        # Customizable via configs/qa_scorecards/*.yaml. Hybrid rule+LLM.
        # Output in results["qa"] / ["compliance_qa"] with overall, passed/failed, evidence, risk.
        # LLM only used for hybrid/llm criteria when deep path active (logged).
        try:
            from .compliance_qa import score_call_with_default_scorecard

            use_llm_qa = bool(self.use_mistral_llm or (results.get("llm") or {}).get("meta", {}).get("llm_used"))
            qa_analyzer = None
            if use_llm_qa:
                if self.provider == "groq":
                    from .llm.groq_analyzer import GroqAnalyzer
                    qa_analyzer = GroqAnalyzer(
                        model=self.llm_model,
                        api_key=self.llm_api_key,
                        groq_eu_residency=self.groq_eu_residency,
                    )
                else:
                    from .llm.mistral_analyzer import ConversationMistralAnalyzer
                    qa_analyzer = ConversationMistralAnalyzer(
                        model=self.llm_model,
                        api_key=self.llm_api_key,
                    )
            qa_res = score_call_with_default_scorecard(
                segments=transcript.segments or [],
                role_map=role_map if isinstance(role_map, dict) else {},
                local_signals={"agent_performance": results.get("agent_performance"), "agent_assessment": results.get("agent_assessment")},
                profile_name=self.profile,
                use_llm=use_llm_qa,
                analyzer=qa_analyzer,
            )
            results["qa"] = qa_res
            results["compliance_qa"] = qa_res
            if isinstance(qa_res, dict) and qa_res.get("llm_criteria_used"):
                logger.info("Fas 4.2 QA scoring used LLM for criteria=%s (hybrid)", qa_res.get("llm_criteria_used"))
            logger.info("Fas 4.2 QA complete | score=%.1f passed=%s risk=%s",
                        qa_res.get("overall_qa_score", 0) if isinstance(qa_res, dict) else 0,
                        qa_res.get("passed") if isinstance(qa_res, dict) else False,
                        qa_res.get("risk_level") if isinstance(qa_res, dict) else "?")
        except Exception as e:
            logger.warning("Fas 4.2 QA scoring failed (non-fatal): %s", e)
            results["qa"] = {"error": str(e), "overall_qa_score": 0.0, "passed": False}

        # --- Fas 4.4.2: Alerting & Workflow Engine (explicit integration) ---
        # Regelbaserade alerts från Fas4 data (sentiment, agent, qa, aggregator).
        # Producerar evidence-based Alerts med recommended_actions.
        # Kan triggas per call eller från aggregate_insights().
        try:
            from .alerting import run_alerts_on_results
            alerts = run_alerts_on_results(results)
            if alerts:
                results["alerts"] = alerts
                logger.info("Fas 4.4.2 alerts triggered: %d (highest severity in first)", len(alerts))
        except Exception as e:
            logger.warning("Fas 4.4.2 alerting failed (non-fatal): %s", e)

        proc_time = round(time.time() - t0, 2)

        # 3. Construct backwards-compatible report
        segments_dict = [s.to_dict() for s in transcript.segments]
        diarization_data = transcript.diarization

        return CallAnalysisReport(
            segments=segments_dict,
            sentiment_results=results.get("sentiment", []),
            intent_results=results.get("intent", []),
            diarization=diarization_data,
            summary=results.get("summary", {}),
            topics=results.get("topics", {}),
            insights=results.get("insights", {}),
            risks=results.get(
                "predictive", {}
            ),  # Map predictive risks to risks for backward-compatibility
            processing_time_s=proc_time,
            results=results,
            llm=llm_result,
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
        pii_log = None
        try:
            from .llm.pii_redactor import redact_segments

            seg_dicts = [s.to_dict() for s in typed_segments]
            redacted_dicts, pii_log = redact_segments(
                seg_dicts, profile_name=self.profile, return_log=True
            )
            if pii_log and pii_log.total_redacted > 0:
                logger.info(
                    "Fas 4.4.1 PII redaction (early, segments): %d events, types=%s",
                    pii_log.total_redacted, pii_log.types_redacted
                )
            if pii_log and pii_log.total_redacted > 0:
                typed_segments = [Segment.from_dict(d) for d in redacted_dicts]
        except Exception as e:
            logger.debug("Early PII redaction (segments) skipped: %s", e)
            pii_log = None

        # Create context and run text analysis
        ctx = AnalysisContext(
            transcript=None,
            segments=typed_segments,
        )

        results = run_analyzers(
            ctx,
            selected=selected_analyzers,
            analyzer_configs=self._build_analyzer_configs(),
        )

        # Attach early PII redaction log (segments path)
        if pii_log is not None and pii_log.total_redacted > 0:
            results["pii_redaction"] = pii_log.model_dump()

        # Pre-extract role for 4.1/4.2 (supports richer output from extended role_classifier)
        role_res = results.get("role") or {}
        role_map = role_res.get("roles", role_res) if isinstance(role_res, dict) else {}

        # --- Fas 4.1: Agent Performance & Customer Metrics (explicit integration, segments path) ---
        # MUST run BEFORE LLM (see comment + reorder in analyze_audio). Same merge into results.
        try:
            from .agent_performance import compute_call_agent_performance

            sent_res = results.get("sentiment") or []
            agent_perf = compute_call_agent_performance(
                segments=typed_segments or [],
                role_map=role_map if isinstance(role_map, dict) else {},
                sentiment_results=sent_res,
                profile_name=self.profile,
            )
            results["agent_performance"] = agent_perf.model_dump()
            local_assess = {
                "empathy_score": agent_perf.agent.empathy_score,
                "compliance_flags": agent_perf.agent.compliance_flags,
                "strengths": [],
                "weaknesses": [],
                "specific_coaching_recommendations": [],
                "overall_assessment": None,
                "source": "local_rules_fas4.1",
                "talk_listen_ratio": agent_perf.agent.talk_listen_ratio,
                "intervention_count": agent_perf.agent.intervention_count,
                "evidence_spans": [],
            }
            results["agent_assessment_local"] = local_assess
            results["agent_assessment"] = local_assess
            results["customer_metrics"] = agent_perf.customer.model_dump()
            logger.info(
                "Fas 4.1 agent_performance (segments) computed | empathy=%.2f flags=%s",
                agent_perf.agent.empathy_score,
                agent_perf.agent.compliance_flags,
            )
        except Exception as e:
            logger.warning("Fas 4.1 agent_performance (segments) failed (non-fatal): %s", e)
            results["agent_performance"] = {"error": str(e), "fallback": True}
            local_assess = None

        # --- LLM holistic layer (Fas 3.2) ---
        # Now runs *after* agent_performance so local metrics reach the LLM for detailed assessment.
        llm_result: dict[str, Any] = {}
        if self._should_use_any_llm(typed_segments or []):
            llm_result = self._run_llm_holistic(typed_segments or [], results)
            results["llm"] = llm_result
            if llm_result.get("meta", {}).get("llm_used"):
                logger.info("Pipeline used %s LLM for holistic analysis (model=%s, cached=%s)",
                            self.provider,
                            llm_result.get("meta", {}).get("model"),
                            llm_result.get("meta", {}).get("cached"))

        # Merge LLM agent_assessment (if present) over local (segments path)
        llm_assess = (results.get("llm") or {}).get("agent_assessment")
        if llm_assess and isinstance(llm_assess, dict) and llm_assess.get("empathy_score") is not None:
            results["agent_assessment"] = llm_assess
            logger.debug("Merged LLM agent_assessment into results (segments path)")

        # --- Fas 4.2 QA (segments path, explicit) ---
        try:
            from .compliance_qa import score_call_with_default_scorecard

            use_llm_qa = bool(self.use_mistral_llm or (results.get("llm") or {}).get("meta", {}).get("llm_used"))
            qa_analyzer = None
            if use_llm_qa:
                if self.provider == "groq":
                    from .llm.groq_analyzer import GroqAnalyzer
                    qa_analyzer = GroqAnalyzer(
                        model=self.llm_model,
                        api_key=self.llm_api_key,
                        groq_eu_residency=self.groq_eu_residency,
                    )
                else:
                    from .llm.mistral_analyzer import ConversationMistralAnalyzer
                    qa_analyzer = ConversationMistralAnalyzer(
                        model=self.llm_model,
                        api_key=self.llm_api_key,
                    )
            qa_res = score_call_with_default_scorecard(
                segments=typed_segments or [],
                role_map=role_map if isinstance(role_map, dict) else {},
                local_signals={"agent_performance": results.get("agent_performance"), "agent_assessment": results.get("agent_assessment")},
                profile_name=self.profile,
                use_llm=use_llm_qa,
                analyzer=qa_analyzer,
            )
            results["qa"] = qa_res
            results["compliance_qa"] = qa_res
            if isinstance(qa_res, dict) and qa_res.get("llm_criteria_used"):
                logger.info("Fas 4.2 (segments) QA used LLM for %s", qa_res.get("llm_criteria_used"))
        except Exception as e:
            logger.warning("Fas 4.2 QA (segments) failed (non-fatal): %s", e)
            results["qa"] = {"error": str(e), "overall_qa_score": 0.0, "passed": False}

        # --- Fas 4.4.2: Alerting & Workflow Engine (explicit integration) ---
        # Regelbaserade alerts från Fas4 data (sentiment, agent, qa, aggregator).
        # Producerar evidence-based Alerts med recommended_actions.
        # Kan triggas per call eller från aggregate_insights().
        try:
            from .alerting import run_alerts_on_results
            alerts = run_alerts_on_results(results)
            if alerts:
                results["alerts"] = alerts
                logger.info("Fas 4.4.2 alerts triggered: %d (highest severity in first)", len(alerts))
        except Exception as e:
            logger.warning("Fas 4.4.2 alerting failed (non-fatal): %s", e)

        proc_time = round(time.time() - t0, 2)

        return CallAnalysisReport(
            segments=[s.to_dict() for s in typed_segments],
            sentiment_results=results.get("sentiment", []),
            intent_results=results.get("intent", []),
            diarization=None,
            summary=results.get("summary", {}),
            topics=results.get("topics", {}),
            insights=results.get("insights", {}),
            risks=results.get(
                "predictive", {}
            ),  # Map predictive risks to risks for backward-compatibility
            processing_time_s=proc_time,
            results=results,
            llm=llm_result,
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
            except Exception:
                pass

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
