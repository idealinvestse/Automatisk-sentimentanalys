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
        llm_model: Specific Mistral model on OpenRouter (default mistralai/mistral-medium-3.5).
        deep_analysis: Alias / stronger signal to enable the LLM deep path (profile + length + low conf heuristics also apply).
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

        # Task 3.2.3: profile-driven LLM defaults (callcenter enables by default)
        try:
            from .profiles import resolve_profile
            _, spec = resolve_profile(profile=profile)
            llm_spec = (spec or {}).get("llm", {}) or {}
            if not self.use_mistral_llm and not self.deep_analysis:
                self.use_mistral_llm = bool(llm_spec.get("enabled", False))
            if self.llm_model is None:
                self.llm_model = llm_spec.get("default_model")
        except Exception:
            # Profile system optional for pure LLM usage; non-fatal
            pass

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

            mistral = ConversationMistralAnalyzer(model=self.llm_model)
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
            transcript = transcriber.transcribe(
                audio_path=audio_path,
                language=language,
                diarize=run_diarization,
                num_speakers=num_speakers,
                hotwords=hotwords,
                initial_prompt=initial_prompt,
                preprocess=preprocess,
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

        # --- Mistral / LLM holistic layer (Fas 3.2) ---
        # Now runs *after* agent_performance so local_ctx in mistral_analyzer receives
        # the metrics for detailed assessment + hybrid coaching recommendations (4.1.2).
        llm_result: dict[str, Any] = {}
        if self._should_use_mistral_llm(transcript.segments or []):
            llm_result = self._run_mistral_holistic(transcript.segments or [], results)
            results["llm"] = llm_result
            if llm_result.get("meta", {}).get("llm_used"):
                logger.info("Pipeline used Mistral LLM for holistic analysis (model=%s, cached=%s)",
                            llm_result.get("meta", {}).get("model"), llm_result.get("meta", {}).get("cached"))

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
            from .llm.mistral_analyzer import ConversationMistralAnalyzer

            use_llm_qa = bool(self.use_mistral_llm or (results.get("llm") or {}).get("meta", {}).get("llm_used"))
            qa_analyzer = None
            if use_llm_qa:
                qa_analyzer = ConversationMistralAnalyzer(model=self.llm_model)
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

        # --- Mistral / LLM holistic layer (Fas 3.2) ---
        # Now runs *after* agent_performance so local metrics reach the LLM for detailed assessment.
        llm_result: dict[str, Any] = {}
        if self._should_use_mistral_llm(typed_segments or []):
            llm_result = self._run_mistral_holistic(typed_segments or [], results)
            results["llm"] = llm_result
            if llm_result.get("meta", {}).get("llm_used"):
                logger.info("Pipeline used Mistral LLM for holistic analysis (model=%s, cached=%s)",
                            llm_result.get("meta", {}).get("model"), llm_result.get("meta", {}).get("cached"))

        # Merge LLM agent_assessment (if present) over local (segments path)
        llm_assess = (results.get("llm") or {}).get("agent_assessment")
        if llm_assess and isinstance(llm_assess, dict) and llm_assess.get("empathy_score") is not None:
            results["agent_assessment"] = llm_assess
            logger.debug("Merged LLM agent_assessment into results (segments path)")

        # --- Fas 4.2 QA (segments path, explicit) ---
        try:
            from .compliance_qa import score_call_with_default_scorecard
            from .llm.mistral_analyzer import ConversationMistralAnalyzer

            use_llm_qa = bool(self.use_mistral_llm or (results.get("llm") or {}).get("meta", {}).get("llm_used"))
            qa_analyzer = ConversationMistralAnalyzer(model=self.llm_model) if use_llm_qa else None
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
