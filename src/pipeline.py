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
            # === TASK-01: LLM Judge for low-confidence sentiment segments ===
            "llm_judge": {
                "min_confidence": 0.6,
                "max_segments_per_call": 5,
                "max_cost_usd": 0.10,
                "provider": self.provider,
                "model": self.llm_model or "llama-3.1-8b-instant",
                "api_key": self.llm_api_key,
            },
        }