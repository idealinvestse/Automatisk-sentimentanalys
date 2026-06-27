    def fetch_and_save_models_catalog(
        output_path: str | Path = "data/openrouter_models_catalog.json",
        api_key: str | None = None,
        include_pricing: bool = True,
    ) -> dict[str, Any]:
        """Scan OpenRouter for all available models and save catalog with cost + short info.

        Uses the public /api/v1/models endpoint (no key required for basic info, but key recommended
        for accurate pricing and to avoid rate limits).

        Saves a JSON with:
        - scanned_at (ISO)
        - count
        - models: list of {id, name, description (short), context_length, pricing {prompt, completion per token USD}, architecture, capabilities...}

        This enables dynamic model selection in dashboard/launcher and cost estimation beyond the
        hardcoded Mistral subset in PRICING.

        Call from CLI: python -m src.cli scan-openrouter-models
        Or from dashboard settings button.
        """
        import urllib.request
        import urllib.error

        url = "https://openrouter.ai/api/v1/models"
        headers = {}
        key = api_key or get_openrouter_api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"

        logger.info("Scanning OpenRouter for available models...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw = json.loads(resp.read().decode("utf-8"))

            models_raw = raw.get("data", [])
            catalog_models = []

            for m in models_raw:
                pricing = m.get("pricing", {}) or {}
                # pricing values are USD per token (e.g. 0.0000015 = $1.5/M)
                model_entry = {
                    "id": m.get("id"),
                    "name": m.get("name"),
                    "description": (m.get("description") or "")[:600].strip(),  # short info
                    "context_length": m.get("context_length"),
                    "pricing": {
                        "prompt": float(pricing.get("prompt", 0)) if pricing.get("prompt") else 0.0,
                        "completion": float(pricing.get("completion", 0)) if pricing.get("completion") else 0.0,
                        "prompt_per_million": round(float(pricing.get("prompt", 0)) * 1_000_000, 4) if pricing.get("prompt") else 0,
                        "completion_per_million": round(float(pricing.get("completion", 0)) * 1_000_000, 4) if pricing.get("completion") else 0,
                    },
                    "architecture": m.get("architecture", {}),
                    "top_provider": m.get("top_provider", {}),
                    "per_request_limits": m.get("per_request_limits"),
                }
                catalog_models.append(model_entry)

            catalog = {
                "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": url,
                "count": len(catalog_models),
                "models": catalog_models,
            }

            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(catalog, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ Saved {len(catalog_models)} OpenRouter models to {out_path} (with pricing + description)")
            return catalog

        except urllib.error.HTTPError as e:
            logger.error(f"OpenRouter models API error: {e.code} {e.reason}")
            raise LLMError(f"Failed to fetch models from OpenRouter: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error scanning OpenRouter models: {e}", exc_info=True)
            raise LLMError(f"Model catalog scan failed: {e}") from e

    @classmethod
    def load_models_catalog(cls, path: str | Path = "data/openrouter_models_catalog.json") -> dict[str, Any] | None:
        """Load previously saved catalog (for dashboard/launcher model picker)."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load model catalog {p}: {e}")
            return None
