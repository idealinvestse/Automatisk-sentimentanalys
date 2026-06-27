
    def refresh_pricing_from_catalog(self, catalog_path: str | Path = "data/openrouter_models_catalog.json") -> int:
        """Ladda pricing dynamiskt från model catalog (scannad via CLI eller dashboard).

        Uppdaterar self.PRICING med live-värden från OpenRouter.
        Returnerar antal modeller som uppdaterades.
        """
        from .model_catalog import load_catalog

        cat = load_catalog(catalog_path)
        if not cat or not cat.get("models"):
            logger.warning("Ingen model catalog hittades – använder hardcoded PRICING")
            return 0

        updated = 0
        for m in cat["models"]:
            mid = m.get("id")
            if not mid:
                continue
            p = m.get("pricing", {})
            prompt = p.get("prompt_per_token_usd") or p.get("prompt") or 0.0
            comp = p.get("completion_per_token_usd") or p.get("completion") or 0.0
            if prompt or comp:
                self.PRICING[mid] = {
                    "input": float(prompt),
                    "output": float(comp),
                }
                updated += 1

        # Också uppdatera default om Mistral finns
        if "mistralai/mistral-medium-3.5" in self.PRICING:
            self.DEFAULT_MODEL = "mistralai/mistral-medium-3.5"

        logger.info(f"[pricing] Uppdaterade {updated} modeller från catalog")
        return updated

    def _compute_approx_cost(self, usage: Any, model: str) -> float | None:
        """Approximate USD cost from usage tokens + dynamic or hardcoded pricing."""
        if usage is None:
            return None
        try:
            pt = getattr(usage, "prompt_tokens", 0) or 0
            ct = getattr(usage, "completion_tokens", 0) or 0
            # Först försök dynamic från catalog
            prices = self.PRICING.get(model)
            if not prices:
                # Försök refresh en gång
                self.refresh_pricing_from_catalog()
                prices = self.PRICING.get(model, self.PRICING.get("default", {"input": 1.5e-6, "output": 7.5e-6}))
            cost = (pt * prices["input"]) + (ct * prices["output"])
            return round(cost, 6)
        except Exception:
            return None
