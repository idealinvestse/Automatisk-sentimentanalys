    async def get_alerting_status(self) -> dict[str, Any]:
        """GET /alerting/status – webhook and circuit breaker health."""
        return await self._get("/alerting/status")