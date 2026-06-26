    def __init__(
        self,
        rules: list[dict[str, Any]] | None = None,
        mistral_analyzer: Any | None = None,
        config: dict[str, Any] | None = None,
        state: AlertingState | None = None,
    ):
        self.rules = rules or DEFAULT_RULES
        self.mistral_analyzer = mistral_analyzer
        self.config = config or load_alerting_config()
        self.state = state or AlertingState()  # defaults to in-memory

        # Keep old instance vars for backward compatibility (will be phased out)
        self._consecutive_failures = 0
        self._webhook_disabled = False