"""Token budgeting for long-context LLM requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..core.errors import LLMError


class ChatTokenizer(Protocol):
    """Minimal tokenizer surface required for exact context accounting."""

    def count_chat_tokens(self, messages: list[dict[str, str]]) -> int: ...


@dataclass(frozen=True)
class ContextBudget:
    """Resolved token allocation for one structured LLM request."""

    requested_context: int
    loaded_context: int
    effective_context: int
    input_tokens: int
    output_tokens: int
    safety_margin_tokens: int

    @property
    def remaining_tokens(self) -> int:
        """Tokens remaining after the complete request allocation."""
        return (
            self.effective_context
            - self.input_tokens
            - self.output_tokens
            - self.safety_margin_tokens
        )

    @property
    def fits(self) -> bool:
        """Whether the request fits without truncation."""
        return self.remaining_tokens >= 0

    def to_dict(self) -> dict[str, int | bool]:
        """Return metadata suitable for an analysis report."""
        return {
            "requested_context": self.requested_context,
            "loaded_context": self.loaded_context,
            "effective_context": self.effective_context,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "remaining_tokens": self.remaining_tokens,
            "fits": self.fits,
        }


def resolve_context_budget(
    tokenizer: ChatTokenizer,
    messages: list[dict[str, str]],
    *,
    requested_context: int,
    loaded_context: int,
    output_tokens: int,
    safety_margin_tokens: int = 2048,
) -> ContextBudget:
    """Count the formatted chat and fail closed when it cannot fit."""
    if min(requested_context, loaded_context, output_tokens) <= 0:
        raise LLMError(
            "Invalid LM Studio context configuration",
            error_code="llm_context_invalid",
            details={
                "requested_context": requested_context,
                "loaded_context": loaded_context,
                "output_tokens": output_tokens,
            },
        )
    budget = ContextBudget(
        requested_context=requested_context,
        loaded_context=loaded_context,
        effective_context=min(requested_context, loaded_context),
        input_tokens=tokenizer.count_chat_tokens(messages),
        output_tokens=output_tokens,
        safety_margin_tokens=max(0, safety_margin_tokens),
    )
    if loaded_context < requested_context:
        raise LLMError(
            f"LM Studio model is loaded with {loaded_context} tokens; {requested_context} required",
            error_code="llm_context_not_ready",
            details=budget.to_dict(),
        )
    if not budget.fits:
        raise LLMError(
            "LLM input exceeds the configured context budget; input was not truncated",
            error_code="llm_context_budget_exceeded",
            details=budget.to_dict(),
        )
    return budget
