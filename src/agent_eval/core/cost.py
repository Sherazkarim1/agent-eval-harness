"""Cost estimation.

Only *API* cost is estimated. Local models (Ollama) are priced at zero because
they cost no API calls; the compute and electricity cost of local hardware is
real but out of scope, and reports say so explicitly rather than pretending
local inference is free in every sense.
"""

from __future__ import annotations

from agent_eval.config import PricingConfig
from agent_eval.domain.models import TokenUsage

#: Price applied when a model is not listed in the pricing configuration.
FALLBACK_INPUT_PER_MILLION = 0.0
FALLBACK_OUTPUT_PER_MILLION = 0.0


def estimate_cost(usage: TokenUsage | None, pricing: PricingConfig, model: str | None) -> float:
    """Estimate the API cost of a token usage record.

    Args:
        usage: Token usage, or ``None`` when the agent reports no usage.
        pricing: Configured per-model pricing.
        model: Model identifier used to look up pricing.

    Returns:
        Estimated cost in the configured currency (USD by default). Unknown
        models fall back to zero-cost pricing rather than guessing a price.
    """
    if usage is None:
        return 0.0
    entry = pricing.models.get(model or "")
    if entry is None:
        entry = pricing.models.get("*")
    if entry is None:
        return _cost_with(
            usage,
            FALLBACK_INPUT_PER_MILLION,
            FALLBACK_OUTPUT_PER_MILLION,
        )
    return _cost_with(usage, entry.input_per_million, entry.output_per_million)


def _cost_with(usage: TokenUsage, input_price: float, output_price: float) -> float:
    """Apply per-million-token prices to a usage record."""
    billable_input = max(usage.input_tokens - usage.cached_input_tokens, 0)
    return (billable_input * input_price + usage.output_tokens * output_price) / 1_000_000


def format_cost(cost_usd: float) -> str:
    """Render a cost for terminal output."""
    if cost_usd == 0:
        return "$0.00"
    if cost_usd < 0.01:
        return f"${cost_usd:.6f}"
    return f"${cost_usd:.4f}"
