"""Small statistics helpers used by the aggregator.

Implemented locally rather than pulling in numpy/pandas: the harness must stay
dependency-light and these calculations are trivial.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from agent_eval.domain.models import LatencyStats, TokenUsage


def percentile(values: Sequence[float], q: float) -> float:
    """Return the ``q``-th percentile (0-100) using linear interpolation.

    Args:
        values: Non-empty sequence of numbers.
        q: Percentile in the range 0-100.

    Returns:
        The interpolated percentile value.

    Raises:
        ValueError: If ``values`` is empty or ``q`` is out of range.
    """
    if not values:
        raise ValueError("percentile() requires at least one value")
    if not 0 <= q <= 100:
        raise ValueError(f"q must be between 0 and 100, got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def latency_stats(values: Sequence[float]) -> LatencyStats:
    """Summarise a latency distribution.

    Args:
        values: Per-case latencies in milliseconds.

    Returns:
        Count, average, p50, p95 and max. An empty input yields zeroed stats
        rather than raising, so an empty run still reports cleanly.
    """
    if not values:
        return LatencyStats()
    return LatencyStats(
        count=len(values),
        average_ms=sum(values) / len(values),
        p50_ms=percentile(values, 50),
        p95_ms=percentile(values, 95),
        max_ms=max(values),
    )


def mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or ``None`` for an empty sequence."""
    return sum(values) / len(values) if values else None


def mean_or_zero(values: Sequence[float]) -> float:
    """Arithmetic mean, or ``0.0`` for an empty sequence."""
    result = mean(values)
    return result if result is not None else 0.0


def sum_tokens(usages: Sequence[TokenUsage | None]) -> TokenUsage:
    """Sum token usage records, skipping ``None`` entries."""
    total = TokenUsage()
    for usage in usages:
        if usage is not None:
            total = total + usage
    return total
