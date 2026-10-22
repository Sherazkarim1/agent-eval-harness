"""Performance assertions: latency, tokens and cost budgets."""

from __future__ import annotations

from agent_eval.domain.assertions.context import AssertionContext
from agent_eval.domain.models import AssertionKind, AssertionResult, StepType


def check_max_latency_ms(ctx: AssertionContext, limit_ms: float) -> AssertionResult:
    """Assert total case latency stayed within the budget."""
    actual = ctx.latency_ms
    passed = actual <= limit_ms
    return AssertionResult(
        assertion=AssertionKind.MAX_LATENCY_MS,
        passed=passed,
        expected=limit_ms,
        actual=round(actual, 3),
        message=f"Latency {actual:.1f}ms is within the {limit_ms:.0f}ms budget"
        if passed
        else f"Latency {actual:.1f}ms exceeds the {limit_ms:.0f}ms budget",
    )


def check_max_tool_latency_ms(
    ctx: AssertionContext, limits: dict[str, float]
) -> list[AssertionResult]:
    """Assert each named tool stayed within its own latency budget.

    Tools that were never called produce a passing result with ``actual=None``
    so that a budget on an unused tool cannot fail a case.
    """
    results: list[AssertionResult] = []
    for tool_name, limit_ms in limits.items():
        durations = [
            call.duration_ms
            for call in ctx.result.tool_calls
            if call.name == tool_name and call.duration_ms is not None
        ]
        if not durations:
            results.append(
                AssertionResult(
                    assertion=AssertionKind.MAX_TOOL_LATENCY_MS,
                    passed=True,
                    expected={tool_name: limit_ms},
                    actual=None,
                    message=f"Tool {tool_name!r} was not called, so no latency budget applies",
                )
            )
            continue
        worst = max(durations)
        passed = worst <= limit_ms
        results.append(
            AssertionResult(
                assertion=AssertionKind.MAX_TOOL_LATENCY_MS,
                passed=passed,
                expected={tool_name: limit_ms},
                actual={tool_name: round(worst, 3)},
                message=f"Tool {tool_name!r} worst latency {worst:.1f}ms is within {limit_ms:.0f}ms"
                if passed
                else f"Tool {tool_name!r} worst latency {worst:.1f}ms exceeds {limit_ms:.0f}ms",
            )
        )
    return results


def check_max_tokens(ctx: AssertionContext, limit: int) -> AssertionResult:
    """Assert total token usage stayed within the budget."""
    tokens = ctx.tokens
    actual = tokens.total_tokens
    passed = actual <= limit
    return AssertionResult(
        assertion=AssertionKind.MAX_TOKENS,
        passed=passed,
        expected=limit,
        actual=actual,
        message=f"Token usage {actual} is within the {limit} budget"
        if passed
        else f"Token usage {actual} exceeds the {limit} budget",
    )


def check_max_cost(ctx: AssertionContext, limit_usd: float) -> AssertionResult:
    """Assert estimated API cost stayed within the budget."""
    actual = ctx.cost_usd
    passed = actual <= limit_usd
    return AssertionResult(
        assertion=AssertionKind.MAX_COST,
        passed=passed,
        expected=limit_usd,
        actual=round(actual, 8),
        message=f"Estimated API cost ${actual:.6f} is within the ${limit_usd:.6f} budget"
        if passed
        else f"Estimated API cost ${actual:.6f} exceeds the ${limit_usd:.6f} budget",
    )


def llm_step_tokens(ctx: AssertionContext) -> int:
    """Highest token usage recorded by a single LLM step."""
    return max(
        (
            step.tokens.total_tokens
            for step in ctx.result.steps
            if step.type is StepType.LLM and step.tokens is not None
        ),
        default=0,
    )
