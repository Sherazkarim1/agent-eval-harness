"""Tests for performance and trajectory assertions."""

from __future__ import annotations

from agent_eval.domain.assertions.context import AssertionContext
from agent_eval.domain.assertions.performance import (
    check_max_cost,
    check_max_latency_ms,
    check_max_tokens,
    check_max_tool_latency_ms,
)
from agent_eval.domain.assertions.trajectory import (
    check_forbidden_steps,
    check_max_steps,
    check_required_steps,
)
from agent_eval.domain.models import AssertionKind, TokenUsage
from tests.conftest import make_result


def ctx_for(
    output: str = "out",
    *,
    tools: list[tuple[str, dict, object]] | None = None,
    tokens: TokenUsage | None = None,
    llm_steps: int = 1,
    latency_ms: float = 10.0,
    cost_usd: float = 0.0,
    tool_latency_ms: float = 10.0,
) -> AssertionContext:
    return AssertionContext(
        output=output,
        result=make_result(
            output,
            tools=tools,
            tokens=tokens,
            llm_steps=llm_steps,
            tool_latency_ms=tool_latency_ms,
        ),
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )


class TestLatency:
    def test_within_budget(self) -> None:
        result = check_max_latency_ms(ctx_for(latency_ms=100.0), 5000.0)
        assert result.passed is True
        assert result.assertion is AssertionKind.MAX_LATENCY_MS

    def test_over_budget(self) -> None:
        result = check_max_latency_ms(ctx_for(latency_ms=6000.0), 5000.0)
        assert result.passed is False
        assert "exceeds" in result.message

    def test_boundary_is_inclusive(self) -> None:
        assert check_max_latency_ms(ctx_for(latency_ms=5000.0), 5000.0).passed is True


class TestToolLatency:
    def test_uncalled_tool_passes(self) -> None:
        results = check_max_tool_latency_ms(ctx_for(), {"search": 100.0})
        assert results[0].passed is True
        assert results[0].actual is None

    def test_uses_worst_call(self) -> None:
        ctx = ctx_for(tools=[("a", {}, None), ("a", {}, None)], tool_latency_ms=50.0)
        results = check_max_tool_latency_ms(ctx, {"a": 100.0})
        assert results[0].passed is True
        assert results[0].actual == {"a": 50.0}

    def test_over_budget(self) -> None:
        ctx = ctx_for(tools=[("a", {}, None)], tool_latency_ms=500.0)
        assert check_max_tool_latency_ms(ctx, {"a": 100.0})[0].passed is False


class TestTokens:
    def test_within_budget(self) -> None:
        ctx = ctx_for(tokens=TokenUsage(input_tokens=10, output_tokens=5))
        result = check_max_tokens(ctx, 100)
        assert result.passed is True
        assert result.actual == 15

    def test_over_budget(self) -> None:
        ctx = ctx_for(tokens=TokenUsage(input_tokens=100, output_tokens=100))
        assert check_max_tokens(ctx, 50).passed is False

    def test_missing_usage_counts_as_zero(self) -> None:
        result = check_max_tokens(ctx_for(tokens=None), 0)
        assert result.passed is True
        assert result.actual == 0


class TestCost:
    def test_within_budget(self) -> None:
        assert check_max_cost(ctx_for(cost_usd=0.001), 0.01).passed is True

    def test_over_budget(self) -> None:
        result = check_max_cost(ctx_for(cost_usd=0.5), 0.01)
        assert result.passed is False
        assert "$0.500000" in result.message

    def test_zero_budget_allows_free_run(self) -> None:
        assert check_max_cost(ctx_for(cost_usd=0.0), 0.0).passed is True


class TestTrajectory:
    def test_max_steps_within(self) -> None:
        result = check_max_steps(ctx_for(llm_steps=2), 5)
        assert result.passed is True
        assert result.assertion is AssertionKind.MAX_STEPS
        assert result.actual == 3  # 2 llm + 1 final

    def test_max_steps_exceeded(self) -> None:
        assert check_max_steps(ctx_for(llm_steps=10), 5).passed is False

    def test_required_step_matches_tool_label(self) -> None:
        ctx = ctx_for(tools=[("search", {}, None)])
        assert check_required_steps(ctx, ["tool:search"])[0].passed is True

    def test_required_step_matches_bare_tool_name(self) -> None:
        ctx = ctx_for(tools=[("search", {}, None)])
        assert check_required_steps(ctx, ["search"])[0].passed is True

    def test_required_step_matches_type(self) -> None:
        assert check_required_steps(ctx_for(llm_steps=1), ["llm", "final"])[0].passed is True

    def test_missing_required_step_fails(self) -> None:
        result = check_required_steps(ctx_for(), ["tool:search"])[0]
        assert result.passed is False
        assert "missing" in result.message

    def test_forbidden_step_absent(self) -> None:
        assert check_forbidden_steps(ctx_for(), ["tool:delete"])[0].passed is True

    def test_forbidden_step_present(self) -> None:
        ctx = ctx_for(tools=[("delete", {}, None)])
        result = check_forbidden_steps(ctx, ["delete"])[0]
        assert result.passed is False
        assert result.assertion is AssertionKind.FORBIDDEN_STEP
