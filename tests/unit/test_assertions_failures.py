"""Unit tests for failure-handling assertions.

The behaviour that matters here is directional: a resilience case must fail
when the agent *succeeds*, not only when it fails.
"""

from __future__ import annotations

from agent_eval.domain.assertions import AssertionEngine, AssertionContext
from agent_eval.domain.models import AssertionKind, Expectation
from tests.conftest import make_result


def context(output: str = "done", error: str | None = None) -> AssertionContext:
    """An assertion context over a successful or failed result."""
    return AssertionContext(
        output=output,
        result=make_result(output, error=error),
        latency_ms=1.0,
        cost_usd=0.0,
    )


def only(ctx: AssertionContext, expect_error: bool):
    """Evaluate a single ``expect_error`` expectation."""
    results = AssertionEngine().evaluate(
        Expectation(expect_error=expect_error),
        ctx.result,
        latency_ms=ctx.latency_ms,
        cost_usd=ctx.cost_usd,
    )
    assert len(results) == 1
    return results[0]


class TestExpectErrorTrue:
    def test_passes_when_the_agent_failed(self) -> None:
        result = only(context(error="upstream_tool_unavailable"), True)
        assert result.passed is True
        assert result.assertion is AssertionKind.EXPECT_ERROR

    def test_reports_the_error_in_the_message(self) -> None:
        result = only(context(error="upstream_tool_unavailable"), True)
        assert "upstream_tool_unavailable" in result.message

    def test_fails_when_the_agent_answered(self) -> None:
        result = only(context("Here is your refund"), True)
        assert result.passed is False

    def test_failure_message_says_it_should_have_failed(self) -> None:
        result = only(context("Here is your refund"), True)
        assert "Expected the agent to fail" in result.message
        assert "Here is your refund" in result.message

    def test_records_expected_and_actual(self) -> None:
        result = only(context(error="boom"), True)
        assert result.expected is True
        assert result.actual is True

    def test_blank_error_still_counts_as_a_failure(self) -> None:
        # `ok` is what decides, not whether the message is populated.
        assert only(context("partial", error=""), True).passed is True


class TestExpectErrorFalse:
    def test_passes_when_the_agent_answered(self) -> None:
        result = only(context("All good"), False)
        assert result.passed is True
        assert "answered without error" in result.message

    def test_fails_when_the_agent_failed(self) -> None:
        result = only(context(error="boom"), False)
        assert result.passed is False

    def test_failure_message_says_it_should_have_answered(self) -> None:
        result = only(context(error="boom"), False)
        assert "Expected the agent to answer" in result.message
        assert "boom" in result.message


class TestDeclarationIsOptional:
    def test_absent_expect_error_produces_no_assertion(self) -> None:
        ctx = context()
        results = AssertionEngine().evaluate(
            Expectation(), ctx.result, latency_ms=ctx.latency_ms, cost_usd=ctx.cost_usd
        )
        assert results == []

    def test_composes_with_other_assertions(self) -> None:
        expected = Expectation(expect_error=True, not_contains=["invented status"])
        ctx = context("I could not reach the tool", error="down")
        results = AssertionEngine().evaluate(
            expected, ctx.result, latency_ms=ctx.latency_ms, cost_usd=ctx.cost_usd
        )
        kinds = {r.assertion for r in results}
        assert kinds == {AssertionKind.EXPECT_ERROR, AssertionKind.NOT_CONTAINS}
        assert all(r.passed for r in results)
