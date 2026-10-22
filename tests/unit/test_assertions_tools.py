"""Tests for tool-behaviour assertions."""

from __future__ import annotations

from agent_eval.domain.assertions.context import AssertionContext
from agent_eval.domain.assertions.tools import (
    check_expected_tool_order,
    check_tool_args,
    check_tool_call_count,
    check_tool_called,
    check_tool_not_called,
)
from agent_eval.domain.models import AssertionKind
from tests.conftest import make_result


def ctx_for(*tools: tuple[str, dict, object]) -> AssertionContext:
    return AssertionContext(
        output="out", result=make_result("out", tools=list(tools)), latency_ms=1.0, cost_usd=0.0
    )


class TestToolCalled:
    def test_passes_when_called(self) -> None:
        results = check_tool_called(ctx_for(("search", {}, None)), ["search"])
        assert results[0].passed is True
        assert results[0].assertion is AssertionKind.TOOL_CALLED

    def test_fails_when_not_called(self) -> None:
        result = check_tool_called(ctx_for(("other", {}, None)), ["search"])[0]
        assert result.passed is False
        assert "never called" in result.message

    def test_one_result_per_tool(self) -> None:
        results = check_tool_called(ctx_for(("a", {}, None)), ["a", "b", "c"])
        assert len(results) == 3
        assert [r.passed for r in results] == [True, False, False]


class TestToolNotCalled:
    def test_passes_when_absent(self) -> None:
        assert check_tool_not_called(ctx_for(("a", {}, None)), ["b"])[0].passed is True

    def test_fails_when_present(self) -> None:
        result = check_tool_not_called(ctx_for(("a", {}, None)), ["a"])[0]
        assert result.passed is False
        assert "should not have been" in result.message


class TestToolCallCount:
    def test_total_count(self) -> None:
        ctx = ctx_for(("a", {}, None), ("b", {}, None))
        result = check_tool_call_count(ctx, 2)[0]
        assert result.passed is True
        assert result.actual == 2

    def test_total_count_mismatch(self) -> None:
        assert check_tool_call_count(ctx_for(("a", {}, None)), 3)[0].passed is False

    def test_per_tool_count(self) -> None:
        ctx = ctx_for(("a", {}, None), ("a", {}, None), ("b", {}, None))
        results = check_tool_call_count(ctx, {"a": 2, "b": 1})
        assert all(r.passed for r in results)

    def test_per_tool_count_ignores_absent_tools(self) -> None:
        results = check_tool_call_count(ctx_for(("a", {}, None)), {"b": 5})
        assert results[0].passed is False


class TestToolArgs:
    def test_subset_match_passes(self) -> None:
        ctx = ctx_for(("search", {"q": "x", "limit": 10}, None))
        result = check_tool_args(ctx, {"search": {"q": "x"}})[0]
        assert result.passed is True

    def test_mismatched_value_fails(self) -> None:
        ctx = ctx_for(("search", {"q": "y"}, None))
        result = check_tool_args(ctx, {"search": {"q": "x"}})[0]
        assert result.passed is False

    def test_missing_argument_fails(self) -> None:
        ctx = ctx_for(("search", {"q": "x"}, None))
        assert check_tool_args(ctx, {"search": {"limit": 5}})[0].passed is False

    def test_uncalled_tool_fails(self) -> None:
        result = check_tool_args(ctx_for(), {"search": {"q": "x"}})[0]
        assert result.passed is False
        assert "never called" in result.message

    def test_all_calls_must_match(self) -> None:
        ctx = ctx_for(("search", {"q": "x"}, None), ("search", {"q": "z"}, None))
        assert check_tool_args(ctx, {"search": {"q": "x"}})[0].passed is False


class TestToolOrder:
    def test_exact_order_passes(self) -> None:
        ctx = ctx_for(("a", {}, None), ("b", {}, None))
        result = check_expected_tool_order(ctx, ["a", "b"])
        assert result.passed is True
        assert result.assertion is AssertionKind.EXPECTED_TOOL_ORDER

    def test_interleaved_calls_tolerated(self) -> None:
        ctx = ctx_for(("a", {}, None), ("x", {}, None), ("b", {}, None))
        assert check_expected_tool_order(ctx, ["a", "b"]).passed is True

    def test_wrong_order_fails(self) -> None:
        ctx = ctx_for(("b", {}, None), ("a", {}, None))
        result = check_expected_tool_order(ctx, ["a", "b"])
        assert result.passed is False
        assert "expected order" in result.message

    def test_missing_tool_fails(self) -> None:
        result = check_expected_tool_order(ctx_for(("a", {}, None)), ["a", "b"])
        assert result.passed is False
        assert "never called" in result.message

    def test_duplicate_calls_use_first_occurrence(self) -> None:
        ctx = ctx_for(("a", {}, None), ("b", {}, None), ("a", {}, None))
        assert check_expected_tool_order(ctx, ["a", "b"]).passed is True
