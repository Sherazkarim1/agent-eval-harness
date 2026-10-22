"""Tests for text assertions."""

from __future__ import annotations

from agent_eval.domain.assertions import AssertionEngine
from agent_eval.domain.assertions.context import AssertionContext
from agent_eval.domain.assertions.text import (
    check_contains,
    check_ends_with,
    check_exact_match,
    check_not_contains,
    check_regex,
    check_starts_with,
)
from agent_eval.domain.models import AssertionKind
from tests.conftest import make_result


def ctx_for(output: str, **kwargs) -> AssertionContext:
    return AssertionContext(
        output=output, result=make_result(output, **kwargs), latency_ms=10.0, cost_usd=0.0
    )


class TestContains:
    def test_passes_when_present(self) -> None:
        results = check_contains(ctx_for("Your refund is approved"), ["refund"])
        assert results[0].passed is True
        assert results[0].assertion is AssertionKind.CONTAINS

    def test_is_case_insensitive(self) -> None:
        assert check_contains(ctx_for("REFUND"), ["refund"])[0].passed is True

    def test_fails_when_absent(self) -> None:
        result = check_contains(ctx_for("Order shipped"), ["refund"])[0]
        assert result.passed is False
        assert "missing" in result.message

    def test_one_result_per_needle(self) -> None:
        assert len(check_contains(ctx_for("a b c"), ["a", "b", "z"])) == 3


class TestNotContains:
    def test_passes_when_absent(self) -> None:
        assert check_not_contains(ctx_for("All good"), ["error"])[0].passed is True

    def test_fails_when_present(self) -> None:
        result = check_not_contains(ctx_for("An error occurred"), ["error"])[0]
        assert result.passed is False
        assert "present" in result.message


class TestExactMatch:
    def test_ignores_surrounding_whitespace(self) -> None:
        assert check_exact_match(ctx_for("  hello  "), "hello").passed is True

    def test_fails_on_different_text(self) -> None:
        assert check_exact_match(ctx_for("hello there"), "hello").passed is False


class TestRegex:
    def test_passes_on_match(self) -> None:
        assert check_regex(ctx_for("order ORD-1001"), [r"ORD-\d+"])[0].passed is True

    def test_fails_on_no_match(self) -> None:
        assert check_regex(ctx_for("nothing"), [r"ORD-\d+"])[0].passed is False

    def test_invalid_pattern_fails_without_raising(self) -> None:
        result = check_regex(ctx_for("x"), ["[unclosed"])[0]
        assert result.passed is False
        assert "not a valid regular expression" in result.message

    def test_matches_across_newlines(self) -> None:
        assert check_regex(ctx_for("first\nsecond"), [r"first.+second"])[0].passed is True


class TestAffixes:
    def test_starts_with(self) -> None:
        assert check_starts_with(ctx_for("Refund approved"), "refund").passed is True
        assert check_starts_with(ctx_for("Approved refund"), "refund").passed is False

    def test_ends_with(self) -> None:
        assert check_ends_with(ctx_for("within 5 days"), "5 DAYS").passed is True
        assert check_ends_with(ctx_for("within 5 days"), "refund").passed is False


class TestEngineDispatch:
    def test_only_declared_assertions_run(self) -> None:
        result = AssertionEngine().evaluate(
            expected={"contains": ["refund"]},
            result=make_result("refund granted"),
            latency_ms=1.0,
        )
        assert len(result) == 1
        assert result[0].assertion is AssertionKind.CONTAINS

    def test_empty_expectation_yields_no_assertions(self) -> None:
        assert (
            AssertionEngine().evaluate(expected={}, result=make_result("x"), latency_ms=1.0) == []
        )

    def test_declaration_order_is_preserved(self) -> None:
        results = AssertionEngine().evaluate(
            expected={"contains": ["a"], "not_contains": ["z"], "starts_with": "a"},
            result=make_result("abc"),
            latency_ms=1.0,
        )
        assert [r.assertion for r in results] == [
            AssertionKind.CONTAINS,
            AssertionKind.NOT_CONTAINS,
            AssertionKind.STARTS_WITH,
        ]
