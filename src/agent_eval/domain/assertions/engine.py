"""The assertion engine.

Maps every field of :class:`~agent_eval.domain.models.Expectation` onto the
matching assertion function and returns structured results. The engine never
raises: a broken assertion becomes ``passed=False`` with an explanatory
message, because a crashing assertion must not abort a run.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from agent_eval.domain.assertions import failures, performance, structure, text, tools, trajectory
from agent_eval.domain.assertions.context import AssertionContext
from agent_eval.domain.models import (
    AgentResult,
    AssertionKind,
    AssertionResult,
    Expectation,
)

logger = logging.getLogger(__name__)


class AssertionEngine:
    """Evaluates an :class:`Expectation` against an :class:`AgentResult`."""

    def evaluate(
        self,
        expected: Expectation | Mapping[str, Any],
        result: AgentResult,
        *,
        latency_ms: float,
        cost_usd: float = 0.0,
    ) -> list[AssertionResult]:
        """Run every assertion declared by ``expected``.

        Args:
            expected: The case expectations, either as an :class:`Expectation`
                or as a raw mapping (as read from a dataset file).
            result: The (already redacted) agent result.
            latency_ms: Harness-measured latency for the case.
            cost_usd: Harness-estimated API cost for the case.

        Returns:
            Structured results, one per declared assertion, in declaration order.
        """
        if not isinstance(expected, Expectation):
            expected = Expectation.model_validate(dict(expected))

        ctx = AssertionContext(
            output=result.output,
            result=result,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )
        collected: list[AssertionResult] = []

        try:
            collected.extend(self._run(expected, ctx))
        except Exception as exc:
            logger.exception("Assertion evaluation failed unexpectedly")
            collected.append(
                AssertionResult(
                    assertion=AssertionKind.UNKNOWN,
                    passed=False,
                    expected=None,
                    actual=None,
                    message=f"Assertion evaluation raised {type(exc).__name__}: {exc}",
                )
            )
        return collected

    def _run(self, expected: Expectation, ctx: AssertionContext) -> list[AssertionResult]:
        """Dispatch each declared expectation to its assertion function."""
        results: list[AssertionResult] = []

        # --- text ---
        if expected.contains:
            results.extend(text.check_contains(ctx, expected.contains))
        if expected.not_contains:
            results.extend(text.check_not_contains(ctx, expected.not_contains))
        if expected.exact_match is not None:
            results.append(text.check_exact_match(ctx, expected.exact_match))
        if expected.regex:
            results.extend(text.check_regex(ctx, expected.regex))
        if expected.starts_with is not None:
            results.append(text.check_starts_with(ctx, expected.starts_with))
        if expected.ends_with is not None:
            results.append(text.check_ends_with(ctx, expected.ends_with))

        # --- structure ---
        if expected.valid_json:
            results.append(structure.check_valid_json(ctx))
        if expected.required_fields:
            results.extend(structure.check_required_fields(ctx, expected.required_fields))
        if expected.json_schema:
            results.append(structure.check_schema(ctx, expected.json_schema))

        # --- tools ---
        if expected.tool_called:
            results.extend(tools.check_tool_called(ctx, expected.tool_called))
        if expected.tool_not_called:
            results.extend(tools.check_tool_not_called(ctx, expected.tool_not_called))
        if expected.tool_call_count is not None:
            results.extend(tools.check_tool_call_count(ctx, expected.tool_call_count))
        if expected.tool_args:
            results.extend(tools.check_tool_args(ctx, expected.tool_args))
        if expected.expected_tool_order:
            results.append(tools.check_expected_tool_order(ctx, expected.expected_tool_order))

        # --- performance ---
        if expected.max_latency_ms is not None:
            results.append(performance.check_max_latency_ms(ctx, expected.max_latency_ms))
        if expected.max_tool_latency_ms:
            results.extend(performance.check_max_tool_latency_ms(ctx, expected.max_tool_latency_ms))
        if expected.max_tokens is not None:
            results.append(performance.check_max_tokens(ctx, expected.max_tokens))
        if expected.max_cost_usd is not None:
            results.append(performance.check_max_cost(ctx, expected.max_cost_usd))

        # --- trajectory ---
        if expected.max_steps is not None:
            results.append(trajectory.check_max_steps(ctx, expected.max_steps))
        if expected.required_step:
            results.extend(trajectory.check_required_steps(ctx, expected.required_step))
        if expected.forbidden_step:
            results.extend(trajectory.check_forbidden_steps(ctx, expected.forbidden_step))

        # --- failure handling ---
        if expected.expect_error is not None:
            results.append(failures.check_expect_error(ctx, expected.expect_error))

        return results
