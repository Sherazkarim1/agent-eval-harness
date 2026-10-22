"""Tool-behaviour assertions."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from agent_eval.domain.assertions.context import AssertionContext, preview
from agent_eval.domain.models import AssertionKind, AssertionResult, ToolCall


def check_tool_called(ctx: AssertionContext, names: list[str]) -> list[AssertionResult]:
    """Assert each named tool was called at least once."""
    called = set(ctx.result.tool_names())
    return [
        AssertionResult(
            assertion=AssertionKind.TOOL_CALLED,
            passed=name in called,
            expected=name,
            actual=sorted(called),
            message=f"Tool {name!r} was called"
            if name in called
            else f"Tool {name!r} was never called",
        )
        for name in names
    ]


def check_tool_not_called(ctx: AssertionContext, names: list[str]) -> list[AssertionResult]:
    """Assert none of the named tools were called."""
    called = set(ctx.result.tool_names())
    return [
        AssertionResult(
            assertion=AssertionKind.TOOL_NOT_CALLED,
            passed=name not in called,
            expected=name,
            actual=sorted(called),
            message=f"Tool {name!r} was correctly avoided"
            if name not in called
            else f"Tool {name!r} was called but should not have been",
        )
        for name in names
    ]


def check_tool_call_count(
    ctx: AssertionContext, expected: int | dict[str, int]
) -> list[AssertionResult]:
    """Assert the total call count, or the per-tool count.

    An ``int`` checks ``len(result.tool_calls)``. A mapping checks the number of
    calls of each named tool (tools absent from the mapping are ignored).
    """
    calls = ctx.result.tool_calls
    results: list[AssertionResult] = []

    if isinstance(expected, int):
        actual = len(calls)
        results.append(
            AssertionResult(
                assertion=AssertionKind.TOOL_CALL_COUNT,
                passed=actual == expected,
                expected=expected,
                actual=actual,
                message=f"Expected exactly {expected} tool call(s), observed {actual}",
            )
        )
        return results

    for tool_name, count in expected.items():
        actual_count = sum(1 for call in calls if call.name == tool_name)
        results.append(
            AssertionResult(
                assertion=AssertionKind.TOOL_CALL_COUNT,
                passed=actual_count == count,
                expected={tool_name: count},
                actual={tool_name: actual_count},
                message=f"Expected {count} call(s) to {tool_name!r}, observed {actual_count}",
            )
        )
    return results


def check_tool_args(
    ctx: AssertionContext, expected: dict[str, dict[str, Any]]
) -> list[AssertionResult]:
    """Assert that calls to each tool used at least the given arguments.

    Argument matching is a subset match: expected key/value pairs must be
    present, extra arguments are allowed. Every named tool must have been
    called at least once with matching arguments.
    """
    results: list[AssertionResult] = []
    for tool_name, expected_args in expected.items():
        matches = [c for c in ctx.result.tool_calls if c.name == tool_name]
        if not matches:
            results.append(
                AssertionResult(
                    assertion=AssertionKind.TOOL_ARGS,
                    passed=False,
                    expected={tool_name: expected_args},
                    actual=None,
                    message=f"Tool {tool_name!r} was never called, so arguments cannot match",
                )
            )
            continue
        mismatches = [
            {"call_index": index, "missing": _missing(call.arguments, expected_args)}
            for index, call in enumerate(matches)
            if _missing(call.arguments, expected_args)
        ]
        results.append(
            AssertionResult(
                assertion=AssertionKind.TOOL_ARGS,
                passed=not mismatches,
                expected={tool_name: expected_args},
                actual=[call.arguments for call in matches],
                message=f"All calls to {tool_name!r} used the expected arguments"
                if not mismatches
                else f"Calls to {tool_name!r} had unexpected arguments: {mismatches}",
            )
        )
    return results


def check_expected_tool_order(ctx: AssertionContext, order: list[str]) -> AssertionResult:
    """Assert the listed tools were called in the given relative order.

    The assertion is satisfied when ``order`` is a subsequence of the observed
    call order, so extra unrelated calls in between are tolerated.
    """
    observed = ctx.result.tool_names()
    positions: list[int | None] = []
    for name in order:
        positions.append(observed.index(name) if name in observed else None)

    missing = [name for name, pos in zip(order, positions, strict=True) if pos is None]
    if missing:
        return AssertionResult(
            assertion=AssertionKind.EXPECTED_TOOL_ORDER,
            passed=False,
            expected=order,
            actual=observed,
            message=f"Tools never called: {', '.join(missing)}",
        )

    passed = all(
        earlier < later
        for earlier, later in pairwise(positions)
        if earlier is not None and later is not None
    )
    return AssertionResult(
        assertion=AssertionKind.EXPECTED_TOOL_ORDER,
        passed=passed,
        expected=order,
        actual=observed,
        message="Tool calls were in the expected order"
        if passed
        else f"Tools were called as {observed}, which does not respect the expected order {order}",
    )


def tool_call_summary(calls: list[ToolCall]) -> list[dict[str, Any]]:
    """Summarise tool calls for report output."""
    return [{"name": call.name, "arguments": call.arguments, "error": call.error} for call in calls]


def _missing(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Return the subset of ``expected`` pairs absent from ``actual``."""
    return {key: val for key, val in expected.items() if actual.get(key) != val}


def format_tool_calls(calls: list[ToolCall]) -> str:
    """Render tool calls as a compact human-readable string."""
    return ", ".join(f"{c.name}({preview(c.arguments, 60)})" for c in calls) or "<no tool calls>"
