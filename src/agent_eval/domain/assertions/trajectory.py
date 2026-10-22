"""Trajectory assertions: step counts and required/forbidden steps."""

from __future__ import annotations

from agent_eval.domain.assertions.context import AssertionContext
from agent_eval.domain.models import AssertionKind, AssertionResult, StepType


def check_max_steps(ctx: AssertionContext, limit: int) -> AssertionResult:
    """Assert the trajectory did not exceed ``limit`` steps."""
    actual = len(ctx.result.steps)
    passed = actual <= limit
    return AssertionResult(
        assertion=AssertionKind.MAX_STEPS,
        passed=passed,
        expected=limit,
        actual=actual,
        message=f"Trajectory used {actual} step(s), within the limit of {limit}"
        if passed
        else f"Trajectory used {actual} step(s), exceeding the limit of {limit}",
    )


def check_required_steps(ctx: AssertionContext, labels: list[str]) -> list[AssertionResult]:
    """Assert each labelled step appears in the trajectory.

    A label matches either a step label (``"llm"``, ``"tool:search_order"``) or a
    bare tool name, so datasets can stay concise.
    """
    observed = ctx.result.step_labels()
    tool_names = set(ctx.result.tool_names())
    results = []
    for label in labels:
        matched = label in observed or label in tool_names
        results.append(
            AssertionResult(
                assertion=AssertionKind.REQUIRED_STEP,
                passed=matched,
                expected=label,
                actual=observed,
                message=f"Required step {label!r} is present"
                if matched
                else f"Required step {label!r} is missing from the trajectory",
            )
        )
    return results


def check_forbidden_steps(ctx: AssertionContext, labels: list[str]) -> list[AssertionResult]:
    """Assert none of the labelled steps appear in the trajectory."""
    observed = ctx.result.step_labels()
    tool_names = set(ctx.result.tool_names())
    results = []
    for label in labels:
        matched = label in observed or label in tool_names
        results.append(
            AssertionResult(
                assertion=AssertionKind.FORBIDDEN_STEP,
                passed=not matched,
                expected=label,
                actual=observed,
                message=f"Forbidden step {label!r} is absent"
                if not matched
                else f"Forbidden step {label!r} appears in the trajectory",
            )
        )
    return results


def trajectory_summary(ctx: AssertionContext) -> dict[str, int]:
    """Count steps by type plus the tool names that were called."""
    counts: dict[str, int] = {}
    for step in ctx.result.steps:
        counts[step.type.value] = counts.get(step.type.value, 0) + 1
    counts["tools"] = len(ctx.result.tool_calls)
    return counts


def has_error_step(ctx: AssertionContext) -> bool:
    """Whether the trajectory recorded an explicit error step."""
    return any(step.type is StepType.ERROR for step in ctx.result.steps)
