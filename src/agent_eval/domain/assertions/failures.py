"""Failure-handling assertions.

A resilience case — "when the upstream tool is down, the agent should fail
cleanly rather than invent an answer" — needs the harness to be able to *pass* a
case whose agent errored. Without these assertions such a case is unassertable:
every errored case counts as a failure, so a dataset cannot express that failing
is the correct behaviour.
"""

from __future__ import annotations

from agent_eval.domain.assertions.context import AssertionContext, preview
from agent_eval.domain.models import AssertionKind, AssertionResult


def check_expect_error(ctx: AssertionContext, expected: bool) -> AssertionResult:
    """Assert whether the agent failed, as the case requires.

    Args:
        ctx: The assertion context.
        expected: ``True`` when the case requires the agent to fail, ``False``
            when it requires the agent to answer.

    Returns:
        A structured result. Failure messages distinguish "should have failed
        but answered" from "should have answered but failed", because the two
        need opposite fixes.
    """
    actual = not ctx.result.ok
    error = ctx.result.error
    passed = actual is expected

    if passed:
        message = (
            f"Agent failed as expected ({preview(error)})"
            if expected
            else "Agent answered without error, as expected"
        )
    elif expected:
        message = (
            "Expected the agent to fail, but it answered successfully. "
            f"It returned: {preview(ctx.output)}"
        )
    else:
        message = f"Expected the agent to answer, but it failed: {preview(error)}"

    return AssertionResult(
        assertion=AssertionKind.EXPECT_ERROR,
        passed=passed,
        expected=expected,
        actual=actual,
        message=message,
    )
