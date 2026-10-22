"""Text-based assertions."""

from __future__ import annotations

import re

from agent_eval.domain.assertions.context import AssertionContext, preview
from agent_eval.domain.models import AssertionKind, AssertionResult


def _result(
    kind: AssertionKind,
    passed: bool,
    expected: object,
    actual: object,
    message: str,
) -> AssertionResult:
    return AssertionResult(
        assertion=kind,
        passed=passed,
        expected=expected,
        actual=actual,
        message=message,
    )


def check_contains(ctx: AssertionContext, needles: list[str]) -> list[AssertionResult]:
    """Assert every needle appears in the output (case-insensitive)."""
    haystack = ctx.output.casefold()
    results = []
    for needle in needles:
        present = needle.casefold() in haystack
        results.append(
            _result(
                AssertionKind.CONTAINS,
                present,
                needle,
                ctx.output,
                f"Expected text {needle!r} was found"
                if present
                else f"Expected text {needle!r} is missing from the output",
            )
        )
    return results


def check_not_contains(ctx: AssertionContext, needles: list[str]) -> list[AssertionResult]:
    """Assert none of the needles appear in the output (case-insensitive)."""
    haystack = ctx.output.casefold()
    results = []
    for needle in needles:
        present = needle.casefold() in haystack
        results.append(
            _result(
                AssertionKind.NOT_CONTAINS,
                not present,
                needle,
                ctx.output,
                f"Forbidden text {needle!r} is absent"
                if not present
                else f"Forbidden text {needle!r} is present in the output",
            )
        )
    return results


def check_exact_match(ctx: AssertionContext, expected: str) -> AssertionResult:
    """Assert the output equals ``expected`` after trimming whitespace."""
    actual = ctx.output.strip()
    passed = actual == expected.strip()
    return _result(
        AssertionKind.EXACT_MATCH,
        passed,
        expected,
        actual,
        "Output matches the expected text exactly"
        if passed
        else "Output does not match the expected text",
    )


def check_regex(ctx: AssertionContext, patterns: list[str]) -> list[AssertionResult]:
    """Assert every pattern matches somewhere in the output."""
    results = []
    for pattern in patterns:
        try:
            matched = re.search(pattern, ctx.output, flags=re.DOTALL) is not None
            message = (
                f"Pattern {pattern!r} matched"
                if matched
                else f"Pattern {pattern!r} did not match the output"
            )
        except re.error as exc:
            matched = False
            message = f"Pattern {pattern!r} is not a valid regular expression: {exc}"
        results.append(_result(AssertionKind.REGEX_MATCH, matched, pattern, ctx.output, message))
    return results


def check_starts_with(ctx: AssertionContext, prefix: str) -> AssertionResult:
    """Assert the output starts with ``prefix`` (case-insensitive)."""
    passed = ctx.output.strip().casefold().startswith(prefix.casefold())
    return _result(
        AssertionKind.STARTS_WITH,
        passed,
        prefix,
        preview(ctx.output),
        "Output starts with the expected prefix"
        if passed
        else f"Output does not start with {prefix!r}",
    )


def check_ends_with(ctx: AssertionContext, suffix: str) -> AssertionResult:
    """Assert the output ends with ``suffix`` (case-insensitive)."""
    passed = ctx.output.strip().casefold().endswith(suffix.casefold())
    return _result(
        AssertionKind.ENDS_WITH,
        passed,
        suffix,
        preview(ctx.output),
        "Output ends with the expected suffix"
        if passed
        else f"Output does not end with {suffix!r}",
    )
