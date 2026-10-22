"""Scoring engine.

Four components are computed per case:

* ``deterministic`` - ratio of passed deterministic assertions
* ``judge``        - the LLM judge's score (when a verdict is available)
* ``trajectory``   - tool-behaviour assertions only
* ``performance``  - latency/token/cost budget assertions only

Weights are configurable and are renormalised over the components that were
actually computed, so a disabled judge does not deflate the score. When no
component is computable, ``overall`` stays ``None`` to distinguish "not
measured" from "scored zero".
"""

from __future__ import annotations

from agent_eval.config import ScoringConfig
from agent_eval.domain.models import (
    AssertionKind,
    AssertionResult,
    JudgeVerdict,
    ScoreBreakdown,
)

#: Assertion kinds that describe agent trajectory shape.
TRAJECTORY_KINDS: frozenset[AssertionKind] = frozenset(
    {
        AssertionKind.TOOL_CALLED,
        AssertionKind.TOOL_NOT_CALLED,
        AssertionKind.TOOL_CALL_COUNT,
        AssertionKind.TOOL_ARGS,
        AssertionKind.EXPECTED_TOOL_ORDER,
        AssertionKind.MAX_STEPS,
        AssertionKind.REQUIRED_STEP,
        AssertionKind.FORBIDDEN_STEP,
    }
)

#: Assertion kinds that describe resource budgets.
PERFORMANCE_KINDS: frozenset[AssertionKind] = frozenset(
    {
        AssertionKind.MAX_LATENCY_MS,
        AssertionKind.MAX_TOOL_LATENCY_MS,
        AssertionKind.MAX_TOKENS,
        AssertionKind.MAX_COST,
    }
)

#: Assertion kinds scored into the deterministic component (all of them).
DETERMINISTIC_KINDS: frozenset[AssertionKind] = frozenset(AssertionKind)


class ScoringEngine:
    """Turns assertions and judge verdicts into a weighted score."""

    def __init__(self, config: ScoringConfig) -> None:
        """Create an engine bound to the given scoring configuration."""
        self._config = config

    def score_case(
        self,
        assertions: list[AssertionResult],
        judge: JudgeVerdict | None,
    ) -> ScoreBreakdown:
        """Score a single case.

        Args:
            assertions: Structured assertion results for the case.
            judge: The judge verdict, or ``None`` when judging was skipped.

        Returns:
            A :class:`ScoreBreakdown` whose ``overall`` is ``None`` when nothing
            could be scored.
        """
        deterministic = _ratio([a for a in assertions if a.assertion in DETERMINISTIC_KINDS])
        trajectory = _ratio([a for a in assertions if a.assertion in TRAJECTORY_KINDS])
        performance = _ratio([a for a in assertions if a.assertion in PERFORMANCE_KINDS])

        components: dict[str, float | None] = {
            "deterministic": deterministic,
            "judge": judge.score if judge is not None and judge.score is not None else None,
            "trajectory": trajectory,
            "performance": performance,
        }

        present = {name for name, value in components.items() if value is not None}
        weights = self._config.weights.normalised(present)
        overall: float | None = None
        if present:
            total = 0.0
            for name in present:
                value = components[name]
                if value is not None:
                    total += value * weights[name]
            overall = total

        return ScoreBreakdown(
            deterministic=deterministic,
            judge=components["judge"],
            trajectory=trajectory,
            performance=performance,
            overall=overall,
            weights=weights,
        )

    def meets_threshold(self, overall: float | None) -> bool:
        """Whether ``overall`` satisfies ``scoring.min_overall_score``."""
        threshold = self._config.min_overall_score
        if threshold is None:
            return True
        if overall is None:
            return False
        return overall >= threshold


def _ratio(assertions: list[AssertionResult]) -> float | None:
    """Ratio of passed assertions, or ``None`` when there are none."""
    if not assertions:
        return None
    passed = sum(1 for a in assertions if a.passed)
    return passed / len(assertions)


def aggregate_component(breakdowns: list[ScoreBreakdown], component: str) -> float | None:
    """Mean of one component across cases, ignoring cases where it is absent."""
    values: list[float] = []
    for breakdown in breakdowns:
        value = getattr(breakdown, component, None)
        if isinstance(value, int | float):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_overall(breakdowns: list[ScoreBreakdown]) -> float | None:
    """Mean of the overall score across cases, ignoring unscored cases."""
    values = [b.overall for b in breakdowns if b.overall is not None]
    if not values:
        return None
    return sum(values) / len(values)


def component_names() -> tuple[str, ...]:
    """Names of the four score components."""
    return ("deterministic", "judge", "trajectory", "performance")
