"""Run aggregation: turns per-case results into a :class:`RunAggregate`."""

from __future__ import annotations

from agent_eval.core.scoring import aggregate_component, aggregate_overall
from agent_eval.domain.models import (
    CaseResult,
    CaseStatus,
    CostStats,
    RunAggregate,
    ScoreBreakdown,
    TokenUsage,
)
from agent_eval.domain.statistics import latency_stats, sum_tokens


def aggregate_results(cases: list[CaseResult]) -> RunAggregate:
    """Aggregate case results into run-level metrics.

    Component scores are means over the cases where that component was
    actually measured; ``overall_score`` is ``None`` when no case was scorable,
    so an empty run is never reported as a score of zero.
    """
    if not cases:
        return RunAggregate()

    breakdowns: list[ScoreBreakdown] = [c.score for c in cases if c.score is not None]

    tokens: TokenUsage = sum_tokens([c.result.token_usage for c in cases])
    cost_usd = sum(c.cost_usd for c in cases)

    return RunAggregate(
        total=len(cases),
        passed=sum(1 for c in cases if c.status is CaseStatus.PASSED),
        failed=sum(1 for c in cases if c.status is CaseStatus.FAILED),
        errored=sum(1 for c in cases if c.status is CaseStatus.ERRORED),
        deterministic_score=aggregate_component(breakdowns, "deterministic"),
        judge_score=aggregate_component(breakdowns, "judge"),
        trajectory_score=aggregate_component(breakdowns, "trajectory"),
        performance_score=aggregate_component(breakdowns, "performance"),
        overall_score=aggregate_overall(breakdowns),
        latency=latency_stats([c.latency_ms for c in cases]),
        tokens=tokens,
        cost=CostStats(api_cost_usd=cost_usd, tokens=tokens.total_tokens),
        failing_case_ids=[c.case_id for c in cases if c.status is CaseStatus.FAILED],
        errored_case_ids=[c.case_id for c in cases if c.status is CaseStatus.ERRORED],
    )


def weighted_pass_rate(cases: list[CaseResult]) -> float | None:
    """Pass rate over cases that produced at least one assertion.

    Returns ``None`` when no case was scored, keeping "not measured" distinct
    from "0% pass rate".
    """
    scored = [c for c in cases if c.assertions]
    if not scored:
        return None
    return sum(1 for c in scored if c.passed) / len(scored)
