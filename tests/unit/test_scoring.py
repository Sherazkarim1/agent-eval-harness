"""Tests for the scoring engine and cost estimation."""

from __future__ import annotations

import pytest

from agent_eval.config import PricingConfig, PricingEntry, ScoringConfig, ScoringWeights
from agent_eval.core.aggregate import aggregate_results, weighted_pass_rate
from agent_eval.core.cost import estimate_cost, format_cost
from agent_eval.core.scoring import ScoringEngine, aggregate_component, aggregate_overall
from agent_eval.domain.models import (
    AssertionKind,
    AssertionResult,
    CaseResult,
    CaseStatus,
    JudgeVerdict,
    ScoreBreakdown,
    TokenUsage,
)
from tests.conftest import make_case, make_result


def passed(kind: AssertionKind) -> AssertionResult:
    return AssertionResult(assertion=kind, passed=True)


def failed(kind: AssertionKind) -> AssertionResult:
    return AssertionResult(assertion=kind, passed=False)


class TestWeights:
    def test_normalised_sums_to_one(self) -> None:
        weights = ScoringWeights(deterministic=2.0, judge=2.0, performance=0.0)
        normalised = weights.normalised({"deterministic", "judge"})
        assert normalised["deterministic"] == 0.5
        assert normalised["judge"] == 0.5

    def test_absent_components_excluded(self) -> None:
        weights = ScoringWeights(deterministic=1.0, judge=3.0)
        normalised = weights.normalised({"deterministic"})
        assert normalised == {"deterministic": 1.0}

    def test_all_zero_selected_returns_zeros(self) -> None:
        assert ScoringWeights(deterministic=0.0, judge=0.0).normalised(
            {"deterministic", "judge"}
        ) == {"deterministic": 0.0, "judge": 0.0}

    def test_rejects_all_zero_weights(self) -> None:
        with pytest.raises(ValueError, match="at least one scoring weight"):
            ScoringWeights(deterministic=0.0, judge=0.0, trajectory=0.0, performance=0.0)


class TestScoringEngine:
    def test_no_assertions_and_no_judge_is_unscored(self) -> None:
        score = ScoringEngine(ScoringConfig()).score_case([], None)
        assert score.overall is None

    def test_deterministic_only(self) -> None:
        score = ScoringEngine(ScoringConfig()).score_case(
            [passed(AssertionKind.CONTAINS), failed(AssertionKind.CONTAINS)], None
        )
        assert score.deterministic == 0.5
        assert score.overall == 0.5

    def test_judge_disabled_redistributes_weight(self) -> None:
        engine = ScoringEngine(
            ScoringConfig(weights=ScoringWeights(deterministic=0.5, judge=0.5, performance=0.0))
        )
        score = engine.score_case([passed(AssertionKind.CONTAINS)], None)
        assert score.judge is None
        assert score.overall == 1.0
        assert score.weights == {"deterministic": 1.0}

    def test_judge_included_when_available(self) -> None:
        engine = ScoringEngine(
            ScoringConfig(weights=ScoringWeights(deterministic=0.5, judge=0.5, performance=0.0))
        )
        score = engine.score_case([passed(AssertionKind.CONTAINS)], JudgeVerdict(score=0.5))
        assert score.overall == pytest.approx(0.75)

    def test_judge_error_is_ignored(self) -> None:
        engine = ScoringEngine(ScoringConfig())
        score = engine.score_case(
            [passed(AssertionKind.CONTAINS)], JudgeVerdict(error="provider down")
        )
        assert score.judge is None
        assert score.overall == 1.0

    def test_tool_assertions_map_to_trajectory(self) -> None:
        engine = ScoringEngine(ScoringConfig())
        score = engine.score_case(
            [passed(AssertionKind.TOOL_CALLED), failed(AssertionKind.TOOL_CALLED)], None
        )
        assert score.trajectory == 0.5
        assert score.performance is None

    def test_performance_assertions_map_to_performance(self) -> None:
        engine = ScoringEngine(ScoringConfig())
        score = engine.score_case([passed(AssertionKind.MAX_TOKENS)], None)
        assert score.performance == 1.0
        assert score.trajectory is None

    def test_components_are_independent(self) -> None:
        engine = ScoringEngine(ScoringConfig())
        score = engine.score_case(
            [
                passed(AssertionKind.CONTAINS),
                failed(AssertionKind.TOOL_CALLED),
                passed(AssertionKind.MAX_LATENCY_MS),
            ],
            None,
        )
        assert score.deterministic == pytest.approx(2 / 3)
        assert score.trajectory == 0.0
        assert score.performance == 1.0

    def test_weights_are_all_components(self) -> None:
        engine = ScoringEngine(ScoringConfig())
        score = engine.score_case(
            [
                passed(AssertionKind.CONTAINS),
                passed(AssertionKind.TOOL_CALLED),
                passed(AssertionKind.MAX_TOKENS),
            ],
            JudgeVerdict(score=1.0),
        )
        assert score.overall == pytest.approx(1.0)


class TestThreshold:
    def test_no_threshold_always_passes(self) -> None:
        assert ScoringEngine(ScoringConfig()).meets_threshold(0.0) is True

    def test_unscored_fails_threshold(self) -> None:
        engine = ScoringEngine(ScoringConfig(min_overall_score=0.5))
        assert engine.meets_threshold(None) is False

    def test_meets_threshold(self) -> None:
        engine = ScoringEngine(ScoringConfig(min_overall_score=0.9))
        assert engine.meets_threshold(0.91) is True
        assert engine.meets_threshold(0.89) is False


class TestAggregationHelpers:
    def test_component_ignores_absent(self) -> None:
        breakdowns = [
            ScoreBreakdown(deterministic=1.0, overall=1.0),
            ScoreBreakdown(overall=0.0),
        ]
        assert aggregate_component(breakdowns, "deterministic") == 1.0
        assert aggregate_component(breakdowns, "judge") is None

    def test_overall_ignores_unscored(self) -> None:
        breakdowns = [ScoreBreakdown(overall=1.0), ScoreBreakdown(overall=None)]
        assert aggregate_overall(breakdowns) == 1.0

    def test_empty_input_is_none(self) -> None:
        assert aggregate_overall([]) is None


def case_result(
    case_id: str,
    status: CaseStatus,
    *,
    assertions: list[AssertionResult] | None = None,
    overall: float | None = None,
    latency_ms: float = 100.0,
    tokens: TokenUsage | None = None,
    cost_usd: float = 0.0,
) -> CaseResult:
    return CaseResult(
        run_id="run_1",
        case_id=case_id,
        case=make_case(id=case_id),
        result=make_result("out", tokens=tokens),
        assertions=assertions or [],
        score=ScoreBreakdown(overall=overall),
        status=status,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )


class TestAggregateResults:
    def test_empty_run(self) -> None:
        agg = aggregate_results([])
        assert agg.total == 0
        assert agg.overall_score is None
        assert agg.latency.count == 0

    def test_counts_by_status(self) -> None:
        agg = aggregate_results(
            [
                case_result("a", CaseStatus.PASSED),
                case_result("b", CaseStatus.FAILED),
                case_result("c", CaseStatus.ERRORED),
            ]
        )
        assert (agg.total, agg.passed, agg.failed, agg.errored) == (3, 1, 1, 1)
        assert agg.failing_case_ids == ["b"]
        assert agg.errored_case_ids == ["c"]

    def test_pass_rate_excludes_errors(self) -> None:
        agg = aggregate_results(
            [
                case_result("a", CaseStatus.PASSED),
                case_result("b", CaseStatus.FAILED),
                case_result("c", CaseStatus.ERRORED),
            ]
        )
        assert agg.pass_rate == 0.5

    def test_tokens_and_cost_are_summed(self) -> None:
        agg = aggregate_results(
            [
                case_result(
                    "a",
                    CaseStatus.PASSED,
                    tokens=TokenUsage(input_tokens=10, output_tokens=5),
                    cost_usd=0.001,
                ),
                case_result(
                    "b",
                    CaseStatus.PASSED,
                    tokens=TokenUsage(input_tokens=1, output_tokens=1),
                    cost_usd=0.002,
                ),
            ]
        )
        assert agg.tokens.total_tokens == 17
        assert agg.cost.api_cost_usd == pytest.approx(0.003)
        assert agg.cost.tokens == 17

    def test_latency_percentiles(self) -> None:
        agg = aggregate_results(
            [
                case_result(str(i), CaseStatus.PASSED, latency_ms=float(i * 100))
                for i in range(1, 11)
            ]
        )
        assert agg.latency.count == 10
        assert agg.latency.average_ms == pytest.approx(550.0)
        assert agg.latency.max_ms == 1000.0

    def test_weighted_pass_rate_ignores_unasserted_cases(self) -> None:
        cases = [
            case_result("a", CaseStatus.PASSED, assertions=[passed(AssertionKind.CONTAINS)]),
            case_result("b", CaseStatus.FAILED, assertions=[failed(AssertionKind.CONTAINS)]),
            case_result("c", CaseStatus.PASSED, assertions=[]),
        ]
        assert weighted_pass_rate(cases) == 0.5

    def test_weighted_pass_rate_none_when_nothing_scored(self) -> None:
        assert weighted_pass_rate([case_result("a", CaseStatus.PASSED)]) is None


class TestCost:
    def test_local_model_is_free(self) -> None:
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert estimate_cost(usage, PricingConfig(), "llama3.2") == 0.0

    def test_priced_model(self) -> None:
        pricing = PricingConfig(
            models={"m": PricingEntry(input_per_million=3.0, output_per_million=15.0)}
        )
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert estimate_cost(usage, pricing, "m") == pytest.approx(18.0)

    def test_wildcard_pricing(self) -> None:
        pricing = PricingConfig(
            models={"*": PricingEntry(input_per_million=1.0, output_per_million=2.0)}
        )
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=500_000)
        assert estimate_cost(usage, pricing, "unknown") == pytest.approx(2.0)

    def test_cached_tokens_are_discounted(self) -> None:
        pricing = PricingConfig(
            models={"m": PricingEntry(input_per_million=10.0, output_per_million=0.0)}
        )
        usage = TokenUsage(input_tokens=1_000_000, cached_input_tokens=900_000)
        assert estimate_cost(usage, pricing, "m") == pytest.approx(1.0)

    def test_none_usage_is_zero(self) -> None:
        assert estimate_cost(None, PricingConfig(), "m") == 0.0

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.0, "$0.00"), (0.000123, "$0.000123"), (1.5, "$1.5000")],
    )
    def test_format_cost(self, value: float, expected: str) -> None:
        assert format_cost(value) == expected
