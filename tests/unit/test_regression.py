"""Unit tests for regression detection.

Each detection rule is exercised on its own, plus the cross-cutting behaviour
that matters more than any single threshold: a comparison the engine cannot
trust must never be reported as a clean pass.
"""

from __future__ import annotations

import pytest

from agent_eval.config import RegressionConfig
from agent_eval.core.regression import (
    MIN_LATENCY_DELTA_MS,
    RegressionDetector,
    compare_records,
)
from agent_eval.domain.models import (
    AgentResult,
    AgentVersion,
    CaseResult,
    CaseStatus,
    CostStats,
    EvalCase,
    ExecutionMode,
    FindingKind,
    JudgeVersion,
    LatencyStats,
    RegressionSeverity,
    RegressionStatus,
    RegressionReport,
    RunAggregate,
    RunRecord,
    TokenUsage,
    ToolCall,
)
from tests.conftest import make_result
from tests.unit.test_baselines import make_run


def aggregate(
    *,
    total: int = 10,
    passed: int = 10,
    failed: int = 0,
    errored: int = 0,
    overall: float | None = 0.9,
    latency_ms: float = 100.0,
    tokens: int = 1000,
    cost_usd: float = 0.0,
    failing: list[str] | None = None,
    errored_ids: list[str] | None = None,
) -> RunAggregate:
    """Build a :class:`RunAggregate` with the given metrics."""
    return RunAggregate(
        total=total,
        passed=passed,
        failed=failed,
        errored=errored,
        deterministic_score=overall,
        judge_score=overall,
        trajectory_score=overall,
        performance_score=overall,
        overall_score=overall,
        latency=LatencyStats(
            average_ms=latency_ms,
            p50_ms=latency_ms,
            p95_ms=latency_ms,
            min_ms=latency_ms,
            max_ms=latency_ms,
        ),
        tokens=TokenUsage(
            input_tokens=tokens // 2,
            output_tokens=tokens - tokens // 2,
            total_tokens=tokens,
        ),
        cost=CostStats(api_cost_usd=cost_usd),
        failing_case_ids=failing or [],
        errored_case_ids=errored_ids or [],
    )


def detect(
    *,
    _baseline: RunAggregate | None = None,
    _current: RunAggregate | None = None,
    **overrides: object,
) -> RegressionReport:
    """Compare a degraded current run against a healthy baseline.

    ``_baseline`` and ``_current`` are separate from ``overrides`` so a typo in
    a threshold name surfaces as a configuration error instead of being
    silently swallowed.
    """
    current = _current if _current is not None else aggregate(overall=0.5, passed=5, failed=5)
    baseline = _baseline if _baseline is not None else aggregate()
    return RegressionDetector(RegressionConfig(**overrides)).compare(
        current, baseline, baseline_name="gold"
    )


def kinds(report) -> list[FindingKind]:
    """The distinct finding kinds in a report."""
    return [f.kind for f in report.findings]


def case_result(case_id: str, tools: list[str]) -> CaseResult:
    """A case result whose trajectory calls ``tools`` in order."""
    calls = [
        ToolCall(name=name, arguments={}, result="ok", step_index=index + 1)
        for index, name in enumerate(tools)
    ]
    return CaseResult(
        run_id="run_1",
        case_id=case_id,
        case=EvalCase(id=case_id, input="hello"),
        result=AgentResult(output="out", tool_calls=calls),
        status=CaseStatus.PASSED,
    )


class TestSkippedComparisons:
    def test_no_baseline_is_skipped_not_failed(self) -> None:
        report = RegressionDetector(RegressionConfig()).compare(aggregate(), None)
        assert report.status is RegressionStatus.SKIPPED
        assert report.has_baseline is False
        assert report.findings == []

    def test_disabled_detection_is_skipped(self) -> None:
        report = RegressionDetector(RegressionConfig(enabled=False)).compare(
            aggregate(overall=0.1), aggregate()
        )
        assert report.status is RegressionStatus.SKIPPED
        assert report.findings == []


class TestScoreRegressions:
    def test_detects_a_drop_beyond_tolerance(self) -> None:
        report = detect()
        assert FindingKind.SCORE_DROP in kinds(report)
        assert report.status is RegressionStatus.FAILED

    def test_reports_signed_delta_and_threshold(self) -> None:
        finding = next(f for f in detect().findings if f.kind is FindingKind.SCORE_DROP)
        assert finding.delta == pytest.approx(-0.4)
        assert finding.threshold == pytest.approx(-0.05)
        assert finding.baseline_value == pytest.approx(0.9)
        assert finding.current_value == pytest.approx(0.5)

    def test_drop_within_tolerance_passes(self) -> None:
        report = detect(_current=aggregate(overall=0.87))
        assert FindingKind.SCORE_DROP not in kinds(report)
        assert report.status is RegressionStatus.PASSED

    def test_improvement_is_reported_as_improved(self) -> None:
        report = detect(_current=aggregate(overall=0.95))
        assert report.status is RegressionStatus.IMPROVED
        assert report.score_delta == pytest.approx(0.05)

    def test_identical_scores_pass(self) -> None:
        report = detect(_current=aggregate(overall=0.9))
        assert report.status is RegressionStatus.PASSED
        assert report.score_delta == pytest.approx(0.0)

    def test_missing_scores_produce_no_finding(self) -> None:
        report = detect(_current=aggregate(overall=None), _baseline=aggregate(overall=None))
        assert report.findings == []
        assert report.score_delta is None

    def test_newly_unscored_run_produces_no_finding(self) -> None:
        # Going from scored to unscored is a reporting gap, not a regression.
        report = detect(_current=aggregate(overall=None))
        assert FindingKind.SCORE_DROP not in kinds(report)

    def test_threshold_is_configurable(self) -> None:
        report = detect(max_score_drop=0.5, _current=aggregate(overall=0.5))
        assert FindingKind.SCORE_DROP not in kinds(report)

    def test_zero_tolerance_flags_any_drop(self) -> None:
        report = detect(max_score_drop=0.0, _current=aggregate(overall=0.89))
        assert FindingKind.SCORE_DROP in kinds(report)


class TestLatencyRegressions:
    def test_detects_an_increase_beyond_tolerance(self) -> None:
        report = detect(_current=aggregate(latency_ms=200.0), _baseline=aggregate())
        assert FindingKind.LATENCY_INCREASE in kinds(report)

    def test_reports_percentage_delta(self) -> None:
        finding = next(
            f for f in detect(
                _current=aggregate(latency_ms=200.0), _baseline=aggregate()
            ).findings
            if f.kind is FindingKind.LATENCY_INCREASE
        )
        assert finding.delta == pytest.approx(100.0)
        assert finding.threshold == pytest.approx(20.0)

    def test_increase_within_tolerance_passes(self) -> None:
        report = detect(_current=aggregate(latency_ms=110.0), _baseline=aggregate())
        assert FindingKind.LATENCY_INCREASE not in kinds(report)

    def test_sub_millisecond_jitter_is_noise(self) -> None:
        # A mock agent answers in ~0.1ms, where percentage noise is meaningless.
        report = detect(
            _current=aggregate(latency_ms=0.2),
            _baseline=aggregate(latency_ms=0.1),
        )
        assert FindingKind.LATENCY_INCREASE not in kinds(report)

    def test_noise_guard_does_not_hide_real_regressions(self) -> None:
        report = detect(
            _current=aggregate(latency_ms=MIN_LATENCY_DELTA_MS * 10),
            _baseline=aggregate(latency_ms=MIN_LATENCY_DELTA_MS),
        )
        assert FindingKind.LATENCY_INCREASE in kinds(report)

    def test_faster_is_never_a_regression(self) -> None:
        report = detect(_current=aggregate(latency_ms=50.0), _baseline=aggregate())
        assert FindingKind.LATENCY_INCREASE not in kinds(report)

    def test_zero_baseline_latency_is_not_divided_by(self) -> None:
        report = detect(_current=aggregate(latency_ms=100.0), _baseline=aggregate(latency_ms=0.0))
        assert FindingKind.LATENCY_INCREASE not in kinds(report)


class TestTokenAndCostRegressions:
    def test_detects_token_increase(self) -> None:
        report = detect(_current=aggregate(tokens=2000), _baseline=aggregate(tokens=1000))
        assert FindingKind.TOKEN_INCREASE in kinds(report)

    def test_token_increase_within_tolerance_passes(self) -> None:
        report = detect(_current=aggregate(tokens=1200), _baseline=aggregate(tokens=1000))
        assert FindingKind.TOKEN_INCREASE not in kinds(report)

    def test_token_threshold_is_configurable(self) -> None:
        report = detect(
            max_token_increase_percent=5.0,
            _current=aggregate(tokens=1200),
            _baseline=aggregate(tokens=1000),
        )
        assert FindingKind.TOKEN_INCREASE in kinds(report)

    def test_detects_cost_increase(self) -> None:
        report = detect(
            _current=aggregate(cost_usd=0.02), _baseline=aggregate(cost_usd=0.01)
        )
        assert FindingKind.COST_INCREASE in kinds(report)

    def test_zero_cost_runs_never_report_cost_regression(self) -> None:
        # Local models cost $0.00, so a $0 -> $0 change must stay silent.
        report = detect(
            _current=aggregate(cost_usd=0.0), _baseline=aggregate(cost_usd=0.0)
        )
        assert FindingKind.COST_INCREASE not in kinds(report)

    def test_cost_increase_from_zero_is_not_measured(self) -> None:
        report = detect(
            _current=aggregate(cost_usd=0.01), _baseline=aggregate(cost_usd=0.0)
        )
        assert FindingKind.COST_INCREASE not in kinds(report)


class TestFailureRegressions:
    def test_detects_new_failures(self) -> None:
        report = detect(
            _current=aggregate(
                total=10, passed=8, failed=2, failing=["c9", "c10"]
            ),
            _baseline=aggregate(),
        )
        finding = next(f for f in report.findings if f.kind is FindingKind.NEW_FAILURES)
        assert finding.case_ids == ["c9", "c10"]
        assert report.status is RegressionStatus.FAILED

    def test_ignores_failures_the_baseline_already_had(self) -> None:
        report = detect(
            _current=aggregate(total=10, passed=9, failed=1, failing=["c9"]),
            _baseline=aggregate(total=10, passed=9, failed=1, failing=["c9"]),
        )
        assert FindingKind.NEW_FAILURES not in kinds(report)

    def test_counts_new_errors_as_new_failures(self) -> None:
        report = detect(
            _current=aggregate(
                total=10, passed=8, failed=1, errored=1, errored_ids=["e1"]
            ),
            _baseline=aggregate(),
        )
        finding = next(f for f in report.findings if f.kind is FindingKind.NEW_FAILURES)
        assert finding.case_ids == ["e1"]

    def test_allows_a_configured_number_of_new_failures(self) -> None:
        report = detect(
            max_new_failures=2,
            _current=aggregate(total=10, passed=8, failed=2, failing=["c9", "c10"]),
            _baseline=aggregate(),
        )
        assert FindingKind.NEW_FAILURES not in kinds(report)

    def test_exceeding_the_allowance_fails(self) -> None:
        report = detect(
            max_new_failures=1,
            _current=aggregate(total=10, passed=7, failed=3, failing=["c8", "c9", "c10"]),
            _baseline=aggregate(),
        )
        assert FindingKind.NEW_FAILURES in kinds(report)

    def test_recovered_cases_produce_no_finding(self) -> None:
        report = detect(
            _current=aggregate(),
            _baseline=aggregate(total=10, passed=9, failed=1, failing=["c9"]),
        )
        assert report.findings == []


class TestComparabilityFindings:
    def test_dataset_version_change_voids_the_comparison(self) -> None:
        report = RegressionDetector(RegressionConfig()).compare(
            aggregate(),
            aggregate(),
            baseline_dataset_version="1.0.0",
            current_dataset_version="2.0.0",
        )
        assert FindingKind.DATASET_CHANGED in kinds(report)
        assert report.status is RegressionStatus.FAILED

    def test_config_change_voids_the_comparison(self) -> None:
        report = RegressionDetector(RegressionConfig()).compare(
            aggregate(),
            aggregate(),
            baseline_config_hash="aaa",
            current_config_hash="bbb",
        )
        assert FindingKind.CONFIG_CHANGED in kinds(report)
        assert report.status is RegressionStatus.FAILED

    def test_matching_identity_produces_no_finding(self) -> None:
        report = RegressionDetector(RegressionConfig()).compare(
            aggregate(),
            aggregate(),
            baseline_dataset_version="1.0.0",
            current_dataset_version="1.0.0",
            baseline_config_hash="aaa",
            current_config_hash="aaa",
        )
        assert report.findings == []

    def test_unknown_identity_is_not_treated_as_a_mismatch(self) -> None:
        report = RegressionDetector(RegressionConfig()).compare(aggregate(), aggregate())
        assert report.findings == []


class TestToolBehaviorFindings:
    def test_detects_a_changed_tool_sequence(self) -> None:
        report = RegressionDetector(RegressionConfig()).compare(
            aggregate(),
            aggregate(),
            current_cases=[case_result("c1", ["search", "fetch"])],
            baseline_cases=[case_result("c1", ["search"])],
        )
        assert FindingKind.TOOL_BEHAVIOR_CHANGE in kinds(report)

    def test_tool_change_is_a_warning_not_a_failure(self) -> None:
        report = RegressionDetector(RegressionConfig()).compare(
            aggregate(),
            aggregate(),
            current_cases=[case_result("c1", ["search"])],
            baseline_cases=[case_result("c1", ["lookup"])],
        )
        assert report.findings[0].severity is RegressionSeverity.WARNING
        assert report.status is RegressionStatus.PASSED

    def test_reordering_counts_as_a_change(self) -> None:
        report = RegressionDetector(RegressionConfig()).compare(
            aggregate(),
            aggregate(),
            current_cases=[case_result("c1", ["b", "a"])],
            baseline_cases=[case_result("c1", ["a", "b"])],
        )
        assert FindingKind.TOOL_BEHAVIOR_CHANGE in kinds(report)

    def test_unchanged_tools_produce_no_finding(self) -> None:
        report = RegressionDetector(RegressionConfig()).compare(
            aggregate(),
            aggregate(),
            current_cases=[case_result("c1", ["search", "fetch"])],
            baseline_cases=[case_result("c1", ["search", "fetch"])],
        )
        assert report.findings == []

    def test_new_cases_are_not_flagged(self) -> None:
        report = RegressionDetector(RegressionConfig()).compare(
            aggregate(),
            aggregate(),
            current_cases=[case_result("c1", ["search"]), case_result("c2", ["x"])],
            baseline_cases=[case_result("c1", ["search"])],
        )
        assert report.findings == []

    def test_can_be_disabled(self) -> None:
        report = RegressionDetector(RegressionConfig(detect_tool_behavior_change=False)).compare(
            aggregate(),
            aggregate(),
            current_cases=[case_result("c1", ["search"])],
            baseline_cases=[case_result("c1", ["other"])],
        )
        assert report.findings == []

    def test_requires_cases_on_both_sides(self) -> None:
        report = RegressionDetector(RegressionConfig()).compare(
            aggregate(),
            aggregate(),
            current_cases=[case_result("c1", ["search"])],
        )
        assert report.findings == []


class TestReportMetadata:
    def test_records_the_baseline_identity(self) -> None:
        report = RegressionDetector(RegressionConfig()).compare(
            aggregate(), aggregate(), baseline_name="gold", baseline_run_id="run_9"
        )
        assert report.baseline_name == "gold"
        assert report.baseline_run_id == "run_9"
        assert report.has_baseline is True

    def test_error_outranks_warning(self) -> None:
        report = RegressionDetector(RegressionConfig()).compare(
            aggregate(overall=0.1),
            aggregate(),
            current_cases=[case_result("c1", ["search"])],
            baseline_cases=[case_result("c1", ["other"])],
        )
        assert report.status is RegressionStatus.FAILED
        assert report.has_blocking_findings is True


class TestCompareRecords:
    def test_compares_two_run_records(self) -> None:
        current = make_run(
            "run_2",
            aggregate=aggregate(overall=0.4, passed=4, failed=6),
        )
        baseline = make_run("run_1", aggregate=aggregate(overall=0.9))
        report = compare_records(RegressionConfig(), current, baseline, baseline_name="gold")

        assert report.status is RegressionStatus.FAILED
        assert report.baseline_run_id == "run_1"

    def test_first_run_is_skipped(self) -> None:
        report = compare_records(RegressionConfig(), make_run("run_1"), None)
        assert report.status is RegressionStatus.SKIPPED

    def test_reads_identity_from_the_records(self) -> None:
        current = make_run("run_2", dataset_version="2.0.0", config_hash="b")
        baseline = make_run("run_1", dataset_version="1.0.0", config_hash="a")
        report = compare_records(RegressionConfig(), current, baseline)
        assert kinds(report) == [FindingKind.DATASET_CHANGED, FindingKind.CONFIG_CHANGED]

    def test_does_not_crash_without_cases(self) -> None:
        report = compare_records(RegressionConfig(), make_run("run_2"), make_run("run_1"))
        assert report.has_baseline is True


def test_detector_accepts_any_case_result_shape() -> None:
    """The detector reads tool names, not the full trajectory."""
    result = make_result("done", tools=[("search", {"q": "x"}, "ok")])
    case = CaseResult(
        run_id="run_1",
        case_id="c1",
        case=EvalCase(id="c1", input="hi"),
        result=result,
    )
    report = RegressionDetector(RegressionConfig()).compare(
        aggregate(), aggregate(), current_cases=[case], baseline_cases=[]
    )
    assert report.findings == []


def test_agent_and_judge_versions_are_not_compared() -> None:
    """Version drift is legitimate; only scores and behaviour are compared."""
    current = RunRecord(
        run_id="run_2",
        dataset_name="unit",
        dataset_version="1.0.0",
        agent=AgentVersion(name="agent", version="2.0.0"),
        judge=JudgeVersion(name="judge", version="2.0.0"),
        aggregate=aggregate(),
        execution_mode=ExecutionMode.MOCK,
        config_hash="same",
    )
    report = compare_records(RegressionConfig(), current, make_run("run_1", config_hash="same"))
    assert report.findings == []
