"""Unit tests for the exit-code policy.

The exit code is the product's CI contract, so these tests pin the exact number
for each outcome rather than only checking success or failure.
"""

from __future__ import annotations

import pytest

from agent_eval.config import HarnessConfig, RegressionConfig, ScoringConfig
from agent_eval.core.exit_policy import (
    EXIT_EVALUATION_FAILED,
    EXIT_INTERNAL,
    EXIT_OK,
    EXIT_USAGE,
    ExitDecision,
    decide_run_exit,
    describe_regression,
    worst_severity,
)
from agent_eval.domain.models import (
    FindingKind,
    RegressionFinding,
    RegressionReport,
    RegressionSeverity,
    RegressionStatus,
    RunAggregate,
)
from tests.unit.test_regression import aggregate


def config(**regression: object) -> HarnessConfig:
    """A harness configuration with only ``regression`` overridden."""
    return HarnessConfig(
        scoring=ScoringConfig(),
        regression=RegressionConfig(**regression),  # type: ignore[arg-type]
    )


def finding(severity: RegressionSeverity) -> RegressionFinding:
    """A single finding at ``severity``."""
    return RegressionFinding(
        kind=FindingKind.SCORE_DROP,
        severity=severity,
        metric="overall_score",
        message="dropped",
    )


def report(*findings: RegressionFinding) -> RegressionReport:
    """A passed report carrying ``findings``."""
    return RegressionReport(
        status=RegressionStatus.PASSED,
        has_baseline=True,
        baseline_name="gold",
        findings=list(findings),
    )


class TestExitCodes:
    def test_clean_run_exits_zero(self) -> None:
        decision = decide_run_exit(config(), aggregate(), report())
        assert decision.code == EXIT_OK
        assert decision.ok is True
        assert decision.reason == ""

    def test_no_baseline_is_not_a_failure(self) -> None:
        skipped = RegressionReport(status=RegressionStatus.SKIPPED)
        assert decide_run_exit(config(), aggregate(), skipped).code == EXIT_OK

    def test_no_regression_report_is_not_a_failure(self) -> None:
        assert decide_run_exit(config(), aggregate(), None).code == EXIT_OK


class TestFailureExit:
    def test_failed_case_exits_one(self) -> None:
        result = aggregate(total=10, passed=9, failed=1)
        decision = decide_run_exit(config(), result, None)
        assert decision.code == EXIT_EVALUATION_FAILED
        assert "1 failed" in decision.reason

    def test_errored_case_exits_one(self) -> None:
        result = aggregate(total=10, passed=9, errored=1)
        decision = decide_run_exit(config(), result, None)
        assert decision.code == EXIT_EVALUATION_FAILED
        assert "errored" in decision.reason

    def test_empty_run_is_a_usage_error(self) -> None:
        # Zero cases means the dataset or filters were wrong, not that the
        # agent performed badly.
        decision = decide_run_exit(config(), RunAggregate(), None)
        assert decision.code == EXIT_USAGE
        assert "No cases" in decision.reason

    def test_score_below_minimum_exits_one(self) -> None:
        strict = HarnessConfig(scoring=ScoringConfig(min_overall_score=0.95))
        decision = decide_run_exit(strict, aggregate(overall=0.9), None)
        assert decision.code == EXIT_EVALUATION_FAILED
        assert "below the minimum" in decision.reason

    def test_score_at_the_minimum_passes(self) -> None:
        strict = HarnessConfig(scoring=ScoringConfig(min_overall_score=0.9))
        assert decide_run_exit(strict, aggregate(overall=0.9), None).code == EXIT_OK

    def test_unscored_run_does_not_fail_the_minimum(self) -> None:
        strict = HarnessConfig(scoring=ScoringConfig(min_overall_score=0.95))
        result = aggregate(overall=None)
        assert decide_run_exit(strict, result, None).code == EXIT_OK

    def test_reasons_accumulate(self) -> None:
        result = aggregate(total=10, passed=8, failed=2, overall=0.5)
        strict = HarnessConfig(scoring=ScoringConfig(min_overall_score=0.9))
        decision = decide_run_exit(strict, result, report(finding(RegressionSeverity.ERROR)))

        assert decision.code == EXIT_EVALUATION_FAILED
        assert "2 failed" in decision.reason
        assert "below the minimum" in decision.reason
        assert "regression finding" in decision.reason


class TestRegressionGating:
    def test_error_finding_exits_one(self) -> None:
        decision = decide_run_exit(config(), aggregate(), report(finding(RegressionSeverity.ERROR)))
        assert decision.code == EXIT_EVALUATION_FAILED
        assert "error" in decision.reason

    def test_warning_finding_passes_by_default(self) -> None:
        warnings = report(finding(RegressionSeverity.WARNING))
        assert decide_run_exit(config(), aggregate(), warnings).code == EXIT_OK

    def test_warning_fails_when_configured_to(self) -> None:
        strict = config(fail_on=["error", "warning"])
        warnings = report(finding(RegressionSeverity.WARNING))
        assert decide_run_exit(strict, aggregate(), warnings).code == EXIT_EVALUATION_FAILED

    def test_info_finding_never_blocks(self) -> None:
        strict = config(fail_on=["error", "warning"])
        assert decide_run_exit(strict, aggregate(), report(finding(RegressionSeverity.INFO))).code == (
            EXIT_OK
        )

    def test_empty_report_passes(self) -> None:
        strict = config(fail_on=["error", "warning"])
        assert decide_run_exit(strict, aggregate(), report()).code == EXIT_OK


class TestDescribeRegression:
    def test_skipped_reads_as_a_normal_first_run(self) -> None:
        text = describe_regression(RegressionReport(status=RegressionStatus.SKIPPED))
        assert "skipped" in text
        assert "No baseline" in text

    def test_no_findings_names_the_baseline(self) -> None:
        text = describe_regression(report())
        assert "No differences" in text
        assert "gold" in text

    def test_findings_are_summarised_by_kind(self) -> None:
        text = describe_regression(report(finding(RegressionSeverity.ERROR)))
        assert "1 finding" in text
        assert "score_drop" in text

    def test_duplicate_kinds_are_collapsed(self) -> None:
        text = describe_regression(
            report(finding(RegressionSeverity.ERROR), finding(RegressionSeverity.ERROR))
        )
        assert "2 finding(s)" in text
        assert text.count("score_drop") == 1


class TestWorstSeverity:
    def test_none_when_there_are_no_findings(self) -> None:
        assert worst_severity(report()) is None

    def test_reports_the_highest_severity(self) -> None:
        found = worst_severity(
            report(finding(RegressionSeverity.INFO), finding(RegressionSeverity.ERROR))
        )
        assert found is RegressionSeverity.ERROR

    def test_warning_beats_info(self) -> None:
        found = worst_severity(
            report(finding(RegressionSeverity.INFO), finding(RegressionSeverity.WARNING))
        )
        assert found is RegressionSeverity.WARNING


class TestExitDecision:
    def test_ok_reflects_the_code(self) -> None:
        assert ExitDecision(EXIT_OK).ok is True
        assert ExitDecision(EXIT_EVALUATION_FAILED).ok is False

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (EXIT_OK, 0),
            (EXIT_EVALUATION_FAILED, 1),
            (EXIT_USAGE, 2),
            (EXIT_INTERNAL, 3),
        ],
    )
    def test_codes_are_stable(self, code: int, expected: int) -> None:
        # CI depends on these exact numbers, so they are pinned deliberately.
        assert code == expected
