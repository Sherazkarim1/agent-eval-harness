"""Integration tests for the worked regression example.

These assert the property that makes the example meaningful: two real agent
versions run through the real pipeline, and the engine reports the difference.
A test that only checked the printed output would pass even if the demo had
hard-coded its numbers, so the assertions here read the aggregates directly.
"""

from __future__ import annotations

import pytest

from agent_eval.config import HarnessConfig
from agent_eval.core.baselines import snapshot_run
from agent_eval.core.exit_policy import EXIT_EVALUATION_FAILED, EXIT_OK, decide_run_exit
from agent_eval.core.regression import RegressionDetector
from agent_eval.core.runner import Runner
from agent_eval.domain.models import FindingKind, RegressionStatus
from agent_eval.judges.mock import HeuristicJudge
from agent_eval.storage.memory import InMemoryRepository
from examples.regression_demo import DATASET, SupportAgentV1, SupportAgentV2, main


@pytest.fixture(scope="module")
def dataset():
    """The bundled dataset the example evaluates."""
    from agent_eval.datasets import load_dataset_file

    return load_dataset_file(DATASET)


async def execute(agent, dataset, config: HarnessConfig):
    """Run one agent version against the dataset."""
    return await Runner(
        agent, config, judge=HeuristicJudge(), repository=InMemoryRepository()
    ).run(dataset)


@pytest.fixture(scope="module")
async def versions(dataset):
    """Execute both agent versions once for the whole module."""
    from examples.regression_demo import _config

    config = _config()
    v1 = await execute(SupportAgentV1(), dataset, config)
    v2 = await execute(SupportAgentV2(), dataset, config)
    return config, v1, v2


def compare(config: HarnessConfig, v1, v2):
    """Compare the two executed runs exactly as the example does."""
    baseline = snapshot_run(v1.record, "v1")
    return RegressionDetector(config.regression).compare(
        v2.aggregate,
        baseline.aggregate,
        baseline_name=baseline.name,
        baseline_run_id=v1.run_id,
        current_cases=v2.cases,
        baseline_cases=v1.cases,
        baseline_dataset_version=baseline.dataset_version,
        current_dataset_version=v2.record.dataset_version,
        baseline_config_hash=baseline.config_hash,
        current_config_hash=v2.record.config_hash,
    )


class TestBothVersionsReallyRun:
    async def test_v1_executes_every_case(self, versions) -> None:
        _config, v1, _v2 = versions
        assert v1.aggregate.total == len(v1.cases) > 0
        assert v1.aggregate.total == 7

    async def test_v1_has_no_failed_cases(self, versions) -> None:
        _config, v1, _v2 = versions
        # The resilience case expects the failure, so it passes; v1 is clean.
        assert v1.aggregate.failed == 0
        assert v1.aggregate.errored == 0
        assert v1.aggregate.passed == v1.aggregate.total

    async def test_v1_produces_a_real_score(self, versions) -> None:
        _config, v1, _v2 = versions
        assert v1.aggregate.overall_score is not None
        assert 0.9 < v1.aggregate.overall_score <= 1.0

    async def test_v2_degrades_measurably(self, versions) -> None:
        _config, v1, v2 = versions
        assert v2.aggregate.overall_score < v1.aggregate.overall_score
        assert v2.aggregate.failed >= 3

    async def test_scores_are_measured_not_declared(self, versions) -> None:
        _config, v1, v2 = versions
        # Both runs are timed for real; neither may report a placeholder.
        assert v1.aggregate.latency.average_ms >= 0.0
        assert v1.aggregate.tokens.total_tokens > 0
        assert v2.aggregate.tokens.total_tokens > 0


class TestRegressionIsDetected:
    async def test_status_is_failed(self, versions) -> None:
        config, v1, v2 = versions
        assert compare(config, v1, v2).status is RegressionStatus.FAILED

    async def test_reports_a_score_drop(self, versions) -> None:
        config, v1, v2 = versions
        report = compare(config, v1, v2)
        assert FindingKind.SCORE_DROP in {f.kind for f in report.findings}

    async def test_score_drop_exceeds_the_configured_tolerance(self, versions) -> None:
        config, v1, v2 = versions
        report = compare(config, v1, v2)
        assert abs(report.score_delta) > config.regression.max_score_drop

    async def test_names_the_newly_failing_cases(self, versions) -> None:
        config, v1, v2 = versions
        report = compare(config, v1, v2)
        finding = next(f for f in report.findings if f.kind is FindingKind.NEW_FAILURES)
        assert set(finding.case_ids) >= {"refund_request_001", "order_status_001"}

    async def test_flags_tool_behaviour_change(self, versions) -> None:
        config, v1, v2 = versions
        report = compare(config, v1, v2)
        assert FindingKind.TOOL_BEHAVIOR_CHANGE in {f.kind for f in report.findings}

    async def test_does_not_flag_cases_that_did_not_change(self, versions) -> None:
        config, v1, v2 = versions
        finding = next(
            f
            for f in compare(config, v1, v2).findings
            if f.kind is FindingKind.NEW_FAILURES
        )
        assert "account_balance_001" not in finding.case_ids
        assert "tool_failure_001" not in finding.case_ids


class TestCiOutcome:
    async def test_v1_alone_would_pass_ci(self, versions) -> None:
        config, v1, _v2 = versions
        assert decide_run_exit(config, v1.aggregate, None).code == EXIT_OK

    async def test_v2_fails_ci(self, versions) -> None:
        config, v1, v2 = versions
        decision = decide_run_exit(config, v2.aggregate, compare(config, v1, v2))
        assert decision.code == EXIT_EVALUATION_FAILED
        assert decision.reason

    async def test_demo_entry_point_returns_the_ci_code(self) -> None:
        # The example must exit non-zero, otherwise `python examples/...` in a
        # CI step would silently pass.
        assert await main() == EXIT_EVALUATION_FAILED


class TestComparisonIsExplainable:
    async def test_identical_version_does_not_regress(self, dataset) -> None:
        from examples.regression_demo import _config

        config = _config()
        v1 = await execute(SupportAgentV1(), dataset, config)
        report = compare(config, v1, v1)

        assert report.status is RegressionStatus.PASSED
        assert report.findings == []
        assert report.score_delta == pytest.approx(0.0)

    async def test_improvement_is_reported_as_improved(self, versions) -> None:
        config, v1, v2 = versions
        report = RegressionDetector(config.regression).compare(
            v1.aggregate,
            v2.aggregate,
            baseline_name="v2",
        )
        assert report.status is RegressionStatus.IMPROVED
