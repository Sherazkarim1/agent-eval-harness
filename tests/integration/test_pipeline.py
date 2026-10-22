"""Integration tests: the full pipeline over the bundled dataset."""

from __future__ import annotations

import pytest

from agent_eval.config import HarnessConfig, ScoringConfig
from agent_eval.core.aggregate import aggregate_results
from agent_eval.core.runner import Runner
from agent_eval.core.scoring import ScoringEngine
from agent_eval.domain.models import AssertionKind, CaseStatus, EvalCase, ExecutionMode
from agent_eval.integrations.mock import default_mock_agent
from agent_eval.judges.mock import HeuristicJudge, MockJudge
from agent_eval.storage.memory import InMemoryRepository


@pytest.fixture
def config() -> HarnessConfig:
    return HarnessConfig(scoring=ScoringConfig())


class TestFullPipeline:
    async def test_example_dataset_end_to_end(self, example_dataset, config) -> None:
        repo = InMemoryRepository()
        outcome = await Runner(
            default_mock_agent(), config, judge=HeuristicJudge(), repository=repo
        ).run(example_dataset)

        agg = outcome.aggregate
        assert agg.total == len(example_dataset.cases)
        assert agg.failed == 0
        assert agg.errored == 0
        assert agg.passed == agg.total
        assert agg.overall_score is not None
        assert 0.0 < agg.overall_score <= 1.0

    async def test_expected_failure_passes_the_case(self, example_dataset, config) -> None:
        # tool_failure_001 declares expect_error, so the agent failing is the
        # correct behaviour and the case must not count as an error.
        outcome = await Runner(default_mock_agent(), config).run(example_dataset)
        case = next(c for c in outcome.cases if c.case_id == "tool_failure_001")

        assert case.status is CaseStatus.PASSED
        assert case.result.error is not None
        assertion = next(a for a in case.assertions if a.assertion is AssertionKind.EXPECT_ERROR)
        assert assertion.passed is True

    async def test_unexpected_failure_still_errors_the_case(self, example_dataset, config) -> None:
        # A case that does not declare expect_error must still fail loudly.
        dataset = example_dataset.model_copy(
            update={
                "cases": [
                    EvalCase(id="plain", input="Please crash the internal server error handler")
                ]
            }
        )
        outcome = await Runner(default_mock_agent(), config).run(dataset)
        assert [c.case_id for c in outcome.cases if c.status is CaseStatus.ERRORED] == ["plain"]

    async def test_every_case_produces_a_result(self, example_dataset, config) -> None:
        outcome = await Runner(default_mock_agent(), config).run(example_dataset)
        assert [c.case_id for c in outcome.cases] == example_dataset.case_ids()

    async def test_run_is_persisted_with_cases(self, example_dataset, config) -> None:
        repo = InMemoryRepository()
        outcome = await Runner(default_mock_agent(), config, repository=repo).run(example_dataset)
        stored = repo.get_cases(outcome.run_id)
        assert len(stored) == agg_total(outcome)
        assert all(c.result.steps for c in stored), "trajectories must be captured"

    async def test_trajectories_are_captured(self, example_dataset, config) -> None:
        outcome = await Runner(default_mock_agent(), config).run(example_dataset)
        order_case = next(c for c in outcome.cases if c.case_id == "order_status_001")
        assert order_case.result.tool_names() == ["get_order_status"]
        assert order_case.result.steps
        assert order_case.result.token_usage is not None

    async def test_latency_percentiles_are_computed(self, example_dataset, config) -> None:
        outcome = await Runner(default_mock_agent(), config).run(example_dataset)
        assert outcome.aggregate.latency.count == len(example_dataset.cases)
        assert outcome.aggregate.latency.max_ms >= outcome.aggregate.latency.p50_ms

    async def test_mock_mode_is_recorded(self, example_dataset, config) -> None:
        outcome = await Runner(default_mock_agent(), config).run(example_dataset)
        assert outcome.record.execution_mode is ExecutionMode.MOCK

    async def test_runs_are_reproducible(self, example_dataset, config) -> None:
        first = await Runner(default_mock_agent(), config).run(example_dataset)
        second = await Runner(default_mock_agent(), config).run(example_dataset)
        assert first.aggregate.overall_score == second.aggregate.overall_score
        assert first.aggregate.passed == second.aggregate.passed
        assert first.aggregate.deterministic_score == second.aggregate.deterministic_score

    async def test_scores_are_computed_not_hardcoded(self, example_dataset, config) -> None:
        """A degraded agent must score strictly lower than the healthy one."""
        from agent_eval.integrations.mock import MockAgent

        healthy = await Runner(default_mock_agent(), config).run(example_dataset)
        degraded = await Runner(MockAgent(default=_useless_scenario()), config).run(example_dataset)

        assert degraded.aggregate.overall_score < healthy.aggregate.overall_score
        assert degraded.aggregate.failed > healthy.aggregate.failed

    async def test_judge_contributes_to_score(self, example_dataset) -> None:
        with_judge = await Runner(
            default_mock_agent(), HarnessConfig(), judge=HeuristicJudge()
        ).run(example_dataset)
        without = await Runner(default_mock_agent(), HarnessConfig()).run(example_dataset)
        assert with_judge.aggregate.judge_score is not None
        assert without.aggregate.judge_score is None

    async def test_leaderboard_reflects_real_runs(self, example_dataset) -> None:
        repo = InMemoryRepository()
        await Runner(default_mock_agent(), HarnessConfig(), repository=repo).run(
            example_dataset, agent_version=_agent("v1.0.0")
        )
        await Runner(default_mock_agent(), HarnessConfig(), repository=repo).run(
            example_dataset, agent_version=_agent("v2.0.0")
        )
        rows = repo.leaderboard("customer_support")
        assert len(rows) == 2
        assert all(row.total_cases == len(example_dataset.cases) for row in rows)


class TestAggregation:
    def test_empty_case_list(self) -> None:
        assert aggregate_results([]).total == 0

    def test_scoring_engine_handles_no_assertions(self) -> None:
        score = ScoringEngine(ScoringConfig()).score_case([], None)
        assert score.overall is None

    def test_mock_judge_is_stable(self, example_dataset) -> None:
        first = _run_with_mock_judge(example_dataset)
        second = _run_with_mock_judge(example_dataset)
        assert first.aggregate.judge_score == second.aggregate.judge_score

    def test_mock_judge_scores_every_judged_case(self, example_dataset) -> None:
        outcome = _run_with_mock_judge(example_dataset)
        judged = [c for c in outcome.cases if c.judge is not None]
        assert judged, "cases with a reference answer must be judged"
        assert all(c.judge.score == pytest.approx(0.85) for c in judged)  # type: ignore[union-attr]


def agg_total(outcome) -> int:
    return outcome.aggregate.total


def _useless_scenario():
    from agent_eval.integrations.mock import MockScenario

    return MockScenario("I have no idea.", input_tokens=5, output_tokens=3)


def _agent(version: str):
    from agent_eval.domain.models import AgentVersion

    return AgentVersion(name="mock-customer-support", version=version, framework="mock")


def _run_with_mock_judge(dataset):
    import asyncio

    return asyncio.run(
        Runner(default_mock_agent(), HarnessConfig(), judge=MockJudge(0.85)).run(dataset)
    )
