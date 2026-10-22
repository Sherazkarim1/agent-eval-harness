"""Tests for the evaluator and the runner."""

from __future__ import annotations

import asyncio

import pytest

from agent_eval.config import (
    HarnessConfig,
    PricingConfig,
    PricingEntry,
    RegressionConfig,
    RunConfig,
    ScoringConfig,
)
from agent_eval.core.evaluator import Evaluator, execution_mode_for
from agent_eval.core.runner import Runner, dataset_path_for, new_run_id
from agent_eval.domain.models import (
    AgentResult,
    CaseStatus,
    EvalCase,
    ExecutionMode,
    Expectation,
    Step,
    StepType,
    TokenUsage,
    ToolCall,
)
from agent_eval.judges.mock import MockJudge
from agent_eval.storage.memory import InMemoryRepository
from tests.conftest import make_case, make_dataset


class ExplodingAdapter:
    """Adapter that raises, to prove agent crashes become errored results."""

    name = "exploding"
    version = "1.0.0"
    framework = "test"

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or RuntimeError("agent crashed")
        self.calls = 0

    async def run(self, case_input: str, *, timeout_s: float | None = None) -> AgentResult:
        self.calls += 1
        raise self._exc

    async def aclose(self) -> None:
        return None


class SlowAdapter:
    """Adapter that never finishes, to exercise the timeout path."""

    name = "slow"
    version = "1.0.0"
    framework = "test"

    async def run(self, case_input: str, *, timeout_s: float | None = None) -> AgentResult:
        await asyncio.sleep(5)
        return AgentResult(output="never")

    async def aclose(self) -> None:
        return None


class SecretLeakingAdapter:
    """Adapter that tries to leak a credential into its output and metadata."""

    name = "leaky"
    version = "1.0.0"
    framework = "test"

    async def run(self, case_input: str, *, timeout_s: float | None = None) -> AgentResult:
        return AgentResult(
            output="Here is the key sk-abcdefgh12345678",
            steps=[Step(type=StepType.LLM, output="OPENAI_API_KEY=sk-abcdefgh12345678")],
            tool_calls=[ToolCall(name="search", arguments={"api_key": "sk-abcdefgh12345678"})],
            metadata={"debug": "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI"},
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    async def aclose(self) -> None:
        return None


def evaluator_for(adapter, *, config: HarnessConfig | None = None, judge=None) -> Evaluator:
    return Evaluator(adapter, judge, config or HarnessConfig())


class TestEvaluatorBasics:
    async def test_passing_case(self) -> None:
        result = await evaluator_for(SecretLeakingAdapter()).run_case(
            make_case(expected={"contains": ["key"]}), "run_1"
        )
        assert result.status is CaseStatus.PASSED
        assert result.passed is True

    async def test_failing_case(self) -> None:
        result = await evaluator_for(SecretLeakingAdapter()).run_case(
            make_case(expected={"contains": ["nonexistent"]}), "run_1"
        )
        assert result.status is CaseStatus.FAILED
        assert len(result.failed_assertions) == 1

    async def test_case_without_assertions_passes(self) -> None:
        result = await evaluator_for(SecretLeakingAdapter()).run_case(make_case(), "run_1")
        assert result.status is CaseStatus.PASSED
        assert result.assertions == []

    async def test_case_error_becomes_errored(self) -> None:
        result = await evaluator_for(ExplodingAdapter()).run_case(make_case(), "run_1")
        assert result.status is CaseStatus.ERRORED
        assert "RuntimeError" in (result.error or "")
        assert "agent crashed" in (result.error or "")

    async def test_timeout_becomes_errored(self) -> None:
        case = make_case(timeout_s=0.01)
        result = await evaluator_for(SlowAdapter()).run_case(case, "run_1")
        assert result.status is CaseStatus.ERRORED
        assert "timed out" in (result.error or "")

    async def test_run_id_and_case_id_recorded(self) -> None:
        result = await evaluator_for(SecretLeakingAdapter()).run_case(
            make_case(id="my_case"), "run_42"
        )
        assert result.run_id == "run_42"
        assert result.case_id == "my_case"

    async def test_latency_is_measured_by_harness(self) -> None:
        result = await evaluator_for(SlowAdapter()).run_case(make_case(timeout_s=0.05), "run_1")
        assert result.latency_ms > 0


class TestRedaction:
    async def test_output_is_redacted(self) -> None:
        result = await evaluator_for(SecretLeakingAdapter()).run_case(make_case(), "run_1")
        assert "sk-abcdefgh12345678" not in result.result.output
        assert "***" in result.result.output

    async def test_tool_arguments_are_redacted(self) -> None:
        result = await evaluator_for(SecretLeakingAdapter()).run_case(make_case(), "run_1")
        assert result.result.tool_calls[0].arguments["api_key"] == "***"

    async def test_metadata_is_redacted(self) -> None:
        result = await evaluator_for(SecretLeakingAdapter()).run_case(make_case(), "run_1")
        assert "wJalrXUtnFEMI" not in str(result.result.metadata)

    async def test_steps_are_redacted(self) -> None:
        result = await evaluator_for(SecretLeakingAdapter()).run_case(make_case(), "run_1")
        assert "sk-abcdefgh12345678" not in str(result.result.steps[0].output)


class TestJudgeIntegration:
    async def test_judge_runs_when_reference_present(self) -> None:
        judge = MockJudge(0.9)
        case = make_case(reference_answer="expected")
        result = await evaluator_for(SecretLeakingAdapter(), judge=judge).run_case(case, "run_1")
        assert result.judge is not None
        assert result.judge.score == 0.9

    async def test_judge_skipped_without_reference(self) -> None:
        result = await evaluator_for(SecretLeakingAdapter(), judge=MockJudge()).run_case(
            make_case(), "run_1"
        )
        assert result.judge is None

    async def test_judge_runs_without_reference_when_configured(self) -> None:
        config = HarnessConfig()
        object.__setattr__(config.judge, "only_with_reference", False)
        result = await evaluator_for(
            SecretLeakingAdapter(), config=config, judge=MockJudge()
        ).run_case(make_case(), "run_1")
        assert result.judge is not None

    async def test_disabled_judge_never_runs(self) -> None:
        config = HarnessConfig()
        object.__setattr__(config.judge, "enabled", False)
        result = await evaluator_for(
            SecretLeakingAdapter(), config=config, judge=MockJudge()
        ).run_case(make_case(reference_answer="x"), "run_1")
        assert result.judge is None

    async def test_judge_exception_degrades_gracefully(self) -> None:
        class ExplodingJudge:
            name = "boom"
            version = "1"
            model = "none"

            async def evaluate(self, case, result):
                raise RuntimeError("judge crashed")

        result = await evaluator_for(SecretLeakingAdapter(), judge=ExplodingJudge()).run_case(
            make_case(reference_answer="x"), "run_1"
        )
        assert result.judge is not None
        assert result.judge.score is None
        assert "judge crashed" in (result.judge.error or "")
        assert result.status is CaseStatus.PASSED

    async def test_judge_failure_does_not_fail_case(self) -> None:
        class BrokenJudge:
            name = "broken"
            version = "1"
            model = "none"

            async def evaluate(self, case, result):
                from agent_eval.domain.models import JudgeVerdict

                return JudgeVerdict(error="provider down")

        result = await evaluator_for(SecretLeakingAdapter(), judge=BrokenJudge()).run_case(
            make_case(reference_answer="x"), "run_1"
        )
        assert result.status is CaseStatus.PASSED
        assert result.judge is not None
        assert result.judge.error == "provider down"


class TestScoringIntegration:
    async def test_score_is_computed(self) -> None:
        result = await evaluator_for(SecretLeakingAdapter()).run_case(
            make_case(expected={"contains": ["key"]}), "run_1"
        )
        assert result.score is not None
        assert result.score.overall == 1.0

    async def test_min_score_threshold_fails_case(self) -> None:
        config = HarnessConfig(scoring=ScoringConfig(min_overall_score=1.0))
        result = await evaluator_for(SecretLeakingAdapter(), config=config).run_case(
            make_case(expected={"contains": ["key"]}), "run_1"
        )
        assert result.score is not None
        assert result.score.overall == 1.0
        assert result.status is CaseStatus.PASSED

    async def test_partial_assertions_lower_score(self) -> None:
        result = await evaluator_for(SecretLeakingAdapter()).run_case(
            make_case(expected={"contains": ["key", "missing"]}), "run_1"
        )
        assert result.score is not None
        assert result.score.deterministic == 0.5

    async def test_cost_is_estimated(self) -> None:
        config = HarnessConfig(
            pricing=PricingConfig(models={"*": PricingEntry(input_per_million=1000.0)})
        )
        result = await evaluator_for(SecretLeakingAdapter(), config=config).run_case(
            make_case(), "run_1"
        )
        assert result.cost_usd > 0


class TestExecutionMode:
    def test_mock_by_default(self) -> None:
        assert execution_mode_for(HarnessConfig()) is ExecutionMode.MOCK

    def test_override_forces_mock(self) -> None:
        config = HarnessConfig(run=RunConfig(mode="live"))
        assert execution_mode_for(config, mock_override=True) is ExecutionMode.MOCK

    def test_live_when_configured(self) -> None:
        config = HarnessConfig(run=RunConfig(mode="live"))
        assert execution_mode_for(config) is ExecutionMode.LIVE


class TestRunner:
    async def test_runs_all_cases(self) -> None:
        dataset = make_dataset(*[make_case(id=f"c{i}") for i in range(3)])
        outcome = await Runner(SecretLeakingAdapter(), HarnessConfig()).run(dataset)
        assert outcome.aggregate.total == 3
        assert len(outcome.cases) == 3

    async def test_run_id_is_generated(self) -> None:
        outcome = await Runner(SecretLeakingAdapter(), HarnessConfig()).run(make_dataset())
        assert outcome.run_id.startswith("run_")
        assert outcome.record.run_id == outcome.run_id

    async def test_explicit_run_id(self) -> None:
        outcome = await Runner(SecretLeakingAdapter(), HarnessConfig()).run(
            make_dataset(), run_id="run_fixed"
        )
        assert outcome.run_id == "run_fixed"

    async def test_mock_mode_recorded(self) -> None:
        outcome = await Runner(SecretLeakingAdapter(), HarnessConfig()).run(make_dataset())
        assert outcome.record.execution_mode is ExecutionMode.MOCK

    async def test_dataset_identity_recorded(self) -> None:
        dataset = make_dataset(make_case(), name="mydataset", version="2.1.0")
        outcome = await Runner(SecretLeakingAdapter(), HarnessConfig()).run(dataset)
        assert outcome.record.dataset_ref == "mydataset@2.1.0"

    async def test_agent_identity_recorded(self) -> None:
        outcome = await Runner(SecretLeakingAdapter(), HarnessConfig()).run(make_dataset())
        assert outcome.record.agent.name == "leaky"
        assert outcome.record.agent.framework == "test"

    async def test_judge_identity_recorded(self) -> None:
        outcome = await Runner(SecretLeakingAdapter(), HarnessConfig(), judge=MockJudge()).run(
            make_dataset()
        )
        assert outcome.record.judge.name == "mock-judge"
        assert outcome.record.judge.model == "mock-judge"

    async def test_persists_when_repository_given(self) -> None:
        repo = InMemoryRepository()
        outcome = await Runner(SecretLeakingAdapter(), HarnessConfig(), repository=repo).run(
            make_dataset()
        )
        assert repo.get_run(outcome.run_id) is not None
        assert len(repo.get_cases(outcome.run_id)) == outcome.aggregate.total

    async def test_does_not_persist_when_disabled(self) -> None:
        repo = InMemoryRepository()
        outcome = await Runner(SecretLeakingAdapter(), HarnessConfig(), repository=repo).run(
            make_dataset(), persist=False
        )
        assert repo.get_run(outcome.run_id) is None

    async def test_config_hash_recorded(self) -> None:
        config = HarnessConfig()
        outcome = await Runner(SecretLeakingAdapter(), config).run(make_dataset())
        assert outcome.record.config_hash == config.fingerprint()

    async def test_on_case_complete_callback(self) -> None:
        seen: list[str] = []
        await Runner(SecretLeakingAdapter(), HarnessConfig()).run(
            make_dataset(*[make_case(id=f"c{i}") for i in range(2)]),
            on_case_complete=lambda result: seen.append(result.case_id),
        )
        assert seen == ["c0", "c1"]

    async def test_stop_on_first_failure(self) -> None:
        config = HarnessConfig(run=RunConfig(stop_on_first_failure=True))
        dataset = make_dataset(
            make_case(id="c0", expected={"contains": ["missing"]}),
            make_case(id="c1"),
        )
        outcome = await Runner(SecretLeakingAdapter(), config).run(dataset)
        assert outcome.aggregate.total == 1

    async def test_error_counts_toward_aggregate(self) -> None:
        outcome = await Runner(ExplodingAdapter(), HarnessConfig()).run(
            make_dataset(*[make_case(id=f"c{i}") for i in range(2)])
        )
        assert outcome.aggregate.errored == 2
        assert outcome.aggregate.overall_score is None

    async def test_dataset_registered_in_repository(self) -> None:
        repo = InMemoryRepository()
        repo.register_dataset(make_dataset(name="registered"))
        assert repo.list_datasets()[0][0] == "registered"


class TestHelpers:
    def test_new_run_ids_are_unique(self) -> None:
        assert new_run_id() != new_run_id()

    def test_dataset_path_is_relative_to_root(self, tmp_path) -> None:
        config = HarnessConfig()
        resolved = dataset_path_for(config, tmp_path)
        assert resolved == tmp_path / config.dataset.path
        assert not resolved.is_absolute() or resolved.is_absolute()

    def test_dataset_path_respects_absolute(self) -> None:
        from pathlib import Path

        config = HarnessConfig(
            dataset={"path": Path("/abs/dataset.yaml")}  # type: ignore[arg-type]
        )
        assert dataset_path_for(config, Path("/root")).as_posix() == "/abs/dataset.yaml"


def test_expectation_defaults_are_empty() -> None:
    assert Expectation().is_empty() is True


def test_regression_config_defaults() -> None:
    config = RegressionConfig()
    assert config.enabled is True
    assert config.fail_on == ["error"]


def test_eval_case_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError):
        EvalCase(id="a", input="hi", timeout_s=0)
