"""Integration tests for the storage layer (in-memory repository)."""

from __future__ import annotations

import pytest

from agent_eval.config import HarnessConfig
from agent_eval.core.runner import Runner
from agent_eval.domain.models import Baseline, ExecutionMode
from agent_eval.integrations.mock import default_mock_agent
from agent_eval.judges.mock import MockJudge
from agent_eval.storage.memory import InMemoryRepository


@pytest.fixture
def repo() -> InMemoryRepository:
    return InMemoryRepository()


async def make_run(repo: InMemoryRepository, dataset, version: str = "1.0.0"):
    from agent_eval.domain.models import AgentVersion

    return await Runner(
        default_mock_agent(),
        HarnessConfig(),
        judge=MockJudge(0.85),
        repository=repo,
    ).run(
        dataset,
        execution_mode=ExecutionMode.MOCK,
        agent_version=AgentVersion(name="mock-agent", version=version, framework="mock"),
    )


class TestRepository:
    async def test_initialize_is_idempotent(self, repo: InMemoryRepository) -> None:
        repo.initialize()
        repo.initialize()

    async def test_save_and_get_run(self, repo: InMemoryRepository, example_dataset) -> None:
        outcome = await make_run(repo, example_dataset)
        stored = repo.get_run(outcome.run_id)
        assert stored is not None
        assert stored.dataset_ref == "customer_support@1.0.0"
        assert stored.aggregate.total == outcome.aggregate.total

    async def test_get_unknown_run_returns_none(self, repo: InMemoryRepository) -> None:
        assert repo.get_run("nope") is None

    async def test_cases_are_stored(self, repo: InMemoryRepository, example_dataset) -> None:
        outcome = await make_run(repo, example_dataset)
        cases = repo.get_cases(outcome.run_id)
        assert len(cases) == len(example_dataset.cases)
        assert all(c.result.output for c in cases)

    async def test_list_runs_newest_first(self, repo: InMemoryRepository, example_dataset) -> None:
        first = await make_run(repo, example_dataset)
        second = await make_run(repo, example_dataset)
        runs = repo.list_runs(dataset_name="customer_support")
        assert {r.run_id for r in runs} == {first.run_id, second.run_id}

    async def test_list_runs_filters_by_mode(self, repo, example_dataset) -> None:
        await make_run(repo, example_dataset)
        assert repo.list_runs(execution_mode="mock")
        assert repo.list_runs(execution_mode="live") == []

    async def test_get_latest_run_matches_filters(self, repo, example_dataset) -> None:
        outcome = await make_run(repo, example_dataset)
        latest = repo.get_latest_run(dataset_name="customer_support")
        assert latest is not None
        assert latest.run_id == outcome.run_id

    async def test_get_latest_run_respects_config_hash(self, repo, example_dataset) -> None:
        await make_run(repo, example_dataset)
        assert repo.get_latest_run(dataset_name="customer_support", config_hash="other") is None

    async def test_baselines_round_trip(self, repo: InMemoryRepository, example_dataset) -> None:
        outcome = await make_run(repo, example_dataset)
        baseline = Baseline(
            name="main",
            run_id=outcome.run_id,
            dataset_name=outcome.record.dataset_name,
            dataset_version=outcome.record.dataset_version,
            aggregate=outcome.aggregate,
        )
        repo.save_baseline(baseline)
        stored = repo.get_baseline("main")
        assert stored is not None
        assert stored.aggregate.total == outcome.aggregate.total

    async def test_unknown_baseline_returns_none(self, repo: InMemoryRepository) -> None:
        assert repo.get_baseline("nope") is None

    async def test_baseline_snapshot_is_serialisable(self) -> None:
        from agent_eval.domain.models import RunAggregate

        baseline = Baseline(
            name="b",
            dataset_name="d",
            dataset_version="1.0.0",
            aggregate=RunAggregate(total=1, passed=1),
        )
        snapshot = baseline.snapshot()
        assert snapshot["dataset_name"] == "d"
        assert snapshot["aggregate"]["total"] == 1

    async def test_dataset_registration(self, repo: InMemoryRepository, example_dataset) -> None:
        repo.register_dataset(example_dataset)
        rows = repo.list_datasets()
        assert rows[0][0] == "customer_support"
        assert rows[0][1] == "1.0.0"
        assert rows[0][2] == len(example_dataset.cases)

    async def test_leaderboard_empty_without_runs(self, repo: InMemoryRepository) -> None:
        assert repo.leaderboard("customer_support") == []

    async def test_close_is_safe(self, repo: InMemoryRepository) -> None:
        assert repo.close() is None
