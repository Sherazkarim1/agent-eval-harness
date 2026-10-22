"""Unit tests for baseline capture and resolution."""

from __future__ import annotations

import pytest

from agent_eval.core.baselines import (
    BaselineSource,
    resolve_baseline,
    save_baseline_from_run,
    snapshot_run,
)
from agent_eval.domain.models import (
    AgentVersion,
    Baseline,
    Dataset,
    ExecutionMode,
    JudgeVersion,
    RunAggregate,
    RunRecord,
)
from agent_eval.storage.memory import InMemoryRepository
from tests.conftest import make_case, make_dataset


def make_run(
    run_id: str = "run_1",
    *,
    dataset_name: str = "unit",
    dataset_version: str = "1.0.0",
    agent_version: str = "1.0.0",
    config_hash: str | None = "cfg1",
    mode: ExecutionMode = ExecutionMode.MOCK,
    aggregate: RunAggregate | None = None,
    started_offset: int = 0,
) -> RunRecord:
    """Build a :class:`RunRecord` for tests."""
    from datetime import datetime, timedelta

    return RunRecord(
        run_id=run_id,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        agent=AgentVersion(name="agent", version=agent_version),
        judge=JudgeVersion(name="judge"),
        aggregate=aggregate or RunAggregate(total=10, passed=9, failed=1),
        execution_mode=mode,
        config_hash=config_hash,
        started_at=datetime(2026, 1, 1, tzinfo=None).astimezone()
        + timedelta(minutes=started_offset),
    )


@pytest.fixture
def repo() -> InMemoryRepository:
    """Empty in-memory repository."""
    return InMemoryRepository()


@pytest.fixture
def dataset() -> Dataset:
    """A minimal dataset."""
    return make_dataset(make_case(), name="unit", version="1.0.0")


class TestSnapshotRun:
    def test_copies_identity(self) -> None:
        run = make_run("run_x", agent_version="3.1.4", config_hash="abc")
        baseline = snapshot_run(run, "main")
        assert baseline.name == "main"
        assert baseline.run_id == "run_x"
        assert baseline.agent_version == "3.1.4"
        assert baseline.config_hash == "abc"
        assert baseline.dataset_ref == "unit@1.0.0"

    def test_copies_execution_mode(self) -> None:
        assert snapshot_run(make_run(mode=ExecutionMode.LIVE), "b").execution_mode is (
            ExecutionMode.LIVE
        )

    def test_snapshot_is_independent_of_later_mutation(self) -> None:
        run = make_run()
        baseline = snapshot_run(run, "main")
        assert baseline.aggregate is run.aggregate

    def test_aggregate_is_carried_over(self) -> None:
        run = make_run(aggregate=RunAggregate(total=3, passed=3))
        assert snapshot_run(run, "b").aggregate.total == 3


class TestSaveBaselineFromRun:
    def test_persists_to_repository(self, repo: InMemoryRepository) -> None:
        run = make_run()
        save_baseline_from_run(repo, run, "main")
        stored = repo.get_baseline("main")
        assert stored is not None
        assert stored.run_id == run.run_id

    def test_overwrites_existing_name(self, repo: InMemoryRepository) -> None:
        save_baseline_from_run(repo, make_run("run_a"), "main")
        save_baseline_from_run(repo, make_run("run_b"), "main")
        stored = repo.get_baseline("main")
        assert stored is not None
        assert stored.run_id == "run_b"
        assert len(repo.list_baselines()) == 1

    def test_distinct_names_coexist(self, repo: InMemoryRepository) -> None:
        save_baseline_from_run(repo, make_run("run_a"), "a")
        save_baseline_from_run(repo, make_run("run_b"), "b")
        assert {b.name for b in repo.list_baselines()} == {"a", "b"}


class TestResolveBaseline:
    def test_named_baseline_wins(self, repo: InMemoryRepository) -> None:
        repo.save_run(make_run("run_latest", aggregate=RunAggregate(total=1, passed=1)))
        repo.save_baseline(
            Baseline(
                name="pinned",
                dataset_name="unit",
                dataset_version="1.0.0",
                aggregate=RunAggregate(total=99, passed=99),
            )
        )
        source = resolve_baseline(repo, "unit", "1.0.0", name="pinned")
        assert source.kind == "named"
        assert source.aggregate is not None
        assert source.aggregate.total == 99

    def test_explicit_name_beats_configured_name(self, repo: InMemoryRepository) -> None:
        for name, total in (("first", 1), ("second", 2)):
            repo.save_baseline(
                Baseline(
                    name=name,
                    dataset_name="unit",
                    dataset_version="1.0.0",
                    aggregate=RunAggregate(total=total, passed=total),
                )
            )
        source = resolve_baseline(repo, "unit", "1.0.0", name="first", configured_name="second")
        assert source.name == "first"

    def test_falls_back_to_configured(self, repo: InMemoryRepository) -> None:
        repo.save_baseline(
            Baseline(
                name="cfg",
                dataset_name="unit",
                dataset_version="1.0.0",
                aggregate=RunAggregate(total=5, passed=5),
            )
        )
        source = resolve_baseline(repo, "unit", "1.0.0", configured_name="cfg")
        assert source.kind == "configured"
        assert source.name == "cfg"

    def test_falls_back_to_latest_run(self, repo: InMemoryRepository) -> None:
        repo.save_run(make_run("run_match", config_hash="same"))
        source = resolve_baseline(repo, "unit", "1.0.0", config_hash="same")
        assert source.kind == "run"
        assert source.run is not None
        assert source.run.run_id == "run_match"

    def test_no_baseline_found_on_empty_store(self, repo: InMemoryRepository) -> None:
        source = resolve_baseline(repo, "unit", "1.0.0")
        assert source.found is False
        assert source.name is None
        assert source.baseline is None
        assert source.run is None

    def test_config_hash_mismatch_yields_nothing(self, repo: InMemoryRepository) -> None:
        repo.save_run(make_run(config_hash="old"))
        source = resolve_baseline(repo, "unit", "1.0.0", config_hash="new")
        assert source.found is False

    def test_dataset_version_must_match(self, repo: InMemoryRepository) -> None:
        repo.save_run(make_run(dataset_version="2.0.0", config_hash="c"))
        source = resolve_baseline(repo, "unit", "1.0.0", config_hash="c")
        assert source.found is False

    def test_execution_mode_filter(self, repo: InMemoryRepository) -> None:
        repo.save_run(make_run("live_run", mode=ExecutionMode.LIVE, config_hash="c"))
        assert (
            resolve_baseline(
                repo, "unit", "1.0.0", config_hash="c", execution_mode=ExecutionMode.MOCK
            ).found
            is False
        )
        assert (
            resolve_baseline(
                repo, "unit", "1.0.0", config_hash="c", execution_mode=ExecutionMode.LIVE
            ).found
            is True
        )

    def test_baseline_name_reports_run_id(self, repo: InMemoryRepository) -> None:
        repo.save_run(make_run("run_abc", config_hash="c"))
        source = resolve_baseline(repo, "unit", "1.0.0", config_hash="c")
        assert source.name == "run:run_abc"

    def test_stale_named_baseline_still_used(self, repo: InMemoryRepository) -> None:
        """A pinned baseline is honoured even when its dataset version is stale."""
        repo.save_baseline(
            Baseline(
                name="old",
                dataset_name="unit",
                dataset_version="0.1.0",
                aggregate=RunAggregate(total=2, passed=2),
            )
        )
        source = resolve_baseline(repo, "unit", "1.0.0", name="old")
        assert source.found is True
        assert source.baseline is not None
        assert source.baseline.dataset_version == "0.1.0"


class TestBaselineSource:
    def test_empty_source(self) -> None:
        source = BaselineSource(kind="run")
        assert source.found is False
        assert source.name is None

    def test_prefers_baseline_name(self) -> None:
        source = BaselineSource(
            kind="named",
            aggregate=RunAggregate(total=1, passed=1),
            baseline=Baseline(
                name="b",
                dataset_name="d",
                dataset_version="1.0.0",
                aggregate=RunAggregate(total=1, passed=1),
            ),
        )
        assert source.name == "b"

    def test_falls_back_to_run_name(self) -> None:
        source = BaselineSource(
            kind="run", aggregate=RunAggregate(total=1, passed=1), run=make_run("run_z")
        )
        assert source.name == "run:run_z"
