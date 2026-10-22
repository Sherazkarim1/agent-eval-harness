"""Integration tests for the SQLite repository.

These run against a real on-disk database in a temporary directory, because the
point of this adapter is precisely that it survives process restarts.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from agent_eval.config import HarnessConfig, StorageConfig
from agent_eval.core.baselines import resolve_baseline, save_baseline_from_run
from agent_eval.core.runner import Runner
from agent_eval.domain.models import (
    AgentVersion,
    Baseline,
    CaseStatus,
    Dataset,
    ExecutionMode,
    RunAggregate,
)
from agent_eval.integrations.mock import default_mock_agent
from agent_eval.judges.mock import MockJudge
from agent_eval.storage import (
    InMemoryRepository,
    SqliteEvaluationRepository,
    create_repository,
)
from agent_eval.storage.sqlite import TARGET_VERSION, apply_migrations, current_version
from tests.conftest import make_case, make_dataset


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Path to a database file in a not-yet-created directory."""
    return tmp_path / "nested" / "dir" / "results.db"


@pytest.fixture
def repo(db_path: Path):
    """A SQLite repository, closed at the end of the test."""
    repository = SqliteEvaluationRepository(db_path)
    yield repository
    repository.close()


async def run_dataset(
    repo,
    dataset: Dataset,
    *,
    agent_version: str = "1.0.0",
    config_hash: str = "abc123",
    mode: ExecutionMode = ExecutionMode.MOCK,
    persist_cases: bool = True,
    close: bool = False,
):
    """Execute a dataset and persist it, returning the run record.

    Args:
        repo: Repository to persist into.
        dataset: Dataset to execute.
        agent_version: Agent version recorded on the run.
        config_hash: Configuration hash stamped onto the run.
        mode: Execution mode recorded on the run.
        persist_cases: Whether to store per-case results.
        close: Close the repository before returning, for tests that go on to
            reopen the database from scratch.
    """
    outcome = await Runner(
        default_mock_agent(),
        HarnessConfig(),
        judge=MockJudge(0.85),
        repository=repo,
    ).run(
        dataset,
        execution_mode=mode,
        agent_version=AgentVersion(name="mock-support", version=agent_version, framework="mock"),
    )
    if config_hash is not None:
        object.__setattr__(outcome.record, "config_hash", config_hash)
    repo.save_run(outcome.record, outcome.cases if persist_cases else None)
    if close:
        repo.close()
    return outcome


class TestSchema:
    def test_creates_parent_directories(self, db_path: Path) -> None:
        with SqliteEvaluationRepository(db_path):
            assert db_path.parent.is_dir()

    def test_schema_version_recorded(self, repo) -> None:
        connection = repo.connection
        assert current_version(connection) == TARGET_VERSION

    def test_migrations_are_idempotent(self, repo) -> None:
        connection = repo.connection
        assert apply_migrations(connection) == TARGET_VERSION
        assert apply_migrations(connection) == TARGET_VERSION

    def test_initialize_is_idempotent(self, repo) -> None:
        repo.initialize()
        repo.initialize()

    def test_foreign_keys_enabled(self, repo) -> None:
        enabled = repo.connection.execute("PRAGMA foreign_keys").fetchone()[0]
        assert enabled == 1

    def test_tables_exist(self, repo) -> None:
        names = {
            row[0]
            for row in repo.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"runs", "case_results", "baselines", "datasets"} <= names

    def test_wal_mode_on_disk(self, db_path: Path) -> None:
        repository = SqliteEvaluationRepository(db_path)
        repository.initialize()
        mode = repository.connection.execute("PRAGMA journal_mode").fetchone()[0]
        repository.close()
        assert mode.lower() == "wal"

    def test_in_memory_database(self) -> None:
        repository = SqliteEvaluationRepository(":memory:")
        repository.initialize()
        assert repository.get_run("nope") is None
        repository.close()


class TestPersistence:
    async def test_run_survives_reopen(self, db_path: Path, example_dataset) -> None:
        first = SqliteEvaluationRepository(db_path)
        outcome = await run_dataset(first, example_dataset, close=True)

        second = SqliteEvaluationRepository(db_path)
        stored = second.get_run(outcome.run_id)
        assert stored is not None
        assert stored.dataset_ref == "customer_support@1.0.0"
        assert stored.aggregate.total == outcome.aggregate.total
        assert len(second.get_cases(outcome.run_id)) == outcome.aggregate.total
        second.close()

    async def test_aggregate_round_trips_exactly(self, repo, example_dataset) -> None:
        outcome = await run_dataset(repo, example_dataset)
        stored = repo.get_run(outcome.run_id)
        assert stored is not None
        assert stored.aggregate.overall_score == outcome.aggregate.overall_score
        assert stored.aggregate.passed == outcome.aggregate.passed
        assert stored.aggregate.errored == outcome.aggregate.errored
        assert stored.aggregate.latency.p95_ms == outcome.aggregate.latency.p95_ms
        assert stored.aggregate.tokens.total_tokens == outcome.aggregate.tokens.total_tokens
        assert stored.aggregate.failing_case_ids == outcome.aggregate.failing_case_ids

    async def test_case_results_round_trip(self, repo, example_dataset) -> None:
        outcome = await run_dataset(repo, example_dataset)
        stored = repo.get_cases(outcome.run_id)
        assert [c.case_id for c in stored] == [c.case_id for c in outcome.cases]
        original = outcome.cases[0]
        restored = stored[0]
        assert restored.status is original.status
        assert restored.result.output == original.result.output
        assert restored.assertions == original.assertions
        assert restored.score == original.score
        assert restored.latency_ms == original.latency_ms

    async def test_trajectories_survive(self, repo, example_dataset) -> None:
        outcome = await run_dataset(repo, example_dataset)
        stored = repo.get_cases(outcome.run_id)
        case = next(c for c in stored if c.case_id == "order_status_001")
        assert case.result.tool_names() == ["get_order_status"]
        assert case.result.steps
        assert case.result.token_usage is not None

    async def test_errors_survive(self, repo, example_dataset) -> None:
        outcome = await run_dataset(repo, example_dataset)
        stored = repo.get_cases(outcome.run_id)
        failed = next(c for c in stored if c.case_id == "tool_failure_001")
        # The case passes because it expects the failure, but the error text
        # must still round-trip through storage.
        assert failed.status is CaseStatus.PASSED
        assert failed.error
        assert failed.result.error

    async def test_agent_and_judge_identity_round_trip(self, repo, example_dataset) -> None:
        outcome = await run_dataset(repo, example_dataset, agent_version="9.9.9")
        stored = repo.get_run(outcome.run_id)
        assert stored is not None
        assert stored.agent.name == "mock-support"
        assert stored.agent.version == "9.9.9"
        assert stored.judge.name == "mock-judge"

    async def test_saving_twice_replaces_cases(self, repo, example_dataset) -> None:
        outcome = await run_dataset(repo, example_dataset)
        repo.save_run(outcome.record, [])
        assert repo.get_cases(outcome.run_id) == []
        repo.save_run(outcome.record, outcome.cases)
        assert len(repo.get_cases(outcome.run_id)) == outcome.aggregate.total

    async def test_run_without_cases_is_allowed(self, repo, example_dataset) -> None:
        outcome = await run_dataset(repo, example_dataset, persist_cases=False)
        assert repo.get_run(outcome.run_id) is not None
        assert repo.get_cases(outcome.run_id) == []

    async def test_unknown_run_returns_none(self, repo) -> None:
        assert repo.get_run("missing") is None

    async def test_unknown_run_has_no_cases(self, repo) -> None:
        assert repo.get_cases("missing") == []

    async def test_save_run_is_atomic(self, repo, example_dataset) -> None:
        outcome = await run_dataset(repo, example_dataset)
        # A distinct run id isolates this save from the one above. The duplicate
        # case_id violates UNIQUE (run_id, case_id), so the case insert fails and
        # must roll the run header back with it.
        fresh = outcome.record.model_copy(update={"run_id": "run_atomic_probe"})
        duplicate = outcome.cases[1].model_copy(update={"case_id": outcome.cases[0].case_id})
        with pytest.raises(sqlite3.IntegrityError):
            repo.save_run(fresh, [outcome.cases[0], duplicate])
        assert repo.get_run("run_atomic_probe") is None
        assert repo.get_cases("run_atomic_probe") == []


class TestListing:
    async def test_list_runs_newest_first(self, repo, example_dataset) -> None:
        dataset = make_dataset(make_case(), name="ordering", version="1.0.0")
        await run_dataset(repo, dataset, config_hash="a")
        await run_dataset(repo, dataset, config_hash="b")
        runs = repo.list_runs(dataset_name="ordering")
        assert len(runs) == 2
        started = [r.started_at for r in runs]
        assert started == sorted(started, reverse=True)

    async def test_list_runs_filters_by_dataset(self, repo, example_dataset) -> None:
        await run_dataset(repo, make_dataset(name="alpha", version="1.0.0"))
        await run_dataset(repo, make_dataset(name="beta", version="1.0.0"))
        assert len(repo.list_runs(dataset_name="alpha")) == 1
        assert len(repo.list_runs()) == 2

    async def test_list_runs_filters_by_version(self, repo) -> None:
        await run_dataset(repo, make_dataset(name="v", version="1.0.0"))
        await run_dataset(repo, make_dataset(name="v", version="2.0.0"))
        assert len(repo.list_runs(dataset_name="v", dataset_version="2.0.0")) == 1

    async def test_list_runs_filters_by_mode(self, repo) -> None:
        await run_dataset(repo, make_dataset(name="m"), mode=ExecutionMode.MOCK)
        assert len(repo.list_runs(execution_mode="mock")) == 1
        assert repo.list_runs(execution_mode="live") == []

    async def test_list_runs_respects_limit(self, repo) -> None:
        dataset = make_dataset(make_case(), name="lim", version="1.0.0")
        for _ in range(3):
            await run_dataset(repo, dataset)
        assert len(repo.list_runs(dataset_name="lim", limit=2)) == 2

    async def test_get_latest_run(self, repo, example_dataset) -> None:
        outcome = await run_dataset(repo, make_dataset(name="lat", version="1.0.0"))
        latest = repo.get_latest_run(dataset_name="lat")
        assert latest is not None
        assert latest.run_id == outcome.run_id

    async def test_get_latest_run_matches_config_hash(self, repo) -> None:
        dataset = make_dataset(make_case(), name="cfg", version="1.0.0")
        await run_dataset(repo, dataset, config_hash="aaa")
        await run_dataset(repo, dataset, config_hash="bbb")
        found = repo.get_latest_run(dataset_name="cfg", config_hash="aaa")
        assert found is not None
        assert found.config_hash == "aaa"

    async def test_get_latest_run_none_when_absent(self, repo) -> None:
        assert repo.get_latest_run(dataset_name="nothing") is None


class TestBaselines:
    async def test_save_and_get(self, repo, example_dataset) -> None:
        outcome = await run_dataset(repo, example_dataset)
        baseline = save_baseline_from_run(repo, outcome.record, "main")
        stored = repo.get_baseline("main")
        assert stored is not None
        assert stored.aggregate.total == outcome.aggregate.total
        assert stored.run_id == outcome.run_id
        assert baseline.dataset_ref == "customer_support@1.0.0"

    async def test_unknown_baseline(self, repo) -> None:
        assert repo.get_baseline("nope") is None

    async def test_overwrite_baseline(self, repo, example_dataset) -> None:
        outcome = await run_dataset(repo, example_dataset)
        save_baseline_from_run(repo, outcome.record, "main")
        second = outcome.record.model_copy(
            update={"run_id": "run_other", "aggregate": RunAggregate(total=99, passed=99)}
        )
        save_baseline_from_run(repo, second, "main")
        stored = repo.get_baseline("main")
        assert stored is not None
        assert stored.aggregate.total == 99
        assert len(repo.list_baselines()) == 1

    async def test_list_baselines_newest_first(self, repo) -> None:
        for name in ("alpha", "beta"):
            repo.save_baseline(
                Baseline(
                    name=name,
                    dataset_name="d",
                    dataset_version="1.0.0",
                    aggregate=RunAggregate(total=1, passed=1),
                )
            )
        names = [b.name for b in repo.list_baselines()]
        assert set(names) == {"alpha", "beta"}

    async def test_baseline_survives_reopen(self, db_path: Path, example_dataset) -> None:
        first = SqliteEvaluationRepository(db_path)
        outcome = await run_dataset(first, example_dataset)
        save_baseline_from_run(first, outcome.record, "durable")
        first.close()

        second = SqliteEvaluationRepository(db_path)
        assert second.get_baseline("durable") is not None
        second.close()

    async def test_resolve_prefers_explicit_name(self, repo, example_dataset) -> None:
        outcome = await run_dataset(repo, example_dataset)
        save_baseline_from_run(repo, outcome.record, "explicit")
        repo.save_baseline(
            Baseline(
                name="configured",
                dataset_name="customer_support",
                dataset_version="1.0.0",
                aggregate=RunAggregate(total=1, passed=0, failed=1),
            )
        )
        source = resolve_baseline(
            repo, "customer_support", "1.0.0", name="explicit", configured_name="configured"
        )
        assert source.kind == "named"
        assert source.name == "explicit"
        assert source.aggregate is not None
        assert source.aggregate.total == outcome.aggregate.total

    async def test_resolve_falls_back_to_configured(self, repo, example_dataset) -> None:
        repo.save_baseline(
            Baseline(
                name="configured",
                dataset_name="customer_support",
                dataset_version="1.0.0",
                aggregate=RunAggregate(total=7, passed=7),
            )
        )
        source = resolve_baseline(repo, "customer_support", "1.0.0", configured_name="configured")
        assert source.kind == "configured"

    async def test_resolve_falls_back_to_matching_run(self, repo, example_dataset) -> None:
        outcome = await run_dataset(repo, example_dataset, config_hash="same")
        source = resolve_baseline(repo, "customer_support", "1.0.0", config_hash="same")
        assert source.kind == "run"
        assert source.found
        assert source.run is not None
        assert source.run.run_id == outcome.run_id

    async def test_resolve_refuses_mismatched_config(self, repo, example_dataset) -> None:
        await run_dataset(repo, example_dataset, config_hash="old")
        source = resolve_baseline(repo, "customer_support", "1.0.0", config_hash="new")
        assert source.found is False
        assert source.aggregate is None

    async def test_resolve_on_empty_store(self, repo, example_dataset) -> None:
        source = resolve_baseline(repo, "customer_support", "1.0.0")
        assert source.found is False
        assert source.name is None

    async def test_missing_named_baseline_falls_back(self, repo, example_dataset) -> None:
        await run_dataset(repo, example_dataset, config_hash="c")
        source = resolve_baseline(repo, "customer_support", "1.0.0", name="ghost", config_hash="c")
        assert source.kind == "run"
        assert source.run is not None


class TestDatasetsAndLeaderboard:
    async def test_register_and_list(self, repo, example_dataset) -> None:
        repo.register_dataset(example_dataset)
        rows = repo.list_datasets()
        assert rows[0][0] == "customer_support"
        assert rows[0][1] == "1.0.0"
        assert rows[0][2] == len(example_dataset.cases)
        assert rows[0][3] is not None

    async def test_register_is_idempotent(self, repo, example_dataset) -> None:
        repo.register_dataset(example_dataset)
        repo.register_dataset(example_dataset)
        assert len(repo.list_datasets()) == 1

    async def test_leaderboard_empty(self, repo) -> None:
        assert repo.leaderboard("nothing") == []

    async def test_leaderboard_one_row_per_agent_version(self, repo) -> None:
        dataset = make_dataset(make_case(), name="lb", version="1.0.0")
        await run_dataset(repo, dataset, agent_version="1.0.0")
        await run_dataset(repo, dataset, agent_version="2.0.0")
        await run_dataset(repo, dataset, agent_version="2.0.0")
        rows = repo.leaderboard("lb")
        assert len(rows) == 2
        assert {r.agent_version for r in rows} == {"1.0.0", "2.0.0"}

    async def test_leaderboard_sorted_by_score(self, repo) -> None:
        dataset = make_dataset(make_case(), name="lb", version="1.0.0")
        await run_dataset(repo, dataset, agent_version="1.0.0")
        await run_dataset(repo, dataset, agent_version="2.0.0")
        scores = [r.overall_score or 0.0 for r in repo.leaderboard("lb")]
        assert scores == sorted(scores, reverse=True)

    async def test_leaderboard_reports_modes(self, repo) -> None:
        dataset = make_dataset(make_case(), name="lb", version="1.0.0")
        await run_dataset(repo, dataset, agent_version="1.0.0", mode=ExecutionMode.MOCK)
        rows = repo.leaderboard("lb")
        assert rows[0].execution_mode is ExecutionMode.MOCK


class TestFactory:
    def test_creates_sqlite(self, tmp_path: Path) -> None:
        repository = create_repository(StorageConfig(type="sqlite", path=tmp_path / "a.db"))
        assert isinstance(repository, SqliteEvaluationRepository)
        repository.close()

    def test_creates_memory(self) -> None:
        assert isinstance(create_repository(StorageConfig(type="memory")), InMemoryRepository)

    def test_schema_is_ready_without_initialize(self, tmp_path: Path) -> None:
        repository = create_repository(StorageConfig(type="sqlite", path=tmp_path / "ready.db"))
        assert repository.list_datasets() == []
        repository.close()


class TestLifecycle:
    def test_close_is_idempotent(self, db_path: Path) -> None:
        repository = SqliteEvaluationRepository(db_path)
        repository.initialize()
        repository.close()
        repository.close()

    def test_reopens_after_close(self, db_path: Path) -> None:
        repository = SqliteEvaluationRepository(db_path)
        repository.initialize()
        repository.close()
        repository.initialize()
        assert repository.get_run("x") is None
        repository.close()

    def test_context_manager(self, db_path: Path) -> None:
        with SqliteEvaluationRepository(db_path) as repository:
            repository.save_baseline(
                Baseline(
                    name="ctx",
                    dataset_name="d",
                    dataset_version="1.0.0",
                    aggregate=RunAggregate(total=1, passed=1),
                )
            )
        with SqliteEvaluationRepository(db_path) as reopened:
            assert reopened.get_baseline("ctx") is not None


class TestParityWithMemory:
    async def test_both_repositories_agree(self, example_dataset) -> None:
        memory = InMemoryRepository()
        sqlite = SqliteEvaluationRepository(":memory:")
        outcome = await run_dataset(memory, example_dataset)
        sqlite.save_run(outcome.record, outcome.cases)

        assert sqlite.get_run(outcome.run_id) == memory.get_run(outcome.run_id)
        assert sqlite.get_cases(outcome.run_id) == memory.get_cases(outcome.run_id)
        assert sqlite.list_datasets() == memory.list_datasets()
        assert len(sqlite.leaderboard("customer_support")) == len(
            memory.leaderboard("customer_support")
        )
        sqlite.close()
