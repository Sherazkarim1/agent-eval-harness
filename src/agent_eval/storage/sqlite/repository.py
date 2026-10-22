"""SQLite implementation of :class:`~agent_eval.ports.repository.EvaluationRepository`.

Uses only the standard library ``sqlite3`` module: no ORM, no driver, no server.
The database file is created on first use along with its parent directories.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_eval.domain.models import (
    AgentVersion,
    Baseline,
    CaseResult,
    Dataset,
    ExecutionMode,
    JudgeVersion,
    LeaderboardRow,
    RunAggregate,
    RunRecord,
)
from agent_eval.storage.sqlite.migrations import apply_migrations

logger = logging.getLogger(__name__)


def _dumps(value: Any) -> str:
    """Serialise a model to compact JSON for storage."""
    return json.dumps(value, separators=(",", ":"), default=str)


class SqliteEvaluationRepository:
    """Persistent evaluation store backed by SQLite.

    Args:
        path: Database file path. Parent directories are created as needed.
            Use ``":memory:"`` for an ephemeral database.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path) if str(path) != ":memory:" else Path(":memory:")
        self._is_memory = str(path) == ":memory:"
        self._connection: sqlite3.Connection | None = None

    # ---------------------------------------------------------------- lifecycle

    @property
    def connection(self) -> sqlite3.Connection:
        """The open connection, initialising the schema on first access."""
        if self._connection is None:
            self._connection = self._connect()
        return self._connection

    def _connect(self) -> sqlite3.Connection:
        if not self._is_memory:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            str(self.path),
            detect_types=0,
            isolation_level=None,  # explicit transaction control
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        if not self._is_memory:
            connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        apply_migrations(connection)
        return connection

    def initialize(self) -> None:
        """Create the database file and schema if they do not exist."""
        _ = self.connection

    def close(self) -> None:
        """Close the connection. Safe to call more than once."""
        if self._connection is not None:
            try:
                self._connection.close()
            except sqlite3.Error:  # pragma: no cover - defensive
                logger.debug("Ignoring error while closing connection", exc_info=True)
            self._connection = None

    def __del__(self) -> None:
        """Release the connection when the repository is garbage collected.

        Without this a forgotten ``close()`` leaks a file handle until the
        interpreter shuts down, which surfaces as a ``ResourceWarning`` in tests.
        """
        self.close()

    def __enter__(self) -> SqliteEvaluationRepository:
        """Enter a context that closes the connection on exit."""
        self.initialize()
        return self

    def __exit__(self, *_: object) -> None:
        """Close the connection when leaving the context."""
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block inside a single transaction, rolling back on error."""
        connection = self.connection
        connection.execute("BEGIN")
        try:
            yield connection
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")

    # -------------------------------------------------------------------- runs

    def save_run(self, run: RunRecord, cases: list[CaseResult] | None = None) -> None:
        """Persist a run and, optionally, its per-case results atomically.

        Args:
            run: The run header plus aggregate metrics.
            cases: Per-case results, stored in the same transaction as the run.
        """
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, dataset_name, dataset_version, execution_mode,
                    config_hash, agent_name, agent_version, agent_framework,
                    judge_name, judge_version, judge_model, judge_provider,
                    started_at, duration_ms, git_commit, branch, aggregate_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._run_params(run),
            )
            if cases:
                connection.execute("DELETE FROM case_results WHERE run_id = ?", (run.run_id,))
                rows = [
                    self._case_params(run.run_id, index, case) for index, case in enumerate(cases)
                ]
                connection.executemany(
                    """
                    INSERT INTO case_results (
                        run_id, case_id, sequence, status, passed,
                        latency_ms, cost_usd, started_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        logger.debug("Saved run %s with %d cases", run.run_id, len(cases or []))

    @staticmethod
    def _run_params(run: RunRecord) -> tuple[Any, ...]:
        return (
            run.run_id,
            run.dataset_name,
            run.dataset_version,
            run.execution_mode.value,
            run.config_hash,
            run.agent.name,
            run.agent.version,
            run.agent.framework,
            run.judge.name,
            run.judge.version,
            run.judge.model,
            run.judge.provider,
            run.started_at.isoformat(),
            run.duration_ms,
            run.git_commit,
            run.branch,
            _dumps(run.aggregate.model_dump(mode="json")),
        )

    @staticmethod
    def _case_params(run_id: str, sequence: int, case: CaseResult) -> tuple[Any, ...]:
        return (
            run_id,
            case.case_id,
            sequence,
            case.status.value,
            int(case.passed),
            case.latency_ms,
            case.cost_usd,
            case.started_at.isoformat(),
            _dumps(case.model_dump(mode="json")),
        )

    def get_run(self, run_id: str) -> RunRecord | None:
        """Load a run header by id, or ``None`` when unknown."""
        row = self.connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._to_run(row) if row else None

    def list_runs(
        self,
        *,
        dataset_name: str | None = None,
        dataset_version: str | None = None,
        execution_mode: str | None = None,
        limit: int = 50,
    ) -> list[RunRecord]:
        """List runs newest first, optionally filtered.

        Args:
            dataset_name: Restrict to one dataset.
            dataset_version: Restrict to one dataset version.
            execution_mode: Restrict to ``"mock"`` or ``"live"``.
            limit: Maximum number of runs to return.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if dataset_name is not None:
            clauses.append("dataset_name = ?")
            params.append(dataset_name)
        if dataset_version is not None:
            clauses.append("dataset_version = ?")
            params.append(dataset_version)
        if execution_mode is not None:
            clauses.append("execution_mode = ?")
            params.append(ExecutionMode(execution_mode).value)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self.connection.execute(
            f"SELECT * FROM runs {where} ORDER BY started_at DESC, rowid DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._to_run(row) for row in rows]

    def get_latest_run(
        self,
        *,
        dataset_name: str,
        dataset_version: str | None = None,
        config_hash: str | None = None,
        execution_mode: str | None = None,
    ) -> RunRecord | None:
        """Most recent run matching every filter, or ``None``.

        Unlike :meth:`list_runs` this keeps scanning backwards through matches
        until it finds one with the requested ``config_hash``, because a run with
        a different configuration is not a valid comparison source.
        """
        clauses = ["dataset_name = ?"]
        params: list[Any] = [dataset_name]
        if dataset_version is not None:
            clauses.append("dataset_version = ?")
            params.append(dataset_version)
        if execution_mode is not None:
            clauses.append("execution_mode = ?")
            params.append(ExecutionMode(execution_mode).value)
        if config_hash is not None:
            clauses.append("config_hash = ?")
            params.append(config_hash)

        row = self.connection.execute(
            f"SELECT * FROM runs WHERE {' AND '.join(clauses)} "
            "ORDER BY started_at DESC, rowid DESC LIMIT 1",
            params,
        ).fetchone()
        return self._to_run(row) if row else None

    @staticmethod
    def _to_run(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            dataset_name=row["dataset_name"],
            dataset_version=row["dataset_version"],
            agent=AgentVersion(
                name=row["agent_name"],
                version=row["agent_version"],
                framework=row["agent_framework"],
            ),
            judge=JudgeVersion(
                name=row["judge_name"],
                version=row["judge_version"],
                model=row["judge_model"],
                provider=row["judge_provider"],
            ),
            aggregate=RunAggregate.model_validate_json(row["aggregate_json"]),
            execution_mode=ExecutionMode(row["execution_mode"]),
            started_at=datetime.fromisoformat(row["started_at"]),
            config_hash=row["config_hash"],
            duration_ms=row["duration_ms"],
            git_commit=row["git_commit"],
            branch=row["branch"],
        )

    # ------------------------------------------------------------------- cases

    def get_cases(self, run_id: str) -> list[CaseResult]:
        """Case results of a stored run, in execution order."""
        rows = self.connection.execute(
            "SELECT payload_json FROM case_results WHERE run_id = ? ORDER BY sequence",
            (run_id,),
        ).fetchall()
        return [CaseResult.model_validate_json(row["payload_json"]) for row in rows]

    # ---------------------------------------------------------------- baselines

    def save_baseline(self, baseline: Baseline) -> None:
        """Store or overwrite a named baseline snapshot."""
        self.connection.execute(
            """
            INSERT OR REPLACE INTO baselines (
                name, run_id, dataset_name, dataset_version, agent_version,
                config_hash, execution_mode, created_at, aggregate_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                baseline.name,
                baseline.run_id,
                baseline.dataset_name,
                baseline.dataset_version,
                baseline.agent_version,
                baseline.config_hash,
                baseline.execution_mode.value,
                baseline.created_at.isoformat(),
                _dumps(baseline.aggregate.model_dump(mode="json")),
            ),
        )

    def get_baseline(self, name: str) -> Baseline | None:
        """Load a named baseline, or ``None`` when unknown."""
        row = self.connection.execute("SELECT * FROM baselines WHERE name = ?", (name,)).fetchone()
        return self._to_baseline(row) if row else None

    def list_baselines(self) -> list[Baseline]:
        """All stored baselines, newest first."""
        rows = self.connection.execute(
            "SELECT * FROM baselines ORDER BY created_at DESC, name"
        ).fetchall()
        return [self._to_baseline(row) for row in rows]

    @staticmethod
    def _to_baseline(row: sqlite3.Row) -> Baseline:
        return Baseline(
            name=row["name"],
            run_id=row["run_id"],
            dataset_name=row["dataset_name"],
            dataset_version=row["dataset_version"],
            agent_version=row["agent_version"],
            config_hash=row["config_hash"],
            execution_mode=ExecutionMode(row["execution_mode"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            aggregate=RunAggregate.model_validate_json(row["aggregate_json"]),
        )

    # ----------------------------------------------------------------- datasets

    def register_dataset(self, dataset: Dataset) -> None:
        """Record dataset metadata so datasets can be listed later."""
        self.connection.execute(
            """
            INSERT OR REPLACE INTO datasets (name, version, case_count, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                dataset.name,
                dataset.version,
                len(dataset.cases),
                datetime.now().astimezone().isoformat(),
            ),
        )

    def list_datasets(self) -> list[tuple[str, str, int, str | None]]:
        """Return ``(name, version, case_count, created_at)`` per dataset version."""
        rows = self.connection.execute(
            "SELECT name, version, case_count, created_at FROM datasets ORDER BY name, version"
        ).fetchall()
        return [(row["name"], row["version"], row["case_count"], row["created_at"]) for row in rows]

    # -------------------------------------------------------------- leaderboard

    def leaderboard(self, dataset_name: str) -> list[LeaderboardRow]:
        """Best run per ``(agent_name, agent_version)`` for a dataset.

        Ties are broken by preferring the newer run, so repeated evaluations of
        the same agent version do not produce duplicate rows.
        """
        rows = self.connection.execute(
            "SELECT * FROM runs WHERE dataset_name = ? ORDER BY started_at DESC, rowid DESC",
            (dataset_name,),
        ).fetchall()

        best: dict[tuple[str, str], LeaderboardRow] = {}
        for row in rows:
            run = self._to_run(row)
            key = (run.agent.name, run.agent.version)
            if key in best:
                continue
            aggregate = run.aggregate
            best[key] = LeaderboardRow(
                agent_name=run.agent.name,
                agent_version=run.agent.version,
                dataset_name=run.dataset_name,
                dataset_version=run.dataset_version,
                overall_score=aggregate.overall_score,
                pass_rate=aggregate.pass_rate,
                avg_latency_ms=aggregate.latency.average_ms,
                total_cases=aggregate.total,
                run_id=run.run_id,
                run_timestamp=run.started_at,
                execution_mode=run.execution_mode,
            )
        return sorted(
            best.values(),
            key=lambda row: (row.overall_score or 0.0, row.run_timestamp),
            reverse=True,
        )
