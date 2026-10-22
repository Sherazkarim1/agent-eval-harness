"""Persistence port.

The default implementation is SQLite (stdlib only), but any storage backend can
be plugged in by implementing this protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_eval.domain.models import (
    Baseline,
    CaseResult,
    Dataset,
    LeaderboardRow,
    RunRecord,
)


@runtime_checkable
class EvaluationRepository(Protocol):
    """Stores runs, baselines and dataset metadata."""

    def initialize(self) -> None:
        """Create the schema if it does not exist yet."""
        ...

    def save_run(self, run: RunRecord, cases: list[CaseResult] | None = None) -> None:
        """Persist a run and, optionally, its per-case results atomically.

        Args:
            run: The run header plus aggregate metrics.
            cases: ``list[CaseResult]`` for full run persistence.
        """
        ...

    def get_run(self, run_id: str) -> RunRecord | None:
        """Load a run header by id, or ``None`` when unknown."""
        ...

    def get_cases(self, run_id: str) -> list[CaseResult]:
        """Load the per-case results of a run, in execution order."""
        ...

    def list_runs(
        self,
        *,
        dataset_name: str | None = None,
        dataset_version: str | None = None,
        execution_mode: str | None = None,
        limit: int = 50,
    ) -> list[RunRecord]:
        """List runs, newest first."""
        ...

    def get_latest_run(
        self,
        *,
        dataset_name: str,
        dataset_version: str | None = None,
        config_hash: str | None = None,
        execution_mode: str | None = None,
    ) -> RunRecord | None:
        """Return the most recent matching run, or ``None``."""
        ...

    def save_baseline(self, baseline: Baseline) -> None:
        """Store (or overwrite) a named baseline snapshot."""
        ...

    def get_baseline(self, name: str) -> Baseline | None:
        """Load a named baseline, or ``None`` when unknown."""
        ...

    def list_baselines(self) -> list[Baseline]:
        """List all stored baselines, newest first."""
        ...

    def register_dataset(self, dataset: Dataset) -> None:
        """Record dataset metadata so datasets can be listed later."""
        ...

    def list_datasets(self) -> list[tuple[str, str, int, str | None]]:
        """Return ``(name, version, case_count, created_at)`` per dataset version."""
        ...

    def leaderboard(self, dataset_name: str) -> list[LeaderboardRow]:
        """Best run per ``(agent_name, agent_version)`` for a dataset."""
        ...

    def close(self) -> None:
        """Release underlying resources."""
        ...
