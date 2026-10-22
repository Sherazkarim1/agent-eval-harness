"""In-memory repository, used by tests and ``storage.type: memory``."""

from __future__ import annotations

from agent_eval.domain.models import (
    Baseline,
    CaseResult,
    Dataset,
    ExecutionMode,
    LeaderboardRow,
    RunRecord,
)


class InMemoryRepository:
    """Non-persistent :class:`~agent_eval.ports.repository.EvaluationRepository`."""

    def __init__(self) -> None:
        self.runs: dict[str, RunRecord] = {}
        self.cases: dict[str, list[CaseResult]] = {}
        self.baselines: dict[str, Baseline] = {}
        self.datasets: dict[tuple[str, str], Dataset] = {}

    def initialize(self) -> None:
        """No schema to create."""
        return None

    def save_run(self, run: RunRecord, cases: list[CaseResult] | None = None) -> None:
        """Store a run and its case results."""
        self.runs[run.run_id] = run
        if cases is not None:
            self.cases[run.run_id] = list(cases)

    def get_run(self, run_id: str) -> RunRecord | None:
        """Load a run by id."""
        return self.runs.get(run_id)

    def get_cases(self, run_id: str) -> list[CaseResult]:
        """Case results of a stored run."""
        return self.cases.get(run_id, [])

    def list_runs(
        self,
        *,
        dataset_name: str | None = None,
        dataset_version: str | None = None,
        execution_mode: str | None = None,
        limit: int = 50,
    ) -> list[RunRecord]:
        """List runs newest first, optionally filtered."""
        runs = list(self.runs.values())
        if dataset_name is not None:
            runs = [r for r in runs if r.dataset_name == dataset_name]
        if dataset_version is not None:
            runs = [r for r in runs if r.dataset_version == dataset_version]
        if execution_mode is not None:
            runs = [r for r in runs if r.execution_mode == ExecutionMode(execution_mode)]
        runs.sort(key=lambda r: r.started_at, reverse=True)
        return runs[:limit]

    def get_latest_run(
        self,
        *,
        dataset_name: str,
        dataset_version: str | None = None,
        config_hash: str | None = None,
        execution_mode: str | None = None,
    ) -> RunRecord | None:
        """Most recent run matching the filters, or ``None``."""
        matches = self.list_runs(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            execution_mode=execution_mode,
            limit=len(self.runs) or 1,
        )
        for run in matches:
            if config_hash is not None and run.config_hash != config_hash:
                continue
            return run
        return None

    def save_baseline(self, baseline: Baseline) -> None:
        """Store a baseline snapshot."""
        self.baselines[baseline.name] = baseline

    def get_baseline(self, name: str) -> Baseline | None:
        """Load a baseline by name."""
        return self.baselines.get(name)

    def list_baselines(self) -> list[Baseline]:
        """All baselines, newest first."""
        return sorted(self.baselines.values(), key=lambda b: b.created_at, reverse=True)

    def register_dataset(self, dataset: Dataset) -> None:
        """Record dataset metadata."""
        self.datasets[(dataset.name, dataset.version)] = dataset

    def list_datasets(self) -> list[tuple[str, str, int, str | None]]:
        """``(name, version, case_count, created_at)`` per dataset version."""
        return [
            (name, version, len(dataset.cases), None)
            for (name, version), dataset in sorted(self.datasets.items())
        ]

    def leaderboard(self, dataset_name: str) -> list[LeaderboardRow]:
        """Best run per agent version for a dataset."""
        best: dict[tuple[str, str], LeaderboardRow] = {}
        for run in self.runs.values():
            if run.dataset_name != dataset_name:
                continue
            row = LeaderboardRow(
                agent_name=run.agent.name,
                agent_version=run.agent.version,
                dataset_name=run.dataset_name,
                dataset_version=run.dataset_version,
                overall_score=run.aggregate.overall_score,
                pass_rate=run.aggregate.pass_rate,
                avg_latency_ms=run.aggregate.latency.average_ms,
                total_cases=run.aggregate.total,
                run_id=run.run_id,
                run_timestamp=run.started_at,
                execution_mode=run.execution_mode,
            )
            key = (row.agent_name, row.agent_version)
            current = best.get(key)
            if current is None or (row.overall_score or 0) > (current.overall_score or 0):
                best[key] = row
        return sorted(best.values(), key=lambda r: r.overall_score or 0.0, reverse=True)

    def close(self) -> None:
        """Nothing to release."""
        return None
