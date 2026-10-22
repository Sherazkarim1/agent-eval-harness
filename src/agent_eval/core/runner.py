"""Run orchestration: load dataset, execute cases, aggregate, persist.

The runner owns the run lifecycle and nothing else; per-case work belongs to
:class:`~agent_eval.core.evaluator.Evaluator`.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from agent_eval.config import HarnessConfig
from agent_eval.core.aggregate import aggregate_results
from agent_eval.core.evaluator import Evaluator, execution_mode_for
from agent_eval.domain.models import (
    AgentVersion,
    CaseResult,
    CaseStatus,
    Dataset,
    EvalCase,
    ExecutionMode,
    JudgeVersion,
    RunAggregate,
    RunRecord,
)
from agent_eval.ports.agent import AgentAdapter
from agent_eval.ports.clock import Clock, SystemClock
from agent_eval.ports.judge import Judge
from agent_eval.ports.repository import EvaluationRepository

logger = logging.getLogger(__name__)


def new_run_id() -> str:
    """Generate a sortable run identifier."""
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


@dataclass(slots=True)
class RunOutcome:
    """Everything produced by one run."""

    record: RunRecord
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        """Identifier of the run."""
        return self.record.run_id

    @property
    def aggregate(self) -> RunAggregate:
        """Aggregated metrics for the run."""
        return self.record.aggregate


class Runner:
    """Executes a dataset against an agent and records the result."""

    def __init__(
        self,
        adapter: AgentAdapter,
        config: HarnessConfig,
        *,
        judge: Judge | None = None,
        repository: EvaluationRepository | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Create a runner.

        Args:
            adapter: The agent under test.
            config: Harness configuration.
            judge: Optional judge; ``None`` disables LLM judging.
            repository: Optional store; when given, runs are persisted.
            clock: Clock used for timestamps and latency measurement.
        """
        self._adapter = adapter
        self._config = config
        self._judge = judge
        self._repository = repository
        self._clock = clock or SystemClock()

    async def run(
        self,
        dataset: Dataset,
        *,
        run_id: str | None = None,
        execution_mode: ExecutionMode | None = None,
        persist: bool = True,
        agent_version: AgentVersion | None = None,
        judge_version: JudgeVersion | None = None,
        config_hash: str | None = None,
        git_commit: str | None = None,
        branch: str | None = None,
        on_case_complete: Callable[[CaseResult], None] | None = None,
    ) -> RunOutcome:
        """Execute every case of ``dataset``.

        Args:
            dataset: The dataset to run.
            run_id: Explicit run id; generated when omitted.
            execution_mode: Override for the recorded execution mode.
            persist: Whether to write the run to the repository.
            agent_version: Agent identity recorded with the run.
            judge_version: Judge identity recorded with the run.
            config_hash: Configuration fingerprint recorded with the run.
            git_commit: Git commit recorded with the run.
            branch: Git branch recorded with the run.
            on_case_complete: Optional callback invoked with each
                :class:`CaseResult`, used by the CLI for progress output.

        Returns:
            A :class:`RunOutcome` with the persisted record and all case results.
        """
        run_id = run_id or new_run_id()
        mode = execution_mode or execution_mode_for(self._config)
        evaluator = Evaluator(
            self._adapter,
            self._judge,
            self._config,
            clock=self._clock,
        )

        started_at = self._clock.now()
        started = self._clock.monotonic()
        cases = await self._execute_cases(
            evaluator, dataset, run_id, on_case_complete=on_case_complete
        )
        duration_ms = (self._clock.monotonic() - started) * 1000.0

        record = RunRecord(
            run_id=run_id,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            agent=agent_version
            or AgentVersion(
                name=self._adapter.name,
                version=self._adapter.version,
                framework=self._adapter.framework,
            ),
            judge=judge_version or _judge_version(self._judge),
            aggregate=aggregate_results(cases),
            execution_mode=mode,
            started_at=started_at,
            git_commit=git_commit,
            branch=branch,
            config_hash=config_hash if config_hash is not None else self._config.fingerprint(),
            duration_ms=duration_ms,
        )

        if persist and self._repository is not None:
            self._repository.save_run(record, cases)

        return RunOutcome(record=record, cases=cases)

    async def _execute_cases(
        self,
        evaluator: Evaluator,
        dataset: Dataset,
        run_id: str,
        *,
        on_case_complete: Callable[[CaseResult], None] | None = None,
    ) -> list[CaseResult]:
        """Run all cases, honouring concurrency and stop-on-first-failure."""
        cases = dataset.cases
        stop = self._config.run.stop_on_first_failure
        results: list[CaseResult] = []

        if self._config.run.max_concurrency > 1:
            semaphore = asyncio.Semaphore(self._config.run.max_concurrency)

            async def run_one(case: EvalCase) -> CaseResult:
                async with semaphore:
                    return await evaluator.run_case(case, run_id)

            tasks = [asyncio.create_task(run_one(case)) for case in cases]
            if stop:
                # Cancel still-pending work at the first non-passing case so
                # "stop on first failure" also bounds latency under concurrency.
                try:
                    for task in asyncio.as_completed(tasks):
                        result = task.result()
                        results.append(result)
                        if on_case_complete is not None:
                            on_case_complete(result)
                        if result.status is not CaseStatus.PASSED:
                            break
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                return results

            for result in await asyncio.gather(*tasks):
                results.append(result)
                if on_case_complete is not None:
                    on_case_complete(result)
            return results

        for case in cases:
            result = await evaluator.run_case(case, run_id)
            results.append(result)
            if on_case_complete is not None:
                on_case_complete(result)
            if stop and result.status is not CaseStatus.PASSED:
                logger.info("Stopping run %s after failure on %s", run_id, case.id)
                break
        return results


def _judge_version(judge: Judge | None) -> JudgeVersion:
    """Build a :class:`JudgeVersion` from the active judge."""
    if judge is None:
        return JudgeVersion(name="none")
    model = getattr(judge, "model", None)
    provider = getattr(judge, "name", None)
    return JudgeVersion(
        name=getattr(judge, "name", "judge"),
        version=getattr(judge, "version", "0.1.0"),
        model=str(model) if model is not None else None,
        provider=str(provider) if provider is not None else None,
    )


def dataset_path_for(config: HarnessConfig, project_root: Path | None = None) -> Path:
    """Resolve the configured dataset path relative to the project root."""
    root = project_root or Path.cwd()
    path = config.dataset.path
    return path if path.is_absolute() else root / path
