"""The shared evaluation pipeline.

Every entry point - the CLI, the pytest plugin, the dashboard - performs the same
four steps in the same order:

1. resolve the comparison target **before** the run, so a run is never compared
   against itself,
2. execute the dataset,
3. compare the finished run against that target,
4. decide the exit code.

That logic lives here so no caller re-implements it, and so every caller fails a
run for exactly the same reasons.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from agent_eval.config import HarnessConfig
from agent_eval.core.baselines import resolve_baseline
from agent_eval.core.exit_policy import ExitDecision, decide_run_exit
from agent_eval.core.regression import RegressionDetector
from agent_eval.core.runner import Runner, RunOutcome
from agent_eval.domain.models import (
    AgentVersion,
    CaseResult,
    Dataset,
    ExecutionMode,
    RegressionReport,
    RunAggregate,
)
from agent_eval.ports.agent import AgentAdapter
from agent_eval.ports.judge import Judge
from agent_eval.ports.repository import EvaluationRepository
from agent_eval.reporting import write_reports

logger = logging.getLogger(__name__)


def mode_for(config: HarnessConfig) -> ExecutionMode:
    """Execution mode declared by configuration."""
    return ExecutionMode.MOCK if config.run.mode == "mock" else ExecutionMode.LIVE


@dataclass(frozen=True)
class ComparisonTarget:
    """A baseline resolved before the current run was persisted."""

    aggregate: RunAggregate
    name: str
    run_id: str | None = None
    cases: list[CaseResult] = field(default_factory=list)
    dataset_version: str | None = None
    config_hash: str | None = None


@dataclass(frozen=True)
class EvaluationRequest:
    """Everything one evaluation needs."""

    dataset: Dataset
    config: HarnessConfig
    adapter: AgentAdapter
    judge: Judge | None = None
    repository: EvaluationRepository | None = None
    execution_mode: ExecutionMode | None = None
    compare: bool = True
    baseline: str | None = None
    git_commit: str | None = None
    branch: str | None = None
    on_case_complete: Callable[[CaseResult], None] | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """The outcome of a pipeline run, plus its verdict."""

    outcome: RunOutcome
    decision: ExitDecision
    regression: RegressionReport | None = None
    baseline_name: str | None = None
    reports: dict[str, Path] = field(default_factory=dict)

    @property
    def aggregate(self) -> RunAggregate:
        """Aggregated metrics of the run."""
        return self.outcome.aggregate

    @property
    def cases(self) -> list[CaseResult]:
        """Per-case results, in execution order."""
        return self.outcome.cases


def evaluate(
    request: EvaluationRequest,
    *,
    report_dir: Path | None = None,
    report_formats: list[str] | None = None,
    config_path: str | None = None,
    write_report: bool = True,
) -> EvaluationResult:
    """Run one evaluation end to end and decide whether it passes.

    The baseline is resolved before execution on purpose: the current run is
    persisted during execution, so resolving afterwards would let a run become
    its own baseline.

    Args:
        request: The dataset, configuration, components and comparison options.
        report_dir: Where to write reports; defaults to ``config.report.output_dir``.
        report_formats: Formats to write; defaults to ``config.report.formats``.
        config_path: Configuration path recorded in the report for traceability.
        write_report: Set ``False`` to skip writing report files.

    Returns:
        An :class:`EvaluationResult` holding the run, its regression report and
        the exit decision.

    Raises:
        Exception: Whatever the agent, storage or reporting layer raised.
            Callers map this onto their own error conventions.
    """
    config = request.config
    mode = request.execution_mode or mode_for(config)

    target = _resolve_target(request, mode) if request.compare else None
    outcome = _execute(request, mode)

    regression = (
        None
        if target is None
        else RegressionDetector(config.regression).compare(
            outcome.aggregate,
            target.aggregate,
            baseline_name=target.name,
            baseline_run_id=target.run_id,
            current_cases=outcome.cases,
            baseline_cases=target.cases,
            baseline_dataset_version=target.dataset_version,
            current_dataset_version=outcome.record.dataset_version,
            baseline_config_hash=target.config_hash,
            current_config_hash=outcome.record.config_hash,
        )
    )

    reports = (
        {}
        if not write_report
        else _write_reports(request, outcome, regression, report_dir, report_formats, config_path)
    )

    return EvaluationResult(
        outcome=outcome,
        decision=decide_run_exit(config, outcome.aggregate, regression),
        regression=regression,
        baseline_name=target.name if target is not None else None,
        reports=reports,
    )


def _resolve_target(request: EvaluationRequest, mode: ExecutionMode) -> ComparisonTarget | None:
    """Locate the baseline to compare against.

    Returns:
        The resolved target, or ``None`` when regression detection is disabled,
        no repository was supplied, or nothing has been stored yet. Having no
        baseline is a normal first-run state, not an error.
    """
    if request.repository is None or not request.config.regression.enabled:
        return None

    dataset = request.dataset
    source = resolve_baseline(
        request.repository,
        dataset.name,
        dataset.version,
        name=request.baseline,
        configured_name=request.config.regression.baseline,
        execution_mode=mode,
        config_hash=request.config.fingerprint(),
    )
    if not source.found or source.aggregate is None:
        logger.debug("No baseline available for %s@%s", dataset.name, dataset.version)
        return None

    stored, run = source.baseline, source.run
    return ComparisonTarget(
        aggregate=source.aggregate,
        name=source.name or "baseline",
        run_id=run.run_id if run is not None else None,
        # Per-case results only exist for a stored run, not for a snapshot, so
        # new-failure and tool-behaviour checks run only in that case.
        cases=request.repository.get_cases(run.run_id) if run is not None else [],
        dataset_version=_identity(stored, run, "dataset_version"),
        config_hash=_identity(stored, run, "config_hash"),
    )


def _identity(stored: object, run: object, attribute: str) -> str | None:
    """Read a comparison-identity field from the baseline snapshot or its run."""
    for source in (stored, run):
        if source is not None:
            return str(getattr(source, attribute))
    return None


def _execute(request: EvaluationRequest, mode: ExecutionMode) -> RunOutcome:
    """Execute the dataset, persisting when a repository was supplied."""
    config = request.config
    adapter = request.adapter
    return asyncio.run(
        Runner(
            adapter,
            config,
            judge=request.judge,
            repository=request.repository,
        ).run(
            request.dataset,
            execution_mode=mode,
            agent_version=AgentVersion(
                name=config.agent.name or adapter.name,
                version=config.agent.version,
                framework=adapter.framework,
            ),
            git_commit=request.git_commit,
            branch=request.branch,
            on_case_complete=request.on_case_complete,
        )
    )


def _write_reports(
    request: EvaluationRequest,
    outcome: RunOutcome,
    regression: RegressionReport | None,
    report_dir: Path | None,
    report_formats: list[str] | None,
    config_path: str | None,
) -> dict[str, Path]:
    """Write the report files configured for this run."""
    config = request.config
    chosen: list[str] = list(report_formats) if report_formats else list(config.report.formats)
    if not chosen:
        return {}
    return write_reports(
        report_dir or config.report.output_dir,
        outcome.record,
        outcome.cases,
        chosen,
        stem="latest",
        regression=regression,
        include_trajectories=config.report.include_trajectories,
        max_failures=config.report.max_failures_in_report,
        config_path=config_path,
    )
