"""`agent-eval compare`: compare two stored runs without executing anything."""

from __future__ import annotations

from pathlib import Path

import typer

from agent_eval.cli.context import Context, build_context
from agent_eval.cli.output import console, error, mock_banner, print_regression
from agent_eval.core.baselines import resolve_baseline
from agent_eval.core.exit_policy import (
    EXIT_EVALUATION_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    ExitDecision,
    decide_run_exit,
    describe_regression,
)
from agent_eval.core.regression import RegressionDetector
from agent_eval.domain.models import RegressionReport, RunRecord
from agent_eval.ports.repository import EvaluationRepository


def run(
    baseline: str | None = typer.Argument(
        None, help="Baseline name or run id; defaults to the latest matching run."
    ),
    current: str | None = typer.Option(
        None, "--current", "-n", help="Run id treated as current (default: latest run)."
    ),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file path."),
    fail_on_warning: bool = typer.Option(
        False, "--fail-on-warning", help="Treat warning findings as blocking too."
    ),
    ignore_config_mismatch: bool = typer.Option(
        False,
        "--ignore-config-mismatch",
        help="Compare even when the two runs used different configurations.",
    ),
) -> None:
    """Compare a stored run against a baseline and detect regressions.

    Both runs are read from storage, so this needs no agent, judge or network.
    Exits ``1`` when the comparison shows a blocking regression.
    """
    try:
        context = build_context(config)
    except (FileNotFoundError, ValueError) as exc:
        error(str(exc))
        raise typer.Exit(EXIT_USAGE) from exc

    if not context.config.regression.enabled:
        error("Regression detection is disabled in configuration.")
        raise typer.Exit(EXIT_USAGE)

    repository = context.storage()
    try:
        current_record = _resolve_run(repository, current, "current")
        baseline_record, baseline_label = _resolve_baseline(
            repository, context, current_record, baseline
        )
        current_cases = repository.get_cases(current_record.run_id)
        baseline_cases = repository.get_cases(baseline_record.run_id)
    except ValueError as exc:
        error(str(exc))
        raise typer.Exit(EXIT_USAGE) from exc
    finally:
        repository.close()

    if baseline_record.run_id == current_record.run_id:
        error("The baseline and the current run are the same run; nothing to compare.")
        raise typer.Exit(EXIT_USAGE)

    mock_banner(current_record.execution_mode)

    report = RegressionDetector(context.config.regression).compare(
        current_record.aggregate,
        baseline_record.aggregate,
        baseline_name=baseline_label,
        baseline_run_id=baseline_record.run_id,
        current_cases=current_cases,
        baseline_cases=baseline_cases,
        baseline_dataset_version=baseline_record.dataset_version,
        current_dataset_version=current_record.dataset_version,
        baseline_config_hash=None if ignore_config_mismatch else baseline_record.config_hash,
        current_config_hash=None if ignore_config_mismatch else current_record.config_hash,
    )
    print_regression(report)

    decision = decide_run_exit(context.config, current_record.aggregate, report)
    if fail_on_warning:
        decision = _escalate(decision, report)
    console.print(describe_regression(report))
    if not decision.ok:
        error(decision.reason)
    raise typer.Exit(decision.code)


def _resolve_run(
    repository: EvaluationRepository,
    run_id: str | None,
    label: str,
) -> RunRecord:
    """Load a run by id, or the most recent run when no id is given.

    Raises:
        ValueError: If the id is unknown or storage holds no runs.
    """
    if run_id is None:
        runs = repository.list_runs(limit=1)
        if not runs:
            raise ValueError(
                f"No stored runs to use as the {label} side. "
                "Run `agent-eval run` first, or pass a run id."
            )
        return runs[0]
    record = repository.get_run(run_id)
    if record is None:
        raise ValueError(f"No run found with id {run_id!r}.")
    return record


def _resolve_baseline(
    repository: EvaluationRepository,
    context: Context,
    current_record: RunRecord,
    baseline: str | None,
) -> tuple[RunRecord, str]:
    """Resolve the baseline run for a comparison.

    A stored baseline is preferred; otherwise the previous run of the same
    dataset and configuration is used.

    Raises:
        ValueError: If no baseline can be found.
    """
    if baseline is not None and baseline.startswith("run:"):
        run_id = baseline.removeprefix("run:")
        record = repository.get_run(run_id)
        if record is None:
            raise ValueError(f"No run found with id {run_id!r}.")
        return record, baseline

    source = resolve_baseline(
        repository,
        current_record.dataset_name,
        current_record.dataset_version,
        name=baseline,
        configured_name=context.config.regression.baseline,
        execution_mode=current_record.execution_mode,
        config_hash=current_record.config_hash,
    )
    if source.run is not None:
        return source.run, source.name or f"run:{source.run.run_id}"
    if source.baseline is not None:
        record = repository.get_run(source.baseline.run_id) if source.baseline.run_id else None
        if record is None:
            raise ValueError(
                f"Baseline {source.baseline.name!r} was found but its run "
                f"{source.baseline.run_id!r} is no longer stored."
            )
        return record, source.baseline.name
    raise ValueError(
        "No baseline found for this dataset and configuration. "
        "Save one with `agent-eval baseline save --name <name>`."
    )


def _escalate(decision: ExitDecision, report: RegressionReport) -> ExitDecision:
    """Downgrade a pass to a failure when warnings should block."""
    if decision.code == EXIT_OK and any(f.severity.value == "warning" for f in report.findings):
        return ExitDecision(
            EXIT_EVALUATION_FAILED,
            "Warning-level regression findings are blocking (--fail-on-warning).",
        )
    return decision
