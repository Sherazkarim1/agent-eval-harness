"""`agent-eval run`: execute a dataset, score it, persist it and report."""

from __future__ import annotations

from pathlib import Path

import typer

from agent_eval.cli.context import Context, build_context, resolve_agent, resolve_judge
from agent_eval.cli.output import (
    console,
    error,
    git_metadata,
    mock_banner,
    print_aggregate,
    print_case_progress,
    print_paths,
    print_regression,
    truncate,
    warn,
)
from agent_eval.config import HarnessConfig
from agent_eval.core.baselines import save_baseline_from_run
from agent_eval.core.exit_policy import EXIT_INTERNAL, EXIT_USAGE
from agent_eval.core.pipeline import EvaluationRequest, EvaluationResult, evaluate, mode_for
from agent_eval.core.runner import RunOutcome
from agent_eval.domain.models import CaseResult, CaseStatus, Dataset, ExecutionMode
from agent_eval.ports.agent import AgentAdapter
from agent_eval.ports.judge import Judge

_VALID_FORMATS = ("json", "markdown", "html")


def run(
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file path."),
    dataset: Path | None = typer.Option(
        None, "--dataset", "-d", help="Dataset file (overrides config)."
    ),
    mock: bool = typer.Option(
        None, "--mock/--live", help="Force mock or live execution (default: config)."
    ),
    tag: list[str] = typer.Option([], "--tag", "-t", help="Only run cases with this tag."),
    case_id: list[str] = typer.Option([], "--case-id", help="Only run these case ids."),
    compare: bool = typer.Option(
        True, "--compare/--no-compare", help="Compare against the baseline after running."
    ),
    baseline: str | None = typer.Option(
        None, "--baseline", "-b", help="Baseline name to compare against."
    ),
    save_baseline: str | None = typer.Option(
        None, "--save-baseline", help="Store this run as a named baseline."
    ),
    report_format: list[str] = typer.Option(
        [], "--format", "-f", help="Report format: json, markdown, html."
    ),
    report_dir: Path | None = typer.Option(
        None, "--report-dir", help="Where to write reports (default: config)."
    ),
    no_report: bool = typer.Option(False, "--no-report", help="Skip writing report files."),
    no_store: bool = typer.Option(False, "--no-store", help="Do not persist to storage."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print the summary."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full case output."),
) -> None:
    """Run an evaluation and report the results.

    Exits non-zero when any case fails or errors, when the overall score is below
    the configured minimum, or when a regression is detected at a severity listed
    in ``regression.fail_on``. This makes the command usable as a CI gate.
    """
    try:
        context = build_context(config)
    except (FileNotFoundError, ValueError) as exc:
        error(str(exc))
        raise typer.Exit(2) from exc

    try:
        harness_config, dataset_obj = _prepare(context, dataset, tag, case_id)
    except (FileNotFoundError, ValueError) as exc:
        error(str(exc))
        raise typer.Exit(2) from exc

    mode = _resolve_mode(harness_config, mock)
    mock_banner(mode)

    try:
        adapter = resolve_agent(harness_config)
        judge = resolve_judge(harness_config, force_mock=mock)
    except ValueError as exc:
        error(str(exc))
        raise typer.Exit(2) from exc

    if not quiet:
        console.print(
            f"[bold]Running[/bold] {dataset_obj.name}@{dataset_obj.version} "
            f"({len(dataset_obj.cases)} cases, mode={mode.value})"
        )

    git_commit, branch = git_metadata(context.project_root)
    result = _pipeline(
        context=context,
        config=harness_config,
        dataset_obj=dataset_obj,
        adapter=adapter,
        judge=judge,
        mode=mode,
        compare=compare and not no_store,
        baseline=baseline,
        no_store=no_store,
        no_report=no_report,
        report_formats=report_format,
        report_dir=report_dir,
        git_commit=git_commit,
        branch=branch,
        quiet=quiet,
        verbose=verbose,
    )
    outcome = result.outcome
    report = result.regression

    _print_problems(outcome, quiet=quiet, verbose=verbose)
    print_aggregate(outcome.aggregate)
    if report is not None:
        console.print()
        print_regression(report)

    if save_baseline:
        _save_baseline(context, outcome, save_baseline, no_store=no_store)
    if result.reports:
        console.print("\n[bold]Reports:[/bold]")
        print_paths(result.reports)

    decision = result.decision
    if not decision.ok:
        error(decision.reason)
    raise typer.Exit(decision.code)


def _pipeline(
    *,
    context: Context,
    config: HarnessConfig,
    dataset_obj: Dataset,
    adapter: AgentAdapter,
    judge: Judge | None,
    mode: ExecutionMode,
    compare: bool,
    baseline: str | None,
    no_store: bool,
    no_report: bool,
    report_formats: list[str],
    report_dir: Path | None,
    git_commit: str | None,
    branch: str | None,
    quiet: bool,
    verbose: bool,
) -> EvaluationResult:
    """Execute the evaluation through the shared pipeline.

    Returns:
        The :class:`~agent_eval.core.pipeline.EvaluationResult` for this run.

    Raises:
        typer.Exit: With code 2 for a usage error, or 3 for an unexpected one.
    """
    unknown = [f for f in report_formats if f not in _VALID_FORMATS]
    if unknown:
        error(
            f"Unknown report format(s): {', '.join(unknown)}. "
            f"Choose from {', '.join(_VALID_FORMATS)}."
        )
        raise typer.Exit(EXIT_USAGE)

    repository = None if no_store else context.storage()
    total = len(dataset_obj.cases)
    state = {"index": 0}

    def progress(case: CaseResult) -> None:
        state["index"] += 1
        if verbose:
            console.print(f"[bold]{case.case_id}[/bold] input: {truncate(case.case.input)}")
            console.print(f"  output: {truncate(case.result.output, 100)}")
        print_case_progress(state["index"], total, case)

    request = EvaluationRequest(
        dataset=dataset_obj,
        config=config,
        adapter=adapter,
        judge=judge,
        repository=repository,
        execution_mode=mode,
        compare=compare,
        baseline=baseline,
        git_commit=git_commit,
        branch=branch,
        on_case_complete=None if quiet else progress,
    )
    try:
        return evaluate(
            request,
            report_dir=_report_dir(context, report_dir),
            report_formats=report_formats,
            config_path=str(context.config_path) if context.config_path else None,
            write_report=not no_report,
        )
    except Exception as exc:
        error(f"Run failed: {type(exc).__name__}: {exc}")
        raise typer.Exit(EXIT_INTERNAL) from exc
    finally:
        if repository is not None:
            repository.close()


def _report_dir(context: Context, report_dir: Path | None) -> Path | None:
    """Resolve the report directory against the project root."""
    resolved = report_dir or context.config.report.output_dir
    if resolved.is_absolute():
        return resolved
    return (context.project_root or context.root) / resolved


def _prepare(
    context: Context,
    dataset_path: Path | None,
    tags: list[str],
    case_ids: list[str],
) -> tuple[HarnessConfig, Dataset]:
    """Apply CLI overrides to the config, then load and filter the dataset."""
    from agent_eval.datasets import load_dataset_file

    config = context.config
    if dataset_path is not None:
        config = config.model_copy(
            update={"dataset": config.dataset.model_copy(update={"path": dataset_path})}
        )
    if tags or case_ids:
        config = config.model_copy(
            update={
                "run": config.run.model_copy(
                    update={"tags": list(tags), "case_ids": list(case_ids)}
                )
            }
        )

    path = config.dataset.path
    if not path.is_absolute():
        path = (context.project_root or context.root) / path
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    dataset_obj = load_dataset_file(path)

    if tags or case_ids:
        dataset_obj = dataset_obj.select(ids=set(case_ids) if case_ids else None, tags=list(tags))
        if not dataset_obj.cases:
            raise ValueError("No cases matched the given --tag/--case-id filters.")
    return config, dataset_obj


def _resolve_mode(config: HarnessConfig, mock: bool | None) -> ExecutionMode:
    """Reconcile the tri-state ``--mock/--live`` flag with configuration.

    ``None`` means the flag was not supplied, so configuration decides.
    """
    if mock is None:
        return mode_for(config)
    return ExecutionMode.MOCK if mock else ExecutionMode.LIVE


def _print_problems(outcome: RunOutcome, *, quiet: bool, verbose: bool) -> None:
    """List the cases that did not pass."""
    if quiet and not verbose:
        return
    problems = [c for c in outcome.cases if c.status is not CaseStatus.PASSED]
    if not problems:
        return
    console.print(f"\n[bold]{len(problems)} problem case(s):[/bold]")
    for case in problems:
        colour = "red" if case.status is CaseStatus.FAILED else "yellow"
        console.print(f"  [{colour}]{case.status.value}[/{colour}] {case.case_id}")
        if case.error:
            console.print(f"    error: {truncate(case.error, 100)}")
        for assertion in case.failed_assertions:
            console.print(
                f"    [red]x[/red] {assertion.assertion.value}: {truncate(assertion.message, 90)}"
            )


def _save_baseline(context: Context, outcome: RunOutcome, name: str, *, no_store: bool) -> None:
    """Persist this run as a named baseline."""
    if no_store:
        warn("--no-store was given, so the baseline was not saved.")
        return
    repository = context.storage()
    try:
        baseline = save_baseline_from_run(repository, outcome.record, name)
    finally:
        repository.close()
    console.print(f"\n[green]Baseline saved:[/green] {baseline.name} ({baseline.dataset_ref})")

