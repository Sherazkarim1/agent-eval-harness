"""`agent-eval report`: re-render reports from a stored run."""

from __future__ import annotations

from pathlib import Path

import typer

from agent_eval.cli.context import Context, build_context
from agent_eval.cli.output import console, error, mock_banner, print_aggregate, print_paths
from agent_eval.core.exit_policy import EXIT_USAGE
from agent_eval.domain.models import CaseResult, RunRecord
from agent_eval.ports.repository import EvaluationRepository
from agent_eval.reporting import build_payload, markdown, write_reports

_VALID_FORMATS = ("json", "markdown", "html")


def run(
    run_id: str | None = typer.Argument(
        None, help="Run id; defaults to the most recent run of the configured dataset."
    ),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file path."),
    report_format: list[str] = typer.Option(
        [], "--format", "-f", help="Report format: json, markdown, html."
    ),
    report_dir: Path | None = typer.Option(
        None, "--report-dir", help="Where to write reports (default: config)."
    ),
    stdout: bool = typer.Option(
        False, "--stdout", help="Print the Markdown report instead of writing files."
    ),
    include_trajectories: bool = typer.Option(
        True,
        "--trajectories/--no-trajectories",
        help="Embed full trajectories in the report.",
    ),
) -> None:
    """Generate a report for an already-stored run.

    No agent is executed, so this is cheap and works without any live provider.
    """
    try:
        context = build_context(config)
    except (FileNotFoundError, ValueError) as exc:
        error(str(exc))
        raise typer.Exit(EXIT_USAGE) from exc

    chosen: list[str] = (
        list(report_format) if report_format else list(context.config.report.formats)
    )
    unknown = [f for f in chosen if f not in _VALID_FORMATS]
    if unknown:
        error(
            f"Unknown report format(s): {', '.join(unknown)}. "
            f"Choose from {', '.join(_VALID_FORMATS)}."
        )
        raise typer.Exit(EXIT_USAGE)

    repository = context.storage()
    try:
        record, cases = _load(repository, context, run_id)
    except ValueError as exc:
        error(str(exc))
        raise typer.Exit(EXIT_USAGE) from exc
    finally:
        repository.close()

    assert record is not None  # narrowed by _load; it raises otherwise
    mock_banner(record.execution_mode)

    console.print(f"[bold]Run[/bold] {record.run_id}  {record.dataset_ref}")
    print_aggregate(record.aggregate)

    if stdout:
        payload = build_payload(
            record,
            cases,
            include_trajectories=include_trajectories,
            max_failures=context.config.report.max_failures_in_report,
            config_path=str(context.config_path) if context.config_path else None,
        )
        console.print()
        console.print(markdown.render(payload))
        return

    output_dir = report_dir or context.config.report.output_dir
    if not output_dir.is_absolute():
        output_dir = (context.project_root or context.root) / output_dir

    paths = write_reports(
        output_dir,
        record,
        cases,
        chosen,
        stem=run_id or record.run_id,
        include_trajectories=include_trajectories,
        max_failures=context.config.report.max_failures_in_report,
        config_path=str(context.config_path) if context.config_path else None,
    )
    console.print("\n[bold]Reports:[/bold]")
    print_paths(paths)


def _load(
    repository: EvaluationRepository,
    context: Context,
    run_id: str | None,
) -> tuple[RunRecord, list[CaseResult]]:
    """Fetch the requested run, or the latest one for the configured dataset.

    Raises:
        ValueError: If the run id is unknown, or no run exists yet.
    """
    if run_id is not None:
        record = repository.get_run(run_id)
        if record is None:
            raise ValueError(f"No run found with id {run_id!r}.")
    else:
        try:
            dataset = context.load_dataset()
            name, version = dataset.name, dataset.version
        except (FileNotFoundError, ValueError):
            name, version = "", None
        record = (
            repository.get_latest_run(dataset_name=name, dataset_version=version)
            if name
            else _latest_any(repository)
        )
        if record is None:
            raise ValueError("No stored runs found. Run `agent-eval run` first, or pass a run id.")
    return record, repository.get_cases(record.run_id)


def _latest_any(repository: EvaluationRepository) -> RunRecord | None:
    """Most recent run regardless of dataset, used when no dataset is loadable."""
    runs = repository.list_runs(limit=1)
    return runs[0] if runs else None
