"""`agent-eval baseline`: manage named baseline snapshots."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from agent_eval.cli.context import Context, build_context
from agent_eval.cli.output import console, error, success
from agent_eval.core.baselines import save_baseline_from_run
from agent_eval.core.baseline_document import (
    document_from_baseline,
    document_to_baseline,
    read_document,
    write_document,
)
from agent_eval.core.exit_policy import EXIT_USAGE
from agent_eval.domain.models import Baseline, RunRecord
from agent_eval.ports.repository import EvaluationRepository

app = typer.Typer(
    help="Save, list, inspect and move named baselines.",
    no_args_is_help=True,
)


def _repository(config: Path | None) -> tuple[Context, EvaluationRepository]:
    """Open storage, raising a clean CLI error when configuration is unusable."""
    try:
        context = build_context(config)
    except (FileNotFoundError, ValueError) as exc:
        error(str(exc))
        raise typer.Exit(EXIT_USAGE) from exc
    return context, context.storage()


def _score_text(baseline: Baseline) -> str:
    score = baseline.aggregate.overall_score
    return "n/a" if score is None else f"{score:.4f}"


@app.command("save")
def save(
    name: str = typer.Option(..., "--name", "-n", help="Name to store the baseline under."),
    run_id: str | None = typer.Option(
        None, "--run", "-r", help="Run id to snapshot (default: latest run)."
    ),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file path."),
) -> None:
    """Capture a stored run as an immutable named baseline."""
    context, repository = _repository(config)
    try:
        record = repository.get_run(run_id) if run_id is not None else _latest(repository, context)
        if record is None:
            error(f"No run found{f' with id {run_id!r}' if run_id else ''}.")
            raise typer.Exit(EXIT_USAGE)
        baseline = save_baseline_from_run(repository, record, name)
    finally:
        repository.close()

    success(f"Baseline {baseline.name!r} saved from run {record.run_id} ({baseline.dataset_ref}).")


@app.command("list")
def list_baselines(
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file path."),
) -> None:
    """List every stored baseline, newest first."""
    _, repository = _repository(config)
    try:
        baselines = repository.list_baselines()
    finally:
        repository.close()

    if not baselines:
        console.print("No baselines stored. Create one with `agent-eval baseline save`.")
        return

    table = Table(title="Baselines")
    table.add_column("name")
    table.add_column("dataset")
    table.add_column("agent")
    table.add_column("mode")
    table.add_column("score")
    table.add_column("created")
    for baseline in baselines:
        table.add_row(
            baseline.name,
            baseline.dataset_ref,
            baseline.agent_version or "-",
            baseline.execution_mode.value,
            _score_text(baseline),
            baseline.created_at.strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)


@app.command("show")
def show(
    name: str = typer.Argument(..., help="Baseline name."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file path."),
) -> None:
    """Print the stored metrics of one baseline."""
    _, repository = _repository(config)
    try:
        baseline = repository.get_baseline(name)
    finally:
        repository.close()

    if baseline is None:
        error(f"No baseline named {name!r}.")
        raise typer.Exit(EXIT_USAGE)

    aggregate = baseline.aggregate
    console.print(f"[bold]{baseline.name}[/bold]")
    console.print(f"  dataset:     {baseline.dataset_ref}")
    console.print(f"  run:         {baseline.run_id or '-'}")
    console.print(f"  agent:       {baseline.agent_version or '-'}")
    console.print(f"  mode:        {baseline.execution_mode.value}")
    console.print(f"  config hash: {baseline.config_hash or '-'}")
    console.print(f"  created:     {baseline.created_at.isoformat()}")
    console.print(
        f"  cases:       {aggregate.total} "
        f"({aggregate.passed} passed, {aggregate.failed} failed, {aggregate.errored} errored)"
    )
    console.print(f"  score:       {_score_text(baseline)}")
    console.print(f"  latency avg: {aggregate.latency.average_ms:.2f} ms")
    console.print(f"  tokens:      {aggregate.tokens.total_tokens}")
    console.print(f"  api cost:    ${aggregate.cost.api_cost_usd:.4f}")


def _latest(repository: EvaluationRepository, context: Context) -> RunRecord | None:
    """Most recent run for the configured dataset, or the newest run overall."""
    try:
        dataset = context.load_dataset()
    except (FileNotFoundError, ValueError):
        runs = repository.list_runs(limit=1)
        return runs[0] if runs else None
    return repository.get_latest_run(dataset_name=dataset.name, dataset_version=dataset.version)


@app.command("export")
def export(
    name: str = typer.Argument(..., help="Name of the stored baseline to export."),
    out: Path = typer.Option(
        None, "--out", "-o", help="Destination file."
    ),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file path."),
) -> None:
    """Write a stored baseline to a portable JSON document.

    The document can be committed to version control, which is how a CI runner
    gets a baseline to compare against: a fresh checkout has no database.
    """
    if out is None:
        out = Path("baselines") / f"{name}.json"
    _, repository = _repository(config)
    try:
        baseline = repository.get_baseline(name)
    finally:
        repository.close()
    if baseline is None:
        error(f"No baseline named {name!r}.")
        raise typer.Exit(EXIT_USAGE)

    path = write_document(out, document_from_baseline(baseline))
    success(f"Exported baseline {name!r} to {path} (score {_score_text(baseline)}).")


@app.command("import")
def import_(
    path: Path = typer.Argument(..., help="Baseline document to load."),
    name: str | None = typer.Option(
        None, "--name", "-n", help="Store under this name (default: the document's name)."
    ),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file path."),
) -> None:
    """Load a baseline document into storage.

    An existing baseline of the same name is overwritten, so re-importing an
    updated document is the normal way to accept a new baseline.
    """
    try:
        document = read_document(path)
    except (FileNotFoundError, ValueError) as exc:
        error(str(exc))
        raise typer.Exit(EXIT_USAGE) from exc

    _, repository = _repository(config)
    try:
        baseline = document_to_baseline(document)
        stored = baseline.model_copy(update={"name": name}) if name else baseline
        repository.save_baseline(stored)
    finally:
        repository.close()
    success(f"Imported baseline {stored.name!r} from {path} (score {_score_text(stored)}).")
