"""`agent-eval datasets`: inspect dataset versions on disk and in storage."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from agent_eval.cli.context import Context, build_context
from agent_eval.cli.output import console, error
from agent_eval.core.exit_policy import EXIT_USAGE
from agent_eval.datasets import discover_versions, latest_version_path, load_dataset_file

app = typer.Typer(
    help="Inspect the dataset files discovered for this project.",
    no_args_is_help=True,
)


def _context(config: Path | None) -> Context:
    """Resolve configuration, exiting cleanly when it is unusable."""
    try:
        return build_context(config)
    except (FileNotFoundError, ValueError) as exc:
        error(str(exc))
        raise typer.Exit(EXIT_USAGE) from exc


@app.command("list")
def list_datasets(
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file path."),
    directory: Path | None = typer.Option(
        None, "--dir", help="Directory to scan (default: the configured dataset's folder)."
    ),
) -> None:
    """List versioned dataset files found next to the configured dataset."""
    context = _context(config)
    configured = context.dataset_path()
    folder = directory or configured.parent
    if not folder.is_absolute():
        folder = (context.project_root or context.root) / folder

    versions = discover_versions(folder, configured.stem)
    if not versions:
        console.print(f"No versioned dataset files matching {configured.stem}*.yaml")
        console.print(f"in {folder}")
        return

    table = Table(title=f"Datasets in {folder}")
    table.add_column("version")
    table.add_column("path")
    table.add_column("cases")
    table.add_column("tags")
    for version, path in versions:
        try:
            dataset = load_dataset_file(path)
        except ValueError as exc:
            table.add_row(version, str(path), f"[red]{exc}[/red]", "-")
            continue
        tags = sorted({tag for case in dataset.cases for tag in case.tags})
        table.add_row(
            version,
            str(path),
            str(len(dataset.cases)),
            ", ".join(tags) if tags else "-",
        )
    console.print(table)

    latest = latest_version_path(folder, configured.stem)
    if latest is not None:
        console.print(f"latest: {latest.name}  (use --dataset {latest} to run it)")


@app.command("show")
def show(
    path: Path | None = typer.Argument(None, help="Dataset file (default: configured)."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file path."),
) -> None:
    """Summarise one dataset file without running it."""
    context = _context(config)
    target = path or context.dataset_path()
    if not target.is_absolute():
        target = (context.project_root or context.root) / target

    try:
        dataset = load_dataset_file(target)
    except (FileNotFoundError, ValueError) as exc:
        error(str(exc))
        raise typer.Exit(EXIT_USAGE) from exc

    tags: dict[str, int] = {}
    for case in dataset.cases:
        for tag in case.tags:
            tags[tag] = tags.get(tag, 0) + 1

    console.print(f"[bold]{dataset.name}[/bold]@{dataset.version}")
    console.print(f"  file:    {target}")
    console.print(f"  cases:   {len(dataset.cases)}")
    references = sum(1 for c in dataset.cases if c.reference_answer)
    console.print(f"  tags:    {', '.join(f'{k} ({v})' for k, v in sorted(tags.items())) or '-'}")
    console.print(f"  judged:  {references} case(s) carry a reference answer")


@app.command("register")
def register(
    path: Path | None = typer.Argument(None, help="Dataset file (default: configured)."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config file path."),
) -> None:
    """Record a dataset's metadata in storage so it can be listed later."""
    context = _context(config)
    target = path or context.dataset_path()
    if not target.is_absolute():
        target = (context.project_root or context.root) / target

    try:
        dataset = load_dataset_file(target)
    except (FileNotFoundError, ValueError) as exc:
        error(str(exc))
        raise typer.Exit(EXIT_USAGE) from exc

    repository = context.storage()
    try:
        repository.register_dataset(dataset)
    finally:
        repository.close()
    console.print(
        f"[green]Registered[/green] {dataset.name}@{dataset.version} ({len(dataset.cases)} cases)"
    )
