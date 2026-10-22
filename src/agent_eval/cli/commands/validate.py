"""`agent-eval validate`: check configuration and dataset without running."""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import ValidationError
from rich.table import Table

from agent_eval.cli.context import build_context
from agent_eval.cli.output import console, error, success
from agent_eval.config import find_config_file
from agent_eval.datasets import summarise_dataset, validate_dataset_file

CONFIG_OPTION = typer.Option(None, "--config", "-c", help="Path to the config file.")


def run(
    config: Path | None = CONFIG_OPTION,
    dataset: Path | None = typer.Option(
        None, "--dataset", "-d", help="Dataset file to validate (defaults to config)."
    ),
) -> None:
    """Validate the configuration file and the dataset.

    Reports every problem found rather than stopping at the first, so a broken
    setup can be fixed in one pass.
    """
    try:
        context = build_context(config)
    except (FileNotFoundError, ValueError, ValidationError) as exc:
        error(str(exc))
        raise typer.Exit(2) from exc

    table = Table(title="Validation", show_header=False, title_justify="left")
    problems = 0

    config_path = context.config_path or find_config_file()
    if config_path is not None:
        table.add_row("config", f"[green]ok[/green] ({config_path})")
    else:
        table.add_row(
            "config",
            "[yellow]using built-in defaults (no config file found)[/yellow]",
        )
    table.add_row("fingerprint", context.config.fingerprint())

    dataset_path = dataset or context.dataset_path()
    issues = validate_dataset_file(dataset_path)
    if issues:
        problems += len(issues)
        table.add_row("dataset", f"[red]{len(issues)} issue(s)[/red] ({dataset_path})")
    else:
        try:
            loaded = context.load_dataset()
        except (FileNotFoundError, ValueError) as exc:
            problems += 1
            table.add_row("dataset", f"[red]unreadable[/red] ({dataset_path}): {exc}")
        else:
            table.add_row(
                "dataset",
                f"[green]ok[/green] ({summarise_dataset(loaded)})",
            )

    console.print(table)

    if issues:
        console.print("\n[bold red]Dataset issues:[/bold red]")
        for issue in issues:
            location = issue.case_id or issue.location
            console.print(f"  [{issue.location}] {location}: {issue.message}")

    if problems:
        console.print(f"\n[red]{problems} problem(s) found.[/red]")
        raise typer.Exit(1)
    success("Everything checks out.")
