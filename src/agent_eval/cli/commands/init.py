"""`agent-eval init`: scaffold a config file and an example dataset."""

from __future__ import annotations

from pathlib import Path

import typer

from agent_eval.cli.output import console, success
from agent_eval.cli.templates import CONFIG_TEMPLATE, DATASET_TEMPLATE
from agent_eval.config import DEFAULT_CONFIG_FILENAMES

FORCE = typer.Option(False, "--force", help="Overwrite existing files.")
PROJECT_NAME = typer.Option("my-agent-project", "--project-name", help="Project name.")
AGENT_NAME = typer.Option("mock-agent", "--agent-name", help="Agent name recorded in runs.")
DATASET_NAME = typer.Option("example", "--dataset-name", help="Dataset name.")


def run(
    directory: Path = typer.Argument(Path.cwd(), help="Where to create the files."),
    force: bool = FORCE,
    project_name: str = PROJECT_NAME,
    agent_name: str = AGENT_NAME,
    dataset_name: str = DATASET_NAME,
) -> None:
    """Create `agent-eval.yaml` and an example dataset.

    Existing files are never overwritten unless ``--force`` is given.
    """
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / DEFAULT_CONFIG_FILENAMES[0]
    dataset_path = directory / "evals" / "dataset.yaml"

    created: list[Path] = []
    skipped: list[Path] = []

    for path, content in (
        (config_path, CONFIG_TEMPLATE.format(project_name=project_name, agent_name=agent_name)),
        (dataset_path, DATASET_TEMPLATE.format(dataset_name=dataset_name)),
    ):
        if path.exists() and not force:
            skipped.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created.append(path)

    if created:
        success(f"Created {len(created)} file(s):")
        for path in created:
            console.print(f"  {path}")
    for path in skipped:
        console.print(f"  skipped (exists): {path}")

    if skipped and not force:
        console.print("\nRe-run with --force to overwrite.")

    console.print("\nNext steps:")
    console.print("  agent-eval validate      # check config and dataset")
    console.print("  agent-eval run --mock    # run the example dataset offline")
