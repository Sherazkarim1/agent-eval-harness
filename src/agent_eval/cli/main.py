"""The ``agent-eval`` command-line application.

Every command is a thin wrapper: it resolves configuration, delegates to the
application layer, and turns the result into output plus an exit code. No
evaluation logic lives here.
"""

from __future__ import annotations

import typer

from agent_eval.cli.commands import baseline as baseline_commands
from agent_eval.cli.commands import compare as compare_commands
from agent_eval.cli.commands import datasets as dataset_commands
from agent_eval.cli.commands import init as init_commands
from agent_eval.cli.commands import report as report_commands
from agent_eval.cli.commands import run as run_commands
from agent_eval.cli.commands import validate as validate_commands

app = typer.Typer(
    name="agent-eval",
    help="A pytest-like evaluation harness for LLM agents.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("init")(init_commands.run)
app.command("validate")(validate_commands.run)
app.command("run")(run_commands.run)
app.command("report")(report_commands.run)
app.command("compare")(compare_commands.run)
app.add_typer(baseline_commands.app, name="baseline")
app.add_typer(dataset_commands.app, name="datasets")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
