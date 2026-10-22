"""Console output helpers shared by the CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from agent_eval.domain.models import (
    CaseResult,
    ExecutionMode,
    RegressionReport,
    RegressionSeverity,
    RunAggregate,
)

#: One console, so every command shares the same terminal behaviour.
console = Console()

#: Diagnostics go to stderr so they never pollute piped stdout.
err_console = Console(stderr=True)

_STATUS_STYLE = {
    "passed": "green",
    "failed": "red",
    "errored": "yellow",
}


def echo(message: str = "") -> None:
    """Print a normal message to stdout."""
    console.print(message)


def error(message: str) -> None:
    """Print an error to stderr."""
    err_console.print(f"[bold red]Error:[/bold red] {message}")


def warn(message: str) -> None:
    """Print a warning to stderr."""
    err_console.print(f"[yellow]{message}[/yellow]")


def success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]{message}[/green]")


def mock_banner(mode: ExecutionMode) -> None:
    """Warn when results come from a mock run.

    Every command that prints scores calls this, so a mock number is never
    presented as a real measurement.
    """
    if mode is ExecutionMode.MOCK:
        warn(
            "MOCK RUN - scores below are simulated, not real measurements. "
            "Use a live agent for real numbers."
        )


def print_aggregate(aggregate: RunAggregate) -> None:
    """Print the headline metrics of a run."""
    table = Table(title="Results", show_header=False, title_justify="left")
    table.add_row("total", str(aggregate.total))
    table.add_row("passed", f"[green]{aggregate.passed}[/green]")
    table.add_row("failed", f"[red]{aggregate.failed}[/red]" if aggregate.failed else "0")
    table.add_row("errored", f"[yellow]{aggregate.errored}[/yellow]" if aggregate.errored else "0")
    table.add_row("pass rate", f"{aggregate.pass_rate * 100:.1f}%")
    if aggregate.overall_score is None:
        table.add_row("overall score", "n/a (nothing scorable)")
    else:
        table.add_row("overall score", f"{aggregate.overall_score:.4f}")
    for label, value in (
        ("deterministic", aggregate.deterministic_score),
        ("judge", aggregate.judge_score),
        ("trajectory", aggregate.trajectory_score),
        ("performance", aggregate.performance_score),
    ):
        table.add_row(f"  {label}", "n/a" if value is None else f"{value:.4f}")
    table.add_row("latency avg", f"{aggregate.latency.average_ms:.2f} ms")
    table.add_row("latency p95", f"{aggregate.latency.p95_ms:.2f} ms")
    table.add_row("tokens", str(aggregate.tokens.total_tokens))
    table.add_row("api cost", f"${aggregate.cost.api_cost_usd:.4f}")
    console.print(table)


def print_case_progress(index: int, total: int, case: CaseResult) -> None:
    """Print one line as a case completes."""
    style = _STATUS_STYLE[case.status.value]
    label = case.status.value
    score = case.score_value
    score_text = "" if score is None else f" score={score:.4f}"
    console.print(f"  [{style}]{label:<7}[/{style}] {case.case_id} ({index}/{total}){score_text}")


def print_regression(regression: RegressionReport) -> None:
    """Print a regression report as a table."""
    if not regression.has_baseline:
        warn("No baseline found; regression check skipped.")
        return

    table = Table(title=f"Regression vs {regression.baseline_name}")
    table.add_column("Severity")
    table.add_column("Kind")
    table.add_column("Detail")
    colours = {
        RegressionSeverity.ERROR: "red",
        RegressionSeverity.WARNING: "yellow",
        RegressionSeverity.INFO: "blue",
    }
    for finding in regression.findings:
        colour = colours[finding.severity]
        table.add_row(
            f"[{colour}]{finding.severity.value}[/{colour}]",
            finding.kind.value,
            finding.message,
        )
    console.print(table)

    delta = regression.score_delta
    delta_text = "n/a" if delta is None else f"{delta:+.4f}"
    colour = "green" if (delta or 0) >= 0 else "red"
    console.print(
        f"status: [{colour}]{regression.status.value}[/{colour}]  "
        f"score delta: {delta_text}  "
        f"findings: {len(regression.findings)}"
    )


def print_paths(paths: dict[str, Path]) -> None:
    """Print the report files that were written."""
    for report_format, path in paths.items():
        console.print(f"  {report_format:<9} {path}")


def fail(message: str, code: int = 2) -> None:
    """Print an error and exit with ``code``."""
    error(message)
    raise typer.Exit(code)


def load_error(exc: Exception) -> None:
    """Report a configuration or dataset failure and exit."""
    error(str(exc))
    raise typer.Exit(2) from exc


def git_metadata(root: Path | None = None) -> tuple[str | None, str | None]:
    """Best-effort ``(commit, branch)`` for the working tree.

    Returns ``(None, None)`` when git is unavailable or the directory is not a
    repository; provenance is a nice-to-have, never a hard requirement.
    """
    import subprocess

    cwd = str(root or Path.cwd())
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
            check=False,
        )
        if commit.returncode != 0:
            return None, None
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    branch_name = branch.stdout.strip() if branch.returncode == 0 else None
    return commit.stdout.strip() or None, branch_name or None


def truncate(text: str, limit: int = 70) -> str:
    """Shorten ``text`` for single-line display."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def console_width() -> int:
    """Current console width, with a safe fallback."""
    return console.width or 100


def summary_line(values: dict[str, Any]) -> str:
    """Render a compact ``k=v`` summary line."""
    return "  ".join(f"{k}={v}" for k, v in values.items())
