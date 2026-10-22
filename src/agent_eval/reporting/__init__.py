"""Report generation in JSON, Markdown and HTML.

All three formats render the same payload, so their numbers cannot disagree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from agent_eval.domain.models import CaseResult, RunRecord
from agent_eval.reporting import html_report, json_report, markdown
from agent_eval.reporting.common import build_payload, round_floats

ReportFormat = Literal["json", "markdown", "html"]

#: Filename suffix used for each format.
SUFFIXES: dict[str, str] = {"json": "json", "markdown": "md", "html": "html"}

_WRITERS = {
    "json": json_report.write,
    "markdown": markdown.write,
    "html": html_report.write,
}


def report_path(output_dir: Path, stem: str, report_format: str) -> Path:
    """Path of a report file for a given format."""
    return Path(output_dir) / f"{stem}.{SUFFIXES[report_format]}"


def write_reports(
    output_dir: Path,
    record: RunRecord,
    cases: list[CaseResult],
    formats: list[str],
    *,
    stem: str = "latest",
    **kwargs: Any,
) -> dict[str, Path]:
    """Write one report per requested format.

    Args:
        output_dir: Directory for the reports; created when absent.
        record: The run record.
        cases: Per-case results.
        formats: Any of ``json``, ``markdown``, ``html``.
        stem: Base filename, without extension.
        **kwargs: Forwarded to :func:`build_payload`.

    Returns:
        A mapping of format name to the path written.

    Raises:
        ValueError: If an unknown format is requested.
    """
    written: dict[str, Path] = {}
    for report_format in formats:
        writer = _WRITERS.get(report_format)
        if writer is None:
            raise ValueError(
                f"Unknown report format {report_format!r}; choose from {sorted(_WRITERS)}."
            )
        path = report_path(Path(output_dir), stem, report_format)
        written[report_format] = writer(path, record, cases, **kwargs)
    return written


__all__ = [
    "SUFFIXES",
    "ReportFormat",
    "build_payload",
    "html_report",
    "json_report",
    "markdown",
    "report_path",
    "round_floats",
    "write_reports",
]
