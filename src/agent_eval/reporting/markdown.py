"""Markdown report writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_eval.domain.models import CaseStatus
from agent_eval.reporting.common import build_payload

_STATUS_LABEL = {
    CaseStatus.PASSED.value: "PASS",
    CaseStatus.FAILED.value: "FAIL",
    CaseStatus.ERRORED.value: "ERR",
}


def _fmt(value: float | None, suffix: str = "") -> str:
    """Render a float or a placeholder for ``None``."""
    return "n/a" if value is None else f"{value}{suffix}"


def _table(rows: list[list[str]], headers: list[str]) -> list[str]:
    """Render a GitHub-flavoured markdown table."""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render(payload: dict[str, Any]) -> str:
    """Render a report payload as Markdown."""
    aggregate = payload["aggregate"]
    scores = aggregate["scores"]
    lines: list[str] = [
        f"# Evaluation report - {payload['dataset']['ref']}",
        "",
        f"- **Run**: `{payload['run_id']}`",
        f"- **Agent**: {payload['agent']['name']} {payload['agent']['version']} "
        f"(`{payload['agent']['framework']}`)",
        f"- **Judge**: {payload['judge']['name']}"
        + (f" via `{payload['judge']['model']}`" if payload["judge"]["model"] else ""),
        f"- **Started**: {payload['started_at']}",
    ]

    if payload["is_mock"]:
        lines += [
            "",
            "> **MOCK RUN.** Scores below come from a mock agent and are not real "
            "measurements. Use `--mock false` for a live run.",
        ]

    lines += [
        "",
        "## Summary",
        "",
        *_table(
            [
                ["Total", str(aggregate["total"])],
                ["Passed", str(aggregate["passed"])],
                ["Failed", str(aggregate["failed"])],
                ["Errored", str(aggregate["errored"])],
                ["Pass rate", _fmt(aggregate["pass_rate"])],
                [
                    "Overall score",
                    "n/a" if scores["overall"] is None else f"{scores['overall']:.4f}",
                ],
                [
                    "Deterministic",
                    "n/a" if scores["deterministic"] is None else f"{scores['deterministic']:.4f}",
                ],
                ["Judge", "n/a" if scores["judge"] is None else f"{scores['judge']:.4f}"],
                [
                    "Trajectory",
                    "n/a" if scores["trajectory"] is None else f"{scores['trajectory']:.4f}",
                ],
                [
                    "Performance",
                    "n/a" if scores["performance"] is None else f"{scores['performance']:.4f}",
                ],
            ],
            ["Metric", "Value"],
        ),
        "",
        "## Latency",
        "",
        *_table(
            [
                ["Average", _fmt(aggregate["latency_ms"]["average"], " ms")],
                ["p50", _fmt(aggregate["latency_ms"]["p50"], " ms")],
                ["p95", _fmt(aggregate["latency_ms"]["p95"], " ms")],
                ["Max", _fmt(aggregate["latency_ms"]["max"], " ms")],
            ],
            ["Percentile", "Latency"],
        ),
        "",
        "## Tokens & cost",
        "",
        *_table(
            [
                ["Input tokens", str(aggregate["tokens"]["input"])],
                ["Output tokens", str(aggregate["tokens"]["output"])],
                ["Total tokens", str(aggregate["tokens"]["total"])],
                ["API cost", f"${aggregate['cost']['api_cost_usd']:.4f}"],
            ],
            ["Metric", "Value"],
        ),
        "",
        f"_{aggregate['cost']['note']}_",
    ]

    cases = payload["cases"]
    lines += [
        "",
        "## Cases",
        "",
        *_table(
            [
                [
                    c["case_id"],
                    _STATUS_LABEL[c["status"]],
                    _fmt(
                        None if c["score"] is None else c["score"]["overall"],
                    ),
                    _fmt(c["latency_ms"], " ms"),
                    c["tags"][0] if c["tags"] else "",
                ]
                for c in cases
            ],
            ["Case", "Status", "Score", "Latency", "Tags"],
        ),
    ]

    problems = [c for c in cases if c["status"] != CaseStatus.PASSED.value]
    if problems:
        lines += ["", f"## Problems ({len(problems)})", ""]
        for case in problems:
            lines.append(f"### {case['case_id']} - {case['status'].upper()}")
            lines.append("")
            lines.append(f"- **Input**: {case['input']}")
            if case["error"]:
                lines.append(f"- **Error**: {case['error']}")
            failed = [a for a in case["assertions"] if not a["passed"]]
            if failed:
                lines.append("- **Failed assertions**:")
                for assertion in failed:
                    lines.append(f"  - `{assertion['assertion']}`: {assertion['message']}")
            if case["judge"] and case["judge"].get("reason"):
                lines.append(f"- **Judge**: {case['judge']['reason']}")
            lines.append("")

    if payload["truncated_problems"]:
        lines += [
            f"_{payload['truncated_problems']} further problem(s) omitted from this "
            "report; see the JSON report for the full set._",
            "",
        ]

    regression = payload["regression"]
    if regression is not None:
        lines += ["## Regression", ""]
        if not regression["has_baseline"]:
            lines.append(
                "No baseline available, so no comparison was performed. "
                "Create one with `agent-eval baseline create`."
            )
        else:
            delta = regression["score_delta"]
            lines.append(
                f"Status **{regression['status']}** vs baseline "
                f"`{regression['baseline_name']}` "
                f"(score delta {'n/a' if delta is None else f'{delta:+.4f}'})."
            )
            if regression["findings"]:
                lines += [
                    "",
                    *_table(
                        [
                            [
                                finding["severity"].upper(),
                                finding["kind"],
                                finding["message"],
                            ]
                            for finding in regression["findings"]
                        ],
                        ["Severity", "Kind", "Detail"],
                    ),
                ]
        lines.append("")

    return "\n".join(lines) + "\n"


def write(path: Path, record: Any, cases: list[Any], **kwargs: Any) -> Path:
    """Write a Markdown report to ``path``.

    Args:
        path: Destination file; parent directories are created as needed.
        record: The run record.
        cases: Per-case results.
        **kwargs: Forwarded to :func:`build_payload`.

    Returns:
        The path written.
    """
    payload = build_payload(record, cases, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(payload), encoding="utf-8")
    return path
