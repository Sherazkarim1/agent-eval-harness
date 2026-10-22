"""Shared helpers for building report payloads.

Every report format renders the *same* payload, so the numbers in the JSON, the
Markdown and the HTML can never disagree.
"""

from __future__ import annotations

import math
from typing import Any

from agent_eval.domain.models import (
    CaseResult,
    CaseStatus,
    ExecutionMode,
    RegressionFinding,
    RegressionReport,
    RegressionSeverity,
    RunAggregate,
    RunRecord,
)

#: Decimal places used for every float in a report. Reporting rounds; the stored
#: values keep full precision.
PRECISION = 4


def round_floats(value: Any, precision: int = PRECISION) -> Any:
    """Recursively round floats so reports never show ``0.7283000000000001``."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        rounded = round(value, precision)
        # Normalise -0.0 to 0.0 so output never depends on sign of zero.
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, dict):
        return {key: round_floats(item, precision) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item, precision) for item in value]
    return value


def _score(value: float | None) -> float | None:
    return None if value is None else round_floats(value)


def aggregate_payload(aggregate: RunAggregate) -> dict[str, Any]:
    """Serialise a run aggregate with rounded floats."""
    return {
        "total": aggregate.total,
        "passed": aggregate.passed,
        "failed": aggregate.failed,
        "errored": aggregate.errored,
        "pass_rate": round_floats(aggregate.pass_rate),
        "scores": {
            "overall": _score(aggregate.overall_score),
            "deterministic": _score(aggregate.deterministic_score),
            "judge": _score(aggregate.judge_score),
            "trajectory": _score(aggregate.trajectory_score),
            "performance": _score(aggregate.performance_score),
        },
        "latency_ms": {
            "average": round_floats(aggregate.latency.average_ms),
            "p50": round_floats(aggregate.latency.p50_ms),
            "p95": round_floats(aggregate.latency.p95_ms),
            "max": round_floats(aggregate.latency.max_ms),
        },
        "tokens": {
            "input": aggregate.tokens.input_tokens,
            "output": aggregate.tokens.output_tokens,
            "cached_input": aggregate.tokens.cached_input_tokens,
            "total": aggregate.tokens.total_tokens,
        },
        "cost": {
            "api_cost_usd": round_floats(aggregate.cost.api_cost_usd),
            "note": aggregate.cost.note,
        },
        "failing_case_ids": list(aggregate.failing_case_ids),
        "errored_case_ids": list(aggregate.errored_case_ids),
    }


def case_payload(case: CaseResult, *, include_trajectory: bool) -> dict[str, Any]:
    """Serialise one case result, optionally including its full trajectory."""
    payload: dict[str, Any] = {
        "case_id": case.case_id,
        "input": case.case.input,
        "status": case.status.value,
        "latency_ms": round_floats(case.latency_ms),
        "cost_usd": round_floats(case.cost_usd),
        "priority": case.case.priority.value,
        "tags": list(case.case.tags),
        "output": case.result.output,
        "error": case.error,
        "score": (
            {
                "overall": _score(case.score.overall),
                "deterministic": _score(case.score.deterministic),
                "judge": _score(case.score.judge),
                "trajectory": _score(case.score.trajectory),
                "performance": _score(case.score.performance),
            }
            if case.score is not None
            else None
        ),
        "assertions": [
            {
                "assertion": a.assertion.value,
                "passed": a.passed,
                "expected": round_floats(a.expected),
                "actual": round_floats(a.actual),
                "message": a.message,
            }
            for a in case.assertions
        ],
        "judge": (
            {
                "score": _score(case.judge.score),
                "reason": case.judge.reason,
                "criteria": round_floats(case.judge.criteria),
                "error": case.judge.error,
                "provider": case.judge.provider,
                "model": case.judge.model,
                "latency_ms": round_floats(case.judge.latency_ms),
            }
            if case.judge is not None
            else None
        ),
    }
    if include_trajectory:
        payload["trajectory"] = {
            "steps": [
                {
                    "type": step.type.value,
                    "name": step.name,
                    "input": round_floats(step.input),
                    "output": round_floats(step.output),
                    "duration_ms": round_floats(step.duration_ms),
                }
                for step in case.result.steps
            ],
            "tool_calls": [
                {
                    "name": call.name,
                    "arguments": round_floats(call.arguments),
                    "result": round_floats(call.result),
                    "error": call.error,
                    "duration_ms": round_floats(call.duration_ms),
                }
                for call in case.result.tool_calls
            ],
            "token_usage": (
                {
                    "input": case.result.token_usage.input_tokens,
                    "output": case.result.token_usage.output_tokens,
                    "total": case.result.token_usage.total_tokens,
                }
                if case.result.token_usage is not None
                else None
            ),
        }
    return payload


def finding_payload(finding: RegressionFinding) -> dict[str, Any]:
    """Serialise one regression finding."""
    return {
        "kind": finding.kind.value,
        "severity": finding.severity.value,
        "metric": finding.metric,
        "baseline_value": round_floats(finding.baseline_value),
        "current_value": round_floats(finding.current_value),
        "delta": round_floats(finding.delta),
        "threshold": round_floats(finding.threshold),
        "message": finding.message,
        "case_ids": list(finding.case_ids),
    }


def regression_payload(regression: RegressionReport | None) -> dict[str, Any] | None:
    """Serialise a regression report."""
    if regression is None:
        return None
    return {
        "status": regression.status.value,
        "has_baseline": regression.has_baseline,
        "baseline_name": regression.baseline_name,
        "baseline_run_id": regression.baseline_run_id,
        "score_delta": round_floats(regression.score_delta),
        "findings": [finding_payload(f) for f in regression.findings],
    }


def build_payload(
    record: RunRecord,
    cases: list[CaseResult],
    *,
    regression: RegressionReport | None = None,
    include_trajectories: bool = True,
    max_failures: int | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Assemble the canonical report payload.

    The same payload backs every output format. ``execution_mode`` is always
    included so a mock score can never be mistaken for a real measurement.

    Args:
        record: The run record.
        cases: Per-case results, in execution order.
        regression: Optional regression report.
        include_trajectories: Whether to embed full trajectories.
        max_failures: Cap on how many failing cases are detailed.
        config_path: Path of the configuration file, for traceability.

    Returns:
        A JSON-serialisable dictionary.
    """
    failing = [c for c in cases if c.status is CaseStatus.FAILED]
    errored = [c for c in cases if c.status is CaseStatus.ERRORED]
    problems = failing + errored
    truncated = 0
    if max_failures is not None and len(problems) > max_failures:
        truncated = len(problems) - max_failures
        problems = problems[:max_failures]

    return {
        "schema_version": 1,
        "run_id": record.run_id,
        "execution_mode": record.execution_mode.value,
        "is_mock": record.execution_mode is ExecutionMode.MOCK,
        "dataset": {
            "name": record.dataset_name,
            "version": record.dataset_version,
            "ref": record.dataset_ref,
        },
        "agent": {
            "name": record.agent.name,
            "version": record.agent.version,
            "framework": record.agent.framework,
        },
        "judge": {
            "name": record.judge.name,
            "version": record.judge.version,
            "model": record.judge.model,
            "provider": record.judge.provider,
        },
        "started_at": record.started_at.isoformat(),
        "duration_ms": round_floats(record.duration_ms),
        "config_hash": record.config_hash,
        "config_path": config_path,
        "git_commit": record.git_commit,
        "branch": record.branch,
        "aggregate": aggregate_payload(record.aggregate),
        "cases": [case_payload(case, include_trajectory=include_trajectories) for case in cases],
        "problem_count": len(failing) + len(errored),
        "truncated_problems": truncated,
        "regression": regression_payload(regression),
    }


def severity_style(severity: RegressionSeverity) -> str:
    """Emoji-free marker used to render a severity consistently."""
    return {
        RegressionSeverity.ERROR: "[error]",
        RegressionSeverity.WARNING: "[warn]",
        RegressionSeverity.INFO: "[info]",
    }[severity]
