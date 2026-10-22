"""Regression detection: compare a run against a baseline.

Every finding carries a severity so the caller can decide what to do:

* ``error``   - a real regression, or an invalid comparison
* ``warning`` - a difference worth watching that is within tolerance
* ``info``    - context only

Dataset-version and configuration-hash mismatches are reported as ``error``.
Comparing across different inputs produces meaningless differences, so the tool
refuses to present such a comparison as a clean pass.
"""

from __future__ import annotations

import logging

from agent_eval.config import RegressionConfig
from agent_eval.domain.models import (
    CaseResult,
    FindingKind,
    RegressionFinding,
    RegressionReport,
    RegressionSeverity,
    RegressionStatus,
    RunAggregate,
    RunRecord,
)

logger = logging.getLogger(__name__)

#: Absolute latency change below which a percentage difference is treated as
#: measurement noise. A mock agent answers in well under a millisecond, so
#: sub-millisecond jitter otherwise trips the percentage threshold on every run.
MIN_LATENCY_DELTA_MS = 1.0


class RegressionDetector:
    """Compares an aggregate against a baseline aggregate."""

    def __init__(self, config: RegressionConfig) -> None:
        """Create a detector bound to the given regression configuration."""
        self._config = config

    def compare(
        self,
        current: RunAggregate,
        baseline: RunAggregate | None,
        *,
        baseline_name: str | None = None,
        baseline_run_id: str | None = None,
        current_cases: list[CaseResult] | None = None,
        baseline_cases: list[CaseResult] | None = None,
        baseline_dataset_version: str | None = None,
        current_dataset_version: str | None = None,
        baseline_config_hash: str | None = None,
        current_config_hash: str | None = None,
    ) -> RegressionReport:
        """Compare ``current`` against ``baseline``.

        Args:
            current: Aggregate of the run being evaluated.
            baseline: Aggregate of the baseline, or ``None`` on a first run.
            baseline_name: Display name of the baseline.
            baseline_run_id: Run the baseline came from, for traceability.
            current_cases: Per-case results of the current run.
            baseline_cases: Per-case results of the baseline run.
            baseline_dataset_version: Dataset version behind the baseline.
            current_dataset_version: Dataset version of the current run.
            baseline_config_hash: Configuration hash behind the baseline.
            current_config_hash: Configuration hash of the current run.

        Returns:
            A :class:`RegressionReport`. ``status`` is ``SKIPPED`` when there is
            no baseline, which is a normal first-run state, not a failure.
        """
        if not self._config.enabled:
            logger.debug("Regression detection disabled by configuration")
            return RegressionReport(status=RegressionStatus.SKIPPED, has_baseline=False)

        if baseline is None:
            return RegressionReport(status=RegressionStatus.SKIPPED, has_baseline=False)

        findings: list[RegressionFinding] = []
        findings.extend(
            self._identity_findings(
                baseline_dataset_version=baseline_dataset_version,
                current_dataset_version=current_dataset_version,
                baseline_config_hash=baseline_config_hash,
                current_config_hash=current_config_hash,
            )
        )
        findings.extend(self._score_findings(current, baseline))
        findings.extend(self._latency_findings(current, baseline))
        findings.extend(self._token_findings(current, baseline))
        findings.extend(self._cost_findings(current, baseline))
        findings.extend(self._failure_findings(current, baseline))
        if self._config.detect_tool_behavior_change:
            findings.extend(self._tool_findings(current_cases or [], baseline_cases or []))

        score_delta = _delta(current.overall_score, baseline.overall_score)
        return RegressionReport(
            status=_status_for(findings, score_delta),
            has_baseline=True,
            baseline_name=baseline_name,
            baseline_run_id=baseline_run_id,
            findings=findings,
            score_delta=score_delta,
        )

    def _identity_findings(
        self,
        *,
        baseline_dataset_version: str | None,
        current_dataset_version: str | None,
        baseline_config_hash: str | None,
        current_config_hash: str | None,
    ) -> list[RegressionFinding]:
        """Report dataset/configuration mismatches that void a comparison."""
        findings: list[RegressionFinding] = []
        if (
            baseline_dataset_version is not None
            and current_dataset_version is not None
            and baseline_dataset_version != current_dataset_version
        ):
            findings.append(
                RegressionFinding(
                    kind=FindingKind.DATASET_CHANGED,
                    severity=RegressionSeverity.ERROR,
                    metric="dataset_version",
                    baseline_value=None,
                    current_value=None,
                    message=(
                        f"Dataset version changed from {baseline_dataset_version} to "
                        f"{current_dataset_version}; metrics are not comparable."
                    ),
                )
            )
        if (
            baseline_config_hash is not None
            and current_config_hash is not None
            and baseline_config_hash != current_config_hash
        ):
            findings.append(
                RegressionFinding(
                    kind=FindingKind.CONFIG_CHANGED,
                    severity=RegressionSeverity.ERROR,
                    metric="config_hash",
                    baseline_value=None,
                    current_value=None,
                    message=(
                        "Configuration changed since the baseline was captured; "
                        "score differences may be caused by the configuration itself."
                    ),
                )
            )
        return findings

    def _score_findings(
        self, current: RunAggregate, baseline: RunAggregate
    ) -> list[RegressionFinding]:
        """Detect overall score drops beyond the configured tolerance."""
        if current.overall_score is None or baseline.overall_score is None:
            return []
        delta = current.overall_score - baseline.overall_score
        if delta >= -self._config.max_score_drop:
            return []
        return [
            RegressionFinding(
                kind=FindingKind.SCORE_DROP,
                severity=RegressionSeverity.ERROR,
                metric="overall_score",
                baseline_value=baseline.overall_score,
                current_value=current.overall_score,
                delta=delta,
                threshold=-self._config.max_score_drop,
                message=(
                    f"Overall score dropped {abs(delta):.4f} "
                    f"(allowed {self._config.max_score_drop:.4f})."
                ),
            )
        ]

    def _latency_findings(
        self, current: RunAggregate, baseline: RunAggregate
    ) -> list[RegressionFinding]:
        """Detect latency regressions beyond a percentage tolerance."""
        limit = self._config.max_latency_increase_percent
        base = baseline.latency.average_ms
        now = current.latency.average_ms
        if base <= 0 or now <= 0:
            return []
        increase = (now - base) / base * 100.0
        if increase <= limit or (now - base) < MIN_LATENCY_DELTA_MS:
            return []
        return [
            RegressionFinding(
                kind=FindingKind.LATENCY_INCREASE,
                severity=RegressionSeverity.ERROR,
                metric="latency.average_ms",
                baseline_value=base,
                current_value=now,
                delta=increase,
                threshold=limit,
                message=(
                    f"Average latency rose {increase:.1f}% "
                    f"({base:.2f}ms -> {now:.2f}ms, allowed {limit:.1f}%)."
                ),
            )
        ]

    def _token_findings(
        self, current: RunAggregate, baseline: RunAggregate
    ) -> list[RegressionFinding]:
        """Detect token-usage regressions beyond a percentage tolerance."""
        limit = self._config.max_token_increase_percent
        base = baseline.tokens.total_tokens
        now = current.tokens.total_tokens
        if base <= 0 or now <= 0:
            return []
        increase = (now - base) / base * 100.0
        if increase <= limit:
            return []
        return [
            RegressionFinding(
                kind=FindingKind.TOKEN_INCREASE,
                severity=RegressionSeverity.ERROR,
                metric="tokens.total_tokens",
                baseline_value=float(base),
                current_value=float(now),
                delta=increase,
                threshold=limit,
                message=(
                    f"Token usage rose {increase:.1f}% "
                    f"({base} -> {now} tokens, allowed {limit:.1f}%)."
                ),
            )
        ]

    def _cost_findings(
        self, current: RunAggregate, baseline: RunAggregate
    ) -> list[RegressionFinding]:
        """Detect API cost regressions beyond a percentage tolerance."""
        limit = self._config.max_cost_increase_percent
        base = baseline.cost.api_cost_usd
        now = current.cost.api_cost_usd
        if base <= 0 or now <= 0:
            return []
        increase = (now - base) / base * 100.0
        if increase <= limit:
            return []
        return [
            RegressionFinding(
                kind=FindingKind.COST_INCREASE,
                severity=RegressionSeverity.ERROR,
                metric="cost.api_cost_usd",
                baseline_value=base,
                current_value=now,
                delta=increase,
                threshold=limit,
                message=(
                    f"API cost rose {increase:.1f}% "
                    f"(${base:.4f} -> ${now:.4f}, allowed {limit:.1f}%)."
                ),
            )
        ]

    def _failure_findings(
        self, current: RunAggregate, baseline: RunAggregate
    ) -> list[RegressionFinding]:
        """Detect newly failing and newly errored cases."""
        findings: list[RegressionFinding] = []
        new_failed = [c for c in current.failing_case_ids if c not in baseline.failing_case_ids]
        new_errored = [c for c in current.errored_case_ids if c not in baseline.errored_case_ids]
        new_ids = new_failed + new_errored
        if not new_ids:
            return findings
        if len(new_ids) > self._config.max_new_failures:
            baseline_total = len(baseline.failing_case_ids) + len(baseline.errored_case_ids)
            current_total = len(current.failing_case_ids) + len(current.errored_case_ids)
            findings.append(
                RegressionFinding(
                    kind=FindingKind.NEW_FAILURES,
                    severity=RegressionSeverity.ERROR,
                    metric="new_failures",
                    baseline_value=float(baseline_total),
                    current_value=float(current_total),
                    delta=float(len(new_ids)),
                    threshold=float(self._config.max_new_failures),
                    message=(
                        f"{len(new_ids)} newly failing case(s): {', '.join(new_ids)} "
                        f"(allowed {self._config.max_new_failures})."
                    ),
                    case_ids=new_ids,
                )
            )
        return findings

    def _tool_findings(
        self, current: list[CaseResult], baseline: list[CaseResult]
    ) -> list[RegressionFinding]:
        """Report cases whose tool-call sequence changed."""
        if not current or not baseline:
            return []
        base_map = {c.case_id: c for c in baseline}
        changed: list[str] = []
        for case in current:
            other = base_map.get(case.case_id)
            if other is None:
                continue
            if case.result.tool_names() != other.result.tool_names():
                changed.append(case.case_id)
        if not changed:
            return []
        return [
            RegressionFinding(
                kind=FindingKind.TOOL_BEHAVIOR_CHANGE,
                severity=RegressionSeverity.WARNING,
                metric="tool_calls",
                message=f"Tool behaviour changed for: {', '.join(changed)}.",
                case_ids=changed,
            )
        ]


def _delta(current: float | None, baseline: float | None) -> float | None:
    """Signed difference between two optional scores."""
    if current is None or baseline is None:
        return None
    return current - baseline


def _status_for(findings: list[RegressionFinding], score_delta: float | None) -> RegressionStatus:
    """Derive the overall verdict from findings and score movement."""
    if any(f.severity is RegressionSeverity.ERROR for f in findings):
        return RegressionStatus.FAILED
    if any(f.severity is RegressionSeverity.WARNING for f in findings):
        return RegressionStatus.PASSED
    if score_delta is not None and score_delta > 0:
        return RegressionStatus.IMPROVED
    return RegressionStatus.PASSED


def compare_records(
    config: RegressionConfig,
    current: RunRecord,
    baseline_run: RunRecord | None,
    *,
    baseline_name: str | None = None,
    current_cases: list[CaseResult] | None = None,
    baseline_cases: list[CaseResult] | None = None,
) -> RegressionReport:
    """Convenience wrapper comparing two :class:`RunRecord` objects."""
    return RegressionDetector(config).compare(
        current.aggregate,
        baseline_run.aggregate if baseline_run is not None else None,
        baseline_name=baseline_name,
        baseline_run_id=baseline_run.run_id if baseline_run is not None else None,
        current_cases=current_cases,
        baseline_cases=baseline_cases,
        baseline_dataset_version=baseline_run.dataset_version if baseline_run else None,
        current_dataset_version=current.dataset_version,
        baseline_config_hash=baseline_run.config_hash if baseline_run else None,
        current_config_hash=current.config_hash,
    )
