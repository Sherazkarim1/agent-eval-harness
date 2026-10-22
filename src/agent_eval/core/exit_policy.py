"""Exit-code policy.

The single place that decides what a command's exit code means, so that
``agent-eval run`` behaves predictably in CI:

===== ===========================================================
Code  Meaning
===== ===========================================================
0     Success; the run met every configured threshold.
1     Evaluation failure: failed/errored cases, or a regression
      whose severity is listed in ``regression.fail_on``.
2     Usage or environment error (bad config, missing dataset).
3     Unexpected internal error.
===== ===========================================================
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_eval.config import HarnessConfig
from agent_eval.domain.models import (
    RegressionReport,
    RegressionSeverity,
    RegressionStatus,
    RunAggregate,
)

EXIT_OK = 0
EXIT_EVALUATION_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERNAL = 3


@dataclass(frozen=True)
class ExitDecision:
    """The outcome of a command, ready to hand to the shell."""

    code: int
    reason: str = ""

    @property
    def ok(self) -> bool:
        """Whether the command succeeded."""
        return self.code == EXIT_OK


def decide_run_exit(
    config: HarnessConfig,
    aggregate: RunAggregate,
    regression: RegressionReport | None,
) -> ExitDecision:
    """Decide the exit code for ``agent-eval run``.

    A run fails when any case failed or errored, when the overall score is below
    ``scoring.min_overall_score``, or when regression detection produced a
    finding whose severity appears in ``regression.fail_on``.

    Args:
        config: Active configuration.
        aggregate: Aggregate of the run that just completed.
        regression: Regression report, or ``None`` when not compared.

    Returns:
        The :class:`ExitDecision` for this run.
    """
    reasons: list[str] = []

    if aggregate.total == 0:
        return ExitDecision(EXIT_USAGE, "No cases were executed.")

    if aggregate.failed or aggregate.errored:
        reasons.append(f"{aggregate.failed} failed and {aggregate.errored} errored case(s).")

    minimum = config.scoring.min_overall_score
    if (
        minimum is not None
        and aggregate.overall_score is not None
        and aggregate.overall_score < minimum
    ):
        reasons.append(
            f"Overall score {aggregate.overall_score:.4f} is below the minimum {minimum:.4f}."
        )

    if regression is not None:
        fail_on = set(config.regression.fail_on)
        blocking = [f for f in regression.findings if f.severity.value in fail_on]
        if blocking:
            severities = sorted({f.severity.value for f in blocking})
            reasons.append(f"{len(blocking)} regression finding(s) at severity {severities}.")

    if reasons:
        return ExitDecision(EXIT_EVALUATION_FAILED, " ".join(reasons))
    return ExitDecision(EXIT_OK)


def describe_regression(regression: RegressionReport) -> str:
    """One-line human summary of a regression report."""
    if regression.status is RegressionStatus.SKIPPED:
        return "No baseline to compare against; regression check skipped."
    if not regression.findings:
        return f"No differences from baseline {regression.baseline_name!r}."
    kinds = ", ".join(sorted({f.kind.value for f in regression.findings}))
    return (
        f"{len(regression.findings)} finding(s) vs baseline {regression.baseline_name!r}: {kinds}."
    )


def worst_severity(regression: RegressionReport) -> RegressionSeverity | None:
    """Highest severity present in a report, or ``None`` when there are none."""
    if any(f.severity is RegressionSeverity.ERROR for f in regression.findings):
        return RegressionSeverity.ERROR
    if any(f.severity is RegressionSeverity.WARNING for f in regression.findings):
        return RegressionSeverity.WARNING
    if regression.findings:
        return RegressionSeverity.INFO
    return None
