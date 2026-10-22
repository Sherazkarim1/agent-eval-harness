"""Baseline capture and resolution.

A baseline is an immutable snapshot of a previous run's aggregate metrics.
Comparison targets are resolved in a fixed order so a comparison is always
explainable:

1. an explicitly named baseline,
2. the configured default baseline name,
3. the latest run of the same ``dataset@version`` **and** the same
   ``config_hash``.

Rule 3 deliberately requires a matching configuration hash. Comparing a run
against one produced under a different configuration produces meaningless
differences, so that case is reported rather than silently scored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agent_eval.domain.models import Baseline, ExecutionMode, RunAggregate, RunRecord
from agent_eval.ports.repository import EvaluationRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineSource:
    """Where a comparison baseline came from.

    Attributes:
        kind: Either ``"named"``, ``"configured"`` or ``"run"``.
        aggregate: Metrics to compare against, or ``None`` when nothing matched.
        baseline: The stored baseline, when one was used by name.
        run: The run a snapshot was taken from, when no stored baseline matched.
    """

    kind: str
    aggregate: RunAggregate | None = None
    baseline: Baseline | None = None
    run: RunRecord | None = None

    @property
    def found(self) -> bool:
        """Whether a usable comparison target was located."""
        return self.aggregate is not None

    @property
    def name(self) -> str | None:
        """Human-readable identifier for the resolved baseline."""
        if self.baseline is not None:
            return self.baseline.name
        if self.run is not None:
            return f"run:{self.run.run_id}"
        return None


def snapshot_run(run: RunRecord, name: str) -> Baseline:
    """Capture an immutable baseline snapshot from a completed run.

    Args:
        run: The run to snapshot.
        name: Name to store the baseline under.

    Returns:
        A frozen :class:`Baseline` carrying a copy of the run's aggregate.
    """
    return Baseline(
        name=name,
        run_id=run.run_id,
        dataset_name=run.dataset_name,
        dataset_version=run.dataset_version,
        agent_version=run.agent.version,
        config_hash=run.config_hash,
        execution_mode=run.execution_mode,
        aggregate=run.aggregate,
    )


def save_baseline_from_run(repository: EvaluationRepository, run: RunRecord, name: str) -> Baseline:
    """Snapshot a run and persist it as a named baseline.

    Args:
        repository: Where to store the baseline.
        run: The run to snapshot.
        name: Name to store the baseline under.

    Returns:
        The stored baseline.
    """
    baseline = snapshot_run(run, name)
    repository.save_baseline(baseline)
    logger.info("Saved baseline %r from run %s", name, run.run_id)
    return baseline


def resolve_baseline(
    repository: EvaluationRepository,
    dataset_name: str,
    dataset_version: str,
    *,
    config_hash: str | None = None,
    name: str | None = None,
    configured_name: str | None = None,
    execution_mode: ExecutionMode | None = None,
) -> BaselineSource:
    """Locate the baseline to compare a run against.

    Args:
        repository: Repository to search.
        dataset_name: Name of the dataset being evaluated.
        dataset_version: Version of the dataset being evaluated.
        config_hash: Configuration hash of the current run.
        name: Explicitly requested baseline name.
        configured_name: Default baseline name from configuration.
        execution_mode: Restrict automatic lookup to a single execution mode.

    Returns:
        A :class:`BaselineSource` describing what was found. ``found`` is
        ``False`` when no baseline exists yet, which is a normal first-run state
        rather than an error.
    """
    for requested, kind in ((name, "named"), (configured_name, "configured")):
        if not requested:
            continue
        baseline = repository.get_baseline(requested)
        if baseline is not None:
            logger.debug("Resolved %s baseline %r", kind, requested)
            return BaselineSource(kind=kind, aggregate=baseline.aggregate, baseline=baseline)
        logger.warning("%s baseline %r not found in storage; falling back", kind, requested)

    run = repository.get_latest_run(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        config_hash=config_hash,
        execution_mode=execution_mode.value if execution_mode else None,
    )
    if run is None:
        return BaselineSource(kind="run")
    return BaselineSource(kind="run", aggregate=run.aggregate, run=run)
