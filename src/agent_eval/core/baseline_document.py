"""Portable baseline documents.

A CI runner starts with an empty database, so a baseline stored only in SQLite
cannot be compared against anything. A baseline *document* is the checked-in
form of a baseline: one JSON file, produced by a real run and committed
alongside the code it describes.

The document carries the aggregate plus the comparison identity (dataset
version and configuration hash). It deliberately does **not** carry per-case
tool trajectories: a baseline is a set of metrics, and committing full
trajectories would bloat the repository with data the detector only uses for a
warning-level check.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_eval.domain.models import Baseline, ExecutionMode, RunAggregate, RunRecord

#: Bumped when the document shape changes incompatibly.
SCHEMA_VERSION = 1


class BaselineDocument(BaseModel):
    """A baseline snapshot written to disk.

    This is the interchange format for baselines that must outlive a single
    machine's database.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = SCHEMA_VERSION
    name: str
    dataset_name: str
    dataset_version: str
    agent_version: str | None = None
    execution_mode: ExecutionMode = ExecutionMode.MOCK
    config_hash: str | None = None
    created_at: str = Field(description="ISO-8601 timestamp of the snapshot.")
    aggregate: RunAggregate


def document_from_baseline(baseline: Baseline) -> BaselineDocument:
    """Convert a stored baseline into a portable document."""
    return BaselineDocument(
        name=baseline.name,
        dataset_name=baseline.dataset_name,
        dataset_version=baseline.dataset_version,
        agent_version=baseline.agent_version,
        execution_mode=baseline.execution_mode,
        config_hash=baseline.config_hash,
        created_at=baseline.created_at.isoformat(),
        aggregate=baseline.aggregate,
    )


def document_from_run(
    run: RunRecord, name: str, *, created_at: str | None = None
) -> BaselineDocument:
    """Snapshot a run directly into a portable document.

    Args:
        run: The completed run to snapshot.
        name: Name to record in the document.
        created_at: ISO-8601 timestamp; defaults to the run's start time.

    Returns:
        The document describing ``run``.
    """
    return BaselineDocument(
        name=name,
        dataset_name=run.dataset_name,
        dataset_version=run.dataset_version,
        agent_version=run.agent.version,
        execution_mode=run.execution_mode,
        config_hash=run.config_hash,
        created_at=created_at or run.started_at.isoformat(),
        aggregate=run.aggregate,
    )


def write_document(path: Path, document: BaselineDocument) -> Path:
    """Write a baseline document to ``path``, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(document.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def read_document(path: Path) -> BaselineDocument:
    """Read and validate a baseline document.

    Args:
        path: The JSON file to read.

    Returns:
        The parsed :class:`BaselineDocument`.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or fails validation.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Baseline document not found: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Baseline document {source} is not valid JSON: {exc}") from exc
    try:
        return BaselineDocument.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Baseline document {source} is invalid: {exc}") from exc


def document_to_baseline(document: BaselineDocument) -> Baseline:
    """Convert a portable document back into a storable baseline."""
    return Baseline(
        name=document.name,
        created_at=datetime.fromisoformat(document.created_at),
        dataset_name=document.dataset_name,
        dataset_version=document.dataset_version,
        agent_version=document.agent_version,
        config_hash=document.config_hash,
        execution_mode=document.execution_mode,
        aggregate=document.aggregate,
    )
