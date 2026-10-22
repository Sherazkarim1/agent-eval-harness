"""Storage adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_eval.storage.memory import InMemoryRepository
from agent_eval.storage.sqlite import SqliteEvaluationRepository

if TYPE_CHECKING:
    from agent_eval.config import StorageConfig
    from agent_eval.ports.repository import EvaluationRepository

__all__ = [
    "InMemoryRepository",
    "SqliteEvaluationRepository",
    "create_repository",
]


def create_repository(config: StorageConfig) -> EvaluationRepository:
    """Build the repository described by a storage configuration.

    Args:
        config: The ``storage`` section of the harness configuration.

    Returns:
        A ready-to-use repository with its schema already initialised.

    Raises:
        ValueError: If ``config.type`` is not a known storage backend.
    """
    repository: EvaluationRepository
    if config.type == "sqlite":
        repository = SqliteEvaluationRepository(config.path)
    elif config.type == "memory":
        repository = InMemoryRepository()
    else:  # pragma: no cover - blocked by config validation
        raise ValueError(f"Unknown storage type: {config.type}")
    repository.initialize()
    return repository
