"""Shared CLI context: configuration, storage, dataset and component building."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import typer

from agent_eval.config import HarnessConfig, load_config
from agent_eval.core.runner import dataset_path_for
from agent_eval.datasets import DatasetError, load_dataset_file
from agent_eval.domain.models import Dataset
from agent_eval.ports.judge import Judge
from agent_eval.ports.repository import EvaluationRepository
from agent_eval.storage import create_repository


@dataclass
class Context:
    """Everything a command needs, resolved once from flags and config."""

    config: HarnessConfig
    config_path: Path | None
    root: Path
    project_root: Path | None = None

    def dataset_path(self) -> Path:
        """Absolute path of the configured dataset file."""
        return dataset_path_for(self.config, self.project_root)

    def storage(self) -> EvaluationRepository:
        """Open the configured storage backend."""
        return create_repository(self.config.storage)

    def load_dataset(self) -> Dataset:
        """Load and validate the configured dataset.

        Raises:
            DatasetError: If the file is missing or invalid.
        """
        path = self.dataset_path()
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return load_dataset_file(path)


def build_context(
    config_path: Path | None = None,
    *,
    project_root: Path | None = None,
    root: Path | None = None,
) -> Context:
    """Resolve configuration for a command.

    Args:
        config_path: Explicit ``--config`` path, or ``None`` to auto-discover.
        project_root: Directory that relative dataset paths resolve against.
        root: Directory to resolve relative paths from; defaults to the cwd.

    Returns:
        A populated :class:`Context`.
    """
    base = root or Path.cwd()
    if config_path is not None:
        resolved = config_path if config_path.is_absolute() else base / config_path
        config = load_config(resolved)
    else:
        config = load_config(None)
        resolved = None
    return Context(
        config=config,
        config_path=resolved,
        root=base,
        project_root=project_root or base,
    )


def resolve_judge(config: HarnessConfig, *, force_mock: bool | None = None) -> Judge | None:
    """Build the judge described by configuration.

    ``force_mock=True`` forces the offline heuristic judge, which is what
    ``--mock`` needs: the agent may be simulated while judging stays offline.
    ``force_mock=False`` simply follows configuration, because the configured
    provider may legitimately be the mock one.

    Args:
        config: Active configuration.
        force_mock: ``True`` forces the offline judge, ``False`` or ``None``
            follows configuration.

    Returns:
        A judge instance, or ``None`` when judging is disabled.

    Raises:
        ValueError: If the configured provider is unknown or unavailable.
    """
    if not config.judge.enabled:
        return None

    provider = "mock" if force_mock else config.judge.provider

    if provider == "mock":
        from agent_eval.judges.mock import HeuristicJudge

        return HeuristicJudge(criteria=config.judge.criteria)

    if provider == "ollama":
        try:
            from agent_eval.judges.ollama import OllamaJudge
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ValueError(
                "The ollama judge needs the 'ollama' extra: pip install agent-eval[ollama]"
            ) from exc
        return cast(
            "Judge",
            OllamaJudge(
                model=config.judge.model,
                base_url=config.judge.base_url,
                criteria=config.judge.criteria,
                timeout_s=config.judge.timeout_s,
                max_retries=config.judge.max_retries,
                prompt_template=config.judge.prompt_template,
            ),
        )
    raise ValueError(f"Unknown judge provider: {provider}")


def resolve_agent(config: HarnessConfig) -> Any:
    """Build the agent adapter described by configuration.

    Returns:
        An object satisfying :class:`~agent_eval.ports.agent.AgentAdapter`.

    Raises:
        ValueError: If the configured agent type or factory cannot be used.
    """
    from agent_eval.integrations.registry import create_agent

    return create_agent(config.agent)


def handle_dataset_error(exc: Exception) -> None:
    """Turn a dataset failure into a clean CLI exit."""
    if isinstance(exc, DatasetError):
        typer.echo(str(exc), err=True)
    else:
        typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(2) from exc
