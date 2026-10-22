"""Configuration models, loading and hashing.

Configuration is validated with Pydantic and ``extra="forbid"``, so a typo in
``agent-eval.yaml`` fails loudly instead of silently disabling a feature.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG_FILENAMES: tuple[str, ...] = (
    "agent-eval.yaml",
    "agent-eval.yml",
    ".agent-eval.yaml",
    ".agent-eval.yml",
)


def _default_fail_on() -> list[Literal["error", "warning"]]:
    """Default regression severities that trigger a non-zero exit code."""
    return ["error"]


def _default_report_formats() -> list[Literal["json", "markdown", "html"]]:
    """Report formats written by default (HTML is opt-in)."""
    return ["json", "markdown"]


def _default_criteria() -> list[str]:
    """Judge criteria scored by default."""
    return ["correctness", "relevance", "completeness"]


class ProjectConfig(BaseModel):
    """Project identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "agent-eval-project"
    version: str = "0.1.0"


class DatasetConfig(BaseModel):
    """Which dataset to evaluate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path = Path("evals/dataset.yaml")


class AgentConfig(BaseModel):
    """Which agent to evaluate and how to construct it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str = "mock"
    name: str | None = None
    version: str = "0.1.0"
    factory: str | None = Field(
        default=None,
        description="Import path 'module:callable' returning an AgentAdapter (or a coroutine "
        "returning one). Used for custom Python agents.",
    )
    options: dict[str, Any] = Field(default_factory=dict)


class JudgeConfig(BaseModel):
    """Judge selection and prompt settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    provider: Literal["mock", "ollama"] = "mock"
    model: str = "mock-judge"
    base_url: str = "http://localhost:11434"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)
    criteria: list[str] = Field(default_factory=_default_criteria)
    prompt_template: str | None = Field(
        default=None, description="Optional custom prompt template with {placeholders}."
    )
    only_with_reference: bool = Field(
        default=True,
        description="Skip judging for cases that declare no reference_answer.",
    )


class ScoringWeights(BaseModel):
    """Relative weight of each score component.

    Weights are normalised over the components that were actually computed, so
    a disabled judge does not silently deflate the overall score.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    deterministic: float = Field(default=0.5, ge=0.0)
    judge: float = Field(default=0.3, ge=0.0)
    trajectory: float = Field(default=0.0, ge=0.0)
    performance: float = Field(default=0.2, ge=0.0)

    @model_validator(mode="after")
    def _at_least_one_weight(self) -> ScoringWeights:
        if self.deterministic + self.judge + self.trajectory + self.performance <= 0:
            raise ValueError("at least one scoring weight must be greater than zero")
        return self

    def normalised(self, present: set[str]) -> dict[str, float]:
        """Return weights restricted to ``present`` components, summing to 1.0."""
        selected = {name: getattr(self, name) for name in present}
        total = sum(selected.values())
        if total <= 0:
            return dict.fromkeys(selected, 0.0)
        return {name: value / total for name, value in selected.items()}


class ScoringConfig(BaseModel):
    """Scoring behaviour."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    min_overall_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Run fails when the overall score is below this value.",
    )
    pass_on_error: bool = Field(
        default=False, description="Whether an errored case counts as a failure (not a pass)."
    )


class RegressionConfig(BaseModel):
    """Thresholds for regression detection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    max_score_drop: float = Field(default=0.05, ge=0.0, le=1.0)
    max_latency_increase_percent: float = Field(default=20.0, ge=0.0)
    max_token_increase_percent: float = Field(default=30.0, ge=0.0)
    max_cost_increase_percent: float = Field(default=20.0, ge=0.0)
    max_new_failures: int = Field(default=0, ge=0)
    detect_tool_behavior_change: bool = True
    fail_on: list[Literal["error", "warning"]] = Field(default_factory=_default_fail_on)
    baseline: str | None = Field(
        default=None, description="Name of the baseline to compare against."
    )


class StorageConfig(BaseModel):
    """Run storage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["sqlite", "memory"] = "sqlite"
    path: Path = Path(".agent_eval/results.db")


class PricingConfig(BaseModel):
    """Per-model API pricing, used for cost estimates.

    Local models are configured with zero pricing, which is why local runs
    report ``$0.00`` *API* cost. Infrastructure cost is not captured.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    currency: str = "USD"
    models: dict[str, PricingEntry] = Field(default_factory=dict)


class PricingEntry(BaseModel):
    """Price per one million tokens for a single model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_per_million: float = Field(default=0.0, ge=0.0)
    output_per_million: float = Field(default=0.0, ge=0.0)
    currency: str = "USD"


class ReportConfig(BaseModel):
    """Report generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    output_dir: Path = Path("reports")
    formats: list[Literal["json", "markdown", "html"]] = Field(
        default_factory=_default_report_formats
    )
    include_trajectories: bool = True
    max_failures_in_report: int = Field(default=50, ge=1)


class RunConfig(BaseModel):
    """Run-level execution settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["mock", "live"] = "mock"
    timeout_s: float = Field(default=60.0, gt=0)
    max_concurrency: int = Field(default=1, ge=1, le=64)
    stop_on_first_failure: bool = False
    tags: list[str] = Field(default_factory=list)
    case_ids: list[str] = Field(default_factory=list)
    seed: int | None = None


class HarnessConfig(BaseModel):
    """Top-level configuration object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    regression: RegressionConfig = Field(default_factory=RegressionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    def fingerprint(self) -> str:
        """Return a stable hash of the behaviour-affecting configuration.

        Paths are normalised to their string form and absolute locations are
        excluded, so the same config produces the same hash on any machine.
        """
        payload = self.model_dump(mode="json", exclude={"report", "run"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]


def default_config() -> HarnessConfig:
    """Return the built-in default configuration."""
    return HarnessConfig()


def find_config_file(start: Path | None = None) -> Path | None:
    """Search ``start`` and its parents for a configuration file."""
    current = (start or Path.cwd()).resolve()
    for directory in [current, *current.parents]:
        for filename in DEFAULT_CONFIG_FILENAMES:
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    return None


def load_config(path: Path | str | None = None, *, required: bool = False) -> HarnessConfig:
    """Load configuration from a YAML file.

    Args:
        path: Explicit config path, or ``None`` to auto-discover.
        required: Raise if no file is found instead of returning defaults.

    Returns:
        The validated configuration. When no file exists and ``required`` is
        ``False``, built-in defaults are returned.

    Raises:
        FileNotFoundError: If no config file is found and ``required=True``.
        ValueError: If the file is not a mapping or fails validation.
    """
    config_path = Path(path) if path is not None else find_config_file()
    if config_path is None:
        if required:
            raise FileNotFoundError(
                "No configuration file found. Run `agent-eval init` to create one, "
                f"or create one of: {', '.join(DEFAULT_CONFIG_FILENAMES)}."
            )
        return default_config()

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        return default_config()
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration file must contain a YAML mapping: {config_path}")
    return HarnessConfig.model_validate(raw)
