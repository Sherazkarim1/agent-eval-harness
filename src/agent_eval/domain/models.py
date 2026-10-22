"""Core domain models for the agent evaluation harness.

All models are Pydantic v2 models. Anything that must not change after
construction is declared ``frozen=True`` so downstream code can rely on
immutability (datasets, expectations, assertions).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------------------
# Tokens, cost and timing
# --------------------------------------------------------------------------------------


class TokenUsage(BaseModel):
    """Token accounting for a single agent invocation."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        """Total tokens (cached inputs included)."""
        return self.input_tokens + self.output_tokens + self.cached_input_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Sum two usage records (used by the aggregator)."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
        )


class LatencyStats(BaseModel):
    """Aggregated latency distribution for a run."""

    model_config = ConfigDict(frozen=True)

    count: int = Field(default=0, ge=0)
    average_ms: float = Field(default=0.0, ge=0.0)
    p50_ms: float = Field(default=0.0, ge=0.0)
    p95_ms: float = Field(default=0.0, ge=0.0)
    max_ms: float = Field(default=0.0, ge=0.0)

    @property
    def average_s(self) -> float:
        """Average latency expressed in seconds (for human-readable reports)."""
        return self.average_ms / 1000.0


class CostStats(BaseModel):
    """Estimated API cost for a run.

    ``api_cost_usd`` is the only cost reported. Local inference reports
    ``0.0`` API cost; the compute/electricity cost of local hardware is real
    but out of scope and explicitly documented as such.
    """

    model_config = ConfigDict(frozen=True)

    api_cost_usd: float = Field(default=0.0, ge=0.0)
    tokens: int = Field(default=0, ge=0)
    note: str = (
        "API cost only. Local inference has non-zero compute cost that this tool does not measure."
    )


# --------------------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------------------


class StepType(StrEnum):
    """Kind of a single step in an agent trajectory."""

    LLM = "llm"
    TOOL = "tool"
    STATE = "state"
    ERROR = "error"
    FINAL = "final"


class Step(BaseModel):
    """One recorded step of an agent execution."""

    model_config = ConfigDict(frozen=True)

    type: StepType
    name: str | None = None
    input: Any | None = None
    output: Any | None = None
    tokens: TokenUsage | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())
    duration_ms: float | None = Field(default=None, ge=0.0)

    @property
    def label(self) -> str:
        """Identifier used by ``required_step`` / ``forbidden_step`` assertions."""
        if self.type is StepType.TOOL and self.name:
            return f"tool:{self.name}"
        return self.type.value


class ToolCall(BaseModel):
    """A single tool invocation performed by the agent."""

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    error: str | None = None
    step_index: int = Field(default=0, ge=0)
    duration_ms: float | None = Field(default=None, ge=0.0)

    @property
    def succeeded(self) -> bool:
        """Whether the tool call completed without an error."""
        return self.error is None


# --------------------------------------------------------------------------------------
# Agent output
# --------------------------------------------------------------------------------------


class AgentResult(BaseModel):
    """Normalised result returned by every agent adapter."""

    model_config = ConfigDict(frozen=True)

    output: str
    steps: list[Step] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    token_usage: TokenUsage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        """``True`` when the agent completed without reporting an error."""
        return self.error is None

    def tool_names(self) -> list[str]:
        """Ordered, de-duplicated list of tool names called by the agent."""
        names: dict[str, None] = {}
        for call in self.tool_calls:
            names.setdefault(call.name, None)
        return list(names)

    def step_labels(self) -> list[str]:
        """Ordered list of step labels (see :attr:`Step.label`)."""
        return [step.label for step in self.steps]

    def total_tool_duration_ms(self) -> float:
        """Sum of all recorded tool call durations."""
        return sum(call.duration_ms for call in self.tool_calls if call.duration_ms is not None)


class AgentVersion(BaseModel):
    """Identifies the evaluated agent implementation."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str = "0.1.0"
    framework: str = "custom"
    extra: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Expectations (deterministic assertions)
# --------------------------------------------------------------------------------------


class Priority(StrEnum):
    """Relative importance of an evaluation case."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Expectation(BaseModel):
    """Declarative expectations for a single evaluation case.

    Every field maps to exactly one assertion kind. All fields are optional;
    an empty expectation produces no assertions and the case passes as long as
    the agent did not error.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- text ---
    contains: list[str] | None = None
    not_contains: list[str] | None = None
    exact_match: str | None = None
    regex: list[str] | None = None
    starts_with: str | None = None
    ends_with: str | None = None

    # --- structure ---
    valid_json: bool = False
    required_fields: list[str] | None = None
    json_schema: dict[str, Any] | None = None

    # --- tool behaviour ---
    tool_called: list[str] | None = None
    tool_not_called: list[str] | None = None
    tool_call_count: int | dict[str, int] | None = None
    tool_args: dict[str, dict[str, Any]] | None = None
    expected_tool_order: list[str] | None = None

    # --- performance ---
    max_latency_ms: float | None = Field(default=None, gt=0)
    max_tool_latency_ms: dict[str, float] | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    max_cost_usd: float | None = Field(default=None, ge=0)

    # --- trajectory ---
    max_steps: int | None = Field(default=None, ge=0)
    required_step: list[str] | None = None
    forbidden_step: list[str] | None = None

    # --- failure handling ---
    # Lets a dataset assert how the agent behaves when it fails. Without this,
    # a resilience case is unassertable: an errored case always counts as a
    # failure, so there is no way to say "erroring here is correct".
    expect_error: bool | None = None

    @field_validator(
        "contains",
        "not_contains",
        "regex",
        "required_fields",
        "tool_called",
        "tool_not_called",
        "expected_tool_order",
        "required_step",
        "forbidden_step",
        mode="after",
    )
    @classmethod
    def _reject_empty_lists(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(value) == 0:
            raise ValueError("must not be an empty list")
        return value

    def is_empty(self) -> bool:
        """Whether the expectation declares no assertions at all."""
        return not self.model_dump(exclude_none=True, exclude_defaults=True)


class EvalCase(BaseModel):
    """A single evaluation case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.\-]+$")
    input: str = Field(min_length=1)
    reference_answer: str | None = None
    expected: Expectation = Field(default_factory=Expectation)
    tags: list[str] = Field(default_factory=list)
    priority: Priority = Priority.MEDIUM
    metadata: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float | None = Field(default=None, gt=0)

    @field_validator("id", mode="before")
    @classmethod
    def _stringify_id(cls, value: Any) -> Any:
        return str(value) if isinstance(value, bool | int | float) else value


class Dataset(BaseModel):
    """A versioned collection of evaluation cases."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(pattern=r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-]+)?$")
    cases: list[EvalCase] = Field(min_length=1)
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def case_ids(self) -> list[str]:
        """Ordered list of case identifiers."""
        return [case.id for case in self.cases]

    def by_tag(self, tag: str) -> list[EvalCase]:
        """All cases carrying ``tag``."""
        return [case for case in self.cases if tag in case.tags]

    def select(self, *, ids: set[str] | None = None, tags: list[str] | None = None) -> Dataset:
        """Return a new dataset restricted to the given ids and/or tags."""
        cases = self.cases
        if ids is not None:
            cases = [case for case in cases if case.id in ids]
        if tags:
            cases = [case for case in cases if any(tag in case.tags for tag in tags)]
        return self.model_copy(update={"cases": cases})


class DatasetIssue(BaseModel):
    """A problem detected while loading or validating a dataset file."""

    model_config = ConfigDict(frozen=True)

    case_id: str | None = None
    location: str
    message: str


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------


class AssertionKind(StrEnum):
    """Identifier of an assertion type."""

    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    EXACT_MATCH = "exact_match"
    REGEX_MATCH = "regex_match"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    VALID_JSON = "valid_json"
    REQUIRED_FIELDS = "required_fields"
    SCHEMA_VALIDATION = "schema_validation"
    TOOL_CALLED = "tool_called"
    TOOL_NOT_CALLED = "tool_not_called"
    TOOL_CALL_COUNT = "tool_call_count"
    TOOL_ARGS = "tool_args"
    EXPECTED_TOOL_ORDER = "expected_tool_order"
    MAX_LATENCY_MS = "max_latency_ms"
    MAX_TOOL_LATENCY_MS = "max_tool_latency_ms"
    MAX_TOKENS = "max_tokens"
    MAX_COST = "max_cost"
    MAX_STEPS = "max_steps"
    REQUIRED_STEP = "required_step"
    FORBIDDEN_STEP = "forbidden_step"
    EXPECT_ERROR = "expect_error"
    UNKNOWN = "unknown"


class CaseStatus(StrEnum):
    """Outcome of a single case."""

    PASSED = "passed"
    FAILED = "failed"
    ERRORED = "errored"


class AssertionResult(BaseModel):
    """Structured result of one assertion."""

    model_config = ConfigDict(frozen=True)

    assertion: AssertionKind
    passed: bool
    expected: Any | None = None
    actual: Any | None = None
    message: str = ""

    @property
    def failed(self) -> bool:
        """``True`` when the assertion did not pass."""
        return not self.passed


class ScoreBreakdown(BaseModel):
    """Weighted score components for a single case.

    ``overall`` is ``None`` when no component could be computed, which keeps
    "not measured" distinguishable from "scored zero".
    """

    model_config = ConfigDict(frozen=True)

    deterministic: float | None = None
    judge: float | None = None
    trajectory: float | None = None
    performance: float | None = None
    overall: float | None = None
    weights: dict[str, float] = Field(default_factory=dict)


class JudgeVerdict(BaseModel):
    """Structured output of the LLM judge.

    A judge implementation must always return a verdict object: on failure it
    returns ``score=None`` plus ``error``. This is what guarantees that free-form
    model output can never break the evaluation pipeline.
    """

    model_config = ConfigDict(frozen=True)

    score: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = None
    criteria: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    provider: str | None = None
    model: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)

    @property
    def ok(self) -> bool:
        """Whether the judge produced a usable score."""
        return self.score is not None and self.error is None


class JudgeVersion(BaseModel):
    """Identifies the judge used for a run."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str = "0.1.0"
    model: str | None = None
    provider: str | None = None


class ExecutionMode(StrEnum):
    """How a run was executed.

    Mock runs are never real measurements; reports must surface this so a mock
    score can never be mistaken for a real one.
    """

    MOCK = "mock"
    LIVE = "live"


class CaseResult(BaseModel):
    """Everything the harness knows about one executed case."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    case_id: str
    case: EvalCase
    result: AgentResult
    assertions: list[AssertionResult] = Field(default_factory=list)
    score: ScoreBreakdown | None = None
    judge: JudgeVerdict | None = None
    status: CaseStatus = CaseStatus.PASSED
    error: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    started_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())

    @property
    def passed(self) -> bool:
        """``True`` when the case status is ``passed``."""
        return self.status is CaseStatus.PASSED

    @property
    def failed_assertions(self) -> list[AssertionResult]:
        """All assertions that failed for this case."""
        return [a for a in self.assertions if a.failed]

    @property
    def score_value(self) -> float | None:
        """Overall case score, or ``None`` when nothing was scorable."""
        return self.score.overall if self.score is not None else None


class RunAggregate(BaseModel):
    """Aggregated metrics for a full run."""

    model_config = ConfigDict(frozen=True)

    total: int = Field(default=0, ge=0)
    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    errored: int = Field(default=0, ge=0)

    deterministic_score: float | None = None
    judge_score: float | None = None
    trajectory_score: float | None = None
    performance_score: float | None = None
    overall_score: float | None = None

    latency: LatencyStats = Field(default_factory=LatencyStats)
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    cost: CostStats = Field(default_factory=CostStats)

    failing_case_ids: list[str] = Field(default_factory=list)
    errored_case_ids: list[str] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Fraction of non-errored cases that passed (0.0-1.0)."""
        scored = self.passed + self.failed
        return self.passed / scored if scored else 0.0

    @property
    def has_failures(self) -> bool:
        """Whether the run contains failures or errors."""
        return bool(self.failed or self.errored)


class RunRecord(BaseModel):
    """A persisted evaluation run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    dataset_name: str
    dataset_version: str
    agent: AgentVersion
    judge: JudgeVersion
    aggregate: RunAggregate
    execution_mode: ExecutionMode = ExecutionMode.MOCK
    started_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())
    git_commit: str | None = None
    branch: str | None = None
    config_hash: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)

    @property
    def dataset_ref(self) -> str:
        """``name@version`` identifier for the dataset used by this run."""
        return f"{self.dataset_name}@{self.dataset_version}"


class Baseline(BaseModel):
    """An immutable snapshot of a previous run, used for regression checks."""

    model_config = ConfigDict(frozen=True)

    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())
    run_id: str | None = None
    dataset_name: str
    dataset_version: str
    agent_version: str | None = None
    config_hash: str | None = None
    execution_mode: ExecutionMode = ExecutionMode.MOCK
    aggregate: RunAggregate

    @property
    def dataset_ref(self) -> str:
        """``name@version`` identifier for the dataset behind this baseline."""
        return f"{self.dataset_name}@{self.dataset_version}"

    def snapshot(self) -> dict[str, Any]:
        """Serialise the comparable subset of the baseline."""
        return {
            "name": self.name,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "agent_version": self.agent_version,
            "config_hash": self.config_hash,
            "execution_mode": self.execution_mode.value,
            "aggregate": self.aggregate.model_dump(mode="json"),
        }


class LeaderboardRow(BaseModel):
    """One row of the leaderboard, built from real stored runs only."""

    model_config = ConfigDict(frozen=True)

    agent_name: str
    agent_version: str
    dataset_name: str
    dataset_version: str
    overall_score: float | None
    pass_rate: float
    avg_latency_ms: float
    total_cases: int
    run_id: str
    run_timestamp: datetime = Field(default_factory=lambda: datetime.now().astimezone())
    execution_mode: ExecutionMode = ExecutionMode.MOCK

    @property
    def accuracy(self) -> float:
        """Pass rate expressed as a percentage."""
        return self.pass_rate * 100.0


class RegressionSeverity(StrEnum):
    """How serious a regression finding is."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class FindingKind(StrEnum):
    """Category of a regression finding."""

    SCORE_DROP = "score_drop"
    LATENCY_INCREASE = "latency_increase"
    NEW_FAILURES = "new_failures"
    TOKEN_INCREASE = "token_increase"
    COST_INCREASE = "cost_increase"
    TOOL_BEHAVIOR_CHANGE = "tool_behavior_change"
    DATASET_CHANGED = "dataset_changed"
    CONFIG_CHANGED = "config_changed"


class RegressionFinding(BaseModel):
    """A single difference detected between a baseline and the current run."""

    model_config = ConfigDict(frozen=True)

    kind: FindingKind
    severity: RegressionSeverity
    metric: str
    baseline_value: float | None = None
    current_value: float | None = None
    delta: float | None = None
    threshold: float | None = None
    message: str = ""
    case_ids: list[str] = Field(default_factory=list)


class RegressionStatus(StrEnum):
    """Overall verdict of a regression comparison."""

    PASSED = "passed"
    FAILED = "failed"
    IMPROVED = "improved"
    SKIPPED = "skipped"


class RegressionReport(BaseModel):
    """Result of comparing a run against a baseline."""

    model_config = ConfigDict(frozen=True)

    status: RegressionStatus
    has_baseline: bool = False
    baseline_name: str | None = None
    baseline_run_id: str | None = None
    findings: list[RegressionFinding] = Field(default_factory=list)
    score_delta: float | None = None

    @property
    def blocking_findings(self) -> list[RegressionFinding]:
        """Findings with ``error`` severity."""
        return [f for f in self.findings if f.severity is RegressionSeverity.ERROR]

    @property
    def has_blocking_findings(self) -> bool:
        """Whether any finding has ``error`` severity."""
        return bool(self.blocking_findings)
