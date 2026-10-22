"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_eval.config import HarnessConfig, ScoringConfig
from agent_eval.domain.models import (
    AgentResult,
    Dataset,
    EvalCase,
    Expectation,
    Step,
    StepType,
    TokenUsage,
    ToolCall,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DATASET = PROJECT_ROOT / "evals" / "dataset.yaml"


@pytest.fixture
def project_root() -> Path:
    """Absolute path of the repository root."""
    return PROJECT_ROOT


@pytest.fixture
def config() -> HarnessConfig:
    """A default configuration with no thresholds."""
    return HarnessConfig(scoring=ScoringConfig())


@pytest.fixture
def example_dataset() -> Dataset:
    """The bundled example dataset."""
    from agent_eval.datasets import load_dataset_file

    return load_dataset_file(EXAMPLE_DATASET)


def make_result(
    output: str = "hello",
    *,
    tools: list[tuple[str, dict, object]] | None = None,
    tokens: TokenUsage | None = None,
    error: str | None = None,
    llm_steps: int = 1,
    tool_latency_ms: float = 10.0,
) -> AgentResult:
    """Build an :class:`AgentResult` for tests."""
    tool_calls = [
        ToolCall(
            name=name, arguments=args, result=res, step_index=i + 1, duration_ms=tool_latency_ms
        )
        for i, (name, args, res) in enumerate(tools or [])
    ]
    steps: list[Step] = [
        Step(
            type=StepType.TOOL,
            name=call.name,
            input=call.arguments,
            output=call.result,
            duration_ms=call.duration_ms,
        )
        for call in tool_calls
    ]
    steps.extend(
        Step(type=StepType.LLM, name="test-model", output=output, tokens=tokens)
        for _ in range(llm_steps)
    )
    steps.append(Step(type=StepType.FINAL, output=output))
    return AgentResult(
        output=output,
        steps=steps,
        tool_calls=tool_calls,
        token_usage=tokens,
        error=error,
    )


def make_case(**kwargs) -> EvalCase:
    """Build an :class:`EvalCase` with sensible defaults."""
    expected = kwargs.pop("expected", None)
    payload: dict = {"id": "case_001", "input": "hello", **kwargs}
    if expected is not None:
        payload["expected"] = (
            expected if isinstance(expected, Expectation) else Expectation(**expected)
        )
    return EvalCase(**payload)


def make_dataset(*cases: EvalCase, name: str = "unit", version: str = "1.0.0") -> Dataset:
    """Build a :class:`Dataset` from cases."""
    return Dataset(name=name, version=version, cases=list(cases) or [make_case()])
