"""Tests for domain model validation and invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_eval.domain.models import (
    AgentResult,
    AssertionResult,
    Dataset,
    EvalCase,
    Expectation,
    Step,
    StepType,
    TokenUsage,
    ToolCall,
)


class TestTokenUsage:
    def test_total_includes_cached(self) -> None:
        usage = TokenUsage(input_tokens=10, output_tokens=5, cached_input_tokens=3)
        assert usage.total_tokens == 18

    def test_addition_sums_fields(self) -> None:
        total = TokenUsage(input_tokens=1, output_tokens=2) + TokenUsage(
            input_tokens=10, output_tokens=20
        )
        assert (total.input_tokens, total.output_tokens) == (11, 22)

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            TokenUsage(input_tokens=-1)


class TestStep:
    def test_tool_label_includes_name(self) -> None:
        assert Step(type=StepType.TOOL, name="search").label == "tool:search"

    def test_llm_label_is_plain(self) -> None:
        assert Step(type=StepType.LLM).label == "llm"

    def test_is_frozen(self) -> None:
        step = Step(type=StepType.LLM)
        with pytest.raises(ValidationError):
            step.type = StepType.TOOL  # type: ignore[misc]


class TestAgentResult:
    def test_ok_reflects_error(self) -> None:
        assert AgentResult(output="x").ok is True
        assert AgentResult(output="x", error="boom").ok is False

    def test_tool_names_are_deduplicated_in_order(self) -> None:
        result = AgentResult(
            output="x",
            tool_calls=[
                ToolCall(name="b"),
                ToolCall(name="a"),
                ToolCall(name="b"),
            ],
        )
        assert result.tool_names() == ["b", "a"]

    def test_total_tool_duration_ignores_missing(self) -> None:
        result = AgentResult(
            output="x",
            tool_calls=[
                ToolCall(name="a", duration_ms=5.0),
                ToolCall(name="b"),
            ],
        )
        assert result.total_tool_duration_ms() == 5.0


class TestExpectation:
    def test_empty_expectation_is_detected(self) -> None:
        assert Expectation().is_empty() is True

    def test_populated_expectation_is_not_empty(self) -> None:
        assert Expectation(contains=["refund"]).is_empty() is False

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            Expectation(contains_typo=["x"])  # type: ignore[call-arg]

    def test_rejects_empty_list(self) -> None:
        with pytest.raises(ValidationError, match="empty list"):
            Expectation(contains=[])

    def test_rejects_negative_latency(self) -> None:
        with pytest.raises(ValidationError):
            Expectation(max_latency_ms=-5)


class TestEvalCase:
    def test_rejects_invalid_id_characters(self) -> None:
        with pytest.raises(ValidationError):
            EvalCase(id="has space", input="hi")

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValidationError):
            EvalCase(id="a", input="")

    def test_coerces_numeric_id(self) -> None:
        assert EvalCase(id=123, input="hi").id == "123"  # type: ignore[arg-type]

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            EvalCase(id="a", input="hi", nope=1)  # type: ignore[call-arg]


class TestDataset:
    def test_requires_cases(self) -> None:
        with pytest.raises(ValidationError):
            Dataset(name="d", version="1.0.0", cases=[])

    def test_rejects_non_semver(self) -> None:
        with pytest.raises(ValidationError):
            Dataset(name="d", version="v1", cases=[EvalCase(id="a", input="hi")])

    def test_select_by_tag(self) -> None:
        dataset = Dataset(
            name="d",
            version="1.0.0",
            cases=[
                EvalCase(id="a", input="hi", tags=["x"]),
                EvalCase(id="b", input="hi", tags=["y"]),
            ],
        )
        assert dataset.select(tags=["x"]).case_ids() == ["a"]

    def test_select_by_ids(self) -> None:
        dataset = Dataset(
            name="d",
            version="1.0.0",
            cases=[EvalCase(id="a", input="hi"), EvalCase(id="b", input="hi")],
        )
        assert dataset.select(ids={"b"}).case_ids() == ["b"]

    def test_by_tag(self) -> None:
        dataset = Dataset(
            name="d",
            version="1.0.0",
            cases=[EvalCase(id="a", input="hi", tags=["x", "y"])],
        )
        assert len(dataset.by_tag("y")) == 1
        assert dataset.by_tag("z") == []


class TestAssertionResult:
    def test_failed_mirrors_passed(self) -> None:
        assert AssertionResult(assertion="contains", passed=False).failed is True
