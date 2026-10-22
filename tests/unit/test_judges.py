"""Tests for judges: mock, heuristic, prompt rendering and response parsing."""

from __future__ import annotations

import pytest

from agent_eval.domain.models import AgentResult, EvalCase, JudgeVerdict, TokenUsage
from agent_eval.judges.base import BaseJudge, heuristic_verdict
from agent_eval.judges.mock import HeuristicJudge, MockJudge
from agent_eval.judges.schema import extract_json_object, parse_judge_response, render_prompt
from agent_eval.ports.llm import (
    LLMOptions,
    LLMProviderError,
    LLMResponse,
    ProviderHealth,
    ProviderState,
)
from tests.conftest import make_result


class FakeProvider:
    """Provider returning canned responses, recording the prompts it received."""

    name = "fake"
    model = "fake-model"

    def __init__(self, responses: list[str], *, fail_times: int = 0) -> None:
        self.responses = responses
        self.fail_times = fail_times
        self.calls: list[str] = []
        self.options: list[LLMOptions | None] = []

    async def generate(self, prompt: str, *, schema=None, options=None) -> LLMResponse:
        self.calls.append(prompt)
        self.options.append(options)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise LLMProviderError("provider unavailable")
        text = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.name,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            latency_ms=1.0,
        )

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider=self.name, state=ProviderState.OK)

    async def aclose(self) -> None:
        return None


class FailingProvider:
    name = "failing"
    model = "n/a"

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or LLMProviderError("connection refused")

    async def generate(self, prompt: str, *, schema=None, options=None) -> LLMResponse:
        raise self._exc

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider=self.name, state=ProviderState.UNAVAILABLE)

    async def aclose(self) -> None:
        return None


def case_with_reference(reference: str | None = "refund within 30 days") -> EvalCase:
    return EvalCase(id="c1", input="I want a refund", reference_answer=reference)


class TestMockJudge:
    async def test_returns_fixed_score(self) -> None:
        verdict = await MockJudge(0.9).evaluate(case_with_reference(), make_result("x"))
        assert verdict.score == 0.9
        assert verdict.provider == "mock"
        assert verdict.ok is True

    async def test_error_result_scores_zero(self) -> None:
        verdict = await MockJudge(0.9).evaluate(
            case_with_reference(), make_result("x", error="boom")
        )
        assert verdict.score == 0.0

    async def test_is_deterministic(self) -> None:
        judge = MockJudge(0.75)
        first = await judge.evaluate(case_with_reference(), make_result("a"))
        second = await judge.evaluate(case_with_reference(), make_result("b"))
        assert first.score == second.score


class TestHeuristicJudge:
    async def test_high_overlap_scores_high(self) -> None:
        result = make_result("refund within 30 days of purchase")
        verdict = await HeuristicJudge().evaluate(case_with_reference(), result)
        assert verdict.score > 0.8

    async def test_unrelated_output_scores_low(self) -> None:
        verdict = await HeuristicJudge().evaluate(
            case_with_reference(), make_result("the weather is nice today")
        )
        assert verdict.score < 0.5

    async def test_agent_error_scores_zero(self) -> None:
        verdict = await HeuristicJudge().evaluate(
            case_with_reference(), make_result("x", error="crash")
        )
        assert verdict.score == 0.0
        assert "crash" in (verdict.reason or "")

    async def test_without_reference_uses_presence(self) -> None:
        verdict = await HeuristicJudge().evaluate(
            case_with_reference(None), make_result("anything")
        )
        assert verdict.score is not None

    async def test_reason_states_it_is_heuristic(self) -> None:
        verdict = heuristic_verdict(case_with_reference(), make_result("refund"))
        assert "Heuristic judge" in (verdict.reason or "")

    async def test_relevance_drops_when_off_topic(self) -> None:
        on_topic = heuristic_verdict(
            EvalCase(id="c", input="where is my refund order"),
            make_result("your refund order ORD-1"),
        )
        off_topic = heuristic_verdict(
            EvalCase(id="c", input="where is my refund order"),
            make_result("banana"),
        )
        assert on_topic.criteria["relevance"] > off_topic.criteria["relevance"]


class TestPromptRendering:
    def test_includes_all_sections(self) -> None:
        prompt = render_prompt(
            case_input="I want a refund",
            agent_output="Approved",
            reference_answer="30 days",
            tool_calls="search(q='refund')",
            criteria=["correctness", "relevance"],
        )
        assert "I want a refund" in prompt
        assert "Approved" in prompt
        assert "30 days" in prompt
        assert "search" in prompt
        assert "correctness" in prompt

    def test_handles_missing_reference_and_tools(self) -> None:
        prompt = render_prompt(
            case_input="x",
            agent_output="y",
            reference_answer=None,
            tool_calls="",
            criteria=["correctness"],
        )
        assert "none provided" in prompt
        assert "no tool calls" in prompt

    def test_custom_template(self) -> None:
        prompt = render_prompt(
            case_input="IN",
            agent_output="OUT",
            reference_answer="REF",
            tool_calls="TOOLS",
            criteria=["c1"],
            template="{case_input}|{agent_output}",
        )
        assert prompt == "IN|OUT"


class TestResponseParsing:
    def test_plain_json(self) -> None:
        score, reason, criteria = parse_judge_response(
            '{"score": 0.8, "reason": "good", "criteria": {"correctness": 0.9}}',
            ["correctness"],
        )
        assert score == 0.8
        assert reason == "good"
        assert criteria["correctness"] == 0.9

    def test_fenced_json(self) -> None:
        score, _, _ = parse_judge_response(
            '```json\n{"score": 0.5, "reason": "ok"}\n```', ["correctness"]
        )
        assert score == 0.5

    def test_json_with_preamble(self) -> None:
        score, _, _ = parse_judge_response(
            'Here is my verdict:\n{"score": 0.7, "reason": "ok"}', ["correctness"]
        )
        assert score == 0.7

    def test_missing_criteria_default_to_score(self) -> None:
        _, _, criteria = parse_judge_response('{"score": 0.6}', ["correctness", "relevance"])
        assert criteria == {"correctness": 0.6, "relevance": 0.6}

    def test_out_of_range_criteria_are_clamped(self) -> None:
        _, _, criteria = parse_judge_response(
            '{"score": 0.5, "criteria": {"correctness": 5.0}}', ["correctness"]
        )
        assert criteria["correctness"] == 1.0

    def test_non_numeric_criteria_ignored(self) -> None:
        _, _, criteria = parse_judge_response(
            '{"score": 0.5, "criteria": {"correctness": "high"}}', ["correctness"]
        )
        assert criteria["correctness"] == 0.5

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="did not contain a JSON object"):
            parse_judge_response("I refuse to answer", ["correctness"])

    def test_score_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_judge_response('{"score": 2.0}', ["correctness"])

    def test_json_array_is_not_an_object(self) -> None:
        with pytest.raises(ValueError):
            parse_judge_response("[1, 2, 3]", ["correctness"])


class TestExtractJsonObject:
    def test_finds_nested_object(self) -> None:
        text = 'prefix {"a": {"b": 1}} suffix'
        assert extract_json_object(text) == {"a": {"b": 1}}

    def test_returns_none_without_object(self) -> None:
        assert extract_json_object("no braces here") is None


class TestBaseJudge:
    async def test_successful_parse(self) -> None:
        provider = FakeProvider(['{"score": 0.75, "reason": "solid"}'])
        judge = BaseJudge(provider)
        verdict = await judge.evaluate(case_with_reference(), make_result("refund"))
        assert verdict.score == 0.75
        assert verdict.model == "fake-model"
        assert verdict.ok is True

    async def test_provider_failure_returns_verdict_not_exception(self) -> None:
        judge = BaseJudge(FailingProvider(), max_retries=0)
        verdict = await judge.evaluate(case_with_reference(), make_result("x"))
        assert verdict.score is None
        assert verdict.error is not None
        assert "connection refused" in verdict.error

    async def test_unexpected_exception_is_caught(self) -> None:
        judge = BaseJudge(FailingProvider(RuntimeError("weird")), max_retries=0)
        verdict = await judge.evaluate(case_with_reference(), make_result("x"))
        assert verdict.error is not None
        assert "RuntimeError" in verdict.error

    async def test_retries_then_succeeds(self) -> None:
        provider = FakeProvider(['{"score": 0.5}'], fail_times=1)
        judge = BaseJudge(provider, max_retries=2)
        verdict = await judge.evaluate(case_with_reference(), make_result("x"))
        assert verdict.score == 0.5
        assert len(provider.calls) == 2

    async def test_malformed_output_is_reported(self) -> None:
        judge = BaseJudge(FakeProvider(["totally not json"]))
        verdict = await judge.evaluate(case_with_reference(), make_result("x"))
        assert verdict.score is None
        assert "malformed judge output" in (verdict.error or "")

    async def test_prompt_contains_case_context(self) -> None:
        provider = FakeProvider(['{"score": 1.0}'])
        judge = BaseJudge(provider)
        await judge.evaluate(case_with_reference(), make_result("refund approved"))
        assert "I want a refund" in provider.calls[0]

    async def test_temperature_and_timeout_forwarded(self) -> None:
        provider = FakeProvider(['{"score": 1.0}'])
        judge = BaseJudge(provider, temperature=0.3, timeout_s=12.0)
        await judge.evaluate(case_with_reference(), make_result("x"))
        options = provider.options[0]
        assert options is not None
        assert options.temperature == 0.3
        assert options.timeout_s == 12.0


class TestJudgeProtocol:
    def test_mock_judge_satisfies_protocol(self) -> None:
        from agent_eval.ports.judge import Judge

        assert isinstance(MockJudge(), Judge)
        assert isinstance(HeuristicJudge(), Judge)

    def test_verdict_always_has_safe_shape(self) -> None:
        verdict = JudgeVerdict()
        assert verdict.score is None
        assert verdict.ok is False
        assert verdict.criteria == {}


def test_agent_result_without_tokens_is_zero() -> None:
    result: AgentResult = make_result("x", tokens=None)
    assert result.token_usage is None
