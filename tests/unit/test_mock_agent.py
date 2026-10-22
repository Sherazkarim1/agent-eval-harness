"""Tests for the mock agent adapter."""

from __future__ import annotations

import pytest

from agent_eval.domain.models import StepType
from agent_eval.integrations.mock import MockAgent, MockScenario, default_mock_agent
from agent_eval.ports.agent import AgentAdapter


class TestMockAgent:
    async def test_satisfies_protocol(self) -> None:
        assert isinstance(MockAgent(), AgentAdapter)

    async def test_default_scenario_for_unknown_input(self) -> None:
        result = await MockAgent().run("something arbitrary")
        assert result.ok is True
        assert result.error is None
        assert "mock agent" in result.output

    async def test_first_matching_rule_wins(self) -> None:
        agent = MockAgent(
            rules=[
                (("refund",), MockScenario("refund answer")),
                (("order",), MockScenario("order answer")),
            ]
        )
        assert (await agent.run("I want a refund")).output == "refund answer"
        assert (await agent.run("my order")).output == "order answer"

    async def test_rules_are_case_insensitive(self) -> None:
        agent = MockAgent(rules=[(("refund",), MockScenario("matched"))])
        assert (await agent.run("I want a REFUND")).output == "matched"

    async def test_any_keyword_in_tuple_matches(self) -> None:
        agent = MockAgent(rules=[(("a", "b"), MockScenario("matched"))])
        assert (await agent.run("contains b")).output == "matched"

    async def test_records_trajectory_steps(self) -> None:
        agent = MockAgent(
            rules=[(("x",), MockScenario("out", tools=[("search", {"q": "x"}, "found")]))]
        )
        result = await agent.run("x")
        assert [step.type for step in result.steps] == [StepType.TOOL, StepType.LLM, StepType.FINAL]
        assert result.steps[0].name == "search"

    async def test_records_tool_calls_with_arguments(self) -> None:
        agent = MockAgent(
            rules=[(("x",), MockScenario("out", tools=[("search", {"q": "x"}, "found")]))]
        )
        result = await agent.run("x")
        assert len(result.tool_calls) == 1
        call = result.tool_calls[0]
        assert call.name == "search"
        assert call.arguments == {"q": "x"}
        assert call.result == "found"
        assert call.step_index == 1
        assert call.succeeded is True

    async def test_step_labels_include_tool_names(self) -> None:
        agent = MockAgent(rules=[(("x",), MockScenario("out", tools=[("search", {}, None)]))])
        result = await agent.run("x")
        assert "tool:search" in result.step_labels()

    async def test_token_usage_is_reported(self) -> None:
        result = await MockAgent(
            rules=[(("x",), MockScenario("out", input_tokens=50, output_tokens=25))]
        ).run("x")
        assert result.token_usage is not None
        assert result.token_usage.input_tokens == 50
        assert result.token_usage.output_tokens == 25

    async def test_error_scenario_adds_error_step(self) -> None:
        agent = MockAgent(rules=[(("x",), MockScenario("", error="tool_failure"))])
        result = await agent.run("x")
        assert result.ok is False
        assert result.error == "tool_failure"
        assert any(step.type is StepType.ERROR for step in result.steps)

    async def test_metadata_marks_mock_mode(self) -> None:
        result = await MockAgent().run("x")
        assert result.metadata["mock"] is True
        assert result.metadata["execution_mode"] == "mock"

    async def test_is_deterministic(self) -> None:
        agent = default_mock_agent()
        first = await agent.run("I want a refund")
        second = await agent.run("I want a refund")
        assert first.output == second.output
        assert first.tool_names() == second.tool_names()
        assert first.token_usage == second.token_usage

    async def test_aclose_is_safe(self) -> None:
        assert await MockAgent().aclose() is None


class TestDefaultMockAgent:
    @pytest.mark.parametrize(
        ("text", "expected_tool"),
        [
            ("I want a refund for my order", "lookup_refund_policy"),
            ("Where is my order?", "get_order_status"),
            ("look up my order details", "search_order"),
            ("Return it as JSON", "get_order_status"),
        ],
    )
    async def test_expected_tool_per_keyword(self, text: str, expected_tool: str) -> None:
        result = await default_mock_agent().run(text)
        assert expected_tool in result.tool_names()

    async def test_refund_mentions_policy(self) -> None:
        result = await default_mock_agent().run("I want a refund")
        assert "30 days" in result.output

    async def test_error_trigger(self) -> None:
        result = await default_mock_agent().run("trigger a crash please")
        assert result.ok is False
        assert result.error == "mock_tool_failure"
