"""Tests for the agent registry that resolves ``agent.type`` to an adapter."""

from __future__ import annotations

from typing import Any

import pytest

from agent_eval.config import AgentConfig
from agent_eval.integrations.registry import create_agent
from agent_eval.ports.agent import AgentAdapter


class Recorder:
    """A minimal hand-written adapter, standing in for user code."""

    name = "recorder"
    version = "9.9.9"
    framework = "custom"

    async def run(self, case_input: str, *, timeout_s: float | None = None) -> Any:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


def recorder_factory() -> Recorder:
    """Factory the registry can import by path."""
    return Recorder()


#: An already-built adapter, proving a factory may return one directly.
recorder_instance = Recorder()


not_an_adapter = object()

#: A string that is not a callable, used to prove the factory check works.
GRAPH_PATH = "not.a.real.graph"


class TestMockAgent:
    def test_builds_the_bundled_mock_agent(self) -> None:
        adapter = create_agent(AgentConfig(type="mock"))

        assert adapter.framework == "mock"
        assert isinstance(adapter, AgentAdapter)

    def test_type_is_case_insensitive(self) -> None:
        assert create_agent(AgentConfig(type="MOCK")).framework == "mock"


class TestCustomFactory:
    def test_resolves_a_module_colon_callable_path(self) -> None:
        adapter = create_agent(
            AgentConfig(
                factory="tests.unit.test_registry:recorder_factory",
            )
        )

        assert isinstance(adapter, Recorder)
        assert adapter.name == "recorder"

    def test_resolves_a_dotted_attribute_path(self) -> None:
        adapter = create_agent(AgentConfig(factory="tests.unit.test_registry.recorder_factory"))

        assert isinstance(adapter, Recorder)

    def test_rejects_a_result_missing_the_adapter_protocol(self) -> None:
        with pytest.raises(ValueError, match="missing required adapter attribute"):
            create_agent(AgentConfig(factory="tests.unit.test_registry:not_an_adapter"))

    def test_accepts_a_factory_returning_a_ready_adapter(self) -> None:
        adapter = create_agent(AgentConfig(factory="tests.unit.test_registry:recorder_instance"))

        assert isinstance(adapter, Recorder)

    def test_reports_an_unknown_module(self) -> None:
        with pytest.raises(ValueError, match="Could not import module"):
            create_agent(AgentConfig(factory="nope_missing:thing"))

    def test_reports_an_unknown_attribute(self) -> None:
        with pytest.raises(ValueError, match="has no attribute"):
            create_agent(AgentConfig(factory="tests.unit.test_registry:absent"))

    def test_rejects_a_malformed_path(self) -> None:
        with pytest.raises(ValueError, match="Invalid import path"):
            create_agent(AgentConfig(factory="just_a_name"))


class TestLangGraph:
    def test_uses_the_bundled_example_graph_by_default(self) -> None:
        pytest.importorskip("langgraph")
        adapter = create_agent(AgentConfig(type="langgraph"))

        assert adapter.framework == "langgraph"

    def test_example_is_an_alias_for_langgraph(self) -> None:
        pytest.importorskip("langgraph")
        adapter = create_agent(AgentConfig(type="example"))

        assert adapter.framework == "langgraph"

    def test_calls_a_graph_factory(self) -> None:
        pytest.importorskip("langgraph")
        adapter = create_agent(
            AgentConfig(
                type="langgraph",
                options={
                    "graph_factory": ("agent_eval.integrations.langgraph.examples:default_graph")
                },
            )
        )

        assert adapter.framework == "langgraph"

    def test_rejects_a_non_callable_graph_factory(self) -> None:
        pytest.importorskip("langgraph")
        with pytest.raises(ValueError, match="not callable"):
            create_agent(
                AgentConfig(
                    type="langgraph",
                    options={"graph_factory": "tests.unit.test_registry:GRAPH_PATH"},
                )
            )

    def test_reads_state_key_options(self) -> None:
        pytest.importorskip("langgraph")
        adapter = create_agent(
            AgentConfig(
                type="langgraph",
                options={
                    "input_key": "question",
                    "output_key": "answer",
                    "recursion_limit": 12,
                },
            )
        )

        assert adapter._input_key == "question"
        assert adapter._output_key == "answer"
        assert adapter._recursion_limit == 12

    def test_ignores_non_string_state_keys(self) -> None:
        """A YAML ``input_key:`` with no value must not break construction."""
        pytest.importorskip("langgraph")
        adapter = create_agent(AgentConfig(type="langgraph", options={"input_key": None}))

        assert adapter._input_key == "input"

    def test_ignores_a_non_integer_recursion_limit(self) -> None:
        pytest.importorskip("langgraph")
        adapter = create_agent(AgentConfig(type="langgraph", options={"recursion_limit": "lots"}))

        assert adapter._recursion_limit is None


class TestUnknownType:
    def test_rejects_an_unknown_agent_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agent(AgentConfig(type="not_a_framework"))
