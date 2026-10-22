"""Tests for the LangGraph adapter.

The adapter is the only place LangGraph is understood, so these tests focus on
the two things the core depends on: a normalised :class:`AgentResult`, and a
trajectory that reflects what the graph actually did.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, TypedDict

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent_eval.domain.models import StepType
from agent_eval.integrations.langgraph.adapter import LangGraphAdapter
from agent_eval.integrations.langgraph.examples import (
    build_support_graph,
    default_graph,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on asyncio only."""
    return "asyncio"


class State(TypedDict, total=False):
    """Minimal message-based graph state."""

    input: str
    messages: Annotated[list[Any], add_messages]
    output: str


def build_tool_graph(result: str = "shipped", *, fail: bool = False) -> Any:
    """A graph with one tool call, mirroring a real tool-using agent."""
    builder = StateGraph(State)
    builder.add_node("plan", _planner)
    builder.add_node("tool", _tool_node(result, fail))
    builder.add_node("respond", _responder)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "tool")
    builder.add_edge("tool", "respond")
    builder.add_edge("respond", END)
    return builder.compile()


def _planner(state: State) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"name": "lookup", "args": {"q": state.get("input", "")}, "id": "c1"}
                ],
            }
        ]
    }


def _tool_node(result: str, fail: bool):
    def node(state: State) -> dict[str, Any]:
        return {
            "messages": [
                {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": "c1",
                    "name": "lookup",
                    "status": "error" if fail else "success",
                }
            ]
        }

    return node


def _responder(state: State) -> dict[str, Any]:
    text = "All done."
    return {"output": text, "messages": [{"role": "assistant", "content": text}]}


class TestTrajectoryCapture:
    async def test_captures_one_tool_call_with_args_and_result(self) -> None:
        adapter = LangGraphAdapter(build_tool_graph())

        result = await adapter.run("where is my order")

        assert len(result.tool_calls) == 1
        call = result.tool_calls[0]
        assert call.name == "lookup"
        assert call.arguments == {"q": "where is my order"}
        assert call.result == "shipped"
        assert call.error is None

    async def test_request_and_result_are_not_double_counted(self) -> None:
        """LangGraph emits a request and a result; they are one invocation."""
        adapter = LangGraphAdapter(build_tool_graph())

        result = await adapter.run("hello")

        assert [call.name for call in result.tool_calls] == ["lookup"]

    async def test_tool_error_is_recorded_on_the_call(self) -> None:
        adapter = LangGraphAdapter(build_tool_graph("boom", fail=True))

        result = await adapter.run("hello")

        assert result.tool_calls[0].error == "boom"
        assert not result.tool_calls[0].succeeded

    async def test_step_order_follows_execution(self) -> None:
        adapter = LangGraphAdapter(build_tool_graph())

        result = await adapter.run("hello")

        labels = [step.label for step in result.steps]
        assert labels.index("tool:lookup") < len(labels) - 1
        assert labels[-1] == StepType.FINAL.value

    async def test_final_step_carries_the_output(self) -> None:
        adapter = LangGraphAdapter(build_tool_graph())

        result = await adapter.run("hello")

        assert result.steps[-1].type is StepType.FINAL
        assert result.steps[-1].output == "All done."

    async def test_conditional_branch_without_tools(self) -> None:
        adapter = LangGraphAdapter(default_graph())

        result = await adapter.run("Hello there")

        assert result.tool_calls == []
        assert "How can I help" in result.output

    async def test_conditional_branch_with_tools(self) -> None:
        adapter = LangGraphAdapter(default_graph())

        result = await adapter.run("Where is my order?")

        assert [call.name for call in result.tool_calls] == ["lookup_order"]
        assert "ORD-1001" in result.output


class TestOutput:
    async def test_reads_the_configured_output_key(self) -> None:
        adapter = LangGraphAdapter(build_tool_graph(), output_key="output")

        result = await adapter.run("hello")

        assert result.output == "All done."

    async def test_falls_back_to_the_last_message(self) -> None:
        adapter = LangGraphAdapter(build_tool_graph(), output_key="missing")

        result = await adapter.run("hello")

        assert result.output == "All done."

    async def test_renders_non_string_output_as_text(self) -> None:
        class DictOutput(TypedDict, total=False):
            input: str
            output: dict[str, Any]

        def node(state: DictOutput) -> dict[str, Any]:
            return {"output": {"status": "ok"}}

        builder = StateGraph(DictOutput)
        builder.add_node("only", node)
        builder.add_edge(START, "only")
        builder.add_edge("only", END)
        adapter = LangGraphAdapter(builder.compile(), messages_key=None)

        result = await adapter.run("hello")

        assert '"status": "ok"' in result.output


class TestTokens:
    async def test_reports_zero_when_the_graph_reports_no_usage(self) -> None:
        """A deterministic graph has no model, so it must not invent tokens."""
        adapter = LangGraphAdapter(build_tool_graph())

        result = await adapter.run("hello")

        assert result.token_usage is not None
        assert result.token_usage.total_tokens == 0

    async def test_sums_usage_across_messages(self) -> None:
        def node(state: State) -> dict[str, Any]:
            return {
                "messages": [
                    AIMessage(
                        content="hi",
                        usage_metadata={
                            "input_tokens": 10,
                            "output_tokens": 4,
                            "total_tokens": 14,
                        },
                    )
                ],
                "output": "hi",
            }

        builder = StateGraph(State)
        builder.add_node("only", node)
        builder.add_edge(START, "only")
        builder.add_edge("only", END)
        adapter = LangGraphAdapter(builder.compile())

        result = await adapter.run("hello")

        assert result.token_usage is not None
        assert result.token_usage.input_tokens == 10
        assert result.token_usage.output_tokens == 4

    async def test_reads_cached_input_tokens(self) -> None:
        def node(state: State) -> dict[str, Any]:
            return {
                "messages": [
                    AIMessage(
                        content="hi",
                        usage_metadata={
                            "input_tokens": 20,
                            "output_tokens": 2,
                            "total_tokens": 22,
                            "input_token_details": {"cache_read": 8},
                        },
                    )
                ],
                "output": "hi",
            }

        builder = StateGraph(State)
        builder.add_node("only", node)
        builder.add_edge(START, "only")
        builder.add_edge("only", END)

        result = await LangGraphAdapter(builder.compile()).run("hello")

        assert result.token_usage is not None
        assert result.token_usage.cached_input_tokens == 8


class TestFailures:
    async def test_graph_exception_becomes_an_errored_result(self) -> None:
        def boom(state: State) -> dict[str, Any]:
            raise RuntimeError("node exploded")

        builder = StateGraph(State)
        builder.add_node("boom", boom)
        builder.add_edge(START, "boom")
        builder.add_edge("boom", END)
        adapter = LangGraphAdapter(builder.compile())

        result = await adapter.run("hello")

        assert result.error is not None
        assert "node exploded" in result.error
        assert not result.ok
        assert result.steps[-1].type is StepType.ERROR

    async def test_timeout_becomes_an_errored_result(self) -> None:
        async def slow(state: State) -> dict[str, Any]:
            await asyncio.sleep(5)
            return {"output": "never"}

        builder = StateGraph(State)
        builder.add_node("slow", slow)
        builder.add_edge(START, "slow")
        builder.add_edge("slow", END)
        adapter = LangGraphAdapter(builder.compile())

        result = await adapter.run("hello", timeout_s=0.05)

        assert result.error is not None
        assert "timeout" in result.error

    async def test_missing_output_key_yields_empty_output(self) -> None:
        def node(state: State) -> dict[str, Any]:
            return {"messages": [{"role": "assistant", "content": "partial"}]}

        builder = StateGraph(State)
        builder.add_node("only", node)
        builder.add_edge(START, "only")
        builder.add_edge("only", END)
        adapter = LangGraphAdapter(builder.compile())

        result = await adapter.run("hello")

        assert result.output == "partial"


class TestMetadata:
    async def test_records_framework_and_state_keys(self) -> None:
        adapter = LangGraphAdapter(build_tool_graph())

        result = await adapter.run("hello")

        assert result.metadata["framework"] == "langgraph"
        assert "messages" in result.metadata["state_keys"]

    async def test_records_configured_identity_options(self) -> None:
        adapter = LangGraphAdapter(
            build_tool_graph(), options={"model": "gpt-oss", "thread_id": "t1"}
        )

        result = await adapter.run("hello")

        assert result.metadata["model"] == "gpt-oss"
        assert result.metadata["thread_id"] == "t1"

    async def test_declares_the_framework_name(self) -> None:
        adapter = LangGraphAdapter(build_tool_graph())

        assert adapter.framework == "langgraph"
        assert adapter.name
        assert adapter.version


class TestLifecycle:
    async def test_aclose_calls_the_graph_closer(self) -> None:
        closed = False

        class Closable:
            async def ainvoke(self, state: Any, **kwargs: Any) -> dict[str, Any]:
                return {"output": "x"}

            async def aclose(self) -> None:
                nonlocal closed
                closed = True

        await LangGraphAdapter(Closable()).aclose()

        assert closed is True

    async def test_aclose_tolerates_a_graph_without_a_closer(self) -> None:
        class Bare:
            async def ainvoke(self, state: Any, **kwargs: Any) -> dict[str, Any]:
                return {"output": "x"}

        assert await LangGraphAdapter(Bare()).aclose() is None


class TestWithoutStreaming:
    async def test_falls_back_to_the_final_state(self) -> None:
        """A graph exposing only ``ainvoke`` still yields a trajectory."""

        class InvokeOnly:
            async def ainvoke(self, state: Any, **kwargs: Any) -> dict[str, Any]:
                return {
                    "output": "done",
                    "messages": [
                        AIMessage(content="thinking"),
                        ToolMessage(content="ok", tool_call_id="c1", name="lookup"),
                        AIMessage(content="done"),
                    ],
                }

        result = await LangGraphAdapter(InvokeOnly()).run("hello")

        assert result.output == "done"
        assert [step.type for step in result.steps][-1] is StepType.FINAL
        assert any(step.type is StepType.LLM for step in result.steps)


class TestExamples:
    def test_example_graph_is_deterministic(self) -> None:
        graph = build_support_graph()
        adapter = LangGraphAdapter(graph)

        first = asyncio.run(adapter.run("Where is my order?"))
        second = asyncio.run(adapter.run("Where is my order?"))

        assert first.output == second.output
        assert [c.name for c in first.tool_calls] == [c.name for c in second.tool_calls]

    def test_default_graph_entry_point(self) -> None:
        assert default_graph() is not None
