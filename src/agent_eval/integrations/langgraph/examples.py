"""A deterministic LangGraph graph used by the examples and tests.

The graph is a small customer-support loop: it inspects the request, calls a
lookup tool when it needs order data, and answers from what the tool returned.
It uses no model, so it runs fully offline and deterministically, which is what
makes it useful for proving trajectory capture works.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

TOOL_CALL_ID = "call-lookup-1"


class SupportState(TypedDict, total=False):
    """State shared by every node of the example graph."""

    input: str
    messages: Annotated[list[Any], add_messages]
    output: str


def _needs_lookup(text: str) -> bool:
    """Whether the request needs order data before it can be answered."""
    lowered = text.casefold()
    return any(word in lowered for word in ("order", "refund", "track", "shipping"))


def plan(state: SupportState) -> dict[str, Any]:
    """Decide whether to call the lookup tool, and record the intent."""
    text = state.get("input", "")
    if _needs_lookup(text):
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "lookup_order",
                            "args": {"query": text},
                            "id": TOOL_CALL_ID,
                        }
                    ],
                }
            ]
        }
    return {"messages": [{"role": "assistant", "content": ""}]}


def lookup(state: SupportState) -> dict[str, Any]:
    """Answer the request from known order data, with no network access."""
    return {
        "messages": [
            {
                "role": "tool",
                "content": '{"order_id": "ORD-1001", "status": "shipped"}',
                "tool_call_id": TOOL_CALL_ID,
                "name": "lookup_order",
            }
        ]
    }


def respond(state: SupportState) -> dict[str, Any]:
    """Produce the final answer from the conversation so far."""
    tool_results = [
        message for message in state.get("messages", []) if type(message).__name__ == "ToolMessage"
    ]
    if tool_results:
        text = "I looked up your order: ORD-1001 has shipped and should arrive tomorrow."
    else:
        text = "Thanks for reaching out. How can I help you today?"
    return {
        "output": text,
        "messages": [{"role": "assistant", "content": text}],
    }


def build_support_graph() -> Any:
    """Compile the example support graph."""
    builder = StateGraph(SupportState)
    builder.add_node("plan", plan)
    builder.add_node("lookup", lookup)
    builder.add_node("respond", respond)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "plan",
        _route_after_plan,
        {"lookup": "lookup", "respond": "respond"},
    )
    builder.add_edge("lookup", "respond")
    builder.add_edge("respond", END)
    return builder.compile()


def _route_after_plan(state: SupportState) -> str:
    """Send requests needing order data to the tool, others straight to the answer."""
    return "lookup" if _needs_lookup(state.get("input", "")) else "respond"


def default_graph() -> Any:
    """Entry point for configuration, e.g. ``agent_eval...examples:default_graph``."""
    return build_support_graph()
