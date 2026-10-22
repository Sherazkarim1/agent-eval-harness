"""Agent adapter port.

An adapter is the only place that knows how a specific agent framework is
driven and how its trajectory is observed. The core never imports LangGraph,
LangChain or any vendor SDK.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_eval.domain.models import AgentResult


@runtime_checkable
class AgentAdapter(Protocol):
    """Runs an agent for a single input and returns a normalised result.

    Implementations **must** populate ``AgentResult.steps`` and
    ``AgentResult.tool_calls`` completely: trajectory capture is part of the
    adapter contract, not something the harness reconstructs afterwards.
    """

    name: str
    version: str
    framework: str

    async def run(self, case_input: str, *, timeout_s: float | None = None) -> AgentResult:
        """Execute the agent.

        Args:
            case_input: The user input for this evaluation case.
            timeout_s: Optional per-case timeout in seconds.

        Returns:
            The normalised :class:`AgentResult`. Adapters should return a
            result carrying ``error`` instead of raising for *agent-level*
            failures (a failed agent is a measurable outcome, not a crash).
        """
        ...

    async def aclose(self) -> None:
        """Release any resources held by the adapter."""
        ...
