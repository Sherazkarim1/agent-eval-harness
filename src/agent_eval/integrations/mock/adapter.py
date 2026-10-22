"""Mock agent adapter: a fully deterministic, offline agent.

This is the adapter that makes ``agent-eval run --mock`` work with no network,
no model and no API key. It follows a small keyword policy, so the same input
always produces the same trajectory, output, tool calls and token counts.

Simulated tool durations are recorded in ``ToolCall.duration_ms`` (trajectory
data is legitimately synthetic in mock mode), but the *case* latency reported
by the harness is always real measured wall-clock time. Latency budgets are
therefore only meaningful in live mode.
"""

from __future__ import annotations

from typing import Any

from agent_eval.domain.models import (
    AgentResult,
    Step,
    StepType,
    TokenUsage,
    ToolCall,
)
from agent_eval.ports.clock import Clock, SystemClock

MOCK_MODEL = "mock-agent-v1"


class MockScenario:
    """A canned agent behaviour: output, tool calls and token usage."""

    def __init__(
        self,
        output: str,
        *,
        tools: list[tuple[str, dict[str, Any], Any]] | None = None,
        input_tokens: int = 120,
        output_tokens: int = 60,
        tool_latency_ms: float = 12.0,
        llm_latency_ms: float = 30.0,
        error: str | None = None,
    ) -> None:
        self.output = output
        self.tools = tools or []
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.tool_latency_ms = tool_latency_ms
        self.llm_latency_ms = llm_latency_ms
        self.error = error


class MockAgent:
    """Deterministic keyword-driven agent used for offline runs and tests."""

    name = "mock-agent"
    version = "1.0.0"
    framework = "mock"

    def __init__(
        self,
        rules: list[tuple[tuple[str, ...], MockScenario]] | None = None,
        default: MockScenario | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        """Create a mock agent.

        Args:
            rules: ``(keywords, scenario)`` pairs; the first rule whose keyword
                appears in the input wins. Comparison is case-insensitive.
            default: Scenario used when no rule matches.
            clock: Clock used for step timestamps.
        """
        self._rules = rules or []
        self._default = default or MockScenario(
            "I am a mock agent. Configure rules to give me domain behaviour."
        )
        self._clock = clock or SystemClock()

    async def run(self, case_input: str, *, timeout_s: float | None = None) -> AgentResult:
        """Return the canned result for ``case_input``."""
        scenario = self._select(case_input)
        steps: list[Step] = []
        tool_calls: list[ToolCall] = []
        total_tokens = TokenUsage(input_tokens=scenario.input_tokens, output_tokens=0)

        for index, (tool_name, arguments, tool_result) in enumerate(scenario.tools):
            tool_calls.append(
                ToolCall(
                    name=tool_name,
                    arguments=arguments,
                    result=tool_result,
                    step_index=index + 1,
                    duration_ms=scenario.tool_latency_ms,
                )
            )
            steps.append(
                Step(
                    type=StepType.TOOL,
                    name=tool_name,
                    input=arguments,
                    output=tool_result,
                    started_at=self._clock.now(),
                    duration_ms=scenario.tool_latency_ms,
                )
            )

        steps.append(
            Step(
                type=StepType.LLM,
                name=MOCK_MODEL,
                input=case_input,
                output=scenario.output,
                tokens=TokenUsage(
                    input_tokens=scenario.input_tokens, output_tokens=scenario.output_tokens
                ),
                started_at=self._clock.now(),
                duration_ms=scenario.llm_latency_ms,
            )
        )
        total_tokens = total_tokens + TokenUsage(
            input_tokens=0, output_tokens=scenario.output_tokens
        )

        if scenario.error is not None:
            steps.append(
                Step(
                    type=StepType.ERROR,
                    name="agent_error",
                    output=scenario.error,
                    started_at=self._clock.now(),
                )
            )

        steps.append(
            Step(
                type=StepType.FINAL,
                output=scenario.output,
                started_at=self._clock.now(),
            )
        )

        return AgentResult(
            output=scenario.output,
            steps=steps,
            tool_calls=tool_calls,
            token_usage=total_tokens,
            metadata={"mock": True, "model": MOCK_MODEL, "execution_mode": "mock"},
            error=scenario.error,
        )

    async def aclose(self) -> None:
        """No resources to release."""
        return None

    def _select(self, case_input: str) -> MockScenario:
        """Pick the first rule whose keyword appears in the input."""
        lowered = case_input.casefold()
        for keywords, scenario in self._rules:
            if any(keyword.casefold() in lowered for keyword in keywords):
                return scenario
        return self._default


def default_mock_agent() -> MockAgent:
    """A mock agent covering the bundled example dataset's behaviours."""
    return MockAgent(
        rules=[
            (
                ("json",),
                MockScenario(
                    '{"order_id": "ORD-1001", "status": "shipped", '
                    '"estimated_delivery": "tomorrow"}',
                    tools=[("get_order_status", {"order_id": "ORD-1001"}, {"status": "shipped"})],
                    output_tokens=40,
                ),
            ),
            (
                ("refund", "return"),
                MockScenario(
                    "I can help with that refund. Our policy allows a full refund within "
                    "30 days of purchase. Your refund will be issued to the original "
                    "payment method within 5 business days.",
                    tools=[("lookup_refund_policy", {"region": "us"}, {"window_days": 30})],
                ),
            ),
            (
                ("order status", "track", "where is my"),
                MockScenario(
                    "Your order ORD-1001 has shipped and is scheduled for delivery tomorrow.",
                    tools=[("get_order_status", {"order_id": "ORD-1001"}, {"status": "shipped"})],
                ),
            ),
            (
                ("order",),
                MockScenario(
                    "I found order ORD-1001: it shipped yesterday via DHL.",
                    tools=[("search_order", {"query": "order"}, {"order_id": "ORD-1001"})],
                ),
            ),
            (
                ("crash", "explode", "internal server error"),
                MockScenario("The agent failed to handle this request.", error="mock_tool_failure"),
            ),
        ]
    )
