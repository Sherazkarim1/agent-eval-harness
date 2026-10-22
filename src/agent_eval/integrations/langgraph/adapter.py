"""LangGraph agent adapter.

Drives a compiled LangGraph graph for one evaluation case and normalises the
run into an :class:`AgentResult`. Everything LangGraph-specific lives here; the
core of the harness never imports LangGraph.

Trajectory capture
------------------
The adapter streams the graph with ``stream_mode="updates"``, so each node's
contribution is observed as it happens rather than reconstructed from the final
state. That distinction matters: LangGraph's ``add_messages`` reducer replays
every intermediate message into the final state, so a state-only reading would
report the same assistant message several times and lose the ordering evidence
that ``step_order`` and ``required_step`` assertions depend on.

Token accounting
----------------
LangGraph reports usage per message in ``AIMessage.usage_metadata``. When a
graph is driven by a real chat model those numbers are real; when it is driven
by deterministic test nodes they are absent, and this adapter then reports zero
tokens rather than inventing a count.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from langchain_core.messages import convert_to_messages

from agent_eval.domain.models import AgentResult, Step, StepType, TokenUsage, ToolCall
from agent_eval.ports.clock import Clock, SystemClock

logger = logging.getLogger(__name__)

DEFAULT_INPUT_KEY = "input"
DEFAULT_OUTPUT_KEY = "output"
DEFAULT_MESSAGES_KEY = "messages"


class LangGraphAdapter:
    """Adapter that evaluates a compiled LangGraph graph.

    The graph is expected to accept a mapping containing the case input and to
    return a mapping containing the final answer. Which keys carry those values
    is configurable, because graphs name their own state fields.
    """

    framework = "langgraph"
    name = "langgraph-agent"
    version = "1.0.0"

    def __init__(
        self,
        graph: Any,
        *,
        options: dict[str, Any] | None = None,
        input_key: str = DEFAULT_INPUT_KEY,
        output_key: str = DEFAULT_OUTPUT_KEY,
        messages_key: str | None = DEFAULT_MESSAGES_KEY,
        recursion_limit: int | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Create the adapter.

        Args:
            graph: A compiled LangGraph graph, or any object exposing
                ``ainvoke``/``astream``.
            options: Raw ``agent.options`` from configuration, recorded on the
                result for provenance.
            input_key: State key the case input is written to.
            output_key: State key the final answer is read from.
            messages_key: State key holding the message list. When ``None``,
                message-based trajectory capture is skipped.
            recursion_limit: Optional LangGraph recursion limit.
            clock: Clock used for step timestamps.
        """
        self._graph = graph
        self._options = options or {}
        self._input_key = input_key
        self._output_key = output_key
        self._messages_key = messages_key
        self._recursion_limit = recursion_limit
        self._clock = clock or SystemClock()

    async def run(self, case_input: str, *, timeout_s: float | None = None) -> AgentResult:
        """Execute the graph once and return the normalised result.

        Agent-level failures are reported through ``AgentResult.error`` rather
        than raised, because a failing agent is a measurable outcome. Only
        genuinely unusable configuration raises.
        """
        state = self._initial_state(case_input)
        run_config = self._run_config()
        started = self._clock.now()

        try:
            steps, tool_calls, tokens, final_state = await self._invoke_with_timeout(
                state, run_config, timeout_s
            )
        except TimeoutError:
            return self._timeout_result(started, timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("LangGraph run failed: %s: %s", type(exc).__name__, exc)
            return self._error_result(started, exc)

        output = self._extract_output(final_state)
        steps.append(
            Step(
                type=StepType.FINAL,
                name=self._output_key,
                output=output,
                started_at=self._clock.now(),
            )
        )
        return AgentResult(
            output=output,
            steps=steps,
            tool_calls=tool_calls,
            token_usage=tokens,
            metadata=self._metadata(final_state),
        )

    async def aclose(self) -> None:
        """Release the graph's resources, if it exposes a closer."""
        for name in ("aclose", "close"):
            closer = getattr(self._graph, name, None)
            if callable(closer):
                result = closer()
                if asyncio.iscoroutine(result):
                    await result
                return None
        return None

    def _initial_state(self, case_input: str) -> dict[str, Any]:
        """Build the graph input state for one case."""
        if self._messages_key is not None:
            return {
                self._input_key: case_input,
                self._messages_key: [{"role": "user", "content": case_input}],
            }
        return {self._input_key: case_input}

    def _run_config(self) -> dict[str, Any] | None:
        """Assemble LangGraph's runtime config, or ``None`` for defaults."""
        if self._recursion_limit is None:
            return None
        return {"recursion_limit": self._recursion_limit}

    async def _invoke_with_timeout(
        self,
        state: dict[str, Any],
        run_config: dict[str, Any] | None,
        timeout_s: float | None,
    ) -> tuple[list[Step], list[ToolCall], TokenUsage, dict[str, Any]]:
        """Stream the graph and collect steps, tool calls, tokens and final state.

        Raises:
            TimeoutError: If ``timeout_s`` elapses before the graph finishes.
        """
        collected: list[Step] = []
        tool_calls: list[ToolCall] = []
        tokens = TokenUsage()
        final_state: dict[str, Any] = {}

        async def consume() -> None:
            nonlocal final_state, tokens
            step_index = 0
            # LangGraph emits a tool *request* on the assistant message and the
            # tool *result* as a separate ToolMessage. They describe one call, so
            # results are merged into their request rather than counted twice.
            pending: dict[str, ToolCall] = {}
            position: dict[str, int] = {}
            async for chunk in self._stream(state, run_config):
                final_state = _merge_state(final_state, chunk)
                for node_name, update in _iter_node_updates(chunk):
                    for message in _messages_of(update, self._messages_key):
                        identifier = _message_identifier(message)
                        for step, call in self._steps_from_message(message, node_name):
                            # Usage is reported on the LLM step; tool steps carry
                            # none, so summing every step's tokens is safe.
                            tokens = tokens + (step.tokens or TokenUsage())
                            if call is not None:
                                if identifier is not None:
                                    pending[identifier] = call
                                    position[identifier] = step_index
                                else:
                                    tool_calls.append(_with_step_index(call, step_index))
                            collected.append(step)
                            step_index += 1
            for identifier, call in pending.items():
                resolved = _resolve_from_messages(final_state, self._messages_key, identifier)
                merged = _merge_result(call, resolved, position.get(identifier, 0))
                tool_calls.append(merged)

        if timeout_s is not None:
            await asyncio.wait_for(consume(), timeout=timeout_s)
        else:
            await consume()

        if self._messages_key is not None and not collected:
            collected = self._steps_from_state(final_state)
        return collected, tool_calls, tokens, final_state

    async def _stream(
        self,
        state: dict[str, Any],
        run_config: dict[str, Any] | None,
    ) -> Any:
        """Yield graph updates, preferring the streaming interface."""
        stream = getattr(self._graph, "astream", None)
        if not callable(stream):
            result = await self._graph.ainvoke(state, config=run_config)
            yield {"__final__": result}
            return
        kwargs: dict[str, Any] = {"stream_mode": "updates"}
        if run_config is not None:
            kwargs["config"] = run_config
        async for chunk in stream(state, **kwargs):
            yield chunk

    def _steps_from_message(
        self,
        message: Any,
        node_name: str,
    ) -> list[tuple[Step, ToolCall | None]]:
        """Translate one LangGraph message into harness steps.

        An assistant message yields one LLM step, plus one step per requested
        tool call. A tool message contributes no step of its own: LangGraph
        already emitted the request, and the result is merged onto it by id so
        one invocation produces exactly one :class:`ToolCall`.
        """
        kind = type(message).__name__
        started = self._clock.now()

        if kind == "AIMessage":
            requested = list(getattr(message, "tool_calls", None) or [])
            content = _content_of(message)
            used = _usage_of(message)
            steps: list[tuple[Step, ToolCall | None]] = []
            if content or not requested:
                steps.append(
                    (
                        Step(
                            type=StepType.LLM,
                            name=node_name,
                            output=content,
                            tokens=used if used.total_tokens else None,
                            started_at=started,
                        ),
                        None,
                    )
                )
            for raw in requested:
                steps.append(
                    (
                        Step(
                            type=StepType.TOOL,
                            name=_tool_name_of(raw),
                            input=_tool_args_of(raw),
                            started_at=started,
                        ),
                        _tool_call_from_request(raw),
                    )
                )
            return steps

        if kind == "ToolMessage":
            content = _content_of(message)
            if not content:
                return []
            return [
                (
                    Step(
                        type=StepType.TOOL,
                        name=str(getattr(message, "name", None) or "unknown_tool"),
                        output=content,
                        started_at=started,
                    ),
                    None,
                )
            ]

        content = _content_of(message)
        if not content:
            return []
        return [
            (
                Step(
                    type=StepType.STATE,
                    name=node_name,
                    output=content,
                    started_at=started,
                ),
                None,
            )
        ]

    def _steps_from_state(self, final_state: dict[str, Any]) -> list[Step]:
        """Fall back to the final state's messages when nothing was streamed.

        A graph invoked without streaming support still yields a usable
        trajectory; only the per-node attribution is lost.
        """
        if self._messages_key is None:
            return []
        steps: list[Step] = []
        for message in _messages_of(final_state, self._messages_key):
            content = _content_of(message)
            if not content:
                continue
            steps.append(
                Step(
                    type=StepType.LLM,
                    name=type(message).__name__,
                    output=content,
                    started_at=self._clock.now(),
                )
            )
        return steps

    def _extract_output(self, final_state: dict[str, Any]) -> str:
        """Read the final answer out of the graph's state."""
        if self._output_key in final_state:
            return _as_text(final_state[self._output_key])
        if self._messages_key and self._messages_key in final_state:
            return self._last_message_text(final_state[self._messages_key])
        return ""

    def _last_message_text(self, messages: Any) -> str:
        """Return the content of the final message carrying text."""
        if not isinstance(messages, list):
            return ""
        for message in reversed(messages):
            text = _content_of(message)
            if text:
                return text
        return ""

    def _metadata(self, final_state: dict[str, Any]) -> dict[str, Any]:
        """Provenance recorded on every result produced by this adapter."""
        metadata: dict[str, Any] = {
            "framework": self.framework,
            "input_key": self._input_key,
            "output_key": self._output_key,
            "state_keys": sorted(final_state),
        }
        for key in ("thread_id", "run_id", "model", "model_name"):
            if key in self._options:
                metadata[key] = self._options[key]
        return metadata

    def _error_result(
        self,
        started: datetime,
        exc: BaseException,
    ) -> AgentResult:
        """Wrap an agent failure as a measurable errored result."""
        message = f"{type(exc).__name__}: {exc}"
        return self._failed_result("langgraph_error", message, started)

    def _timeout_result(
        self,
        started: datetime,
        timeout_s: float | None,
    ) -> AgentResult:
        """Wrap a timeout as a measurable errored result."""
        limit = timeout_s if timeout_s is not None else 0.0
        return self._failed_result(
            "langgraph_timeout",
            f"langgraph_timeout: exceeded {limit:g}s",
            started,
            {"timeout_s": limit},
        )

    def _failed_result(
        self,
        name: str,
        message: str,
        started: datetime,
        extra: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Build the errored result shared by failures and timeouts."""
        ended = self._clock.now()
        return AgentResult(
            output="",
            steps=[
                Step(
                    type=StepType.ERROR,
                    name=name,
                    output=message,
                    started_at=ended,
                    duration_ms=_elapsed_ms(started, ended),
                )
            ],
            metadata={"framework": self.framework, **(extra or {})},
            error=message,
        )


# --------------------------------------------------------------------------------------
# LangGraph payload helpers
#
# These read loosely-typed graph output defensively: a graph is user code, and a
# missing or renamed field must not crash the harness mid-run.
# --------------------------------------------------------------------------------------


def _iter_node_updates(chunk: Any) -> list[tuple[str, Any]]:
    """Yield ``(node_name, update)`` pairs from one streamed chunk."""
    if not isinstance(chunk, dict):
        return []
    final = chunk.get("__final__")
    if "__final__" in chunk:
        return [("", final)]
    return [(str(node), update) for node, update in chunk.items()]


def _merge_state(current: dict[str, Any], chunk: Any) -> dict[str, Any]:
    """Accumulate streamed updates into an approximation of the final state."""
    if not isinstance(chunk, dict):
        return current
    if "__final__" in chunk:
        final = chunk["__final__"]
        return final if isinstance(final, dict) else current
    for update in chunk.values():
        if isinstance(update, dict):
            for key, value in update.items():
                if key.endswith("messages") and isinstance(value, list):
                    current.setdefault(key, []).extend(value)
                else:
                    current[key] = value
    return current


def _messages_of(update: Any, messages_key: str | None) -> list[Any]:
    """Pull the message list out of a node update, normalising it to objects.

    A node may return plain dicts (``{"role": "assistant", ...}``) rather than
    LangChain message instances. ``stream_mode="updates"`` preserves whichever
    the node produced, so they are converted here; otherwise tool calls and
    usage metadata would be invisible for dict-returning graphs.
    """
    if messages_key is None or not isinstance(update, dict):
        return []
    messages = update.get(messages_key)
    if not isinstance(messages, list):
        return []
    if all(isinstance(item, dict) for item in messages):
        try:
            return list(convert_to_messages(messages))
        except (TypeError, ValueError):
            return list(messages)
    return list(messages)


def _content_of(message: Any) -> str:
    """Read a message's textual content.

    Content may be a plain string or LangChain's structured block list; both are
    flattened to text so assertions can match against it.
    """
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return ""


def _error_of(message: Any) -> str | None:
    """Return a tool message's error text, if it represents a failure."""
    status = getattr(message, "status", None)
    if status is None and isinstance(message, dict):
        status = message.get("status")
    if status == "error":
        return _content_of(message) or "tool_error"
    return None


def _usage_of(message: Any) -> TokenUsage:
    """Read ``usage_metadata`` from a message, if the model reported any."""
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        return TokenUsage()
    details = usage.get("input_token_details") or {}
    cached = details.get("cache_read") if isinstance(details, dict) else None
    return TokenUsage(
        input_tokens=_as_int(usage.get("input_tokens")),
        output_tokens=_as_int(usage.get("output_tokens")),
        cached_input_tokens=_as_int(cached),
    )


def _message_identifier(message: Any) -> str | None:
    """Correlation id used to pair a tool request with its result.

    An assistant message's id matches the ``tool_call_id`` of the ToolMessage
    its tool produced. When a message carries a single tool request, that
    request's own id is used instead, so graphs that omit the message id still
    pair correctly.
    """
    identifier = getattr(message, "id", None)
    if isinstance(identifier, str) and identifier:
        return identifier
    requested = list(getattr(message, "tool_calls", None) or [])
    if len(requested) == 1:
        raw_id = requested[0].get("id") if isinstance(requested[0], dict) else None
        if isinstance(raw_id, str) and raw_id:
            return raw_id
    return None


def _resolve_from_messages(
    final_state: dict[str, Any],
    messages_key: str | None,
    identifier: str,
) -> Any:
    """Find the ToolMessage answering ``identifier`` in the final state."""
    if messages_key is None:
        return None
    for message in _messages_of(final_state, messages_key):
        if getattr(message, "tool_call_id", None) == identifier:
            return message
    return None


def _merge_result(call: ToolCall, resolved: Any, step_index: int) -> ToolCall:
    """Attach a tool result (or its error) to the request that produced it."""
    if resolved is None:
        return call.model_copy(update={"step_index": step_index})
    return call.model_copy(
        update={
            "step_index": step_index,
            "result": _content_of(resolved),
            "error": _error_of(resolved),
        }
    )


def _with_step_index(call: ToolCall, step_index: int) -> ToolCall:
    """Return ``call`` positioned at ``step_index``."""
    return call.model_copy(update={"step_index": step_index})


def _tool_name_of(raw: Any) -> str:
    """Read the tool name from a tool-call request."""
    name = raw.get("name") if isinstance(raw, dict) else getattr(raw, "name", None)
    return str(name or "unknown_tool")


def _tool_args_of(raw: Any) -> Any:
    """Read the tool arguments from a tool-call request."""
    arguments = raw.get("args") if isinstance(raw, dict) else getattr(raw, "args", None)
    return arguments if arguments is not None else {}


def _tool_call_from_request(raw: Any) -> ToolCall:
    """Build an unresolved :class:`ToolCall` from a tool-call request."""
    arguments = _tool_args_of(raw)
    return ToolCall(
        name=_tool_name_of(raw),
        arguments=arguments if isinstance(arguments, dict) else {},
    )


def _as_text(value: Any) -> str:
    """Render a state value as comparable text."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _as_int(value: Any) -> int:
    """Coerce a usage counter to a non-negative int."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(0, int(value))


def _elapsed_ms(started: datetime, ended: datetime) -> float:
    """Milliseconds between two timestamps, never negative."""
    return max(0.0, (ended - started).total_seconds() * 1000.0)
