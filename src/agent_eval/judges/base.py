"""Shared judge logic: prompt rendering, provider call, safe parsing.

Any judge subclass only has to provide a provider. Error handling lives here so
that *every* judge degrades the same way: a provider failure becomes a verdict
with ``error`` set, never an exception that would abort a run.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from agent_eval.domain.models import AgentResult, EvalCase, JudgeVerdict, TokenUsage
from agent_eval.domain.statistics import mean
from agent_eval.judges.schema import extract_json_object, parse_judge_response, render_prompt
from agent_eval.ports.clock import Clock, SystemClock
from agent_eval.ports.llm import LLMOptions, LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)


class BaseJudge:
    """Turns a provider into a :class:`Judge`."""

    name = "base-judge"
    version = "0.1.0"

    def __init__(
        self,
        provider: LLMProvider,
        *,
        criteria: list[str] | None = None,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        prompt_template: str | None = None,
        schema: type[BaseModel] | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._provider = provider
        self._criteria = list(criteria or ["correctness", "relevance", "completeness"])
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._prompt_template = prompt_template
        self._schema = schema
        self._clock = clock or SystemClock()

    @property
    def model(self) -> str:
        """Model identifier of the backing provider."""
        return self._provider.model

    async def evaluate(self, case: EvalCase, result: AgentResult) -> JudgeVerdict:
        """Judge one case, never raising."""
        prompt = render_prompt(
            case_input=case.input,
            agent_output=result.output,
            reference_answer=case.reference_answer,
            tool_calls=_render_tool_calls(result),
            criteria=self._criteria,
            template=self._prompt_template,
        )

        started = self._clock.monotonic()
        try:
            response = await self._call(prompt)
        except LLMProviderError as exc:
            return self._error(exc, started)
        except Exception as exc:
            logger.warning("Judge %s failed unexpectedly: %s", self.name, exc)
            return self._error(exc, started)

        latency_ms = (self._clock.monotonic() - started) * 1000.0
        try:
            score, reason, criteria = parse_judge_response(response.text, self._criteria)
        except ValueError as exc:
            return JudgeVerdict(
                error=f"malformed judge output: {exc}",
                provider=self._provider.name,
                model=self._provider.model,
                latency_ms=latency_ms,
            )

        return JudgeVerdict(
            score=score,
            reason=reason,
            criteria=criteria,
            provider=self._provider.name,
            model=self._provider.model,
            latency_ms=latency_ms,
        )

    async def _call(self, prompt: str) -> Any:
        """Call the provider with retries for transient failures."""
        options = LLMOptions(temperature=self._temperature, timeout_s=self._timeout_s, seed=0)
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._provider.generate(prompt, schema=self._schema, options=options)
            except LLMProviderError as exc:
                last_error = exc
                if attempt < self._max_retries:
                    logger.debug("Judge provider retry %d/%d", attempt + 1, self._max_retries)
        raise last_error if last_error else LLMProviderError("judge provider call failed")

    async def aclose(self) -> None:
        """Release the backing provider's resources."""
        await self._provider.aclose()

    def _error(self, exc: Exception, started: float) -> JudgeVerdict:
        """Build a verdict describing a provider failure."""
        return JudgeVerdict(
            error=f"{type(exc).__name__}: {exc}",
            provider=self._provider.name,
            model=self._provider.model,
            latency_ms=(self._clock.monotonic() - started) * 1000.0,
        )


def _render_tool_calls(result: AgentResult) -> str:
    """Render tool calls for the judge prompt."""
    if not result.tool_calls:
        return "(no tool calls)"
    return ", ".join(f"{call.name}({_short_args(call.arguments)})" for call in result.tool_calls)


def _short_args(arguments: dict[str, Any]) -> str:
    """Render tool arguments compactly."""
    parts = [f"{key}={value!r}" for key, value in list(arguments.items())[:5]]
    return ", ".join(parts) if parts else ""


def heuristic_verdict(
    case: EvalCase, result: AgentResult, criteria: list[str] | None = None
) -> JudgeVerdict:
    """Score an answer without any model, using deterministic signals only.

    This is the offline fallback judge. It is deliberately explainable: the
    score comes from keyword overlap with the reference answer, length sanity
    and absence of obvious failure markers. It is *not* a language model and
    must never be presented as one.
    """
    names = list(criteria or ["correctness", "relevance", "completeness"])
    if not result.ok:
        return JudgeVerdict(
            score=0.0,
            reason=f"Agent reported an error: {result.error}",
            criteria=dict.fromkeys(names, 0.0),
            provider="heuristic",
            model="heuristic-v1",
        )

    output = result.output.casefold()
    reference = (case.reference_answer or "").casefold()
    if reference:
        overlap = _overlap(output, reference)
        correctness = min(1.0, overlap)
        completeness = min(1.0, overlap)
    else:
        correctness = 1.0 if output.strip() else 0.0
        completeness = correctness

    relevance = 1.0 if _keyword_overlap(output, case.input.casefold()) > 0 else 0.6

    scores = {
        "correctness": round(correctness, 4),
        "relevance": round(relevance, 4),
        "completeness": round(completeness, 4),
    }
    overall = mean([scores.get(name, correctness) for name in names]) or correctness

    return JudgeVerdict(
        score=round(overall, 4),
        reason=_heuristic_reason(scores, has_reference=bool(reference)),
        criteria=scores,
        provider="heuristic",
        model="heuristic-v1",
    )


def _overlap(output: str, reference: str) -> float:
    """Fraction of reference content words present in the output."""
    ref_words = set(reference.split())
    if not ref_words:
        return 0.0
    out_words = set(output.split())
    return len(ref_words & out_words) / len(ref_words)


def _keyword_overlap(output: str, request: str) -> int:
    """Count of request keywords present in the output."""
    keywords = {word for word in request.split() if len(word) > 3}
    return sum(1 for word in keywords if word in output)


def _heuristic_reason(scores: dict[str, float], *, has_reference: bool) -> str:
    """Explain a heuristic score in one sentence."""
    basis = (
        "keyword overlap with the reference answer"
        if has_reference
        else "output presence (no reference answer supplied)"
    )
    parts = ", ".join(f"{name}={value:.2f}" for name, value in scores.items())
    return f"Heuristic judge ({basis}): {parts}."


def verdict_usage(result: AgentResult) -> TokenUsage:
    """Token usage attributed to a verdict (always the agent's usage)."""
    return result.token_usage or TokenUsage()


def judge_payload(value: str) -> dict[str, Any] | None:
    """Expose JSON extraction so alternative judges can reuse it."""
    return extract_json_object(value)
