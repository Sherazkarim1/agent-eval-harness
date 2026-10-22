"""A fully offline LLM provider.

This provider exists so that CI, unit tests and ``agent-eval run --mock`` never
need a model server. It returns a deterministic, valid judge response, so the
whole judging path is exercised without a network.

The response is *recognisably synthetic*: the reason string says so, and the
model identifier carries a ``mock`` prefix. A verdict produced here must never be
mistaken for a real model judgement.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from agent_eval.domain.models import TokenUsage
from agent_eval.ports.llm import LLMOptions, LLMResponse, ProviderHealth, ProviderState
from agent_eval.providers.base import BaseProvider

MOCK_MODEL_PREFIX = "mock"

#: Reason attached to every synthetic verdict.
MOCK_REASON = "Mock provider returned a deterministic verdict; no model was called."


class MockProvider(BaseProvider):
    """Deterministic offline provider."""

    name = "mock-provider"

    def __init__(
        self,
        model: str = "mock-model",
        *,
        score: float = 0.85,
        latency_ms: float = 0.0,
        text: str | None = None,
        fail_with: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the mock provider.

        Args:
            model: Model identifier reported in responses.
            score: Score the synthetic judge response will carry.
            latency_ms: Latency to report, so performance paths stay testable.
            text: Exact text to return instead of a generated verdict. Use this
                to inject malformed output in tests.
            fail_with: When set, :meth:`generate` raises
                :class:`LLMProviderError` with this message.
            **kwargs: Forwarded to :class:`BaseProvider`.
        """
        super().__init__(model, **kwargs)
        self._score = max(0.0, min(1.0, score))
        self._latency_ms = latency_ms
        self._text = text
        self._fail_with = fail_with

    async def generate(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
        options: LLMOptions | None = None,
    ) -> LLMResponse:
        """Return a deterministic response, never touching the network."""
        if self._fail_with is not None:
            from agent_eval.ports.llm import LLMProviderError

            raise LLMProviderError(self._fail_with)

        text = self._text if self._text is not None else self._verdict(prompt)
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.name,
            usage=self._usage(prompt, text),
            latency_ms=self._latency_ms,
        )

    async def health_check(self) -> ProviderHealth:
        """The mock provider is always usable."""
        return ProviderHealth(
            provider=self.name,
            state=ProviderState.OK,
            detail="Mock provider is always available.",
            models=[self.model],
        )

    def _verdict(self, prompt: str) -> str:
        """Build a valid, deterministic judge response.

        The prompt is not interpreted: this provider ran no model, so any
        grading it produced would be invented. It returns the configured score
        with a reason that says so, and the judge layer fills in per-criterion
        scores from the overall score.
        """
        del prompt
        return json.dumps({"score": round(self._score, 4), "reason": MOCK_REASON})

    def _usage(self, prompt: str, text: str) -> TokenUsage:
        """Report a rough token count derived from text length.

        These are *simulated* counts for a provider that ran no model. They keep
        budget assertions exercisable offline and are labelled as mock wherever
        they are reported.
        """
        input_tokens = max(1, len(prompt) // 4)
        output_tokens = max(1, len(text) // 4)
        return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
