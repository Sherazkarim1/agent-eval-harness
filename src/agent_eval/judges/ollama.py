"""LLM judge backed by a locally running Ollama server.

This is the default *live* judge: it needs no API key and no cloud account, so
the harness can produce real model judgements on a laptop with no network.

Judging never aborts a run. Every failure mode — server down, model missing,
request timed out, unusable model output — becomes a :class:`JudgeVerdict` with
``error`` set, because raising would discard an otherwise complete evaluation
because a judge hiccupped.

The provider is imported lazily so that installing without the ``ollama`` extra
still works for mock runs.
"""

from __future__ import annotations

from typing import Any

from agent_eval.judges.mock import ProviderJudge
from agent_eval.judges.schema import JudgeResponse
from agent_eval.ports.clock import Clock
from agent_eval.ports.llm import LLMProvider, LLMProviderError, ProviderHealth

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


class OllamaJudge(ProviderJudge):
    """Judge that grades each case with a local Ollama model."""

    name = "ollama-judge"
    version = "1.0.0"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        criteria: list[str] | None = None,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        prompt_template: str | None = None,
        keep_alive: str | None = None,
        num_ctx: int | None = None,
        clock: Clock | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        """Create the judge.

        Args:
            model: Ollama model tag, e.g. ``llama3.2``.
            base_url: Ollama server root, without the ``/api`` suffix.
            criteria: Criterion names to score.
            temperature: Sampling temperature. ``0.0`` keeps grading stable.
            timeout_s: Per-request timeout, also applied to the HTTP client.
            max_retries: Extra attempts after a provider failure.
            prompt_template: Optional custom prompt template.
            keep_alive: How long Ollama keeps the model resident, e.g. ``"5m"``.
            num_ctx: Context window size passed to the model.
            clock: Clock used to measure judge latency.
            provider: Pre-built provider, mainly for tests. When omitted a real
                :class:`~agent_eval.providers.ollama.OllamaProvider` is created.

        Raises:
            LLMProviderError: If the ``ollama`` extra is not installed.
        """
        super().__init__(
            provider
            if provider is not None
            else _build_provider(
                model=model,
                base_url=base_url,
                timeout_s=timeout_s,
                keep_alive=keep_alive,
                num_ctx=num_ctx,
                clock=clock,
            ),
            criteria=criteria,
            temperature=temperature,
            timeout_s=timeout_s,
            max_retries=max_retries,
            prompt_template=prompt_template,
            schema=JudgeResponse,
            clock=clock,
        )
        self.name = f"{self._provider.name}-judge"

    async def health_check(self) -> ProviderHealth:
        """Report whether this judge can grade right now.

        Returns:
            The backing provider's health; never raises.
        """
        return await self._provider.health_check()


def _build_provider(
    *,
    model: str,
    base_url: str,
    timeout_s: float,
    keep_alive: str | None,
    num_ctx: int | None,
    clock: Clock | None,
) -> Any:
    """Construct the Ollama provider, reporting a missing extra clearly."""
    try:
        from agent_eval.providers.ollama import OllamaProvider
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise LLMProviderError(
            "The ollama judge needs the 'ollama' extra: pip install agent-eval[ollama]"
        ) from exc
    return OllamaProvider(
        model,
        base_url=base_url,
        timeout_s=timeout_s,
        keep_alive=keep_alive,
        num_ctx=num_ctx,
        clock=clock,
    )
