"""LLM provider port plus the small value objects it exchanges."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent_eval.domain.models import TokenUsage


class LLMOptions(BaseModel):
    """Provider-agnostic generation options."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float | None = Field(default=None, ge=0.0)
    max_tokens: int | None = Field(default=None, gt=0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    seed: int | None = None
    timeout_s: float | None = Field(default=None, gt=0)
    stop: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """A single completion returned by an :class:`LLMProvider`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    model: str
    provider: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(default=0.0, ge=0.0)
    raw: dict[str, Any] | None = None


class ProviderState(StrEnum):
    """Result of a provider health check."""

    OK = "ok"
    UNAVAILABLE = "unavailable"
    MISCONFIGURED = "misconfigured"


class ProviderHealth(BaseModel):
    """Health of an LLM provider, used for actionable error messages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    state: ProviderState
    detail: str | None = None
    models: list[str] = Field(default_factory=list)


class LLMProviderError(RuntimeError):
    """Raised when a provider cannot fulfil a request.

    Callers in the judge path catch this and degrade to a verdict with
    ``error`` set, so a provider outage never aborts an evaluation run.
    """


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal text-generation interface."""

    name: str
    model: str

    async def generate(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
        options: LLMOptions | None = None,
    ) -> LLMResponse:
        """Generate a completion.

        Args:
            prompt: The fully rendered prompt.
            schema: Optional Pydantic model the response must conform to.
            options: Optional generation overrides.

        Returns:
            An :class:`LLMResponse`.

        Raises:
            LLMProviderError: On any provider-level failure.
        """
        ...

    async def health_check(self) -> ProviderHealth:
        """Report whether the provider is usable right now."""
        ...

    async def aclose(self) -> None:
        """Release provider resources (HTTP clients, subprocesses)."""
        ...
