"""Shared provider behaviour: health reporting and resource cleanup.

Subclasses only implement :meth:`generate`; everything a provider inherits from
here is bookkeeping that would otherwise be repeated per backend.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from agent_eval.ports.clock import Clock, SystemClock
from agent_eval.ports.llm import (
    LLMOptions,
    LLMResponse,
    ProviderHealth,
    ProviderState,
)

logger = logging.getLogger(__name__)


class BaseProvider:
    """Common plumbing for LLM providers."""

    name = "base-provider"

    def __init__(
        self,
        model: str,
        *,
        clock: Clock | None = None,
    ) -> None:
        """Create a provider bound to ``model``.

        Args:
            model: Model identifier passed to the backend.
            clock: Clock used to measure provider latency.
        """
        self._model = model
        self._clock = clock or SystemClock()

    @property
    def model(self) -> str:
        """Model identifier this provider calls."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Point the provider at a different model."""
        self._model = value

    async def generate(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
        options: LLMOptions | None = None,
    ) -> LLMResponse:
        """Generate a completion.

        Raises:
            NotImplementedError: Subclasses must override this.
        """
        raise NotImplementedError

    async def health_check(self) -> ProviderHealth:
        """Report provider health.

        The base implementation treats a provider with no transport as
        ``unavailable`` rather than raising, so callers can always ask.
        """
        return ProviderHealth(
            provider=self.name,
            state=ProviderState.UNAVAILABLE,
            detail=f"{self.name} does not implement a health check.",
            models=[self._model],
        )

    async def aclose(self) -> None:
        """Release provider resources."""
        return None

    def _elapsed_ms(self, started: float) -> float:
        """Milliseconds since ``started``."""
        return max(0.0, (self._clock.monotonic() - started) * 1000.0)
