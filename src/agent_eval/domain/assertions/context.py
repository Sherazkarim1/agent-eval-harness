"""Shared input for every assertion function."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agent_eval.domain.models import AgentResult, TokenUsage

#: Text previews are truncated to this length in assertion messages.
PREVIEW_LIMIT = 240


def preview(value: object, limit: int = PREVIEW_LIMIT) -> str:
    """Return a short, safe string representation of ``value``."""
    if value is None:
        return "<none>"
    text = value if isinstance(value, str) else json.dumps(value, default=str, sort_keys=True)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…"


@dataclass(frozen=True, slots=True)
class AssertionContext:
    """Everything an assertion may inspect.

    ``output`` is the redacted agent output, ``result`` is the redacted
    :class:`AgentResult`, and ``latency_ms`` / ``cost_usd`` are measured by the
    harness rather than trusted from the adapter.
    """

    output: str
    result: AgentResult
    latency_ms: float
    cost_usd: float
    _parsed_json: Any = field(default=None, compare=False)
    _json_parsed: bool = field(default=False, compare=False)
    _json_error: str | None = field(default=None, compare=False)

    @property
    def parsed_json(self) -> Any:
        """The output parsed as JSON, or ``None`` when it is not valid JSON."""
        if not self._json_parsed:
            object.__setattr__(self, "_parsed_json", self._parse())
            object.__setattr__(self, "_json_parsed", True)
        return self._parsed_json

    @property
    def json_error(self) -> str | None:
        """Why the output could not be parsed as JSON, or ``None`` if it could."""
        if not self._json_parsed:
            self.parsed_json  # noqa: B018 - triggers parsing
        return self._json_error

    @property
    def tokens(self) -> TokenUsage:
        """Token usage of the run, defaulting to zero when the agent reports none."""
        return self.result.token_usage or TokenUsage()

    def with_json(self, value: Any, error: str | None) -> AssertionContext:
        """Return a copy carrying a pre-computed JSON parse result."""
        return AssertionContext(
            output=self.output,
            result=self.result,
            latency_ms=self.latency_ms,
            cost_usd=self.cost_usd,
            _parsed_json=value,
            _json_parsed=True,
            _json_error=error,
        )

    def _parse(self) -> Any:
        """Parse ``output`` as JSON, recording any error."""
        try:
            return json.loads(self.output)
        except (json.JSONDecodeError, ValueError) as exc:
            object.__setattr__(self, "_json_error", str(exc))
            return None
