"""Ports (protocols) defining every boundary of the harness.

Nothing in this package imports an adapter. The core depends on these
protocols; adapters implement them.
"""

from agent_eval.ports.agent import AgentAdapter
from agent_eval.ports.clock import Clock, SystemClock
from agent_eval.ports.judge import Judge
from agent_eval.ports.llm import (
    LLMOptions,
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    ProviderHealth,
    ProviderState,
)
from agent_eval.ports.repository import EvaluationRepository

__all__ = [
    "AgentAdapter",
    "Clock",
    "EvaluationRepository",
    "Judge",
    "LLMOptions",
    "LLMProvider",
    "LLMProviderError",
    "LLMResponse",
    "ProviderHealth",
    "ProviderState",
    "SystemClock",
]
