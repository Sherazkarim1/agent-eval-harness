"""LLM provider adapters.

Only the offline provider is exported eagerly. :class:`OllamaProvider` lives in
:mod:`agent_eval.providers.ollama` and is imported lazily by callers, because it
depends on the optional ``ollama`` extra.
"""

from agent_eval.providers.base import BaseProvider
from agent_eval.providers.mock import MockProvider

__all__ = ["BaseProvider", "MockProvider"]
