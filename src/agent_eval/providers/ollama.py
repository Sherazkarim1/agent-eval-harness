"""Ollama provider: text generation against a locally running Ollama server.

Ollama is the default *live* judge backend because it needs no API key and no
cloud account, which keeps the promise that the harness is usable offline.

This module imports ``httpx`` lazily so that the core stays dependency-free; the
``ollama`` extra installs it. Every failure is converted into
:class:`LLMProviderError`, because the judge path turns provider errors into a
verdict with ``error`` set rather than aborting a run.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from agent_eval.domain.models import TokenUsage
from agent_eval.ports.clock import Clock
from agent_eval.ports.llm import (
    LLMOptions,
    LLMProviderError,
    LLMResponse,
    ProviderHealth,
    ProviderState,
)
from agent_eval.providers.base import BaseProvider

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
GENERATE_PATH = "/api/generate"
TAGS_PATH = "/api/tags"
VERSION_PATH = "/api/version"


class OllamaProvider(BaseProvider):
    """Talks to a local Ollama server over HTTP."""

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.2",
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 120.0,
        keep_alive: str | None = None,
        num_ctx: int | None = None,
        clock: Clock | None = None,
        client: Any | None = None,
    ) -> None:
        """Create the provider.

        Args:
            model: Model tag, e.g. ``llama3.2``.
            base_url: Ollama server root, without the ``/api`` suffix.
            timeout_s: HTTP timeout for a single request.
            keep_alive: How long the model stays resident, e.g. ``"5m"``.
            num_ctx: Context window size passed to the model.
            clock: Clock used to measure latency.
            client: Pre-built HTTP client, mainly for tests. One is created on
                first use when omitted.
        """
        super().__init__(model, clock=clock)
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._keep_alive = keep_alive
        self._num_ctx = num_ctx
        self._client = client
        self._owns_client = client is None

    async def generate(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,
        options: LLMOptions | None = None,
    ) -> LLMResponse:
        """Generate a completion with ``/api/generate``.

        Raises:
            LLMProviderError: If the server is unreachable, errors, or answers
                with an unusable payload.
        """
        payload = self._build_payload(prompt, schema=schema, options=options)
        started = self._clock.monotonic()
        data = await self._post_json(
            GENERATE_PATH, payload, timeout_s=options.timeout_s if options else None
        )
        latency_ms = self._elapsed_ms(started)

        text = data.get("response")
        if not isinstance(text, str):
            raise LLMProviderError("Ollama response did not contain a string 'response' field.")
        return LLMResponse(
            text=text,
            model=str(data.get("model") or self.model),
            provider=self.name,
            usage=_usage_of(data),
            latency_ms=latency_ms,
            raw=data,
        )

    async def health_check(self) -> ProviderHealth:
        """Report whether the server is up and the model is available.

        This never raises: an unreachable server is reported as
        ``unavailable`` so the CLI can print actionable guidance.
        """
        try:
            version = await self._get_json(VERSION_PATH)
        except LLMProviderError as exc:
            return ProviderHealth(
                provider=self.name,
                state=ProviderState.UNAVAILABLE,
                detail=_unavailable_detail(self._base_url, exc),
                models=[self.model],
            )

        detail: str | None = None
        state = ProviderState.OK
        models: list[str] = []
        try:
            models = await self._list_models()
        except LLMProviderError as exc:
            state = ProviderState.UNAVAILABLE
            detail = f"Could not list models: {exc}"

        if state is ProviderState.OK and models and not _matches(models, self.model):
            state = ProviderState.MISCONFIGURED
            detail = (
                f"Model {self.model!r} is not installed. "
                f"Run: ollama pull {self.model}. Installed: {', '.join(models)}."
            )

        return ProviderHealth(
            provider=self.name,
            state=state,
            detail=detail
            or f"Ollama {' '.join(str(version.get('version', 'unknown')).split())} reachable.",
            models=models,
        )

    async def aclose(self) -> None:
        """Close the HTTP client, if this provider created it."""
        if not self._owns_client:
            return None
        client = self._client
        self._client = None
        if client is None:
            return None
        closer = getattr(client, "aclose", None)
        if callable(closer):
            await closer()
        return None

    def _build_payload(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None,
        options: LLMOptions | None,
    ) -> dict[str, Any]:
        """Assemble the ``/api/generate`` request body."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive

        ollama_options = self._ollama_options(options)
        if schema is not None:
            payload["format"] = _json_schema(schema)
        if ollama_options:
            payload["options"] = ollama_options
        return payload

    def _ollama_options(self, options: LLMOptions | None) -> dict[str, Any]:
        """Translate provider-agnostic options into Ollama's ``options``."""
        merged: dict[str, Any] = {}
        if self._num_ctx is not None:
            merged["num_ctx"] = self._num_ctx
        if options is None:
            return merged

        if options.temperature is not None:
            merged["temperature"] = options.temperature
        if options.top_p is not None:
            merged["top_p"] = options.top_p
        if options.max_tokens is not None:
            merged["num_predict"] = options.max_tokens
        if options.seed is not None:
            merged["seed"] = options.seed
        if options.stop:
            merged["stop"] = list(options.stop)
        merged.update(options.extra)
        return merged

    async def _list_models(self) -> list[str]:
        """Return the model tags installed on the server."""
        data = await self._get_json(TAGS_PATH)
        models = data.get("models")
        if not isinstance(models, list):
            return []
        tags: list[str] = []
        for entry in models:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                tags.append(entry["name"])
        return tags

    async def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """POST JSON and return the decoded object."""
        client = self._ensure_client()
        try:
            response = await client.post(
                f"{self._base_url}{path}",
                json=payload,
                timeout=self._timeout_s if timeout_s is None else timeout_s,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise LLMProviderError(f"Ollama POST {path} failed: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMProviderError(f"Ollama POST {path} returned {type(data).__name__}.")
        return data

    async def _get_json(self, path: str) -> dict[str, Any]:
        """GET JSON and return the decoded object."""
        client = self._ensure_client()
        try:
            response = await client.get(f"{self._base_url}{path}", timeout=self._timeout_s)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise LLMProviderError(f"Ollama GET {path} failed: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMProviderError(f"Ollama GET {path} returned {type(data).__name__}.")
        return data

    def _ensure_client(self) -> Any:
        """Return the HTTP client, creating it on first use."""
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise LLMProviderError(
                    "The ollama provider needs the 'ollama' extra: pip install agent-eval[ollama]"
                ) from exc
            self._client = httpx.AsyncClient()
        return self._client


def _json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Render ``schema`` as a grammar for Ollama's ``format`` field.

    Prose is stripped: a model only needs field names and types, and docstrings
    would be billed as prompt tokens on every judged case.
    """
    stripped = _strip_descriptions(schema.model_json_schema())
    return stripped if isinstance(stripped, dict) else {}


def _strip_descriptions(node: Any) -> Any:
    """Recursively drop ``description`` keys from a JSON schema."""
    if isinstance(node, dict):
        return {
            key: _strip_descriptions(value) for key, value in node.items() if key != "description"
        }
    if isinstance(node, list):
        return [_strip_descriptions(item) for item in node]
    return node


def _usage_of(data: dict[str, Any]) -> TokenUsage:
    """Convert Ollama's token counters into harness token counts."""
    prompt_tokens = _as_int(data.get("prompt_eval_count"))
    output_tokens = _as_int(data.get("eval_count"))
    return TokenUsage(input_tokens=prompt_tokens, output_tokens=output_tokens)


def _as_int(value: Any) -> int:
    """Coerce a JSON counter to a non-negative int."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(0, int(value))


def _matches(models: list[str], wanted: str) -> bool:
    """Whether ``wanted`` is installed, tolerating implicit tag suffixes."""
    if wanted in models:
        return True
    if ":" not in wanted:
        return any(name.split(":")[0] == wanted for name in models)
    base = wanted.split(":")[0]
    return any(name == base or name.split(":")[0] == base for name in models)


def _unavailable_detail(base_url: str, exc: LLMProviderError) -> str:
    """Explain how to start a server that is not running."""
    return (
        f"Ollama is not reachable at {base_url} ({exc}). "
        "Start it with `ollama serve`, then pull the model with "
        "`ollama pull <model>`."
    )
