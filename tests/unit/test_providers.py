"""Tests for the LLM provider layer: offline mock and the Ollama backend.

The Ollama tests never open a socket. A fake client implements the small slice
of ``httpx.AsyncClient`` the provider uses (``post``/``get``/``aclose``) and
records the exact payloads sent, so request shape, option mapping and error
translation are all asserted without a server.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_eval.domain.models import TokenUsage
from agent_eval.judges.schema import JudgeResponse
from agent_eval.ports.llm import (
    LLMOptions,
    LLMProvider,
    LLMProviderError,
    ProviderState,
)
from agent_eval.providers.base import BaseProvider
from agent_eval.providers.mock import MOCK_REASON, MockProvider
from agent_eval.providers.ollama import DEFAULT_BASE_URL, OllamaProvider

BASE_URL = "http://ollama.test:11434"


class FakeResponse:
    """Minimal stand-in for ``httpx.Response``."""

    def __init__(
        self,
        payload: Any,
        *,
        status_error: str | None = None,
        raise_json: bool = False,
    ) -> None:
        self._payload = payload
        self._status_error = status_error
        self._raise_json = raise_json

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise RuntimeError(self._status_error)

    def json(self) -> Any:
        if self._raise_json:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


class FakeClient:
    """Records requests and replays queued responses."""

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.responses = responses or {}
        self.posts: list[tuple[str, dict[str, Any], Any]] = []
        self.gets: list[tuple[str, Any]] = []
        self.closed = False

    async def post(self, url: str, *, json: Any, timeout: Any) -> FakeResponse:
        self.posts.append((url, json, timeout))
        return _as_response(self.responses.get(url, {}))

    async def get(self, url: str, *, timeout: Any) -> FakeResponse:
        self.gets.append((url, timeout))
        return _as_response(self.responses.get(url, {}))

    async def aclose(self) -> None:
        self.closed = True


def _as_response(value: Any) -> FakeResponse:
    if isinstance(value, FakeResponse):
        return value
    if isinstance(value, BaseException):
        raise value
    return FakeResponse(value)


def server(models: list[str] | None = None) -> FakeClient:
    """A fake Ollama server listing ``models`` and answering one generate call."""
    return FakeClient(
        {
            f"{BASE_URL}/api/version": {"version": "0.5.7"},
            f"{BASE_URL}/api/tags": {"models": [{"name": name} for name in models or []]},
            f"{BASE_URL}/api/generate": {
                "model": "llama3.2",
                "response": '{"score": 0.5, "reason": "ok", "criteria": {}}',
                "prompt_eval_count": 41,
                "eval_count": 12,
            },
        }
    )


def provider(client: Any, **kwargs: Any) -> OllamaProvider:
    return OllamaProvider("llama3.2", base_url=BASE_URL, client=client, **kwargs)


class TestMockProvider:
    async def test_satisfies_provider_protocol(self) -> None:
        assert isinstance(MockProvider(), LLMProvider)

    async def test_returns_parseable_judge_response(self) -> None:
        response = await MockProvider(score=0.9).generate("prompt")

        payload = JudgeResponse.model_validate_json(response.text)
        assert payload.score == 0.9
        assert payload.reason == MOCK_REASON

    async def test_labels_every_verdict_as_simulated(self) -> None:
        # Nothing here graded the prompt, so the reason must never read like a
        # real judgement.
        response = await MockProvider().generate("Grade this terrible answer.")
        assert "no model was called" in response.text
        assert response.provider == "mock-provider"

    async def test_is_deterministic(self) -> None:
        first = await MockProvider().generate("prompt")
        second = await MockProvider().generate("prompt")
        assert first.text == second.text

    async def test_clamps_score_into_range(self) -> None:
        low = await MockProvider(score=-1.0).generate("prompt")
        high = await MockProvider(score=4.0).generate("prompt")
        assert JudgeResponse.model_validate_json(low.text).score == 0.0
        assert JudgeResponse.model_validate_json(high.text).score == 1.0

    async def test_reports_reported_latency(self) -> None:
        response = await MockProvider(latency_ms=12.5).generate("prompt")
        assert response.latency_ms == 12.5

    async def test_reports_usage_derived_from_text(self) -> None:
        response = await MockProvider(text="b" * 20).generate("a" * 40)
        assert response.usage == TokenUsage(input_tokens=10, output_tokens=5)

    async def test_usage_is_never_zero_for_non_empty_text(self) -> None:
        response = await MockProvider().generate("hi")
        assert response.usage.input_tokens >= 1
        assert response.usage.output_tokens >= 1

    async def test_injects_fixed_text(self) -> None:
        response = await MockProvider(text="not json at all").generate("prompt")
        assert response.text == "not json at all"

    async def test_raises_configured_failure(self) -> None:
        with pytest.raises(LLMProviderError, match="server on fire"):
            await MockProvider(fail_with="server on fire").generate("prompt")

    async def test_health_is_always_ok(self) -> None:
        health = await MockProvider().health_check()
        assert health.state is ProviderState.OK
        assert health.models == ["mock-model"]

    async def test_never_touches_the_network(self) -> None:
        # No client can be injected, so reaching a socket would raise.
        await MockProvider().generate("prompt")

    async def test_model_can_be_reassigned(self) -> None:
        backend = MockProvider()
        backend.model = "other"
        assert (await backend.generate("prompt")).model == "other"


class TestOllamaProviderGenerate:
    async def test_satisfies_provider_protocol(self) -> None:
        assert isinstance(provider(server()), LLMProvider)

    async def test_returns_text_and_converts_usage(self) -> None:
        response = await provider(server(["llama3.2"])).generate("prompt")

        assert response.text == '{"score": 0.5, "reason": "ok", "criteria": {}}'
        assert response.usage == TokenUsage(input_tokens=41, output_tokens=12)
        assert response.provider == "ollama"
        assert response.raw is not None
        assert response.raw["prompt_eval_count"] == 41

    async def test_posts_to_generate_with_stream_disabled(self) -> None:
        client = server()
        await provider(client).generate("prompt")

        url, payload, _timeout = client.posts[0]
        assert url == f"{BASE_URL}/api/generate"
        assert payload["model"] == "llama3.2"
        assert payload["prompt"] == "prompt"
        assert payload["stream"] is False
        assert "format" not in payload
        assert "options" not in payload

    async def test_omits_options_when_none_are_set(self) -> None:
        client = server()
        await provider(client).generate("prompt", options=LLMOptions())
        assert client.posts[0][1].get("options", {}) == {}

    async def test_translates_options_to_ollama_names(self) -> None:
        client = server()
        await provider(client).generate(
            "prompt",
            options=LLMOptions(
                temperature=0.25,
                top_p=0.9,
                max_tokens=256,
                seed=7,
                stop=["</s>"],
                extra={"mirostat": 2},
            ),
        )
        options = client.posts[0][1]["options"]
        assert options == {
            "temperature": 0.25,
            "top_p": 0.9,
            "num_predict": 256,
            "seed": 7,
            "stop": ["</s>"],
            "mirostat": 2,
        }

    async def test_includes_num_ctx_from_constructor(self) -> None:
        client = server()
        await provider(client, num_ctx=8192).generate("prompt")
        assert client.posts[0][1]["options"] == {"num_ctx": 8192}

    async def test_includes_keep_alive_when_configured(self) -> None:
        client = server()
        await provider(client, keep_alive="5m").generate("prompt")
        assert client.posts[0][1]["keep_alive"] == "5m"

    async def test_passes_schema_as_json_format(self) -> None:
        client = server()
        await provider(client).generate("prompt", schema=JudgeResponse)

        sent = client.posts[0][1]["format"]
        assert sent["title"] == "JudgeResponse"
        assert set(sent["properties"]) == {"score", "reason", "criteria"}

    async def test_schema_format_omits_docstrings(self) -> None:
        client = server()
        await provider(client).generate("prompt", schema=JudgeResponse)
        assert "description" not in json.dumps(client.posts[0][1]["format"])

    async def test_uses_provider_timeout_by_default(self) -> None:
        client = server()
        await provider(client, timeout_s=5.0).generate("prompt")
        assert client.posts[0][2] == 5.0

    async def test_option_timeout_overrides_provider_timeout(self) -> None:
        client = server()
        await provider(client, timeout_s=5.0).generate("prompt", options=LLMOptions(timeout_s=1.5))
        assert client.posts[0][2] == 1.5

    async def test_measures_latency_through_the_clock(self) -> None:
        ticks = iter([10.0, 10.25])
        client = server()
        backend = provider(client, clock=_Clock(ticks))
        response = await backend.generate("prompt")
        assert response.latency_ms == pytest.approx(250.0)

    async def test_missing_response_field_raises(self) -> None:
        client = FakeClient(
            {f"{BASE_URL}/api/generate": {"model": "llama3.2"}},
        )
        with pytest.raises(LLMProviderError, match="did not contain a string 'response'"):
            await provider(client).generate("prompt")

    async def test_non_dict_payload_raises(self) -> None:
        client = FakeClient({f"{BASE_URL}/api/generate": ["nope"]})
        with pytest.raises(LLMProviderError, match="returned list"):
            await provider(client).generate("prompt")

    async def test_transport_error_becomes_provider_error(self) -> None:
        client = FakeClient({f"{BASE_URL}/api/generate": ConnectionRefusedError("no route")})
        with pytest.raises(LLMProviderError, match="POST /api/generate failed"):
            await provider(client).generate("prompt")

    async def test_http_error_becomes_provider_error(self) -> None:
        client = FakeClient(
            {f"{BASE_URL}/api/generate": FakeResponse({}, status_error="503 Server Error")},
        )
        with pytest.raises(LLMProviderError, match="503"):
            await provider(client).generate("prompt")

    async def test_invalid_json_becomes_provider_error(self) -> None:
        client = FakeClient(
            {f"{BASE_URL}/api/generate": FakeResponse(None, raise_json=True)},
        )
        with pytest.raises(LLMProviderError, match="Expecting value"):
            await provider(client).generate("prompt")

    async def test_missing_token_counters_default_to_zero(self) -> None:
        client = FakeClient(
            {f"{BASE_URL}/api/generate": {"response": "hi", "eval_count": -3, "x": True}},
        )
        response = await provider(client).generate("prompt")
        assert response.usage == TokenUsage(input_tokens=0, output_tokens=0)

    async def test_model_falls_back_to_configured_name(self) -> None:
        client = FakeClient({f"{BASE_URL}/api/generate": {"response": "hi"}})
        response = await provider(client).generate("prompt")
        assert response.model == "llama3.2"


class TestOllamaProviderHealth:
    async def test_ok_when_model_installed(self) -> None:
        health = await provider(server(["llama3.2"])).health_check()
        assert health.state is ProviderState.OK
        assert health.models == ["llama3.2"]
        assert "0.5.7" in (health.detail or "")

    async def test_matches_explicit_tag(self) -> None:
        health = await OllamaProvider(
            "llama3.2:3b", base_url=BASE_URL, client=server(["llama3.2:3b"])
        ).health_check()
        assert health.state is ProviderState.OK

    async def test_matches_implicit_tag(self) -> None:
        # `llama3.2` should resolve to the installed `llama3.2:3b`.
        health = await provider(server(["llama3.2:3b"])).health_check()
        assert health.state is ProviderState.OK

    async def test_unavailable_when_server_is_down(self) -> None:
        client = FakeClient(
            {f"{BASE_URL}/api/version": ConnectionRefusedError("connection refused")}
        )
        health = await provider(client).health_check()

        assert health.state is ProviderState.UNAVAILABLE
        assert "ollama serve" in (health.detail or "")
        assert "ollama pull" in (health.detail or "")
        assert health.models == ["llama3.2"]

    async def test_unavailable_when_version_endpoint_errors(self) -> None:
        client = FakeClient({f"{BASE_URL}/api/version": FakeResponse({}, status_error="500")})
        health = await provider(client).health_check()
        assert health.state is ProviderState.UNAVAILABLE

    async def test_misconfigured_when_model_missing(self) -> None:
        health = await provider(server(["qwen2.5"])).health_check()

        assert health.state is ProviderState.MISCONFIGURED
        assert "ollama pull llama3.2" in (health.detail or "")
        assert health.models == ["qwen2.5"]

    async def test_ok_when_server_reports_no_models(self) -> None:
        # Some proxies answer /api/tags with an empty list; that is not proof
        # that the model is missing, so it must not be called misconfigured.
        health = await provider(server()).health_check()
        assert health.state is ProviderState.OK

    async def test_unavailable_when_listing_models_fails(self) -> None:
        client = server(["llama3.2"])
        client.responses[f"{BASE_URL}/api/tags"] = RuntimeError("boom")
        health = await provider(client).health_check()
        assert health.state is ProviderState.UNAVAILABLE


class TestOllamaProviderLifecycle:
    async def test_does_not_close_an_injected_client(self) -> None:
        client = server()
        await provider(client).aclose()
        assert client.closed is False

    async def test_closes_a_client_it_created(self) -> None:
        client = server()
        backend = OllamaProvider("llama3.2", base_url=BASE_URL)
        backend._client = client
        backend._owns_client = True

        await backend.aclose()
        assert client.closed is True
        assert backend._client is None

    async def test_close_is_idempotent(self) -> None:
        client = server()
        backend = OllamaProvider("llama3.2", base_url=BASE_URL)
        backend._client = client
        backend._owns_client = True

        await backend.aclose()
        await backend.aclose()
        assert client.closed is True

    async def test_creates_a_real_client_lazily(self) -> None:
        backend = OllamaProvider("llama3.2", base_url=DEFAULT_BASE_URL)
        assert backend._client is None
        await backend.aclose()

    async def test_uses_installed_httpx(self) -> None:
        client = server()
        backend = provider(client)
        assert backend._ensure_client() is client


class TestBaseProvider:
    async def test_generate_must_be_overridden(self) -> None:
        with pytest.raises(NotImplementedError):
            await BaseProvider("m").generate("prompt")

    async def test_default_health_is_unavailable(self) -> None:
        health = await BaseProvider("m").health_check()
        assert health.state is ProviderState.UNAVAILABLE
        assert health.models == ["m"]

    async def test_aclose_is_a_no_op(self) -> None:
        assert await BaseProvider("m").aclose() is None

    async def test_elapsed_is_never_negative(self) -> None:
        ticks = iter([10.0, 9.0])
        assert BaseProvider("m", clock=_Clock(ticks))._elapsed_ms(10.0) == 0.0


class _Clock:
    """Clock returning a scripted sequence of monotonic readings."""

    def __init__(self, ticks: Any) -> None:
        self._ticks = iter(ticks)

    def now(self) -> Any:
        raise NotImplementedError

    def monotonic(self) -> float:
        return next(self._ticks)
