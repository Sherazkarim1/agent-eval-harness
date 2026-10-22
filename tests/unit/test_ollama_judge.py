"""Tests for the Ollama judge.

The judge is exercised through a fake HTTP transport rather than a real server,
so the contract that matters — a verdict is produced, a provider failure
degrades into a verdict with ``error`` set, and neither can abort a run — is
verified without pulling a model.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_eval.domain.models import EvalCase
from agent_eval.judges.ollama import DEFAULT_MODEL, OllamaJudge
from agent_eval.ports.judge import Judge
from agent_eval.ports.llm import ProviderState
from agent_eval.providers.ollama import OllamaProvider
from tests.conftest import make_result
from tests.unit.test_providers import FakeClient, FakeResponse

BASE_URL = "http://ollama.test:11434"
GENERATE = f"{BASE_URL}/api/generate"

VERDICT = '{"score": 0.75, "reason": "Answer is correct", "criteria": {"correctness": 0.8}}'


class ScriptedClient(FakeClient):
    """Fake client that serves one queued generate response per request.

    The last queued entry repeats, which is what lets a retry test assert that
    a transient failure is followed by a success.
    """

    def __init__(self, responses: dict[str, Any], generate: list[Any]) -> None:
        super().__init__(responses)
        self._generate = generate
        self._served = 0

    async def post(self, url: str, *, json: Any, timeout: Any) -> FakeResponse:
        self.posts.append((url, json, timeout))
        if url == GENERATE:
            value = self._generate[min(self._served, len(self._generate) - 1)]
            self._served += 1
            return _as_response(value)
        return _as_response(self.responses.get(url, {}))


def _as_response(value: Any) -> FakeResponse:
    if isinstance(value, FakeResponse):
        return value
    if isinstance(value, BaseException):
        raise value
    return FakeResponse(value)


class ScriptedServer:
    """A fake Ollama server replaying one response per generate request."""

    def __init__(self, *generate: Any) -> None:
        queue = list(generate) or [{"response": VERDICT}]
        self.client = ScriptedClient(
            {
                f"{BASE_URL}/api/version": {"version": "0.5.7"},
                f"{BASE_URL}/api/tags": {"models": [{"name": DEFAULT_MODEL}]},
            },
            queue,
        )

    @property
    def posts(self) -> list[tuple[str, dict[str, Any], Any]]:
        return self.client.posts


def judge(server: ScriptedServer, **kwargs: Any) -> OllamaJudge:
    model = kwargs.pop("model", DEFAULT_MODEL)
    return OllamaJudge(
        model,
        base_url=BASE_URL,
        provider=OllamaProvider(model, base_url=BASE_URL, client=server.client),
        **kwargs,
    )


def case() -> EvalCase:
    return EvalCase(id="c1", input="How do I get a refund?", reference_answer="Within 30 days")


def result(output: str = "Refunds are available within 30 days.", error: str | None = None):
    return make_result(output, error=error)


class TestOllamaJudgeResolution:
    def test_satisfies_judge_protocol(self) -> None:
        assert isinstance(OllamaJudge(provider=OllamaProvider("m")), Judge)

    def test_model_reports_the_configured_tag(self) -> None:
        # The provider is what actually talks to the server, so its model wins.
        built = OllamaJudge("llama3.2", provider=OllamaProvider("llama3.2:3b"))
        assert built.model == "llama3.2:3b"

    def test_name_derives_from_provider(self) -> None:
        built = OllamaJudge(provider=OllamaProvider("llama3.2"))
        assert built.name == "ollama-judge"
        assert built.version == "1.0.0"

    def test_configured_model_is_reported(self) -> None:
        built = OllamaJudge("qwen2.5", provider=OllamaProvider("qwen2.5"))
        assert built.model == "qwen2.5"

    def test_creates_a_real_provider_by_default(self) -> None:
        built = OllamaJudge("llama3.2")
        assert isinstance(built._provider, OllamaProvider)
        assert built._provider.model == "llama3.2"


class TestOllamaJudgeEvaluate:
    async def test_parses_a_valid_verdict(self) -> None:
        server = ScriptedServer()
        verdict = await judge(server).evaluate(case(), result())

        assert verdict.ok is True
        assert verdict.score == pytest.approx(0.75)
        assert verdict.reason == "Answer is correct"
        assert verdict.criteria["correctness"] == pytest.approx(0.8)
        assert verdict.provider == "ollama"
        assert verdict.model == DEFAULT_MODEL

    async def test_fills_missing_criteria_from_overall_score(self) -> None:
        server = ScriptedServer({"response": '{"score": 0.4, "reason": "partial", "criteria": {}}'})
        verdict = await judge(server, criteria=["correctness", "tone"]).evaluate(case(), result())
        assert verdict.criteria == {"correctness": pytest.approx(0.4), "tone": pytest.approx(0.4)}

    async def test_clamps_out_of_range_criteria(self) -> None:
        server = ScriptedServer(
            {"response": '{"score": 0.5, "reason": "x", "criteria": {"correctness": 7}}'}
        )
        verdict = await judge(server, criteria=["correctness"]).evaluate(case(), result())
        assert verdict.criteria["correctness"] == pytest.approx(1.0)

    async def test_requests_the_structured_schema(self) -> None:
        server = ScriptedServer()
        await judge(server).evaluate(case(), result())

        payload = server.posts[0][1]
        assert payload["format"]["title"] == "JudgeResponse"
        assert payload["model"] == DEFAULT_MODEL
        assert payload["stream"] is False

    async def test_sends_configured_sampling_options(self) -> None:
        server = ScriptedServer()
        await judge(server, temperature=0.2, timeout_s=7.5).evaluate(case(), result())

        payload = server.posts[0][1]
        assert payload["options"]["temperature"] == pytest.approx(0.2)
        assert payload["options"]["seed"] == 0
        assert server.posts[0][2] == 7.5

    async def test_prompt_carries_case_and_reference(self) -> None:
        server = ScriptedServer()
        await judge(server).evaluate(case(), result())

        prompt = server.posts[0][1]["prompt"]
        assert "How do I get a refund?" in prompt
        assert "Within 30 days" in prompt
        assert "Refunds are available within 30 days." in prompt

    async def test_custom_prompt_template_is_used(self) -> None:
        server = ScriptedServer()
        judge_instance = judge(server, prompt_template="Q: {case_input}\nA: {agent_output}\nJSON:")
        await judge_instance.evaluate(case(), result())

        prompt = server.posts[0][1]["prompt"]
        assert prompt.startswith("Q: How do I get a refund?")
        assert "REFERENCE" not in prompt

    async def test_extra_criteria_are_requested(self) -> None:
        server = ScriptedServer()
        await judge(server, criteria=["correctness", "safety"]).evaluate(case(), result())

        prompt = server.posts[0][1]["prompt"]
        assert "- correctness" in prompt
        assert "- safety" in prompt


class TestOllamaJudgeFailureHandling:
    async def test_server_down_becomes_an_error_verdict(self) -> None:
        client = FakeClient({GENERATE: ConnectionRefusedError("connection refused")})
        built = OllamaJudge(provider=OllamaProvider("m", base_url=BASE_URL, client=client))

        verdict = await built.evaluate(case(), result())
        assert verdict.ok is False
        assert "LLMProviderError" in (verdict.error or "")
        assert verdict.score is None
        assert verdict.provider == "ollama"

    async def test_malformed_output_becomes_an_error_verdict(self) -> None:
        server = ScriptedServer({"response": "I think the answer is fine actually"})
        verdict = await judge(server).evaluate(case(), result())

        assert verdict.ok is False
        assert "malformed judge output" in (verdict.error or "")

    async def test_fenced_json_is_accepted(self) -> None:
        server = ScriptedServer(
            {"response": f"Here you go:\n```json\n{VERDICT}\n```\nHope that helps."}
        )
        verdict = await judge(server).evaluate(case(), result())
        assert verdict.score == pytest.approx(0.75)

    async def test_chatty_json_is_accepted(self) -> None:
        server = ScriptedServer({"response": f"Looks good to me. {VERDICT} That is my verdict."})
        verdict = await judge(server).evaluate(case(), result())
        assert verdict.score == pytest.approx(0.75)

    async def test_out_of_range_score_is_reported_as_malformed(self) -> None:
        server = ScriptedServer({"response": '{"score": 9.0, "reason": "x", "criteria": {}}'})
        verdict = await judge(server).evaluate(case(), result())
        assert verdict.ok is False
        assert "malformed judge output" in (verdict.error or "")

    async def test_retries_then_succeeds(self) -> None:
        server = ScriptedServer(
            {"status_error": "502 Bad Gateway"},
            {"response": VERDICT},
        )
        verdict = await judge(server, max_retries=2).evaluate(case(), result())

        assert verdict.ok is True
        assert len(server.posts) == 2

    async def test_gives_up_after_max_retries(self) -> None:
        server = ScriptedServer(FakeResponse({}, status_error="502 Bad Gateway"))
        verdict = await judge(server, max_retries=1).evaluate(case(), result())

        assert verdict.ok is False
        assert len(server.posts) == 2

    async def test_no_retry_when_max_retries_is_zero(self) -> None:
        server = ScriptedServer(FakeResponse({}, status_error="502 Bad Gateway"))
        verdict = await judge(server, max_retries=0).evaluate(case(), result())

        assert verdict.ok is False
        assert len(server.posts) == 1

    async def test_agent_error_still_produces_a_verdict(self) -> None:
        # The judge never raises; an agent that failed simply gets graded on
        # whatever output survived.
        server = ScriptedServer()
        verdict = await judge(server).evaluate(case(), result("", error="boom"))

        assert verdict.provider == "ollama"
        assert verdict.score == pytest.approx(0.75)


class TestOllamaJudgeHealth:
    async def test_healthy_when_server_and_model_are_present(self) -> None:
        server = ScriptedServer()
        health = await judge(server).health_check()
        assert health.state is ProviderState.OK

    async def test_reports_an_unreachable_server(self) -> None:
        client = FakeClient(
            {f"{BASE_URL}/api/version": ConnectionRefusedError("connection refused")}
        )
        built = OllamaJudge(provider=OllamaProvider("m", base_url=BASE_URL, client=client))

        health = await built.health_check()
        assert health.state is ProviderState.UNAVAILABLE
        assert "ollama serve" in (health.detail or "")

    async def test_reports_a_missing_model(self) -> None:
        server = ScriptedServer()
        server.client.responses[f"{BASE_URL}/api/tags"] = {"models": [{"name": "phi3"}]}
        health = await judge(server).health_check()

        assert health.state is ProviderState.MISCONFIGURED
        assert f"ollama pull {DEFAULT_MODEL}" in (health.detail or "")


class TestOllamaJudgeLifecycle:
    async def test_aclose_reaches_the_provider(self) -> None:
        server = ScriptedServer()
        built = OllamaJudge(provider=OllamaProvider("m", base_url=BASE_URL, client=server.client))
        # An injected client is not owned by the provider, so closing is a no-op.
        await built.aclose()
        assert server.client.closed is False
