"""Mock and heuristic judges: fully offline judging.

``MockJudge`` returns a fixed score for every case, which makes it ideal for
CI smoke tests where determinism matters more than signal.

``HeuristicJudge`` derives a score from deterministic signals (keyword overlap
with the reference answer, error state). It is honest about what it is: a
keyword heuristic, not a language model.
"""

from __future__ import annotations

from typing import Any

from agent_eval.domain.models import AgentResult, EvalCase, JudgeVerdict
from agent_eval.judges.base import BaseJudge, heuristic_verdict
from agent_eval.ports.clock import Clock
from agent_eval.ports.llm import LLMProvider

MOCK_JUDGE_MODEL = "mock-judge"


class MockJudge:
    """Returns a fixed score. Zero network calls, zero variance."""

    name = "mock-judge"
    version = "1.0.0"

    def __init__(
        self,
        score: float = 0.85,
        *,
        reason: str = "Mock judge returned a fixed score; no model was called.",
        criteria: list[str] | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._score = score
        self._reason = reason
        self._criteria = list(criteria or ["correctness", "relevance", "completeness"])
        self._clock = clock

    @property
    def model(self) -> str:
        """Model identifier reported in verdicts."""
        return MOCK_JUDGE_MODEL

    async def evaluate(self, case: EvalCase, result: AgentResult) -> JudgeVerdict:
        """Return the configured fixed verdict."""
        if not result.ok:
            return JudgeVerdict(
                score=0.0,
                reason=f"Agent reported an error: {result.error}",
                criteria=dict.fromkeys(self._criteria, 0.0),
                provider="mock",
                model=MOCK_JUDGE_MODEL,
            )
        return JudgeVerdict(
            score=self._score,
            reason=self._reason,
            criteria=dict.fromkeys(self._criteria, self._score),
            provider="mock",
            model=MOCK_JUDGE_MODEL,
        )


class HeuristicJudge:
    """Deterministic offline judge based on keyword overlap."""

    name = "heuristic-judge"
    version = "1.0.0"

    def __init__(
        self,
        *,
        criteria: list[str] | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._criteria = list(criteria or ["correctness", "relevance", "completeness"])
        self._clock = clock

    @property
    def model(self) -> str:
        """Model identifier reported in verdicts."""
        return "heuristic-v1"

    async def evaluate(self, case: EvalCase, result: AgentResult) -> JudgeVerdict:
        """Score the case with deterministic signals only."""
        return heuristic_verdict(case, result, self._criteria)


class ProviderJudge(BaseJudge):
    """Judge backed by any :class:`LLMProvider` (used by the Ollama judge)."""

    def __init__(self, provider: LLMProvider, **kwargs: Any) -> None:
        super().__init__(provider, **kwargs)
        self.name = f"{provider.name}-judge"
