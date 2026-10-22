"""Judge port.

A judge turns an agent execution into a structured score. Implementations must
never raise: any internal failure has to come back as a
:class:`~agent_eval.domain.models.JudgeVerdict` with ``error`` set, which is
what stops free-form model output from breaking the pipeline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_eval.domain.models import AgentResult, EvalCase, JudgeVerdict


@runtime_checkable
class Judge(Protocol):
    """Scores agent output against a reference answer."""

    name: str
    version: str

    async def evaluate(self, case: EvalCase, result: AgentResult) -> JudgeVerdict:
        """Judge a single case.

        Returns:
            A :class:`JudgeVerdict`. Implementations must catch their own
            errors and return ``JudgeVerdict(score=None, error=...)``.
        """
        ...
