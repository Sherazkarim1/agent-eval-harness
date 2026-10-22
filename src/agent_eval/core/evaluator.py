"""Case evaluator: runs one case end to end and scores it.

Order of operations matters and is deliberate:

1. measure latency with the harness clock (never trust the adapter's number),
2. redact the result *before* anything inspects it,
3. estimate API cost,
4. run deterministic assertions,
5. run the judge (only when a reference answer exists or the judge is forced),
6. score.

Agent-level failures are captured as ``status=errored`` results rather than
exceptions, because a failing agent is a measurable outcome.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from agent_eval.config import HarnessConfig
from agent_eval.core.cost import estimate_cost
from agent_eval.core.scoring import ScoringEngine
from agent_eval.domain.assertions import AssertionEngine
from agent_eval.domain.models import (
    AgentResult,
    AssertionResult,
    CaseResult,
    CaseStatus,
    EvalCase,
    ExecutionMode,
    JudgeVerdict,
    ScoreBreakdown,
)
from agent_eval.domain.sanitize import SecretRedactor
from agent_eval.ports.agent import AgentAdapter
from agent_eval.ports.clock import Clock, SystemClock
from agent_eval.ports.judge import Judge

logger = logging.getLogger(__name__)


class Evaluator:
    """Executes and scores individual evaluation cases."""

    def __init__(
        self,
        adapter: AgentAdapter,
        judge: Judge | None,
        config: HarnessConfig,
        *,
        clock: Clock | None = None,
        redactor: SecretRedactor | None = None,
    ) -> None:
        """Create an evaluator.

        Args:
            adapter: The agent under test.
            judge: The judge, or ``None`` to disable LLM judging.
            config: Harness configuration (weights, cost, judge settings).
            clock: Clock used for timestamps and latency measurement.
            redactor: Secret redactor applied to every agent result.
        """
        self._adapter = adapter
        self._judge = judge
        self._config = config
        self._clock = clock or SystemClock()
        self._redactor = redactor or SecretRedactor()
        self._assertions = AssertionEngine()
        self._scoring = ScoringEngine(config.scoring)

    async def run_case(self, case: EvalCase, run_id: str) -> CaseResult:
        """Run, assert, judge and score a single case.

        Args:
            case: The case to execute.
            run_id: Identifier of the owning run.

        Returns:
            A fully populated :class:`CaseResult`.
        """
        started_at = self._clock.now()
        timeout_s = case.timeout_s or self._config.run.timeout_s
        started = self._clock.monotonic()

        try:
            raw_result = await self._invoke(case, timeout_s)
        except Exception as exc:
            logger.warning("Case %s raised: %s", case.id, exc)
            return self._error_result(case, run_id, started_at, started, exc)

        latency_ms = (self._clock.monotonic() - started) * 1000.0
        result = self._redact(raw_result)
        cost_usd = estimate_cost(
            result.token_usage, self._config.pricing, self._agent_model(result)
        )

        assertions = self._assertions.evaluate(
            case.expected, result, latency_ms=latency_ms, cost_usd=cost_usd
        )

        judge_verdict = await self._judge_case(case, result)
        score = self._scoring.score_case(assertions, judge_verdict)

        status = self._status(case, result, assertions, score)

        return CaseResult(
            run_id=run_id,
            case_id=case.id,
            case=case,
            result=result,
            assertions=assertions,
            score=score,
            judge=judge_verdict,
            status=status,
            error=result.error,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            started_at=started_at,
        )

    def _status(
        self,
        case: EvalCase,
        result: AgentResult,
        assertions: list[AssertionResult],
        score: ScoreBreakdown,
    ) -> CaseStatus:
        """Decide the case outcome.

        An agent failure normally makes the case errored. A case that declares
        ``expect_error`` is the exception: there, failing *is* the correct
        behaviour, so the case passes and only the ``expect_error`` assertion
        decides whether it did.
        """
        if not result.ok:
            if case.expected.expect_error:
                return (
                    CaseStatus.PASSED
                    if not any(a.failed for a in assertions)
                    else CaseStatus.FAILED
                )
            return CaseStatus.ERRORED
        if any(a.failed for a in assertions) or not self._scoring.meets_threshold(score.overall):
            return CaseStatus.FAILED
        return CaseStatus.PASSED

    async def _invoke(self, case: EvalCase, timeout_s: float) -> AgentResult:
        """Call the adapter, enforcing the case timeout."""
        try:
            return await asyncio.wait_for(
                self._adapter.run(case.input, timeout_s=timeout_s), timeout=timeout_s
            )
        except TimeoutError:
            return AgentResult(
                output="",
                error=f"Agent timed out after {timeout_s:.1f}s",
                metadata={"timeout_s": timeout_s},
            )

    async def _judge_case(self, case: EvalCase, result: AgentResult) -> JudgeVerdict | None:
        """Run the judge when it is enabled and applicable to this case."""
        if self._judge is None or not self._config.judge.enabled:
            return None
        if case.reference_answer is None and self._config.judge.only_with_reference:
            return None
        try:
            return await self._judge.evaluate(case, result)
        except Exception as exc:
            logger.warning("Judge raised for case %s: %s", case.id, exc)
            return JudgeVerdict(error=f"{type(exc).__name__}: {exc}")

    def _redact(self, result: AgentResult) -> AgentResult:
        """Apply secret redaction to every user-visible field."""
        if not result.steps and not result.tool_calls and not result.metadata:
            return result.model_copy(update={"output": self._redactor.scrub(result.output)})
        return result.model_copy(
            update={
                "output": self._redactor.scrub(result.output),
                "steps": [
                    step.model_copy(
                        update={
                            "input": self._redactor.scrub(step.input),
                            "output": self._redactor.scrub(step.output),
                            "name": self._redactor.scrub(step.name) if step.name else None,
                        }
                    )
                    for step in result.steps
                ],
                "tool_calls": [
                    call.model_copy(
                        update={
                            "arguments": self._redactor.scrub(call.arguments),
                            "result": self._redactor.scrub(call.result),
                            "name": self._redactor.scrub(call.name),
                            "error": self._redactor.scrub(call.error) if call.error else None,
                        }
                    )
                    for call in result.tool_calls
                ],
                "metadata": self._redactor.scrub(result.metadata),
                "error": self._redactor.scrub(result.error) if result.error else None,
            }
        )

    def _agent_model(self, result: AgentResult) -> str | None:
        """Best-effort model identifier for pricing lookup."""
        model = result.metadata.get("model")
        return str(model) if model is not None else None

    def _error_result(
        self,
        case: EvalCase,
        run_id: str,
        started_at: datetime,
        started: float,
        exc: Exception,
    ) -> CaseResult:
        """Build an errored case result for an exception raised by the adapter."""
        latency_ms = (self._clock.monotonic() - started) * 1000.0
        message = f"{type(exc).__name__}: {exc}"
        result = AgentResult(output="", error=message)
        return CaseResult(
            run_id=run_id,
            case_id=case.id,
            case=case,
            result=result,
            assertions=self._assertions.evaluate(
                case.expected, result, latency_ms=latency_ms, cost_usd=0.0
            ),
            score=None,
            judge=None,
            status=CaseStatus.ERRORED,
            error=message,
            latency_ms=latency_ms,
            cost_usd=0.0,
            started_at=started_at,
        )


def execution_mode_for(config: HarnessConfig, *, mock_override: bool = False) -> ExecutionMode:
    """Resolve the execution mode for a run."""
    if mock_override or config.run.mode == "mock":
        return ExecutionMode.MOCK
    return ExecutionMode.LIVE
