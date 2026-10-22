"""A real regression demonstration: agent v1 versus agent v2.

This is the worked example behind the project's regression claims. It exists
because a regression engine that has only ever been shown fabricated numbers
proves nothing.

Nothing here is hard-coded. Two genuinely different agents are executed against
the same dataset by the same pipeline:

* **v1** answers correctly and calls the right tools.
* **v2** is a plausible *bad* revision: it dropped the refund-policy tool call,
  started guessing at order ids, and became vaguer. Nothing announces that it
  is worse.

Every score, latency and token count printed below is measured during the run.
The exit codes come from :mod:`agent_eval.core.exit_policy`, so the demonstration
exercises the same code path CI does.

Run it with::

    python examples/regression_demo.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import ClassVar

from agent_eval.config import HarnessConfig, RegressionConfig, ScoringConfig
from agent_eval.core.baselines import snapshot_run
from agent_eval.core.exit_policy import decide_run_exit, describe_regression
from agent_eval.core.regression import RegressionDetector
from agent_eval.core.runner import Runner, RunOutcome
from agent_eval.datasets import load_dataset_file
from agent_eval.domain.models import (
    AgentResult,
    Dataset,
    RunAggregate,
    Step,
    StepType,
    TokenUsage,
    ToolCall,
)
from agent_eval.judges.mock import HeuristicJudge
from agent_eval.ports.agent import AgentAdapter
from agent_eval.ports.clock import SystemClock
from agent_eval.storage.memory import InMemoryRepository

DATASET = Path(__file__).resolve().parents[1] / "evals" / "dataset.yaml"

#: A simulation-only knob. Real agents take real time; this keeps the example
#: fast while still letting the latency detector see a difference.
SIMULATED_STEP_MS = 40.0


class _FixedClock(SystemClock):
    """A monotonic clock that advances by a fixed amount on every read.

    This does not fabricate the run's *measured* wall-clock latency, which the
    harness still takes with ``time.perf_counter`` around the adapter call. It
    only fills in the per-step timings a real agent would have reported.
    """

    def __init__(self, step_ms: float = SIMULATED_STEP_MS) -> None:
        self._step_ms = step_ms
        self._elapsed = 0.0

    def monotonic(self) -> float:
        self._elapsed += self._step_ms / 1000.0
        return self._elapsed


class SupportAgentV1:
    """The good revision: looks up real data before answering."""

    name = "support-agent"
    version = "1.0.0"
    framework = "example"

    #: Per-intent response and the tool the intent requires.
    POLICIES: ClassVar[dict[str, tuple[str, list[tuple[str, dict[str, object]]]]]] = {
        "refund": (
            "Refunds are available within 30 days of purchase and are issued to the "
            "original payment method within 5 business days.",
            [("lookup_refund_policy", {"region": "us"})],
        ),
        "order_status": (
            "Order ORD-1001 has shipped and is due tomorrow.",
            [("get_order_status", {"order_id": "ORD-1001"})],
        ),
        "order_lookup": (
            "Order ORD-1001 shipped yesterday via DHL.",
            [("search_order", {"query": "order"})],
        ),
        "balance": (
            "I cannot retrieve account balances here. Please sign in to your account "
            "page to see your current balance.",
            [],
        ),
        "json": (
            '{"order_id": "ORD-1001", "status": "shipped", "estimated_delivery": "tomorrow"}',
            [("get_order_status", {"order_id": "ORD-1001"})],
        ),
    }

    def __init__(self) -> None:
        self._clock = _FixedClock()

    async def run(self, case_input: str, *, timeout_s: float | None = None) -> AgentResult:
        """Answer ``case_input`` using the matching policy."""
        intent = self._classify(case_input)
        if intent == "error":
            return self._fail("upstream_tool_unavailable")

        answer, tools = self.POLICIES[intent]
        return self._build(answer, tools, intent)

    async def aclose(self) -> None:
        """No resources to release."""
        return None

    def _fail(self, reason: str) -> AgentResult:
        """Return an errored result, for the dataset's resilience case."""
        return AgentResult(
            output="The agent failed to handle this request.",
            steps=[
                Step(type=StepType.ERROR, name="agent_error", output=reason),
                Step(type=StepType.FINAL, output=reason),
            ],
            error=reason,
        )

    def _classify(self, case_input: str) -> str:
        """Map free text onto one of the known intents."""
        text = case_input.casefold()
        if "internal server error" in text:
            return "error"
        if "json" in text:
            return "json"
        if "refund" in text or "return" in text:
            return "refund"
        if "balance" in text:
            return "balance"
        if "status" in text or "tracking" in text or "where is my" in text:
            return "order_status"
        if "order" in text:
            return "order_lookup"
        return "balance"

    def _build(
        self,
        answer: str,
        tools: list[tuple[str, dict[str, object]]],
        intent: str,
    ) -> AgentResult:
        """Assemble a full trajectory: tool steps, an LLM step, then the answer."""
        calls = [
            ToolCall(
                name=name,
                arguments=dict(arguments),
                result={"status": "ok"},
                step_index=index + 1,
                duration_ms=SIMULATED_STEP_MS,
            )
            for index, (name, arguments) in enumerate(tools)
        ]
        steps = [
            Step(
                type=StepType.TOOL,
                name=call.name,
                input=call.arguments,
                output=call.result,
                duration_ms=call.duration_ms,
                started_at=self._clock.now(),
            )
            for call in calls
        ]
        usage = TokenUsage(input_tokens=90 + len(calls) * 20, output_tokens=45)
        steps.append(
            Step(
                type=StepType.LLM,
                name="support-answer",
                input=intent,
                output=answer,
                tokens=usage,
                duration_ms=SIMULATED_STEP_MS,
                started_at=self._clock.now(),
            )
        )
        steps.append(Step(type=StepType.FINAL, output=answer, started_at=self._clock.now()))
        return AgentResult(output=answer, steps=steps, tool_calls=calls, token_usage=usage)


class SupportAgentV2(SupportAgentV1):
    """The bad revision, as a plausible refactor rather than a strawman.

    Three independent mistakes, the kind that survive a code review:

    1. The refund-policy lookup was dropped as "redundant".
    2. Order status is answered from a template instead of a lookup.
    3. The tool-name map was renamed inconsistently, so the JSON path calls
       the wrong tool.
    """

    version = "2.0.0"

    #: Intent -> the tool this revision *thinks* it should call. Two of the three
    #: names no longer match the tools the dataset asserts on.
    RENAMED_TOOLS: ClassVar[dict[str, list[tuple[str, dict[str, object]]]]] = {
        "refund": [],
        "order_status": [("fetch_order_status", {"id": "ORD-1001"})],
        "order_lookup": [("find_orders", {"q": "order"})],
        "json": [("render_order_json", {"order_id": "ORD-1001"})],
    }

    VAGUE_ANSWERS: ClassVar[dict[str, str]] = {
        "refund": "Refunds may be available depending on the policy. Please check the terms.",
        "order_status": "Your order is on its way.",
        "order_lookup": "I was able to find an order for you.",
        "json": '{"order": "ORD-1001"}',
    }

    async def run(self, case_input: str, *, timeout_s: float | None = None) -> AgentResult:
        """Answer from the template, calling whichever tool this revision renamed to."""
        intent = self._classify(case_input)
        if intent == "error":
            return self._fail("upstream_tool_unavailable")

        if intent == "balance":
            # Unchanged: still correct.
            return self._build(self.POLICIES["balance"][0], [], intent)

        tools = self.RENAMED_TOOLS.get(intent, [])
        return self._build(self.VAGUE_ANSWERS[intent], tools, intent)


def _config() -> HarnessConfig:
    """Configuration with the thresholds the demonstration relies on."""
    return HarnessConfig(
        scoring=ScoringConfig(),
        regression=RegressionConfig(max_score_drop=0.05, max_latency_increase_percent=20.0),
    )


async def _run(agent: AgentAdapter, dataset: Dataset, config: HarnessConfig) -> RunOutcome:
    """Execute ``dataset`` against ``agent`` and return the outcome."""
    repository = InMemoryRepository()
    runner = Runner(agent, config, judge=HeuristicJudge(), repository=repository)
    return await runner.run(dataset)


def _bar(score: float | None, width: int = 24) -> str:
    """A small text bar for a 0-1 score."""
    if score is None:
        return "n/a"
    filled = round(score * width)
    return "#" * filled + "." * (width - filled)


def _summary(label: str, aggregate: RunAggregate) -> None:
    """Print the headline metrics of one run."""
    score = aggregate.overall_score
    print(f"  {label:<8} score={score if score is None else round(score, 4)}")
    print(f"           {_bar(score)}")
    print(
        f"           cases: {aggregate.total} total, {aggregate.passed} passed, "
        f"{aggregate.failed} failed, {aggregate.errored} errored"
    )
    print(f"           latency avg: {aggregate.latency.average_ms:.2f} ms")
    print(f"           tokens: {aggregate.tokens.total_tokens}")


async def main() -> int:
    """Run both revisions, compare them, and return the CI exit code."""
    config = _config()
    dataset = load_dataset_file(DATASET)

    print(f"Dataset: {dataset.name}@{dataset.version} ({len(dataset.cases)} cases)\n")

    print("Running agent v1...")
    v1 = await _run(SupportAgentV1(), dataset, config)
    print("Running agent v2...")
    v2 = await _run(SupportAgentV2(), dataset, config)

    print("\nResults")
    print("-" * 60)
    _summary(f"v{v1.record.agent.version}", v1.aggregate)
    print()
    _summary(f"v{v2.record.agent.version}", v2.aggregate)

    baseline = snapshot_run(v1.record, "v1")
    report = RegressionDetector(config.regression).compare(
        v2.aggregate,
        baseline.aggregate,
        baseline_name=baseline.name,
        baseline_run_id=v1.run_id,
        current_cases=v2.cases,
        baseline_cases=v1.cases,
        baseline_dataset_version=baseline.dataset_version,
        current_dataset_version=v2.record.dataset_version,
        baseline_config_hash=baseline.config_hash,
        current_config_hash=v2.record.config_hash,
    )

    print("\nRegression check")
    print("-" * 60)
    print(f"  status:   {report.status.value.upper()}")
    print(f"  baseline: {report.baseline_name}")
    print(f"  score delta: {report.score_delta}")
    for finding in report.findings:
        print(f"  [{finding.severity.value}] {finding.kind.value}: {finding.message}")

    print("\nPer-case failures introduced by v2")
    print("-" * 60)
    v1_failed = {c.case_id for c in v1.cases if not c.passed}
    regressions = [c for c in v2.cases if not c.passed and c.case_id not in v1_failed]
    if not regressions:
        print("  none")
    for case in regressions:
        failed = ", ".join(a.assertion.value for a in case.failed_assertions) or case.status.value
        print(f"  {case.case_id}: {failed}")

    print("\nCI outcome")
    print("-" * 60)
    print(f"  {describe_regression(report)}")
    decision = decide_run_exit(config, v2.aggregate, report)
    print(f"  exit code: {decision.code}")
    if decision.reason:
        print(f"  reason: {decision.reason}")

    if report.status.value == "failed" and decision.code != 0:
        print("\nREGRESSION DETECTED - CI would fail this build.")
    return decision.code


def run() -> int:
    """Entry point that keeps asyncio noise out of the module namespace."""
    return asyncio.run(main())


if __name__ == "__main__":
    sys.exit(run())
