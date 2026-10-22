"""Ad-hoc end-to-end smoke check of the core engine."""

import asyncio

from agent_eval.config import load_config
from agent_eval.core.runner import Runner
from agent_eval.datasets import load_dataset_file
from agent_eval.integrations.mock import default_mock_agent
from agent_eval.judges.mock import HeuristicJudge
from agent_eval.storage.memory import InMemoryRepository


async def main() -> None:
    config = load_config("agent-eval.yaml")
    dataset = load_dataset_file("evals/dataset.yaml")
    repo = InMemoryRepository()
    runner = Runner(
        default_mock_agent(),
        config,
        judge=HeuristicJudge(),
        repository=repo,
    )
    outcome = await runner.run(dataset)
    agg = outcome.aggregate
    print(f"run_id      : {outcome.run_id}")
    print(
        f"total       : {agg.total} passed={agg.passed} failed={agg.failed} errored={agg.errored}"
    )
    print(f"overall     : {agg.overall_score}")
    print(f"det/judge   : {agg.deterministic_score} / {agg.judge_score}")
    print(f"latency avg : {agg.latency.average_ms:.2f}ms p95={agg.latency.p95_ms:.2f}ms")
    print(f"tokens      : {agg.tokens.total_tokens}")
    for case in outcome.cases:
        failed = [a.assertion.value for a in case.failed_assertions]
        print(f"  {case.status.value:8} {case.case_id:24} failed={failed}")


asyncio.run(main())
