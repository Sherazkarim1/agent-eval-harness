"""Ad-hoc check that runs and baselines survive across separate processes.

Writes to a throwaway database, then re-opens it in a fresh interpreter to prove
the data was actually committed rather than held in memory.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

from agent_eval.config import load_config
from agent_eval.core.baselines import save_baseline_from_run
from agent_eval.core.runner import Runner
from agent_eval.datasets import load_dataset_file
from agent_eval.integrations.mock import default_mock_agent
from agent_eval.judges.mock import HeuristicJudge
from agent_eval.storage import create_repository

DB_PATH = Path("reports/smoke.db")

INSPECT = """
import sys
from pathlib import Path

from agent_eval.storage import SqliteEvaluationRepository

repo = SqliteEvaluationRepository(Path(sys.argv[1]))
runs = repo.list_runs(dataset_name="customer_support")
print(f"runs        : {len(runs)}")
print(f"newest      : {runs[0].run_id} total={runs[0].aggregate.total}")
print(f"cases       : {len(repo.get_cases(runs[0].run_id))}")
print(f"trajectory  : {repo.get_cases(runs[0].run_id)[0].result.step_labels()}")
baseline = repo.get_baseline("smoke")
print(f"baseline    : {baseline.name} total={baseline.aggregate.total} run={baseline.run_id}")
print(f"leaderboard : {[(r.agent_name, r.agent_version) for r in repo.leaderboard('customer_support')]}")
print(f"datasets    : {repo.list_datasets()}")
repo.close()
"""


def main() -> None:
    DB_PATH.unlink(missing_ok=True)
    config = load_config("agent-eval.yaml")
    storage = config.storage.model_copy(update={"path": DB_PATH})
    dataset = load_dataset_file("evals/dataset.yaml")
    repo = create_repository(storage)
    repo.register_dataset(dataset)

    runner = Runner(default_mock_agent(), config, judge=HeuristicJudge(), repository=repo)
    first = asyncio.run(runner.run(dataset))
    repo.save_run(first.record, first.cases)
    save_baseline_from_run(repo, first.record, "smoke")
    print(f"run_1     : {first.run_id}")
    print(f"baseline  : saved total={first.aggregate.total}")

    second = asyncio.run(runner.run(dataset))
    repo.save_run(second.record, second.cases)
    print(f"run_2     : {second.run_id}")
    print(f"aggregate : total={second.aggregate.total} score={second.aggregate.overall_score}")
    repo.close()


if __name__ == "__main__":
    main()
    print("\n--- reopened in a NEW process ---")
    subprocess.run([sys.executable, "-c", INSPECT, str(DB_PATH)], check=True)
