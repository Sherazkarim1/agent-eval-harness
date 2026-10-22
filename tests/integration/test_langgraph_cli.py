"""End-to-end tests running a real LangGraph graph through the CLI.

These prove the adapter satisfies the harness contract, not just its own unit
tests: real assertions over a real graph, persisted through the real storage
layer and rendered into the real reports.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from agent_eval.cli.main import app

pytest.importorskip("langgraph")

runner = CliRunner()

CONFIG = """
project:
  name: langgraph-demo
agent:
  type: langgraph
  name: support-graph
  options:
    graph_factory: "agent_eval.integrations.langgraph.examples:default_graph"
dataset:
  path: evals/dataset.yaml
storage:
  type: sqlite
  path: .agent_eval/results.db
"""

DATASET = """
name: support
version: 1.0.0
description: LangGraph adapter evaluation.
cases:
  - id: order_lookup_001
    input: Where is my order?
    expected:
      contains: ["ORD-1001", "shipped"]
      tool_called: [lookup_order]
      tool_args:
        lookup_order:
          query: "Where is my order?"
  - id: greeting_001
    input: Hello there
    expected:
      contains: ["How can I help"]
      tool_not_called: [lookup_order]
"""


RAISING_GRAPH = '''
"""A graph whose only node fails, used to prove errors stay per-case."""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict, total=False):
    """Minimal state."""

    input: str
    output: str


def boom(state: State) -> dict[str, Any]:
    """Always fail."""
    raise RuntimeError("node exploded")


def build() -> Any:
    """Compile the failing graph."""
    builder = StateGraph(State)
    builder.add_node("boom", boom)
    builder.add_edge(START, "boom")
    builder.add_edge("boom", END)
    return builder.compile()
'''


@pytest.fixture(autouse=True)
def _isolate_cwd():
    """Restore the working directory after every test."""
    previous = Path.cwd()
    yield
    os.chdir(previous)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project configured to evaluate the bundled example graph."""
    os.chdir(tmp_path)
    (tmp_path / "evals").mkdir()
    (tmp_path / "agent-eval.yaml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "evals" / "dataset.yaml").write_text(DATASET, encoding="utf-8")
    return tmp_path


def invoke(*args: str):
    """Run the CLI in the current project directory."""
    return runner.invoke(app, list(args), env={"COLUMNS": "200"})


class TestLangGraphRun:
    def test_both_cases_pass(self, project: Path) -> None:
        result = invoke("run", "--live")

        assert result.exit_code == 0, result.output
        payload = json.loads((project / "reports" / "latest.json").read_text())
        assert payload["aggregate"]["passed"] == 2
        assert payload["aggregate"]["failed"] == 0
        assert payload["aggregate"]["errored"] == 0

    def test_reports_live_mode_not_mock(self, project: Path) -> None:
        invoke("run", "--live")

        payload = json.loads((project / "reports" / "latest.json").read_text())
        assert payload["is_mock"] is False
        assert payload["execution_mode"] == "live"

    def test_tool_call_is_captured_in_the_report(self, project: Path) -> None:
        invoke("run", "--live")

        payload = json.loads((project / "reports" / "latest.json").read_text())
        order_case = next(
            case for case in payload["cases"] if case["case_id"] == "order_lookup_001"
        )
        calls = order_case["trajectory"]["tool_calls"]
        assert [call["name"] for call in calls] == ["lookup_order"]
        assert calls[0]["arguments"]["query"] == "Where is my order?"
        assert "shipped" in calls[0]["result"]

    def test_trajectory_is_recorded(self, project: Path) -> None:
        invoke("run", "--live")

        payload = json.loads((project / "reports" / "latest.json").read_text())
        greeting = next(case for case in payload["cases"] if case["case_id"] == "greeting_001")
        steps = greeting["trajectory"]["steps"]
        assert steps[-1]["type"] == "final"
        assert greeting["trajectory"]["tool_calls"] == []

    def test_tool_arguments_assertion_is_enforced(self, project: Path) -> None:
        """A wrong expected argument must fail, proving tool_args really compares."""
        dataset = project / "evals" / "dataset.yaml"
        dataset.write_text(
            DATASET.replace('query: "Where is my order?"', 'query: "something else"'),
            encoding="utf-8",
        )

        result = invoke("run", "--live")

        assert result.exit_code == 1
        assert "tool_args" in result.output

    def test_missing_extra_is_a_usage_error(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        (tmp_path / "evals").mkdir()
        (tmp_path / "evals" / "dataset.yaml").write_text(DATASET, encoding="utf-8")
        (tmp_path / "agent-eval.yaml").write_text(
            CONFIG.replace(
                'graph_factory: "agent_eval.integrations.langgraph.examples:default_graph"',
                'graph_factory: "nope.missing:build"',
            ),
            encoding="utf-8",
        )

        result = invoke("run", "--live")

        assert result.exit_code == 2
        assert "nope.missing" in result.output

    def test_graph_failure_is_reported_per_case(self, project: Path) -> None:
        """A raising node must error the case, not abort the run."""
        config = yaml.safe_load(CONFIG)
        config["agent"]["options"]["graph_factory"] = "raising_graph:build"
        (project / "agent-eval.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        sys.path.insert(0, str(project))
        try:
            (project / "raising_graph.py").write_text(RAISING_GRAPH, encoding="utf-8")
            result = invoke("run", "--live")
        finally:
            sys.path.remove(str(project))

        assert result.exit_code == 1, result.output
        payload = json.loads((project / "reports" / "latest.json").read_text())
        assert payload["aggregate"]["errored"] == 2
        assert "node exploded" in payload["cases"][0]["error"]


class TestExampleAlias:
    def test_example_type_needs_no_graph_option(self, project: Path) -> None:
        config = yaml.safe_load(CONFIG)
        config["agent"] = {"type": "example", "name": "support-graph"}
        (project / "agent-eval.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

        result = invoke("run", "--live")

        assert result.exit_code == 0, result.output
