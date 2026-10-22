"""Integration tests for the Ollama-backed judge through the real CLI.

Two things are proven here:

1. ``judge.provider: ollama`` resolves to a live judge that degrades gracefully
   when the server is unreachable, so a run still completes and reports *why*
   judging failed instead of crashing.
2. The offline path is untouched: ``--mock`` and the default mock provider never
   import the optional ``ollama`` extra.

No test in this module needs a running Ollama server.
"""

from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from agent_eval.cli.main import app

runner = CliRunner()

DEAD_URL = "http://127.0.0.1:1"


def invoke(project: Path, *args: str):
    """Run the CLI with ``project`` as the working directory."""
    return runner.invoke(app, list(args), env={"COLUMNS": "200"})


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A freshly initialised project directory, which is also the CWD."""
    os.chdir(tmp_path)
    assert invoke(tmp_path, "init", str(tmp_path)).exit_code == 0
    return tmp_path


@pytest.fixture(autouse=True)
def _isolate_cwd():
    """Restore the working directory after every test in this module."""
    previous = Path.cwd()
    yield
    os.chdir(previous)


def set_judge(project: Path, **overrides) -> None:
    """Point the project's config at the Ollama judge."""
    path = project / "agent-eval.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["judge"].update(overrides)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


class TestOllamaJudgeConfiguration:
    def test_config_validates(self, project: Path) -> None:
        set_judge(project, provider="ollama", model="llama3.2", base_url=DEAD_URL)
        result = invoke(project, "validate")
        assert result.exit_code == 0

    def test_unknown_provider_is_rejected(self, project: Path) -> None:
        set_judge(project, provider="openai")
        result = invoke(project, "validate")
        assert result.exit_code == 2


class TestOllamaJudgeFailureHandling:
    def test_unreachable_server_does_not_abort_the_run(self, project: Path) -> None:
        # A dead port fails instantly, so this asserts the *degradation*
        # contract rather than connectivity.
        set_judge(project, provider="ollama", model="llama3.2", base_url=DEAD_URL, max_retries=0)
        result = invoke(project, "run", "--live", "--no-store", "--no-report")

        assert result.exit_code in (0, 1)
        assert "Traceback" not in result.output

    def test_judge_failure_is_recorded_in_the_report(self, project: Path) -> None:
        set_judge(project, provider="ollama", model="llama3.2", base_url=DEAD_URL, max_retries=0)
        invoke(project, "run", "--live", "--format", "json")

        payload = json.loads((project / "reports" / "latest.json").read_text())
        judged = [c for c in payload["cases"] if c.get("judge") is not None]
        assert judged, "cases that declare a reference answer should have been judged"

        errors = [case["judge"]["error"] for case in judged if case["judge"].get("error")]
        assert errors, "an unreachable judge must be reported, not hidden"
        assert all("Ollama" in error for error in errors)
        assert all(case["judge"]["score"] is None for case in judged)

    def test_failure_does_not_abort_before_later_cases(self, project: Path) -> None:
        set_judge(project, provider="ollama", model="llama3.2", base_url=DEAD_URL, max_retries=0)
        result = invoke(project, "run", "--live", "--format", "json")

        payload = json.loads((project / "reports" / "latest.json").read_text())
        assert len(payload["cases"]) > 1
        assert all(case["status"] != "error" for case in payload["cases"])
        assert result.exit_code in (0, 1)


class TestOfflinePathIsUnaffected:
    def test_mock_run_never_reaches_ollama(self, project: Path) -> None:
        # `--mock` must force the offline judge even though config names Ollama.
        set_judge(project, provider="ollama", model="llama3.2", base_url=DEAD_URL)
        result = invoke(project, "run", "--mock", "--format", "json")

        assert result.exit_code == 0
        payload = json.loads((project / "reports" / "latest.json").read_text())
        providers = {c["judge"]["provider"] for c in payload["cases"] if c.get("judge")}
        assert providers == {"heuristic"}

    def test_default_config_needs_no_ollama_extra(self, project: Path) -> None:
        result = invoke(project, "run", "--format", "json")
        assert result.exit_code == 0

    def test_mock_run_works_without_the_httpx_extra(self, project: Path) -> None:
        # Simulates an install that never pulled `[ollama]`.
        code = (
            "import builtins, sys\n"
            "real_import = builtins.__import__\n"
            "def guard(name, *args, **kwargs):\n"
            "    if name == 'httpx':\n"
            "        raise ImportError('no httpx')\n"
            "    return real_import(name, *args, **kwargs)\n"
            "builtins.__import__ = guard\n"
            "sys.argv = ['agent-eval', 'run', '--mock', '--no-store', '--no-report']\n"
            "from agent_eval.cli.main import main\n"
            "main()\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr

    def test_importing_the_package_does_not_import_httpx(self) -> None:
        code = (
            "import sys\n"
            "import agent_eval, agent_eval.judges, agent_eval.providers\n"
            "print('httpx' in sys.modules)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "False"


def test_ollama_is_not_required_by_the_core() -> None:
    """The core must stay importable without any optional extra installed."""
    real_import = builtins.__import__
    blocked = {"httpx", "langgraph", "langchain_core", "fastapi"}

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise ImportError(f"{name} is not installed")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = guard
    try:
        for module in (
            "agent_eval.cli.main",
            "agent_eval.core.runner",
            "agent_eval.providers.mock",
            "agent_eval.judges.mock",
            "agent_eval.reporting",
        ):
            __import__(module)
    finally:
        builtins.__import__ = real_import
