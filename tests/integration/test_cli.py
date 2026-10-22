"""End-to-end tests for the `agent-eval` command-line interface.

These exercise the real command surface through Typer's ``CliRunner``, plus one
subprocess test that proves the packaged entry point and its exit codes behave
correctly outside the test process.
"""

from __future__ import annotations

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


def invoke(project: Path, *args: str):
    """Run the CLI with ``project`` as the working directory."""
    return runner.invoke(app, list(args), env={"COLUMNS": "200"})


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A freshly initialised project directory, which is also the CWD."""
    os.chdir(tmp_path)
    result = invoke(tmp_path, "init", str(tmp_path))
    assert result.exit_code == 0, result.output
    return tmp_path


@pytest.fixture(autouse=True)
def _isolate_cwd():
    """Restore the working directory after every test in this module."""
    previous = Path.cwd()
    yield
    os.chdir(previous)


def read_config(project: Path) -> dict:
    """Parse the project's config file."""
    return yaml.safe_load((project / "agent-eval.yaml").read_text(encoding="utf-8"))


def write_dataset(project: Path, name: str, body: str) -> Path:
    """Write an extra dataset file and return its path."""
    path = project / "evals" / name
    path.write_text(body, encoding="utf-8")
    return path


class TestInit:
    def test_creates_config_and_dataset(self, tmp_path: Path) -> None:
        result = invoke(tmp_path, "init", str(tmp_path))
        assert result.exit_code == 0
        assert (tmp_path / "agent-eval.yaml").is_file()
        assert (tmp_path / "evals" / "dataset.yaml").is_file()

    def test_generated_project_is_valid(self, tmp_path: Path) -> None:
        invoke(tmp_path, "init", str(tmp_path))
        assert (
            yaml.safe_load((tmp_path / "agent-eval.yaml").read_text(encoding="utf-8"))["agent"][
                "type"
            ]
            == "mock"
        )

    def test_does_not_overwrite_without_force(self, project: Path) -> None:
        config = project / "agent-eval.yaml"
        config.write_text("# edited\n", encoding="utf-8")
        result = invoke(project, "init", str(project))
        assert result.exit_code == 0
        assert config.read_text(encoding="utf-8") == "# edited\n"

    def test_force_overwrites(self, project: Path) -> None:
        config = project / "agent-eval.yaml"
        config.write_text("# edited\n", encoding="utf-8")
        result = invoke(project, "init", str(project), "--force")
        assert result.exit_code == 0
        assert config.read_text(encoding="utf-8") != "# edited\n"

    def test_custom_names_are_used(self, tmp_path: Path) -> None:
        result = invoke(
            tmp_path,
            "init",
            str(tmp_path),
            "--project-name",
            "shopbot",
            "--agent-name",
            "shop-agent",
            "--dataset-name",
            "shop",
        )
        assert result.exit_code == 0
        config = read_config(tmp_path)
        assert config["project"]["name"] == "shopbot"
        assert config["agent"]["name"] == "shop-agent"
        dataset = yaml.safe_load((tmp_path / "evals" / "dataset.yaml").read_text())
        assert dataset["name"] == "shop"


class TestValidate:
    def test_initialised_project_passes(self, project: Path) -> None:
        result = invoke(project, "validate")
        assert result.exit_code == 0

    def test_missing_dataset_exits_1(self, project: Path) -> None:
        (project / "evals" / "dataset.yaml").unlink()
        result = invoke(project, "validate")
        assert result.exit_code == 1

    def test_broken_config_exits_2(self, project: Path) -> None:
        (project / "agent-eval.yaml").write_text(
            "agent:\n  type: mock\n  bogus_key: 1\n", encoding="utf-8"
        )
        result = invoke(project, "validate")
        assert result.exit_code == 2

    def test_reports_duplicate_case_ids(self, project: Path) -> None:
        write_dataset(
            project,
            "dupes.yaml",
            "name: dupes\nversion: 1.0.0\ncases:\n"
            "  - id: same\n    input: a\n"
            "  - id: same\n    input: b\n",
        )
        result = invoke(project, "validate", "--dataset", "evals/dupes.yaml")
        assert result.exit_code == 1
        assert "same" in result.output


class TestRun:
    def test_mock_run_succeeds_offline(self, project: Path) -> None:
        result = invoke(project, "run", "--mock")
        assert result.exit_code == 0
        assert "MOCK RUN" in result.output

    def test_run_persists_and_writes_reports(self, project: Path) -> None:
        result = invoke(project, "run", "--mock")
        assert result.exit_code == 0
        assert (project / "reports" / "latest.json").is_file()
        assert (project / "reports" / "latest.md").is_file()
        assert (project / ".agent_eval" / "results.db").is_file()

    def test_json_report_declares_mock_mode(self, project: Path) -> None:
        invoke(project, "run", "--mock")
        payload = json.loads((project / "reports" / "latest.json").read_text())
        assert payload["is_mock"] is True
        assert payload["execution_mode"] == "mock"
        assert payload["run_id"]

    def test_first_run_reports_no_regression(self, project: Path) -> None:
        result = invoke(project, "run", "--mock")
        assert "Regression" not in result.output

    def test_live_flag_overrides_mock_config(self, project: Path) -> None:
        result = invoke(project, "run", "--live")
        assert result.exit_code == 0
        assert "MOCK RUN" not in result.output

    def test_tag_filter_runs_subset(self, project: Path) -> None:
        result = invoke(project, "run", "--mock", "--tag", "safety")
        assert result.exit_code == 0
        payload = json.loads((project / "reports" / "latest.json").read_text())
        assert [c["case_id"] for c in payload["cases"]] == ["must_not_hallucinate_001"]

    def test_unknown_tag_is_a_usage_error(self, project: Path) -> None:
        result = invoke(project, "run", "--mock", "--tag", "nope")
        assert result.exit_code == 2

    def test_no_store_writes_no_database(self, project: Path) -> None:
        result = invoke(project, "run", "--mock", "--no-store", "--no-report")
        assert result.exit_code == 0
        assert not (project / ".agent_eval" / "results.db").exists()

    def test_unknown_report_format_is_a_usage_error(self, project: Path) -> None:
        result = invoke(project, "run", "--mock", "--format", "pdf")
        assert result.exit_code == 2

    def test_failing_case_exits_1(self, project: Path) -> None:
        write_dataset(
            project,
            "failing.yaml",
            "name: failing\nversion: 1.0.0\ncases:\n"
            "  - id: impossible\n    input: Hello\n"
            "    expected:\n      contains: ['zzz-never-emitted']\n",
        )
        result = invoke(project, "run", "--mock", "--dataset", "evals/failing.yaml")
        assert result.exit_code == 1

    def test_missing_dataset_exits_2(self, project: Path) -> None:
        result = invoke(project, "run", "--mock", "--dataset", "evals/absent.yaml")
        assert result.exit_code == 2


class TestBaselineAndRegression:
    def test_save_and_list(self, project: Path) -> None:
        invoke(project, "run", "--mock", "--save-baseline", "gold")
        result = invoke(project, "baseline", "list")
        assert result.exit_code == 0
        assert "gold" in result.output

    def test_show_reports_stored_metrics(self, project: Path) -> None:
        invoke(project, "run", "--mock", "--save-baseline", "gold")
        result = invoke(project, "baseline", "show", "gold")
        assert result.exit_code == 0
        assert "score" in result.output

    def test_show_unknown_baseline_exits_2(self, project: Path) -> None:
        result = invoke(project, "baseline", "show", "ghost")
        assert result.exit_code == 2

    def test_identical_run_against_baseline_passes(self, project: Path) -> None:
        invoke(project, "run", "--mock", "--save-baseline", "gold", "--no-report")
        invoke(project, "run", "--mock", "--no-report")
        result = invoke(project, "compare", "gold")
        assert result.exit_code == 0

    def test_score_drop_exits_1(self, project: Path) -> None:
        dataset = write_dataset(
            project,
            "regress.yaml",
            "name: regress\nversion: 1.0.0\ncases:\n"
            "  - id: c1\n    input: Hello\n"
            "    expected:\n      contains: ['mock']\n",
        )
        first = invoke(
            project,
            "run",
            "--mock",
            "--dataset",
            str(dataset),
            "--save-baseline",
            "good",
            "--no-report",
        )
        assert first.exit_code == 0

        dataset.write_text(
            "name: regress\nversion: 1.0.0\ncases:\n"
            "  - id: c1\n    input: Hello\n"
            "    expected:\n      contains: ['zzz-regressed']\n",
            encoding="utf-8",
        )
        second = invoke(
            project,
            "run",
            "--mock",
            "--dataset",
            str(dataset),
            "--baseline",
            "good",
            "--no-report",
        )
        assert second.exit_code == 1
        assert "score_drop" in second.output

    def test_run_never_compares_against_itself(self, project: Path) -> None:
        invoke(project, "run", "--mock", "--no-report")
        second = invoke(project, "run", "--mock")
        assert second.exit_code == 0
        current_run_id = json.loads((project / "reports" / "latest.json").read_text())["run_id"]
        assert f"run:{current_run_id}" not in second.output

    def test_compare_without_stored_runs_exits_2(self, project: Path) -> None:
        result = invoke(project, "compare")
        assert result.exit_code == 2

    def test_config_mismatch_is_reported(self, project: Path) -> None:
        invoke(project, "run", "--mock", "--save-baseline", "gold", "--no-report")
        config = read_config(project)
        config["judge"]["criteria"] = ["accuracy"]
        (project / "agent-eval.yaml").write_text(yaml.safe_dump(config))
        invoke(project, "run", "--mock", "--no-report")
        result = invoke(project, "compare", "gold")
        assert result.exit_code == 1
        assert "config_changed" in result.output

    def test_config_mismatch_can_be_ignored(self, project: Path) -> None:
        invoke(project, "run", "--mock", "--save-baseline", "gold", "--no-report")
        config = read_config(project)
        config["judge"]["criteria"] = ["accuracy"]
        (project / "agent-eval.yaml").write_text(yaml.safe_dump(config))
        invoke(project, "run", "--mock", "--no-report")
        result = invoke(project, "compare", "gold", "--ignore-config-mismatch")
        assert result.exit_code == 0


class TestReport:
    def test_regenerates_from_storage(self, project: Path) -> None:
        invoke(project, "run", "--mock", "--no-report")
        result = invoke(project, "report", "--format", "html")
        assert result.exit_code == 0
        assert list((project / "reports").glob("*.html"))

    def test_stdout_mode_renders_markdown(self, project: Path) -> None:
        invoke(project, "run", "--mock", "--no-report")
        result = invoke(project, "report", "--stdout")
        assert result.exit_code == 0
        assert "# Evaluation report" in result.output

    def test_unknown_run_id_exits_2(self, project: Path) -> None:
        invoke(project, "run", "--mock", "--no-report")
        result = invoke(project, "report", "run_does_not_exist")
        assert result.exit_code == 2

    def test_no_stored_runs_exits_2(self, project: Path) -> None:
        result = invoke(project, "report")
        assert result.exit_code == 2

    def test_all_three_formats_render_the_same_numbers(self, project: Path) -> None:
        invoke(project, "run", "--mock", "--no-report")
        invoke(project, "report", "--format", "json", "--format", "markdown", "--format", "html")
        payload = json.loads(next((project / "reports").glob("*.json")).read_text())
        assert payload["aggregate"]["total"] == 3
        markdown = next((project / "reports").glob("*.md")).read_text()
        assert payload["run_id"] in markdown


class TestDatasets:
    def test_show_summarises_the_dataset(self, project: Path) -> None:
        result = invoke(project, "datasets", "show")
        assert result.exit_code == 0
        assert "cases:" in result.output

    def test_list_ignores_non_semver_filenames(self, project: Path) -> None:
        write_dataset(
            project,
            "dataset_v2.yaml",
            "name: d\nversion: 2.0.0\ncases:\n  - id: a\n    input: hi\n",
        )
        result = invoke(project, "datasets", "list")
        assert result.exit_code == 0

    def test_register_records_metadata(self, project: Path) -> None:
        result = invoke(project, "datasets", "register")
        assert result.exit_code == 0


class TestHelp:
    @pytest.mark.parametrize(
        "command",
        ["init", "validate", "run", "report", "compare", "baseline", "datasets"],
    )
    def test_every_command_is_reachable(self, project: Path, command: str) -> None:
        result = invoke(project, command, "--help")
        assert result.exit_code == 0
        assert command in result.output


class TestSubprocessEntryPoint:
    """Prove the packaged CLI works outside the test process."""

    def test_python_dash_m_exit_codes(self, project: Path) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "agent_eval", "run", "--mock"],
            cwd=project,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        # The mock warning is a diagnostic, so it belongs on stderr.
        assert "MOCK RUN" in completed.stderr
        assert "passed" in completed.stdout

    def test_failing_run_exits_1(self, project: Path) -> None:
        write_dataset(
            project,
            "failing.yaml",
            "name: failing\nversion: 1.0.0\ncases:\n"
            "  - id: impossible\n    input: Hello\n"
            "    expected:\n      contains: ['zzz-never-emitted']\n",
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "agent_eval",
                "run",
                "--mock",
                "--dataset",
                "evals/failing.yaml",
            ],
            cwd=project,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 1
