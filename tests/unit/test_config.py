"""Tests for configuration loading, validation and hashing."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_eval.config import (
    DEFAULT_CONFIG_FILENAMES,
    HarnessConfig,
    PricingConfig,
    PricingEntry,
    RegressionConfig,
    ScoringConfig,
    ScoringWeights,
    default_config,
    find_config_file,
    load_config,
)

VALID = """
project:
  name: demo
dataset:
  path: evals/dataset.yaml
agent:
  type: mock
judge:
  provider: mock
scoring:
  min_overall_score: 0.8
regression:
  max_score_drop: 0.1
storage:
  type: sqlite
  path: .agent_eval/results.db
"""


def write(tmp_path: Path, content: str, name: str = "agent-eval.yaml") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


class TestDefaults:
    def test_default_config_is_valid(self) -> None:
        config = default_config()
        assert config.agent.type == "mock"
        assert config.judge.provider == "mock"
        assert config.storage.type == "sqlite"

    def test_weights_sum_to_something_positive(self) -> None:
        weights = default_config().scoring.weights
        assert weights.deterministic + weights.judge + weights.performance > 0

    def test_offline_defaults_require_no_credentials(self) -> None:
        config = default_config()
        assert config.judge.provider == "mock"
        assert config.pricing.models == {}
        assert config.run.mode == "mock"


class TestLoadConfig:
    def test_loads_valid_file(self, tmp_path: Path) -> None:
        config = load_config(write(tmp_path, VALID))
        assert config.project.name == "demo"
        assert config.dataset.path == Path("evals/dataset.yaml")
        assert config.scoring.min_overall_score == 0.8
        assert config.regression.max_score_drop == 0.1

    def test_explicit_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    def test_no_config_discovered_returns_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert load_config() == default_config()

    def test_no_config_with_required_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="agent-eval init"):
            load_config(required=True)

    def test_empty_file_uses_defaults(self, tmp_path: Path) -> None:
        assert load_config(write(tmp_path, "")) == default_config()

    def test_non_mapping_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="YAML mapping"):
            load_config(write(tmp_path, "- a\n- b\n"))

    def test_unknown_field_rejected(self, tmp_path: Path) -> None:
        content = VALID + "\nunknown_section:\n  x: 1\n"
        with pytest.raises(ValidationError):
            load_config(write(tmp_path, content))

    def test_typo_in_field_rejected(self, tmp_path: Path) -> None:
        content = VALID.replace("type: mock", "typo: mock")
        with pytest.raises(ValidationError):
            load_config(write(tmp_path, content))

    def test_invalid_judge_provider_rejected(self, tmp_path: Path) -> None:
        content = VALID.replace("provider: mock", "provider: not-a-provider")
        with pytest.raises(ValidationError):
            load_config(write(tmp_path, content))

    def test_partial_config_uses_defaults_for_the_rest(self, tmp_path: Path) -> None:
        config = load_config(write(tmp_path, "project:\n  name: minimal\n"))
        assert config.project.name == "minimal"
        assert config.agent.type == "mock"


class TestValidation:
    def test_rejects_negative_score_drop(self) -> None:
        with pytest.raises(ValidationError):
            RegressionConfig(max_score_drop=-0.1)

    def test_rejects_score_above_one(self) -> None:
        with pytest.raises(ValidationError):
            ScoringConfig(min_overall_score=1.5)

    def test_rejects_non_positive_timeout(self) -> None:
        with pytest.raises(ValidationError):
            HarnessConfig(run={"timeout_s": 0})  # type: ignore[typeddict-item]

    def test_rejects_all_zero_weights(self) -> None:
        with pytest.raises(ValidationError):
            ScoringWeights(deterministic=0, judge=0, trajectory=0, performance=0)

    def test_rejects_negative_pricing(self) -> None:
        with pytest.raises(ValidationError):
            PricingEntry(input_per_million=-1.0)

    def test_unknown_storage_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HarnessConfig(storage={"type": "postgres"})  # type: ignore[typeddict-item]

    def test_pricing_defaults_to_free(self) -> None:
        assert PricingConfig().models == {}


class TestFingerprint:
    def test_is_stable(self) -> None:
        assert default_config().fingerprint() == default_config().fingerprint()

    def test_changes_with_behaviour(self) -> None:
        a = HarnessConfig()
        b = HarnessConfig(agent={"type": "mock", "name": "different"})  # type: ignore[typeddict-item]
        assert a.fingerprint() != b.fingerprint()

    def test_ignores_report_and_run_settings(self) -> None:
        a = HarnessConfig()
        b = HarnessConfig(report={"output_dir": Path("elsewhere")})  # type: ignore[typeddict-item]
        assert a.fingerprint() == b.fingerprint()

    def test_is_short_hex(self) -> None:
        fingerprint = default_config().fingerprint()
        assert len(fingerprint) == 16
        assert all(char in "0123456789abcdef" for char in fingerprint)


class TestDiscovery:
    def test_finds_config_in_parent(self, tmp_path: Path) -> None:
        write(tmp_path, VALID)
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        found = find_config_file(nested)
        assert found is not None
        assert found.name == "agent-eval.yaml"

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert find_config_file(tmp_path) is None

    def test_knows_default_filenames(self) -> None:
        assert "agent-eval.yaml" in DEFAULT_CONFIG_FILENAMES


class TestProjectConfigFile:
    def test_bundled_config_is_valid(self, project_root: Path) -> None:
        config = load_config(project_root / "agent-eval.yaml")
        assert config.project.name == "customer-support-agent"
        assert config.agent.type == "mock"

    def test_bundled_dataset_loads(self, project_root: Path) -> None:
        from agent_eval.datasets import load_dataset_file

        dataset = load_dataset_file(project_root / config_path(project_root))
        assert dataset.name == "customer_support"


def config_path(project_root: Path) -> str:
    return "evals/dataset.yaml"
