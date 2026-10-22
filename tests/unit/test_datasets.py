"""Tests for dataset loading, validation and versioning."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_eval.datasets import (
    DatasetError,
    bump,
    discover_versions,
    is_newer,
    latest_version_path,
    load_dataset_file,
    parse_dataset,
    parse_version,
    summarise_dataset,
    validate_dataset_file,
    version_from_filename,
)
from agent_eval.datasets.versioning import version_key

MINIMAL = """
name: demo
version: 1.0.0
cases:
  - id: a
    input: hello
    expected:
      contains: ["hello"]
"""


def write(tmp_path: Path, content: str, name: str = "dataset.yaml") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


class TestLoadDataset:
    def test_loads_minimal_dataset(self, tmp_path: Path) -> None:
        dataset = load_dataset_file(write(tmp_path, MINIMAL))
        assert dataset.name == "demo"
        assert dataset.version == "1.0.0"
        assert dataset.case_ids() == ["a"]
        assert dataset.cases[0].expected.contains == ["hello"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset_file(tmp_path / "nope.yaml")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = write(tmp_path, "")
        with pytest.raises(DatasetError, match="empty"):
            load_dataset_file(path)

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        path = write(tmp_path, "name: [unclosed\n")
        with pytest.raises(DatasetError):
            load_dataset_file(path)

    def test_non_mapping_raises(self, tmp_path: Path) -> None:
        path = write(tmp_path, "- just\n- a\n- list\n")
        with pytest.raises(DatasetError, match="must be a mapping"):
            load_dataset_file(path)

    def test_duplicate_ids_raise(self, tmp_path: Path) -> None:
        content = """
name: demo
version: 1.0.0
cases:
  - id: a
    input: one
  - id: a
    input: two
"""
        with pytest.raises(DatasetError, match="Duplicate case id: a"):
            load_dataset_file(write(tmp_path, content))

    def test_invalid_version_is_reported(self, tmp_path: Path) -> None:
        content = MINIMAL.replace("version: 1.0.0", "version: not-a-version")
        with pytest.raises(DatasetError):
            load_dataset_file(write(tmp_path, content))

    def test_error_lists_all_issues(self, tmp_path: Path) -> None:
        content = """
name: demo
version: 1.0.0
cases:
  - id: "bad id"
    input: ""
"""
        with pytest.raises(DatasetError) as exc:
            load_dataset_file(write(tmp_path, content))
        assert len(exc.value.issues) >= 2

    def test_issue_locations_include_case_index(self, tmp_path: Path) -> None:
        content = """
name: demo
version: 1.0.0
cases:
  - id: good
    input: hello
  - input: missing id
"""
        with pytest.raises(DatasetError) as exc:
            load_dataset_file(write(tmp_path, content))
        assert any(issue.case_id == "cases[1]" for issue in exc.value.issues)


class TestParseDataset:
    def test_accepts_mapping_directly(self) -> None:
        dataset = parse_dataset(
            {"name": "d", "version": "1.0.0", "cases": [{"id": "a", "input": "x"}]}
        )
        assert dataset.name == "d"

    def test_rejects_list(self) -> None:
        with pytest.raises(DatasetError):
            parse_dataset([1, 2, 3])

    def test_extra_top_level_field_rejected(self) -> None:
        with pytest.raises(DatasetError):
            parse_dataset(
                {"name": "d", "version": "1.0.0", "typo": 1, "cases": [{"id": "a", "input": "x"}]}
            )


class TestValidateDatasetFile:
    def test_valid_file_returns_no_issues(self, tmp_path: Path) -> None:
        assert validate_dataset_file(write(tmp_path, MINIMAL)) == []

    def test_invalid_file_returns_issues(self, tmp_path: Path) -> None:
        path = write(tmp_path, "name: demo\nversion: 1.0.0\ncases: []\n")
        assert validate_dataset_file(path)

    def test_missing_file_returns_issue(self, tmp_path: Path) -> None:
        issues = validate_dataset_file(tmp_path / "nope.yaml")
        assert len(issues) == 1
        assert "does not exist" in issues[0].message


class TestVersioning:
    def test_parse_version(self) -> None:
        assert parse_version("1.2.3") == (1, 2, 3, "")

    def test_parse_prerelease(self) -> None:
        assert parse_version("1.0.0-rc1") == (1, 0, 0, "-rc1")

    def test_parse_invalid(self) -> None:
        with pytest.raises(ValueError):
            parse_version("1.2")

    def test_prerelease_sorts_before_release(self) -> None:
        assert version_key("1.0.0-rc1") < version_key("1.0.0")

    def test_is_newer(self) -> None:
        assert is_newer("1.1.0", "1.0.9") is True
        assert is_newer("1.0.0", "1.0.0") is False
        assert is_newer("1.0.0", "2.0.0") is False

    @pytest.mark.parametrize(
        ("part", "expected"),
        [("major", "2.0.0"), ("minor", "1.3.0"), ("patch", "1.2.4")],
    )
    def test_bump(self, part: str, expected: str) -> None:
        assert bump("1.2.3", part) == expected

    def test_bump_invalid_part(self) -> None:
        with pytest.raises(ValueError):
            bump("1.2.3", "nope")

    def test_version_from_filename(self) -> None:
        assert version_from_filename(Path("dataset_v2.yaml")) == "2"
        assert version_from_filename(Path("dataset_v1.2.3.yaml")) == "1.2.3"
        assert version_from_filename(Path("dataset.yaml")) is None

    def test_discover_versions_sorted(self, tmp_path: Path) -> None:
        for version in ("1.0.0", "1.10.0", "1.2.0"):
            write(tmp_path, MINIMAL, f"dataset_v{version}.yaml")
        found = discover_versions(tmp_path)
        assert [v for v, _ in found] == ["1.0.0", "1.2.0", "1.10.0"]

    def test_discover_ignores_unversioned(self, tmp_path: Path) -> None:
        write(tmp_path, MINIMAL, "dataset.yaml")
        write(tmp_path, MINIMAL, "dataset_v1.0.0.yaml")
        assert len(discover_versions(tmp_path)) == 1

    def test_latest_version_path(self, tmp_path: Path) -> None:
        write(tmp_path, MINIMAL, "dataset_v1.0.0.yaml")
        latest = write(tmp_path, MINIMAL, "dataset_v2.0.0.yaml")
        assert latest_version_path(tmp_path) == latest

    def test_missing_directory(self, tmp_path: Path) -> None:
        assert discover_versions(tmp_path / "nope") == []
        assert latest_version_path(tmp_path / "nope") is None


class TestSummary:
    def test_summarise_lists_tags(self, example_dataset) -> None:
        summary = summarise_dataset(example_dataset)
        assert "customer_support@1.0.0" in summary
        assert "7 cases" in summary
        assert "billing" in summary
