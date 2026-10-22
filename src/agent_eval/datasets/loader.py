"""Dataset loading and validation.

Every failure mode produces an actionable :class:`DatasetIssue` rather than a
bare traceback: a missing file, malformed YAML, a schema violation, or a
duplicate case id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from agent_eval.datasets.versioning import parse_version
from agent_eval.domain.models import Dataset, DatasetIssue


class DatasetError(ValueError):
    """Raised when a dataset cannot be loaded.

    The message includes every collected issue so the user can fix the file in
    one pass.
    """

    def __init__(self, path: Path, issues: list[DatasetIssue]) -> None:
        """Create an error carrying the path and every collected issue."""
        self.path = path
        self.issues = issues
        detail = "\n".join(f"  - [{issue.location}] {issue.message}" for issue in issues)
        super().__init__(f"Invalid dataset {path}:\n{detail}")


def load_dataset_file(path: Path | str) -> Dataset:
    """Load and validate a dataset from a YAML file.

    Args:
        path: Path to the dataset YAML.

    Returns:
        The validated dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        DatasetError: If the YAML is malformed or fails validation.
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    try:
        raw = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DatasetError(
            dataset_path, [DatasetIssue(location="<yaml>", message=str(exc))]
        ) from exc

    if raw is None:
        raise DatasetError(
            dataset_path,
            [DatasetIssue(location="<root>", message="Dataset file is empty")],
        )
    return parse_dataset(raw, source=dataset_path)


def parse_dataset(raw: Any, *, source: Path | str = "<memory>") -> Dataset:
    """Validate a raw dataset mapping.

    Args:
        raw: Parsed YAML content.
        source: Path used in error messages.

    Returns:
        The validated dataset.

    Raises:
        DatasetError: If validation fails or case ids are duplicated.
    """
    source_path = Path(source)
    if not isinstance(raw, dict):
        raise DatasetError(
            source_path,
            [
                DatasetIssue(
                    location="<root>",
                    message=f"Dataset must be a mapping, got {type(raw).__name__}",
                )
            ],
        )

    issues: list[DatasetIssue] = []
    version = raw.get("version")
    if version is not None:
        try:
            parse_version(str(version))
        except ValueError as exc:
            issues.append(DatasetIssue(location="version", message=str(exc)))

    try:
        dataset = Dataset.model_validate(raw)
    except ValidationError as exc:
        issues.extend(_issues_from_validation_error(exc))
        raise DatasetError(source_path, issues) from exc

    duplicates = _duplicate_ids(dataset)
    if duplicates:
        raise DatasetError(
            source_path,
            [
                DatasetIssue(location="cases", message=f"Duplicate case id: {case_id}")
                for case_id in duplicates
            ],
        )
    return dataset


def validate_dataset_file(path: Path | str) -> list[DatasetIssue]:
    """Validate a dataset file without raising.

    Returns:
        An empty list when the file is valid, otherwise one issue per problem.
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        return [DatasetIssue(location=str(dataset_path), message="File does not exist")]
    try:
        load_dataset_file(dataset_path)
    except FileNotFoundError:
        return [DatasetIssue(location=str(dataset_path), message="File does not exist")]
    except DatasetError as exc:
        return exc.issues
    return []


def _duplicate_ids(dataset: Dataset) -> list[str]:
    """Return case ids that appear more than once."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for case_id in dataset.case_ids():
        if case_id in seen and case_id not in duplicates:
            duplicates.append(case_id)
        seen.add(case_id)
    return duplicates


def _issues_from_validation_error(exc: ValidationError) -> list[DatasetIssue]:
    """Flatten a Pydantic error into dataset issues."""
    issues: list[DatasetIssue] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        case_id = _case_id_from_location(error["loc"])
        issues.append(DatasetIssue(case_id=case_id, location=location, message=error["msg"]))
    return issues


def _case_id_from_location(location: tuple[Any, ...]) -> str | None:
    """Extract a case id from a Pydantic error location when present."""
    for index, part in enumerate(location):
        if part == "cases" and index + 1 < len(location):
            candidate = location[index + 1]
            if isinstance(candidate, int):
                return f"cases[{candidate}]"
    return None


def summarise_dataset(dataset: Dataset) -> str:
    """One-line human summary of a dataset."""
    tags = sorted({tag for case in dataset.cases for tag in case.tags})
    return f"{dataset.name}@{dataset.version}: {len(dataset.cases)} cases" + (
        f", tags: {', '.join(tags)}" if tags else ""
    )
