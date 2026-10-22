"""Dataset versioning helpers.

Dataset versions are semantic versions (``major.minor.patch``) declared in the
file itself. A dataset directory may hold several versioned files; these helpers
discover them, sort them and report which is the latest.
"""

from __future__ import annotations

import re
from pathlib import Path

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-+][0-9A-Za-z.\-]+)?$")
_VERSION_IN_NAME_RE = re.compile(r"_v(\d+(?:\.\d+)*)$")


def parse_version(version: str) -> tuple[int, int, int, str]:
    """Split a semver string into a comparable tuple.

    Args:
        version: A semantic version string.

    Returns:
        ``(major, minor, patch, suffix)``.

    Raises:
        ValueError: If the string is not a semantic version.
    """
    match = _SEMVER_RE.match(version)
    if match is None:
        raise ValueError(f"Not a semantic version: {version!r}")
    major, minor, patch = (int(match.group(i)) for i in (1, 2, 3))
    suffix = version[match.end(3) :]
    return major, minor, patch, suffix


def version_key(version: str) -> tuple[int, int, int, int, str]:
    """Sort key for a semantic version; releases sort before pre-releases."""
    major, minor, patch, suffix = parse_version(version)
    # A non-empty suffix denotes a pre-release, which sorts before the release.
    return major, minor, patch, 0 if suffix else 1, suffix


def is_newer(candidate: str, reference: str) -> bool:
    """Whether ``candidate`` is a strictly newer version than ``reference``."""
    return version_key(candidate) > version_key(reference)


def bump(version: str, part: str = "patch") -> str:
    """Return ``version`` with the requested component incremented.

    Args:
        version: Current semantic version.
        part: ``"major"``, ``"minor"`` or ``"patch"``.

    Raises:
        ValueError: For an unknown part or an invalid version.
    """
    major, minor, patch, _ = parse_version(version)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"Unknown version part: {part!r}")


def version_from_filename(path: Path) -> str | None:
    """Extract a version from a filename such as ``dataset_v2.yaml``."""
    match = _VERSION_IN_NAME_RE.search(path.stem)
    return match.group(1) if match else None


def discover_versions(directory: Path, stem: str = "dataset") -> list[tuple[str, Path]]:
    """Find all versioned dataset files in ``directory``.

    Args:
        directory: Directory to scan.
        stem: Base filename, matched against ``<stem>*.yaml``.

    Returns:
        ``(version, path)`` pairs sorted oldest to newest. Files without a
        version in their name are skipped because their version can only be
        known by reading them; names whose version is not semantic (for example
        ``dataset_v2.yaml``) are skipped too, since they cannot be ordered.
    """
    if not directory.is_dir():
        return []
    found: list[tuple[str, Path]] = []
    for path in sorted(directory.glob(f"{stem}*.yaml")):
        version = version_from_filename(path)
        if version is None:
            continue
        try:
            version_key(version)
        except ValueError:
            continue
        found.append((version, path))
    return sorted(found, key=lambda item: version_key(item[0]))


def latest_version_path(directory: Path, stem: str = "dataset") -> Path | None:
    """Path of the highest-numbered versioned dataset file, or ``None``."""
    versions = discover_versions(directory, stem)
    return versions[-1][1] if versions else None
