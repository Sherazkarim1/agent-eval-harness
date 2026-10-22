"""JSON report writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_eval.reporting.common import build_payload


def render(payload: dict[str, Any]) -> str:
    """Render a report payload as pretty-printed JSON."""
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def write(
    path: Path,
    record: Any,
    cases: list[Any],
    **kwargs: Any,
) -> Path:
    """Write a JSON report to ``path``.

    Args:
        path: Destination file; parent directories are created as needed.
        record: The run record.
        cases: Per-case results.
        **kwargs: Forwarded to :func:`build_payload`.

    Returns:
        The path written.
    """
    payload = build_payload(record, cases, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(payload), encoding="utf-8")
    return path
