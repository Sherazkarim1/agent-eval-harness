"""Structured judge prompt and response schema."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_JUDGE_PROMPT = """You are grading the output of a customer-support AI agent.

You will be given the user's request, the agent's answer, the reference answer
when one exists, and a summary of the tools the agent called.

Grade the agent's answer on these criteria, each from 0.0 (unacceptable) to
1.0 (excellent):
{criteria}

Reply with a single JSON object and nothing else:
{{"score": <float 0-1>, "reason": "<one or two sentences>", "criteria": {{{criteria_json}}}}}

USER REQUEST:
{case_input}

AGENT ANSWER:
{agent_output}

REFERENCE ANSWER:
{reference_answer}

TOOLS CALLED:
{tool_calls}

JSON:"""


class JudgeCriteria(BaseModel):
    """Per-criterion scores returned by the judge."""

    model_config = ConfigDict(extra="allow")

    correctness: float | None = Field(default=None, ge=0.0, le=1.0)
    relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    completeness: float | None = Field(default=None, ge=0.0, le=1.0)


class JudgeResponse(BaseModel):
    """The structured verdict a judge must produce.

    ``criteria`` values are left unvalidated here on purpose: a judge that
    returns ``{"correctness": "high"}`` should be partially usable rather than
    failing the whole verdict. Non-numeric entries are dropped by
    :func:`parse_judge_response`.
    """

    model_config = ConfigDict(extra="ignore")

    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    criteria: dict[str, Any] = Field(default_factory=dict)


def render_prompt(
    *,
    case_input: str,
    agent_output: str,
    reference_answer: str | None,
    tool_calls: str,
    criteria: list[str],
    template: str | None = None,
) -> str:
    """Render the judge prompt.

    Args:
        case_input: The evaluation case input.
        agent_output: The (redacted) agent output.
        reference_answer: Reference answer, or ``None``.
        tool_calls: Rendered tool-call summary.
        criteria: Criterion names to score.
        template: Optional custom template; ``None`` uses the built-in prompt.

    Returns:
        The prompt string to send to the judge provider.
    """
    base = template or DEFAULT_JUDGE_PROMPT
    criteria_json = json.dumps(dict.fromkeys(criteria, 0.0))
    criteria_bullets = "\n".join(f"- {name}" for name in criteria) or "- overall quality"
    return base.format(
        criteria=criteria_bullets,
        criteria_json=criteria_json,
        case_input=case_input,
        agent_output=agent_output,
        reference_answer=reference_answer or "(none provided)",
        tool_calls=tool_calls or "(no tool calls)",
    )


def parse_judge_response(text: str, criteria: list[str]) -> tuple[float, str, dict[str, float]]:
    """Parse a judge response.

    Args:
        text: Raw model output.
        criteria: Expected criterion names.

    Returns:
        ``(score, reason, criteria_scores)``.

    Raises:
        ValueError: If the output cannot be parsed into the structured schema.
    """
    payload = extract_json_object(text)
    if payload is None:
        raise ValueError("judge response did not contain a JSON object")
    parsed = JudgeResponse.model_validate(payload)

    scores: dict[str, float] = {}
    for name, value in parsed.criteria.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        scores[name] = max(0.0, min(1.0, float(value)))

    for name in criteria:
        scores.setdefault(name, parsed.score)
    if not scores:
        scores = {"overall": parsed.score}
    return parsed.score, parsed.reason, scores


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a possibly fenced / chatty response."""
    stripped = text.strip()
    if "```" in stripped:
        blocks = stripped.split("```")
        for block in blocks:
            candidate = _try_json(block)
            if candidate is not None:
                return candidate
    candidate = _try_json(stripped)
    if candidate is not None:
        return candidate

    start = stripped.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(stripped)):
            char = stripped[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = _try_json(stripped[start : index + 1])
                    if candidate is not None:
                        return candidate
                    break
        start = stripped.find("{", start + 1)
    return None


def _try_json(text: str) -> dict[str, Any] | None:
    """Parse ``text`` as a JSON object, ignoring language tags around it."""
    candidate = text.strip()
    if candidate.lower().startswith("json"):
        candidate = candidate[4:].strip()
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None
