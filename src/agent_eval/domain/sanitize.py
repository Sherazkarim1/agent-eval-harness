"""Secret redaction for trajectories, logs and reports.

The harness stores whatever the agent produced, so it must scrub values that
look like credentials before anything is persisted or rendered. Redaction is
applied by the evaluator immediately after the agent returns, so no adapter can
accidentally leak a secret into the database.
"""

from __future__ import annotations

import re
from typing import Any, Final

REDACTED: Final[str] = "***"

#: Environment-variable names whose values are always masked.
SENSITIVE_ENV_KEYS: Final[frozenset[str]] = frozenset(
    {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "COHERE_API_KEY",
        "MISTRAL_API_KEY",
        "GROQ_API_KEY",
        "TOGETHER_API_KEY",
        "OPENROUTER_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "STRIPE_SECRET_KEY",
        "STRIPE_API_KEY",
        "SLACK_TOKEN",
        "SLACK_BOT_TOKEN",
        "DATABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
    }
)

#: ``KEY=value`` pairs inside free text (dotenv snippets, command lines).
_ASSIGNMENT_RE = re.compile(
    r"(?i)\b("
    + "|".join(sorted(SENSITIVE_ENV_KEYS))
    + r")\s*[=:]\s*(\"[^\"]*\"|'[^']*'|[^\s,;)\]}]+)"
)

#: Provider-style tokens that carry no key name.
_BEARER_RE = re.compile(r"(?i)\b(bearer|token)\s+([A-Za-z0-9._\-]{8,})")

#: Well-known credential prefixes.
_PREFIXED_RE = re.compile(
    r"\b(sk-[A-Za-z0-9_\-]{8,}"
    r"|sk-ant-[A-Za-z0-9_\-]{8,}"
    r"|gh[pousr]_[A-Za-z0-9]{16,}"
    r"|AKIA[0-9A-Z]{12,}"
    r"|xox[abposr]-[A-Za-z0-9\-]{10,}"
    r"|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,})"
)

#: JSON web tokens (``{"token": "..."}``) and header-style pairs.
_QUOTED_FIELD_RE = re.compile(
    r"(?i)\"([A-Za-z0-9_\-]*(?:api[_-]?key|secret|password|passwd|token|authorization)"
    r"|access[_-]?key|private[_-]?key|credential)s?\"\s*:\s*\"[^\"]*\""
)

_HEADER_RE = re.compile(r"(?i)\b(authorization|x-api-key|x-auth-token)\s*:\s*([^\r\n]+)")


class SecretRedactor:
    """Replaces credential-looking substrings with :data:`REDACTED`.

    Patterns are applied to the whole string, not per line, so a secret split
    across lines cannot slip through.
    """

    def __init__(self, extra_patterns: list[str] | None = None) -> None:
        """Create a redactor.

        Args:
            extra_patterns: Additional regular expressions; every match is
                replaced by :data:`REDACTED`.
        """
        self._extra = [re.compile(p) for p in (extra_patterns or [])]

    def scrub_text(self, text: str) -> str:
        """Return ``text`` with all recognised credentials masked."""
        if not text:
            return text
        scrubbed = _ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}={REDACTED}", text)
        scrubbed = _QUOTED_FIELD_RE.sub(lambda m: f'"{m.group(1)}": "{REDACTED}"', scrubbed)
        scrubbed = _HEADER_RE.sub(lambda m: f"{m.group(1)}: {REDACTED}", scrubbed)
        scrubbed = _BEARER_RE.sub(lambda m: f"{m.group(1)} {REDACTED}", scrubbed)
        scrubbed = _PREFIXED_RE.sub(REDACTED, scrubbed)
        for pattern in self._extra:
            scrubbed = pattern.sub(REDACTED, scrubbed)
        return scrubbed

    def scrub(self, value: Any, *, _depth: int = 0) -> Any:
        """Recursively scrub a JSON-like structure.

        Mapping keys listed in :data:`SENSITIVE_ENV_KEYS` are masked wholesale;
        other values are scrubbed as text.
        """
        if _depth > 12:
            return "<max-depth>"
        if value is None or isinstance(value, bool | int | float):
            return value
        if isinstance(value, str):
            return self.scrub_text(value)
        if isinstance(value, dict):
            out: dict[Any, Any] = {}
            for key, item in value.items():
                if isinstance(key, str) and (
                    key.upper() in SENSITIVE_ENV_KEYS or _is_secret_field(key)
                ):
                    out[key] = REDACTED
                else:
                    out[key] = self.scrub(item, _depth=_depth + 1)
            return out
        if isinstance(value, list | tuple | set):
            return [self.scrub(item, _depth=_depth + 1) for item in value]
        return self.scrub_text(str(value))

    def scrub_env(self, env: dict[str, str]) -> dict[str, str]:
        """Return ``env`` with sensitive values replaced (key names preserved)."""
        return {
            key: (REDACTED if key.upper() in SENSITIVE_ENV_KEYS else self.scrub_text(value))
            for key, value in env.items()
        }


def _is_secret_field(key: str) -> bool:
    """Whether a mapping key names a credential-bearing field."""
    lowered = key.lower()
    return lowered in {
        "api_key",
        "apikey",
        "secret",
        "secret_key",
        "password",
        "passwd",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "private_key",
        "client_secret",
        "credentials",
    }


def default_redactor() -> SecretRedactor:
    """Return a redactor with the default pattern set."""
    return SecretRedactor()
