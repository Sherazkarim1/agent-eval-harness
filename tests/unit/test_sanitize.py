"""Tests for secret redaction."""

from __future__ import annotations

from agent_eval.domain.sanitize import REDACTED, SecretRedactor


class TestTextRedaction:
    def test_env_assignment(self) -> None:
        result = SecretRedactor().scrub_text("OPENAI_API_KEY=sk-abc123def456ghi")
        assert result == f"OPENAI_API_KEY={REDACTED}"
        assert "sk-abc" not in result

    def test_multiple_assignments(self) -> None:
        text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE and AWS_SECRET_ACCESS_KEY=wJalr"
        result = SecretRedactor().scrub_text(text)
        assert "AKIA" not in result
        assert "wJalr" not in result

    def test_json_secret_field(self) -> None:
        result = SecretRedactor().scrub_text('{"api_key": "super-secret-value"}')
        assert "super-secret-value" not in result

    def test_bearer_token(self) -> None:
        result = SecretRedactor().scrub_text("Authorization: Bearer abc123def456ghi")
        assert "abc123def456ghi" not in result

    def test_authorization_header(self) -> None:
        result = SecretRedactor().scrub_text("authorization: token-abcdefghijkl")
        assert "abcdefghijkl" not in result

    def test_prefixed_sk_token(self) -> None:
        assert "sk-ant-abcdefgh" not in SecretRedactor().scrub_text("sk-ant-abcdefghijkl")

    def test_github_token(self) -> None:
        assert "ghp_" not in SecretRedactor().scrub_text("ghp_abcdefghijklmnopqrstuvwx")

    def test_jwt(self) -> None:
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N"
        assert "dozjgNryP4J3jVmNHl0w5N" not in SecretRedactor().scrub_text(f"token={jwt}")

    def test_secret_split_across_lines(self) -> None:
        result = SecretRedactor().scrub_text("OPENAI_API_KEY=\nsk-abcdefgh12345678")
        assert "sk-abcdefgh12345678" not in result

    def test_ordinary_text_untouched(self) -> None:
        text = "Your refund was approved within 30 days."
        assert SecretRedactor().scrub_text(text) == text

    def test_empty_string(self) -> None:
        assert SecretRedactor().scrub_text("") == ""

    def test_extra_pattern(self) -> None:
        redactor = SecretRedactor(extra_patterns=[r"CUSTOM-\d+"])
        assert "CUSTOM-42" not in redactor.scrub_text("id=CUSTOM-42")


class TestStructureRedaction:
    def test_nested_dict(self) -> None:
        data = {"outer": {"password": "hunter2", "name": "alice"}}
        result = SecretRedactor().scrub(data)
        assert result["outer"]["password"] == REDACTED
        assert result["outer"]["name"] == "alice"

    def test_sensitive_key_wholesale(self) -> None:
        result = SecretRedactor().scrub({"OPENAI_API_KEY": "anything"})
        assert result["OPENAI_API_KEY"] == REDACTED

    def test_list_values(self) -> None:
        result = SecretRedactor().scrub(["safe", {"token": "abc123"}])
        assert result[0] == "safe"
        assert result[1]["token"] == REDACTED

    def test_primitives_pass_through(self) -> None:
        redactor = SecretRedactor()
        assert redactor.scrub(None) is None
        assert redactor.scrub(42) == 42
        assert redactor.scrub(True) is True
        assert redactor.scrub(1.5) == 1.5

    def test_scrub_text_inside_values(self) -> None:
        result = SecretRedactor().scrub({"note": "key=sk-abcdefgh12345678"})
        assert "sk-abcdefgh" not in result["note"]

    def test_depth_limit_prevents_infinite_recursion(self) -> None:
        deep: dict = {"a": "b"}
        for _ in range(20):
            deep = {"nested": deep}
        assert SecretRedactor().scrub(deep) is not None

    def test_non_string_keys(self) -> None:
        assert SecretRedactor().scrub({1: "a"}) == {1: "a"}


class TestEnvRedaction:
    def test_masks_sensitive_values_keeps_keys(self) -> None:
        env = {"OPENAI_API_KEY": "sk-abc", "HOME": "/home/user"}
        result = SecretRedactor().scrub_env(env)
        assert result["OPENAI_API_KEY"] == REDACTED
        assert result["HOME"] == "/home/user"

    def test_scrubs_ordinary_values(self) -> None:
        env = {"HOME": "/home/user", "PATH": "/usr/bin"}
        assert SecretRedactor().scrub_env(env) == env
