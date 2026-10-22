"""Tests for structure assertions and the JSON Schema subset."""

from __future__ import annotations

import pytest

from agent_eval.domain.assertions.context import AssertionContext
from agent_eval.domain.assertions.structure import (
    check_required_fields,
    check_schema,
    check_valid_json,
    validate_json_schema,
)
from agent_eval.domain.models import AssertionKind
from tests.conftest import make_result


def ctx_for(output: str) -> AssertionContext:
    return AssertionContext(output=output, result=make_result(output), latency_ms=1.0, cost_usd=0.0)


class TestValidJson:
    def test_passes_for_object(self) -> None:
        result = check_valid_json(ctx_for('{"a": 1}'))
        assert result.passed is True
        assert result.assertion is AssertionKind.VALID_JSON

    def test_fails_with_reason(self) -> None:
        result = check_valid_json(ctx_for("not json"))
        assert result.passed is False
        assert "not valid JSON" in result.message

    def test_parses_only_once(self) -> None:
        ctx = ctx_for('{"a": 1}')
        assert ctx.parsed_json == {"a": 1}
        assert ctx.parsed_json is ctx.parsed_json


class TestRequiredFields:
    def test_passes_when_all_present(self) -> None:
        results = check_required_fields(ctx_for('{"a": 1, "b": 2}'), ["a", "b"])
        assert results[0].passed is True

    def test_lists_missing_fields(self) -> None:
        result = check_required_fields(ctx_for('{"a": 1}'), ["a", "b"])[0]
        assert result.passed is False
        assert "b" in result.message

    def test_fails_for_non_json(self) -> None:
        result = check_required_fields(ctx_for("plain text"), ["a"])[0]
        assert result.passed is False
        assert "not valid JSON" in result.message

    def test_fails_for_non_object_json(self) -> None:
        result = check_required_fields(ctx_for("[1, 2]"), ["a"])[0]
        assert result.passed is False
        assert "not object" in result.message


class TestSchema:
    def test_valid_object(self) -> None:
        schema = {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"type": "string"}},
        }
        assert check_schema(ctx_for('{"id": "a"}'), schema).passed is True

    def test_missing_required_property(self) -> None:
        schema = {"type": "object", "required": ["id"]}
        result = check_schema(ctx_for("{}"), schema)
        assert result.passed is False
        assert "id" in result.message

    def test_wrong_type(self) -> None:
        result = check_schema(
            ctx_for('{"id": 5}'),
            {
                "type": "object",
                "properties": {"id": {"type": "string"}},
            },
        )
        assert result.passed is False
        assert "expected type string" in result.message

    def test_fails_for_non_json(self) -> None:
        result = check_schema(ctx_for("oops"), {"type": "object"})
        assert result.passed is False
        assert "not valid JSON" in result.message


class TestValidateJsonSchema:
    def test_rejects_wrong_type(self) -> None:
        errors = validate_json_schema("x", {"type": "integer"})
        assert errors and "expected type integer" in errors[0]

    def test_boolean_is_not_an_integer(self) -> None:
        assert validate_json_schema(True, {"type": "integer"})

    def test_integer_is_a_number(self) -> None:
        assert validate_json_schema(1, {"type": "number"}) == []

    def test_enum(self) -> None:
        assert validate_json_schema("c", {"enum": ["a", "b"]})
        assert validate_json_schema("a", {"enum": ["a", "b"]}) == []

    def test_const(self) -> None:
        assert validate_json_schema(2, {"const": 1})

    def test_string_bounds(self) -> None:
        assert validate_json_schema("ab", {"type": "string", "minLength": 3})
        assert validate_json_schema("abcd", {"type": "string", "maxLength": 3})
        assert validate_json_schema("abc", {"type": "string", "minLength": 3}) == []

    def test_string_pattern(self) -> None:
        assert validate_json_schema("abc", {"type": "string", "pattern": r"^\d+$"})
        assert validate_json_schema("123", {"type": "string", "pattern": r"^\d+$"}) == []

    def test_invalid_pattern_is_reported(self) -> None:
        errors = validate_json_schema("x", {"type": "string", "pattern": "[bad"})
        assert errors and "invalid pattern" in errors[0]

    def test_number_bounds(self) -> None:
        assert validate_json_schema(1, {"type": "number", "minimum": 2})
        assert validate_json_schema(3, {"type": "number", "maximum": 2})
        assert validate_json_schema(2, {"type": "number", "exclusiveMinimum": 2})
        assert validate_json_schema(2, {"type": "number", "exclusiveMaximum": 2})
        assert validate_json_schema(2, {"type": "number", "minimum": 1, "maximum": 3}) == []

    def test_array_items_and_bounds(self) -> None:
        schema = {"type": "array", "items": {"type": "string"}, "minItems": 1}
        assert validate_json_schema([], schema)
        assert validate_json_schema([1], schema)
        assert validate_json_schema(["a"], schema) == []

    def test_nested_object_path_in_message(self) -> None:
        schema = {
            "type": "object",
            "properties": {"user": {"type": "object", "properties": {"age": {"type": "integer"}}}},
        }
        errors = validate_json_schema({"user": {"age": "x"}}, schema)
        assert errors and "$.user.age" in errors[0]

    def test_additional_properties_false(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": False,
        }
        assert validate_json_schema({"a": "x", "b": 1}, schema)
        assert validate_json_schema({"a": "x"}, schema) == []

    def test_nullable_union_type(self) -> None:
        schema = {"type": ["string", "null"]}
        assert validate_json_schema(None, schema) == []
        assert validate_json_schema("x", schema) == []
        assert validate_json_schema(1, schema)

    def test_unknown_type_keyword_is_ignored(self) -> None:
        assert validate_json_schema("x", {"type": "unknown"}) == []

    @pytest.mark.parametrize(
        "value",
        [1, 1.5, True, None, "s", [], {}],
    )
    def test_valid_values_pass_empty_schema(self, value: object) -> None:
        assert validate_json_schema(value, {}) == []
