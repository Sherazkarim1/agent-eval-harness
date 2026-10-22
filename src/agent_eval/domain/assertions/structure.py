"""Structure assertions: JSON validity, required fields, schema validation.

The schema validator implements the subset of JSON Schema that matters for
agent output checks (types, required properties, enums, numeric and string
bounds, array item types, nested objects). Pulling in a full JSON Schema
library for that subset would violate the minimal-dependency rule.
"""

from __future__ import annotations

from typing import Any

from agent_eval.domain.assertions.context import AssertionContext
from agent_eval.domain.models import AssertionKind, AssertionResult

_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def check_valid_json(ctx: AssertionContext) -> AssertionResult:
    """Assert the output is parseable JSON."""
    value = ctx.parsed_json
    error = ctx.json_error
    return AssertionResult(
        assertion=AssertionKind.VALID_JSON,
        passed=error is None,
        expected="valid JSON",
        actual=value if error is None else ctx.output[:200],
        message="Output is valid JSON" if error is None else f"Output is not valid JSON: {error}",
    )


def check_required_fields(ctx: AssertionContext, fields: list[str]) -> list[AssertionResult]:
    """Assert a JSON object output contains every required field."""
    value = ctx.parsed_json
    if ctx.json_error is not None:
        return [
            AssertionResult(
                assertion=AssertionKind.REQUIRED_FIELDS,
                passed=False,
                expected=fields,
                actual=None,
                message=f"Cannot check required fields: output is not valid JSON "
                f"({ctx.json_error})",
            )
        ]
    if not isinstance(value, dict):
        return [
            AssertionResult(
                assertion=AssertionKind.REQUIRED_FIELDS,
                passed=False,
                expected=fields,
                actual=value,
                message=f"Cannot check required fields: output is "
                f"{type(value).__name__}, not object",
            )
        ]
    missing = [field for field in fields if field not in value]
    return [
        AssertionResult(
            assertion=AssertionKind.REQUIRED_FIELDS,
            passed=not missing,
            expected=fields,
            actual=sorted(value),
            message="All required fields are present"
            if not missing
            else f"Missing required fields: {', '.join(missing)}",
        )
    ]


def check_schema(ctx: AssertionContext, schema: dict[str, Any]) -> AssertionResult:
    """Validate the JSON output against a JSON Schema subset."""
    value = ctx.parsed_json
    if ctx.json_error is not None:
        return AssertionResult(
            assertion=AssertionKind.SCHEMA_VALIDATION,
            passed=False,
            expected=schema,
            actual=None,
            message=f"Cannot validate schema: output is not valid JSON ({ctx.json_error})",
        )
    errors = validate_json_schema(value, schema)
    return AssertionResult(
        assertion=AssertionKind.SCHEMA_VALIDATION,
        passed=not errors,
        expected=schema,
        actual=value,
        message="Output matches the schema"
        if not errors
        else "Schema violations: " + "; ".join(errors[:5]),
    )


def validate_json_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate ``value`` against the supported JSON Schema subset.

    Supported keywords: ``type``, ``enum``, ``const``, ``required``,
    ``properties``, ``additionalProperties``, ``items``, ``minimum``,
    ``maximum``, ``exclusiveMinimum``, ``exclusiveMaximum``, ``minLength``,
    ``maxLength``, ``pattern``, ``minItems``, ``maxItems``.

    Returns:
        A list of human-readable violations; empty means valid.
    """
    errors: list[str] = []

    expected_type = schema.get("type")
    if expected_type is not None:
        types = [expected_type] if isinstance(expected_type, str) else list(expected_type)
        if not any(_matches_type(value, t) for t in types):
            errors.append(f"{path}: expected type {'|'.join(types)}, got {_type_name(value)}")
            return errors

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected constant {schema['const']!r}, got {value!r}")

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']!r}")

    if isinstance(value, str):
        errors.extend(_check_string(value, schema, path))
    if isinstance(value, int | float) and not isinstance(value, bool):
        errors.extend(_check_number(value, schema, path))
    if isinstance(value, list):
        errors.extend(_check_array(value, schema, path))
    if isinstance(value, dict):
        errors.extend(_check_object(value, schema, path))

    return errors


def _matches_type(value: Any, expected: str) -> bool:
    python_type = _TYPE_MAP.get(expected)
    if python_type is None:
        return True
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return isinstance(value, python_type)


def _type_name(value: Any) -> str:
    return "null" if value is None else type(value).__name__


def _check_string(value: str, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    min_length = schema.get("minLength")
    if isinstance(min_length, int) and len(value) < min_length:
        errors.append(f"{path}: shorter than minLength {min_length}")
    max_length = schema.get("maxLength")
    if isinstance(max_length, int) and len(value) > max_length:
        errors.append(f"{path}: longer than maxLength {max_length}")
    pattern = schema.get("pattern")
    if isinstance(pattern, str):
        import re

        try:
            if re.search(pattern, value) is None:
                errors.append(f"{path}: does not match pattern {pattern!r}")
        except re.error as exc:
            errors.append(f"{path}: invalid pattern in schema ({exc})")
    return errors


def _check_number(value: float, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    minimum = schema.get("minimum")
    if isinstance(minimum, int | float) and value < minimum:
        errors.append(f"{path}: below minimum {minimum}")
    maximum = schema.get("maximum")
    if isinstance(maximum, int | float) and value > maximum:
        errors.append(f"{path}: above maximum {maximum}")
    exclusive_min = schema.get("exclusiveMinimum")
    if isinstance(exclusive_min, int | float) and value <= exclusive_min:
        errors.append(f"{path}: not greater than {exclusive_min}")
    exclusive_max = schema.get("exclusiveMaximum")
    if isinstance(exclusive_max, int | float) and value >= exclusive_max:
        errors.append(f"{path}: not less than {exclusive_max}")
    return errors


def _check_array(value: list[Any], schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    min_items = schema.get("minItems")
    if isinstance(min_items, int) and len(value) < min_items:
        errors.append(f"{path}: fewer than minItems {min_items}")
    max_items = schema.get("maxItems")
    if isinstance(max_items, int) and len(value) > max_items:
        errors.append(f"{path}: more than maxItems {max_items}")
    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        for index, item in enumerate(value):
            errors.extend(validate_json_schema(item, item_schema, f"{path}[{index}]"))
    return errors


def _check_object(value: dict[str, Any], schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    required = schema.get("required")
    if isinstance(required, list):
        for key in required:
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for key, sub_schema in properties.items():
            if key in value and isinstance(sub_schema, dict):
                errors.extend(validate_json_schema(value[key], sub_schema, f"{path}.{key}"))
    additional = schema.get("additionalProperties")
    if additional is False and isinstance(properties, dict):
        for key in value:
            if key not in properties:
                errors.append(f"{path}: additional property {key!r} is not allowed")
    return errors
