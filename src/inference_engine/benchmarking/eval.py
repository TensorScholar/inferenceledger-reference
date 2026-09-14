from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .ifstruct_validator import validate_response


class EvalType(StrEnum):
    """Supported deterministic benchmark validators."""

    CONTAINS_ALL = "contains_all"
    EXACT_MATCH = "exact_match"
    JSON_FIELD_EQUALS = "json_field_equals"
    JSON_KEYS = "json_keys"
    IFSTRUCT_STRUCTURE = "ifstruct_structure"


@dataclass(frozen=True)
class IfStructEvalContract:
    """IFStruct JSON-path scoring contract for one published-reference item."""

    json_schema: dict[str, Any]
    top_level_count: int | list[int] | None
    top_level_key: str | None
    require_wrapper_key: bool
    require_code_block: bool
    require_no_commentary: bool
    output_format: str


@dataclass(frozen=True)
class EvalSpec:
    """Deterministic expectation for one workload item."""

    eval_type: EvalType
    expected: str | None = None
    expected_fields: dict[str, str] | None = None
    required: list[str] | None = None
    case_sensitive: bool = False
    ifstruct: IfStructEvalContract | None = None


@dataclass(frozen=True)
class EvalResult:
    """Quality result for one provider response."""

    passed: bool
    score: float
    eval_type: str
    reason: str


def parse_eval_spec(raw: Any) -> EvalSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("eval must be an object")

    eval_type_raw = raw.get("type")
    if not isinstance(eval_type_raw, str):
        raise ValueError("eval.type must be a string")
    eval_type = EvalType(eval_type_raw)

    expected = raw.get("expected")
    if expected is not None and not isinstance(expected, str):
        raise ValueError("eval.expected must be a string")

    expected_fields_raw = raw.get("expected_fields")
    expected_fields: dict[str, str] | None = None
    if expected_fields_raw is not None:
        if not isinstance(expected_fields_raw, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in expected_fields_raw.items()
        ):
            raise ValueError("eval.expected_fields must be an object of strings")
        expected_fields = expected_fields_raw

    required_raw = raw.get("required")
    required: list[str] | None = None
    if required_raw is not None:
        if not isinstance(required_raw, list) or not all(
            isinstance(item, str) for item in required_raw
        ):
            raise ValueError("eval.required must be a list of strings")
        required = required_raw

    case_sensitive = raw.get("case_sensitive", False)
    if not isinstance(case_sensitive, bool):
        raise ValueError("eval.case_sensitive must be a boolean")

    ifstruct: IfStructEvalContract | None = None
    if eval_type == EvalType.IFSTRUCT_STRUCTURE:
        extra = sorted(set(raw) - {"type", "ifstruct"})
        if extra:
            raise ValueError(f"ifstruct_structure eval rejects unknown fields: {extra}")
        ifstruct = _parse_ifstruct_contract(raw.get("ifstruct"))

    return EvalSpec(
        eval_type=eval_type,
        expected=expected,
        expected_fields=expected_fields,
        required=required,
        case_sensitive=case_sensitive,
        ifstruct=ifstruct,
    )


def evaluate_text(text: str, spec: EvalSpec | None) -> EvalResult | None:
    if spec is None:
        return None

    if spec.eval_type == EvalType.EXACT_MATCH:
        return _evaluate_exact_match(text, spec)
    if spec.eval_type == EvalType.CONTAINS_ALL:
        return _evaluate_contains_all(text, spec)
    if spec.eval_type == EvalType.JSON_FIELD_EQUALS:
        return _evaluate_json_field_equals(text, spec)
    if spec.eval_type == EvalType.JSON_KEYS:
        return _evaluate_json_keys(text, spec)
    if spec.eval_type == EvalType.IFSTRUCT_STRUCTURE:
        return _evaluate_ifstruct_structure(text, spec)

    raise ValueError(f"Unsupported eval type: {spec.eval_type}")


def _evaluate_exact_match(text: str, spec: EvalSpec) -> EvalResult:
    expected = spec.expected
    if expected is None:
        raise ValueError("exact_match eval requires expected")
    actual = text.strip()
    expected_value = expected.strip()
    if not spec.case_sensitive:
        actual = actual.lower()
        expected_value = expected_value.lower()
    passed = actual == expected_value
    return EvalResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        eval_type=spec.eval_type.value,
        reason="exact match passed" if passed else "response did not exactly match expected text",
    )


def _evaluate_contains_all(text: str, spec: EvalSpec) -> EvalResult:
    required = spec.required
    if not required:
        raise ValueError("contains_all eval requires required")
    haystack = text if spec.case_sensitive else text.lower()
    missing = [
        item
        for item in required
        if (item if spec.case_sensitive else item.lower()) not in haystack
    ]
    passed = not missing
    score = (len(required) - len(missing)) / len(required)
    return EvalResult(
        passed=passed,
        score=score,
        eval_type=spec.eval_type.value,
        reason="all required substrings present"
        if passed
        else f"missing required substrings: {', '.join(missing)}",
    )


def _evaluate_json_keys(text: str, spec: EvalSpec) -> EvalResult:
    required = spec.required
    if not required:
        raise ValueError("json_keys eval requires required")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return EvalResult(
            passed=False,
            score=0.0,
            eval_type=spec.eval_type.value,
            reason=f"invalid JSON: {exc.msg}",
        )
    if not isinstance(parsed, dict):
        return EvalResult(
            passed=False,
            score=0.0,
            eval_type=spec.eval_type.value,
            reason="JSON response is not an object",
        )

    missing = [key for key in required if key not in parsed]
    passed = not missing
    score = (len(required) - len(missing)) / len(required)
    return EvalResult(
        passed=passed,
        score=score,
        eval_type=spec.eval_type.value,
        reason="all required JSON keys present" if passed else f"missing JSON keys: {', '.join(missing)}",
    )


def _evaluate_json_field_equals(text: str, spec: EvalSpec) -> EvalResult:
    expected_fields = spec.expected_fields
    if not expected_fields:
        raise ValueError("json_field_equals eval requires expected_fields")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return EvalResult(
            passed=False,
            score=0.0,
            eval_type=spec.eval_type.value,
            reason=f"invalid JSON: {exc.msg}",
        )
    if not isinstance(parsed, dict):
        return EvalResult(
            passed=False,
            score=0.0,
            eval_type=spec.eval_type.value,
            reason="JSON response is not an object",
        )

    mismatches: list[str] = []
    matched = 0
    for field, expected in expected_fields.items():
        actual = parsed.get(field)
        if not isinstance(actual, str):
            mismatches.append(field)
            continue
        actual_value = actual.strip()
        expected_value = expected.strip()
        if not spec.case_sensitive:
            actual_value = actual_value.lower()
            expected_value = expected_value.lower()
        if actual_value == expected_value:
            matched += 1
        else:
            mismatches.append(field)

    passed = not mismatches
    score = matched / len(expected_fields)
    return EvalResult(
        passed=passed,
        score=score,
        eval_type=spec.eval_type.value,
        reason="all expected JSON fields matched"
        if passed
        else f"mismatched JSON fields: {', '.join(mismatches)}",
    )


def _parse_ifstruct_contract(raw: Any) -> IfStructEvalContract:
    if not isinstance(raw, dict):
        raise ValueError("eval.ifstruct must be an object")
    allowed = {
        "json_schema",
        "top_level_count",
        "top_level_key",
        "require_wrapper_key",
        "require_code_block",
        "require_no_commentary",
        "output_format",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"eval.ifstruct rejects unknown fields: {unknown}")

    json_schema = raw.get("json_schema")
    if not isinstance(json_schema, dict):
        raise ValueError("eval.ifstruct.json_schema must be an object")

    top_level_count = raw.get("top_level_count")
    resolved_count: int | list[int] | None
    if top_level_count is None:
        resolved_count = None
    elif not _valid_top_level_count(top_level_count):
        raise ValueError("eval.ifstruct.top_level_count must be an int or a two-int range")
    elif isinstance(top_level_count, int):
        resolved_count = top_level_count
    else:
        resolved_count = [int(top_level_count[0]), int(top_level_count[1])]

    top_level_key = raw.get("top_level_key")
    if top_level_key is not None and not isinstance(top_level_key, str):
        raise ValueError("eval.ifstruct.top_level_key must be a string or null")

    require_wrapper_key = raw.get("require_wrapper_key")
    require_code_block = raw.get("require_code_block")
    require_no_commentary = raw.get("require_no_commentary")
    if not isinstance(require_wrapper_key, bool):
        raise ValueError("eval.ifstruct.require_wrapper_key must be a boolean")
    if not isinstance(require_code_block, bool):
        raise ValueError("eval.ifstruct.require_code_block must be a boolean")
    if not isinstance(require_no_commentary, bool):
        raise ValueError("eval.ifstruct.require_no_commentary must be a boolean")

    output_format = raw.get("output_format")
    if output_format != "json":
        raise ValueError("eval.ifstruct.output_format must be 'json' for this adapter")

    return IfStructEvalContract(
        json_schema=json_schema,
        top_level_count=resolved_count,
        top_level_key=top_level_key,
        require_wrapper_key=require_wrapper_key,
        require_code_block=require_code_block,
        require_no_commentary=require_no_commentary,
        output_format=output_format,
    )


def _valid_top_level_count(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _evaluate_ifstruct_structure(text: str, spec: EvalSpec) -> EvalResult:
    contract = spec.ifstruct
    if contract is None:
        raise ValueError("ifstruct_structure eval requires ifstruct")
    result = validate_response(
        response=text,
        json_schema=contract.json_schema,
        top_level_count=contract.top_level_count,
        require_no_commentary=contract.require_no_commentary,
        output_format=contract.output_format,
        top_level_key=contract.top_level_key,
        require_wrapper_key=contract.require_wrapper_key,
        require_code_block=contract.require_code_block,
    )
    reason = "IFStruct structure checks passed" if result.passed else "; ".join(result.errors)
    return EvalResult(
        passed=result.passed,
        score=result.score,
        eval_type=spec.eval_type.value,
        reason=reason,
    )
