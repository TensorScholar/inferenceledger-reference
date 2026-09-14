from __future__ import annotations

import pytest

from inference_engine.benchmarking.eval import EvalType, evaluate_text, parse_eval_spec


def test_parse_eval_spec_for_json_keys() -> None:
    spec = parse_eval_spec({"type": "json_keys", "required": ["status", "reason"]})

    assert spec is not None
    assert spec.eval_type == EvalType.JSON_KEYS
    assert spec.required == ["status", "reason"]


def test_parse_eval_spec_for_json_field_equals() -> None:
    spec = parse_eval_spec(
        {
            "type": "json_field_equals",
            "expected_fields": {"status": "ok", "reason": "ready"},
        }
    )

    assert spec is not None
    assert spec.eval_type == EvalType.JSON_FIELD_EQUALS
    assert spec.expected_fields == {"status": "ok", "reason": "ready"}


def test_json_keys_eval_passes_valid_object() -> None:
    spec = parse_eval_spec({"type": "json_keys", "required": ["status", "reason"]})

    result = evaluate_text('{"status":"ok","reason":"valid"}', spec)

    assert result is not None
    assert result.passed is True
    assert result.score == 1.0


def test_json_keys_eval_fails_invalid_json() -> None:
    spec = parse_eval_spec({"type": "json_keys", "required": ["status"]})

    result = evaluate_text("status: ok", spec)

    assert result is not None
    assert result.passed is False
    assert "invalid JSON" in result.reason


def test_json_field_equals_eval_passes_expected_values() -> None:
    spec = parse_eval_spec(
        {
            "type": "json_field_equals",
            "expected_fields": {"status": "ok", "reason": "ready"},
        }
    )

    result = evaluate_text('{"status":"OK","reason":"ready"}', spec)

    assert result is not None
    assert result.passed is True
    assert result.score == 1.0


def test_json_field_equals_eval_scores_mismatched_fields() -> None:
    spec = parse_eval_spec(
        {
            "type": "json_field_equals",
            "expected_fields": {"status": "ok", "reason": "ready"},
        }
    )

    result = evaluate_text('{"status":"ok","reason":"blocked"}', spec)

    assert result is not None
    assert result.passed is False
    assert result.score == 0.5
    assert "reason" in result.reason


def test_contains_all_eval_scores_partial_match() -> None:
    spec = parse_eval_spec({"type": "contains_all", "required": ["cost", "quality"]})

    result = evaluate_text("Cost-aware routing matters.", spec)

    assert result is not None
    assert result.passed is False
    assert result.score == 0.5


def test_exact_match_eval_ignores_case_by_default() -> None:
    spec = parse_eval_spec({"type": "exact_match", "expected": "positive"})

    result = evaluate_text("Positive", spec)

    assert result is not None
    assert result.passed is True


def test_parse_eval_spec_rejects_invalid_required() -> None:
    with pytest.raises(ValueError, match="eval.required"):
        parse_eval_spec({"type": "contains_all", "required": "cost"})


def test_parse_eval_spec_rejects_invalid_expected_fields() -> None:
    with pytest.raises(ValueError, match="eval.expected_fields"):
        parse_eval_spec({"type": "json_field_equals", "expected_fields": {"status": 1}})


def test_parse_eval_spec_for_ifstruct_structure() -> None:
    spec = parse_eval_spec(
        {
            "type": "ifstruct_structure",
            "ifstruct": {
                "json_schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                        "required": ["id"],
                    },
                },
                "top_level_count": 1,
                "top_level_key": "invoices",
                "require_wrapper_key": True,
                "require_code_block": False,
                "require_no_commentary": True,
                "output_format": "json",
            },
        }
    )

    assert spec is not None
    assert spec.eval_type == EvalType.IFSTRUCT_STRUCTURE
    assert spec.ifstruct is not None
    assert spec.ifstruct.top_level_key == "invoices"


def test_ifstruct_structure_eval_passes_wrapped_object() -> None:
    spec = parse_eval_spec(
        {
            "type": "ifstruct_structure",
            "ifstruct": {
                "json_schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                        "required": ["id"],
                    },
                },
                "top_level_count": 1,
                "top_level_key": "invoices",
                "require_wrapper_key": True,
                "require_code_block": False,
                "require_no_commentary": True,
                "output_format": "json",
            },
        }
    )

    result = evaluate_text('{"invoices": [{"id": "1"}]}', spec)

    assert result is not None
    assert result.passed is True
    assert result.score == 1.0
    assert result.eval_type == "ifstruct_structure"


def test_ifstruct_structure_eval_rejects_yaml_format() -> None:
    with pytest.raises(ValueError, match="output_format"):
        parse_eval_spec(
            {
                "type": "ifstruct_structure",
                "ifstruct": {
                    "json_schema": {"type": "array"},
                    "top_level_count": 1,
                    "top_level_key": "items",
                    "require_wrapper_key": True,
                    "require_code_block": False,
                    "require_no_commentary": False,
                    "output_format": "yaml",
                },
            }
        )
