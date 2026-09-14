from __future__ import annotations

import sys

from inference_engine.benchmarking.ifstruct_validator import (
    check_for_commentary,
    check_uses_code_block,
    extract_json_from_response,
    remove_thinking_tags,
    validate_response,
)

# Captured from a real scoring run; cut off by the model's generation cap.
# Upstream IFStruct tests/test_validator.py at 1948dda22fb08bb1fe15eeb5332119d6087889ac.
RUNAWAY_NUMBER_RESPONSE = (
    '{\n  "product_name": "NVIDIA GeForce RTX 4070 Ti Super",\n  "price_usd": ' + "24" + "9" * 8120
)


def _id_schema() -> dict[str, object]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
        "minItems": 1,
        "maxItems": 1,
    }


def test_unclosed_json_codeblock_is_not_extracted_as_raw_json() -> None:
    response = '```json\n[{"id": "1"}]'

    uses_block, block_type = check_uses_code_block(response, "json")
    data, error = extract_json_from_response(response)

    assert uses_block is False
    assert block_type is None
    assert data is None
    assert error == "Unclosed code block"


def test_validate_response_fails_unclosed_required_json_codeblock() -> None:
    result = validate_response(
        response='```json\n[{"id": "1"}]',
        json_schema=_id_schema(),
        top_level_count=1,
        require_no_commentary=False,
        output_format="json",
        top_level_key=None,
        require_wrapper_key=False,
        require_code_block=True,
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.details["uses_code_block"] is False
    assert "Response must use a code block but none was found" in result.errors
    assert "Unclosed code block" in result.errors


def test_raw_json_still_parses_without_fences() -> None:
    json_data, json_error = extract_json_from_response('[{"id": "1"}]')

    assert json_error is None
    assert json_data == [{"id": "1"}]


def test_raw_json_rejects_trailing_extra_brace() -> None:
    data, error = extract_json_from_response('{"id": "1"}\n}')

    assert data is None
    assert error is not None
    assert "Trailing content after JSON" in error


def test_json_extraction_rejects_yaml_fence() -> None:
    data, error = extract_json_from_response('```yaml\n{"id": "1"}\n```')

    assert data is None
    assert error == "Expected JSON output, got YAML code block"


def test_commentary_checks_flag_short_text_after_code_block() -> None:
    has_json_commentary, json_commentary = check_for_commentary(
        '```json\n[{"id": "1"}]\n```\nok'
    )

    assert has_json_commentary is True
    assert json_commentary == 'Response contains text outside JSON: "ok"'


def test_string_enum_mismatch_fails_validation() -> None:
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "unit": {"type": "string", "enum": ["cup", "tsp"]},
            },
            "required": ["unit"],
        },
        "minItems": 1,
        "maxItems": 1,
    }

    result = validate_response(
        response='[{"unit": "cups"}]',
        json_schema=schema,
        top_level_count=1,
        require_no_commentary=False,
        output_format="json",
        top_level_key=None,
        require_wrapper_key=False,
        require_code_block=False,
    )

    assert result.passed is False
    assert "'cups' not in allowed values ['cup', 'tsp']" in result.errors


def test_validate_response_strips_thinking_tags() -> None:
    result = validate_response(
        response='<think>draft invalid junk</think>[{"id": "1"}]',
        json_schema=_id_schema(),
        top_level_count=1,
        require_no_commentary=True,
        output_format="json",
        top_level_key=None,
        require_wrapper_key=False,
        require_code_block=False,
    )

    assert result.passed is True


def test_remove_thinking_tags_handles_harmony_final_channel() -> None:
    text = 'analysis notes<|start|>assistant<|channel|>final<|message|>[{"id":"1"}]<|end|>'

    assert remove_thinking_tags(text) == '[{"id":"1"}]'


def test_json_extraction_rejects_oversized_number_literal() -> None:
    data, error = extract_json_from_response(RUNAWAY_NUMBER_RESPONSE)

    assert data is None
    limit = sys.get_int_max_str_digits()
    assert error == f"JSON parse error: number literal exceeds the {limit}-digit int limit"


def test_wrapper_key_and_item_count_pass() -> None:
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    }
    result = validate_response(
        response='{"invoices": [{"id": "1"}, {"id": "2"}]}',
        json_schema=schema,
        top_level_count=2,
        require_no_commentary=True,
        output_format="json",
        top_level_key="invoices",
        require_wrapper_key=True,
        require_code_block=False,
    )

    assert result.passed is True
    assert result.score == 1.0


def test_wrapper_required_rejects_bare_list() -> None:
    result = validate_response(
        response='[{"id": "1"}]',
        json_schema=_id_schema(),
        top_level_count=1,
        require_no_commentary=False,
        output_format="json",
        top_level_key="invoices",
        require_wrapper_key=True,
        require_code_block=False,
    )

    assert result.passed is False
    assert "Expected wrapped object with key 'invoices', got bare list" in result.errors


def test_yaml_output_format_is_rejected_by_json_adapter() -> None:
    try:
        validate_response(
            response="- id: '1'",
            json_schema=_id_schema(),
            top_level_count=1,
            require_no_commentary=False,
            output_format="yaml",
            top_level_key=None,
            require_wrapper_key=False,
            require_code_block=False,
        )
    except ValueError as exc:
        assert "non-JSON" in str(exc)
    else:
        raise AssertionError("yaml output_format must fail closed")
