from __future__ import annotations

import json

import pytest

from inference_engine.domain.models.execution import (
    AttemptOutcome,
    CostEvidenceKind,
)
from inference_engine.infrastructure.importers.litellm import (
    LITELLM_STANDARD_LOGGING_SOURCE,
    LiteLLMImportError,
    import_litellm_standard_logging_chain,
    sanitize_standard_logging_payload,
)
from inference_engine.infrastructure.telemetry.request_log import JsonlRequestLog


def _payload(
    *,
    record_id: str = "record-1",
    call_id: str = "call-1",
    trace_id: str = "trace-1",
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    status: str = "success",
    start_time: float = 100.0,
    end_time: float = 100.2,
    response_time: float = 0.2,
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    total_tokens: int = 120,
    response_cost: float | None = 0.0012,
    cost_failure_debug: object = None,
    cache_hit: bool | None = False,
    error_information: object = None,
    error_str: str | None = None,
) -> dict[str, object]:
    return {
        "id": record_id,
        "trace_id": trace_id,
        "litellm_call_id": call_id,
        "custom_llm_provider": provider,
        "model": model,
        "status": status,
        "startTime": start_time,
        "endTime": end_time,
        "response_time": response_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "response_cost": response_cost,
        "response_cost_failure_debug_info": cost_failure_debug,
        "cache_hit": cache_hit,
        "error_information": error_information,
        "error_str": error_str,
    }


def test_imports_successful_standard_log_as_external_reported_cost() -> None:
    trace = import_litellm_standard_logging_chain([_payload()])

    assert trace.request_id == "trace-1"
    assert trace.provider == "openai"
    assert trace.model == "gpt-4o-mini"
    assert trace.latency_ms == 200
    assert trace.estimated_cost_usd == pytest.approx(0.0012)
    assert trace.cost_evidence_complete is True
    assert trace.pricing_table_version == f"external_reported:{LITELLM_STANDARD_LOGGING_SOURCE}"
    assert trace.provider_attempt_count == 1
    assert trace.provider_retry_count == 0
    assert trace.error_type is None

    attempt = trace.provider_attempts[0]
    assert attempt.outcome == AttemptOutcome.SUCCEEDED
    assert attempt.cost_evidence == CostEvidenceKind.REPORTED_BY_EXECUTION_STACK
    assert attempt.reported_cost_usd == pytest.approx(0.0012)
    assert attempt.calculated_cost_usd is None
    assert attempt.cost_source == LITELLM_STANDARD_LOGGING_SOURCE
    assert attempt.cost_source_record_id == "record-1"


def test_orders_fallback_attempts_by_observed_time_and_sums_reported_cost() -> None:
    failed = _payload(
        record_id="failed-leg",
        call_id="call-a",
        provider="openai",
        model="model-a",
        status="failure",
        start_time=100.0,
        end_time=100.2,
        response_time=0.2,
        response_cost=0.0004,
        prompt_tokens=100,
        completion_tokens=0,
        total_tokens=100,
        error_information={
            "error_class": "RateLimitError",
            "error_message": "rate limited",
            "error_code": "429",
        },
    )
    succeeded = _payload(
        record_id="success-leg",
        call_id="call-b",
        provider="anthropic",
        model="model-b",
        status="success",
        start_time=100.3,
        end_time=100.7,
        response_time=0.4,
        response_cost=0.0016,
        prompt_tokens=105,
        completion_tokens=25,
        total_tokens=130,
    )

    trace = import_litellm_standard_logging_chain([succeeded, failed])

    assert trace.request_id == "trace-1"
    assert trace.provider == "anthropic"
    assert trace.model == "model-b"
    assert trace.latency_ms == 700
    assert trace.prompt_tokens == 105
    assert trace.completion_tokens == 25
    assert trace.total_tokens == 130
    assert trace.provider_attempt_count == 2
    assert trace.provider_retry_count == 1
    assert trace.estimated_cost_usd == pytest.approx(0.002)
    assert [attempt.provider for attempt in trace.provider_attempts] == ["openai", "anthropic"]
    assert [attempt.model for attempt in trace.provider_attempts] == ["model-a", "model-b"]
    assert trace.provider_attempts[0].status_code == 429
    assert trace.provider_attempts[0].error_type == "RateLimitError"


def test_cost_tracking_failure_makes_entire_chain_cost_incomplete() -> None:
    failed_cost_leg = _payload(
        record_id="leg-1",
        status="failure",
        response_cost=0.0,
        cost_failure_debug={"error_str": "unknown model pricing"},
    )
    final_leg = _payload(
        record_id="leg-2",
        call_id="call-2",
        start_time=100.3,
        end_time=100.4,
        response_time=0.1,
        response_cost=0.001,
    )

    trace = import_litellm_standard_logging_chain([failed_cost_leg, final_leg])

    assert trace.cost_evidence_complete is False
    assert trace.estimated_cost_usd is None
    assert trace.provider_attempts[0].cost_evidence == CostEvidenceKind.UNKNOWN
    assert trace.provider_attempts[0].cost_usd is None
    assert trace.provider_attempts[1].cost_is_known is True


def test_zero_response_cost_is_preserved_when_litellm_reports_no_cost_failure() -> None:
    trace = import_litellm_standard_logging_chain([_payload(response_cost=0.0)])

    assert trace.cost_evidence_complete is True
    assert trace.estimated_cost_usd == 0.0
    assert trace.provider_attempts[0].reported_cost_usd == 0.0
    assert trace.provider_attempts[0].cost_is_known is True


def test_null_litellm_cache_state_remains_unknown() -> None:
    trace = import_litellm_standard_logging_chain([_payload(cache_hit=None)])

    assert trace.cache_hit is None


def test_failed_terminal_attempt_preserves_failure_identity() -> None:
    trace = import_litellm_standard_logging_chain(
        [
            _payload(
                status="failure",
                response_cost=0.0002,
                error_information={
                    "error_class": "Timeout",
                    "error_message": "provider timeout",
                    "error_code": 504,
                },
            )
        ]
    )

    assert trace.error_type == "Timeout"
    assert trace.error_message == "provider timeout"
    assert trace.provider_attempts[0].outcome == AttemptOutcome.FAILED
    assert trace.provider_attempts[0].status_code == 504
    assert trace.cost_evidence_complete is True


def test_trace_id_is_execution_chain_boundary_even_when_call_ids_differ() -> None:
    trace = import_litellm_standard_logging_chain(
        [
            _payload(
                record_id="a",
                call_id="call-a",
                status="failure",
                end_time=100.1,
                response_time=0.1,
            ),
            _payload(
                record_id="b",
                call_id="call-b",
                start_time=100.2,
                end_time=100.3,
                response_time=0.1,
            ),
        ]
    )

    assert trace.request_id == "trace-1"
    assert trace.provider_attempt_count == 2

    with pytest.raises(LiteLLMImportError, match="share one LiteLLM trace_id"):
        import_litellm_standard_logging_chain(
            [
                _payload(record_id="a", trace_id="trace-a"),
                _payload(record_id="b", trace_id="trace-b"),
            ]
        )


def test_multiple_successes_or_post_success_records_are_rejected_as_ambiguous() -> None:
    with pytest.raises(LiteLLMImportError, match="multiple successful"):
        import_litellm_standard_logging_chain(
            [
                _payload(record_id="a", end_time=100.1, response_time=0.1),
                _payload(
                    record_id="b",
                    call_id="call-b",
                    start_time=100.2,
                    end_time=100.3,
                    response_time=0.1,
                ),
            ]
        )

    with pytest.raises(LiteLLMImportError, match="records after a successful"):
        import_litellm_standard_logging_chain(
            [
                _payload(record_id="a", end_time=100.1, response_time=0.1),
                _payload(
                    record_id="b",
                    call_id="call-b",
                    status="failure",
                    start_time=100.2,
                    end_time=100.3,
                    response_time=0.1,
                ),
            ]
        )


def test_duplicate_source_record_id_is_rejected() -> None:
    with pytest.raises(LiteLLMImportError, match="record ids must be unique"):
        import_litellm_standard_logging_chain(
            [
                _payload(status="failure"),
                _payload(start_time=101.0, end_time=101.2),
            ]
        )


def test_missing_provider_or_model_identity_fails_closed() -> None:
    missing_provider = _payload()
    missing_provider["custom_llm_provider"] = None
    with pytest.raises(LiteLLMImportError, match="custom_llm_provider"):
        import_litellm_standard_logging_chain([missing_provider])

    missing_model = _payload()
    missing_model["model"] = ""
    with pytest.raises(LiteLLMImportError, match="model"):
        import_litellm_standard_logging_chain([missing_model])


def test_invalid_time_and_usage_evidence_is_rejected() -> None:
    with pytest.raises(LiteLLMImportError, match="endTime must not precede"):
        import_litellm_standard_logging_chain([_payload(start_time=2.0, end_time=1.0)])

    with pytest.raises(LiteLLMImportError, match="total_tokens"):
        import_litellm_standard_logging_chain(
            [_payload(prompt_tokens=100, completion_tokens=20, total_tokens=119)]
        )

    with pytest.raises(LiteLLMImportError, match="response_cost must be finite"):
        import_litellm_standard_logging_chain([_payload(response_cost=float("nan"))])


def test_jsonl_round_trip_preserves_external_cost_origin(tmp_path) -> None:
    trace = import_litellm_standard_logging_chain([_payload()])
    log = JsonlRequestLog(tmp_path / "requests.jsonl")

    log.append(trace)
    restored = log.read_all()

    assert restored == [trace]
    attempt = restored[0].provider_attempts[0]
    assert attempt.cost_evidence == CostEvidenceKind.REPORTED_BY_EXECUTION_STACK
    assert attempt.reported_cost_usd == pytest.approx(0.0012)
    assert attempt.cost_source == LITELLM_STANDARD_LOGGING_SOURCE
    assert attempt.cost_source_record_id == "record-1"


def test_jsonl_reader_downgrades_malformed_external_provenance(tmp_path) -> None:
    trace = import_litellm_standard_logging_chain([_payload()])
    log = JsonlRequestLog(tmp_path / "requests.jsonl")
    log.append(trace)

    raw = json.loads(log.path.read_text(encoding="utf-8"))
    raw["provider_attempts"][0]["cost_source_record_id"] = None
    log.path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    restored = log.read_all()[0]

    assert restored.cost_evidence_complete is False
    assert restored.estimated_cost_usd is None
    assert restored.provider_attempts[0].cost_evidence == CostEvidenceKind.UNKNOWN
    assert restored.provider_attempts[0].reported_cost_usd is None


def test_temporally_ambiguous_or_overlapping_trace_is_rejected() -> None:
    with pytest.raises(LiteLLMImportError, match="ambiguous attempt ordering"):
        import_litellm_standard_logging_chain(
            [
                _payload(
                    record_id="failed",
                    call_id="call-a",
                    status="failure",
                    start_time=100.0,
                    end_time=100.1,
                    response_time=0.1,
                ),
                _payload(
                    record_id="success",
                    call_id="call-b",
                    start_time=100.0,
                    end_time=100.2,
                    response_time=0.2,
                ),
            ]
        )

    with pytest.raises(LiteLLMImportError, match="overlapping LLM records"):
        import_litellm_standard_logging_chain(
            [
                _payload(
                    record_id="failed",
                    call_id="call-a",
                    status="failure",
                    start_time=100.0,
                    end_time=100.3,
                    response_time=0.3,
                ),
                _payload(
                    record_id="success",
                    call_id="call-b",
                    start_time=100.2,
                    end_time=100.4,
                    response_time=0.2,
                ),
            ]
        )


def test_sanitize_redacts_prompt_and_response_not_evidence_fields() -> None:
    payload = _payload()
    payload["messages"] = [{"role": "user", "content": "secret prompt"}]
    payload["response"] = {"choices": [{"message": {"content": "secret answer"}}]}

    sanitized = sanitize_standard_logging_payload(payload)

    assert sanitized["id"] == "record-1"
    assert sanitized["trace_id"] == "trace-1"
    assert sanitized["response_cost"] == 0.0012
    assert sanitized["messages"] == [{"role": "user", "content": "[REDACTED_PROMPT]"}]
    assert sanitized["response"] == "[REDACTED_RESPONSE]"
