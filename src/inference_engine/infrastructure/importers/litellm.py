from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite

from ...domain.models.execution import (
    AttemptOutcome,
    CostEvidenceKind,
    ProviderAttempt,
)
from ..telemetry.request_log import RequestTrace

LITELLM_STANDARD_LOGGING_SOURCE = "litellm_standard_logging_payload"


class LiteLLMImportError(ValueError):
    """Raised when LiteLLM evidence is too ambiguous or malformed to import safely."""


@dataclass(frozen=True)
class _StandardLogRecord:
    record_id: str
    trace_id: str
    provider: str
    model: str
    outcome: AttemptOutcome
    start_time: float
    end_time: float
    response_time_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_hit: bool | None
    reported_cost_usd: float | None
    error_type: str | None
    error_message: str | None
    status_code: int | None


def import_litellm_standard_logging_chain(
    payloads: Sequence[Mapping[str, object]],
) -> RequestTrace:
    """Convert one LiteLLM retry/fallback trace into canonical execution evidence.

    Correlation is intentionally based on ``trace_id``. LiteLLM's current standard logging
    contract describes that field as the trace spanning multiple LLM calls and its integrations
    use it for retry/fallback grouping. ``litellm_call_id`` is therefore not assumed to be stable
    across attempts and is not used as the execution-chain key.

    A trace with multiple successful LLM records is rejected because it is ambiguous whether the
    trace represents one retry/fallback execution or a broader multi-call workflow. For a normal
    fallback chain, zero or more failures may precede at most one terminal success.

    ``response_cost`` is treated as externally reported evidence, never as InferenceLedger
    pricing-reconstructed evidence. If LiteLLM supplies ``response_cost_failure_debug_info``, the
    monetary amount is considered unknown even if a numeric ``response_cost`` is present.
    """
    if not payloads:
        raise LiteLLMImportError("at least one LiteLLM StandardLoggingPayload is required")

    records = [_parse_record(payload) for payload in payloads]
    trace_ids = {record.trace_id for record in records}
    if len(trace_ids) != 1:
        raise LiteLLMImportError("all imported records must share one LiteLLM trace_id")
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise LiteLLMImportError("LiteLLM record ids must be unique within one trace")

    ordered = sorted(records, key=lambda record: (record.start_time, record.end_time, record.record_id))
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.start_time <= previous.start_time:
            raise LiteLLMImportError(
                "LiteLLM trace has ambiguous attempt ordering because multiple records share the same start time"
            )
        if current.start_time < previous.end_time:
            raise LiteLLMImportError(
                "LiteLLM trace contains overlapping LLM records and is not an unambiguous retry/fallback chain"
            )

    successful = [record for record in ordered if record.outcome == AttemptOutcome.SUCCEEDED]
    if len(successful) > 1:
        raise LiteLLMImportError(
            "LiteLLM trace contains multiple successful LLM records and is not an unambiguous retry/fallback chain"
        )
    if successful and successful[0] is not ordered[-1]:
        raise LiteLLMImportError(
            "LiteLLM trace contains records after a successful LLM attempt and is not an unambiguous retry/fallback chain"
        )

    attempts = tuple(_provider_attempt(index, record) for index, record in enumerate(ordered, 1))
    final = ordered[-1]

    all_costs_known = all(attempt.cost_is_known for attempt in attempts)
    execution_cost = (
        sum(attempt.cost_usd or 0.0 for attempt in attempts) if all_costs_known else None
    )
    pricing_context = (
        f"external_reported:{LITELLM_STANDARD_LOGGING_SOURCE}"
        if all_costs_known
        else f"incomplete:external:{LITELLM_STANDARD_LOGGING_SOURCE}"
    )
    request_latency_ms = _seconds_to_milliseconds(
        max(record.end_time for record in ordered) - min(record.start_time for record in ordered)
    )
    error_type = None
    error_message = None
    if final.outcome == AttemptOutcome.FAILED:
        error_type = final.error_type or "litellm_failure"
        error_message = final.error_message

    return RequestTrace(
        request_id=final.trace_id,
        provider=final.provider,
        model=final.model,
        latency_ms=request_latency_ms,
        prompt_tokens=final.prompt_tokens,
        completion_tokens=final.completion_tokens,
        total_tokens=final.total_tokens,
        estimated_cost_usd=execution_cost,
        pricing_table_version=pricing_context,
        cache_hit=final.cache_hit,
        error_type=error_type,
        error_message=error_message,
        timestamp=datetime.fromtimestamp(final.end_time, tz=UTC).isoformat(),
        provider_attempt_count=len(attempts),
        provider_retry_count=max(len(attempts) - 1, 0),
        cost_evidence_complete=all_costs_known,
        provider_attempts=attempts,
    )


def _provider_attempt(index: int, record: _StandardLogRecord) -> ProviderAttempt:
    if record.reported_cost_usd is None:
        cost_evidence = CostEvidenceKind.UNKNOWN
        cost_source = None
        cost_source_record_id = None
    else:
        cost_evidence = CostEvidenceKind.REPORTED_BY_EXECUTION_STACK
        cost_source = LITELLM_STANDARD_LOGGING_SOURCE
        cost_source_record_id = record.record_id

    return ProviderAttempt(
        attempt_index=index,
        provider=record.provider,
        model=record.model,
        outcome=record.outcome,
        latency_ms=_seconds_to_milliseconds(record.response_time_seconds),
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        total_tokens=record.total_tokens,
        reported_cost_usd=record.reported_cost_usd,
        cost_evidence=cost_evidence,
        cost_source=cost_source,
        cost_source_record_id=cost_source_record_id,
        error_type=record.error_type,
        status_code=record.status_code,
    )


def _parse_record(payload: Mapping[str, object]) -> _StandardLogRecord:
    record_id = _required_text(payload, "id")
    trace_id = _required_text(payload, "trace_id")
    provider = _required_text(payload, "custom_llm_provider")
    model = _required_text(payload, "model")

    status = _required_text(payload, "status")
    if status == "success":
        outcome = AttemptOutcome.SUCCEEDED
    elif status == "failure":
        outcome = AttemptOutcome.FAILED
    else:
        raise LiteLLMImportError(f"unsupported LiteLLM status: {status!r}")

    start_time = _required_finite_float(payload, "startTime")
    end_time = _required_finite_float(payload, "endTime")
    if end_time < start_time:
        raise LiteLLMImportError("LiteLLM endTime must not precede startTime")
    response_time = _required_finite_float(payload, "response_time")
    if response_time < 0:
        raise LiteLLMImportError("LiteLLM response_time must be non-negative")

    prompt_tokens = _required_non_negative_int(payload, "prompt_tokens")
    completion_tokens = _required_non_negative_int(payload, "completion_tokens")
    total_tokens = _required_non_negative_int(payload, "total_tokens")
    if total_tokens < prompt_tokens + completion_tokens:
        raise LiteLLMImportError(
            "LiteLLM total_tokens must be at least prompt_tokens + completion_tokens"
        )

    cost_failure_debug = payload.get("response_cost_failure_debug_info")
    reported_cost = None
    if cost_failure_debug is None:
        raw_cost = payload.get("response_cost")
        if raw_cost is not None:
            reported_cost = _finite_float(raw_cost, field="response_cost")
            if reported_cost < 0:
                raise LiteLLMImportError("LiteLLM response_cost must be non-negative")

    raw_cache_hit = payload.get("cache_hit")
    if raw_cache_hit is None:
        cache_hit = None
    elif isinstance(raw_cache_hit, bool):
        cache_hit = raw_cache_hit
    else:
        raise LiteLLMImportError("LiteLLM cache_hit must be boolean or null")

    error_information = payload.get("error_information")
    error_type: str | None = None
    error_message = _optional_text(payload.get("error_str"))
    status_code: int | None = None
    if error_information is not None:
        if not isinstance(error_information, Mapping):
            raise LiteLLMImportError("LiteLLM error_information must be an object or null")
        error_type = _optional_text(error_information.get("error_class"))
        error_message = _optional_text(error_information.get("error_message")) or error_message
        status_code = _optional_status_code(error_information.get("error_code"))

    if outcome == AttemptOutcome.FAILED and error_type is None:
        error_type = "litellm_failure"

    return _StandardLogRecord(
        record_id=record_id,
        trace_id=trace_id,
        provider=provider,
        model=model,
        outcome=outcome,
        start_time=start_time,
        end_time=end_time,
        response_time_seconds=response_time,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cache_hit=cache_hit,
        reported_cost_usd=reported_cost,
        error_type=error_type,
        error_message=error_message,
        status_code=status_code,
    )


def _required_text(payload: Mapping[str, object], field: str) -> str:
    value = _optional_text(payload.get(field))
    if value is None:
        raise LiteLLMImportError(f"LiteLLM {field} must be a non-empty string")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LiteLLMImportError("LiteLLM text fields must be strings or null")
    stripped = value.strip()
    return stripped or None


def _required_finite_float(payload: Mapping[str, object], field: str) -> float:
    if field not in payload:
        raise LiteLLMImportError(f"LiteLLM {field} is required")
    return _finite_float(payload[field], field=field)


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LiteLLMImportError(f"LiteLLM {field} must be numeric")
    resolved = float(value)
    if not isfinite(resolved):
        raise LiteLLMImportError(f"LiteLLM {field} must be finite")
    return resolved


def _required_non_negative_int(payload: Mapping[str, object], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise LiteLLMImportError(f"LiteLLM {field} must be an integer")
    if value < 0:
        raise LiteLLMImportError(f"LiteLLM {field} must be non-negative")
    return value


def _optional_status_code(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _seconds_to_milliseconds(value: float) -> int:
    milliseconds = round(value * 1000.0)
    if milliseconds < 0:
        raise LiteLLMImportError("latency cannot be negative")
    return milliseconds


def sanitize_standard_logging_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Redact prompt/response content while preserving evidence-bearing fields.

    The sanitizer is intentionally conservative: unknown nested objects are copied, and only
    documented content-bearing fields are replaced. Identifiers, status, provider/model, timing,
    usage, cost, cache, and error fields remain intact.
    """
    copied = _copy_jsonable(payload)
    if not isinstance(copied, dict):
        raise LiteLLMImportError("LiteLLM payload must be an object")
    sanitized = copied
    _redact_messages(sanitized.get("messages"))
    if "response" in sanitized:
        sanitized["response"] = "[REDACTED_RESPONSE]"
    model_params = sanitized.get("model_parameters")
    if isinstance(model_params, dict) and "messages" in model_params:
        _redact_messages(model_params.get("messages"))
    return sanitized


def _redact_messages(messages: object) -> None:
    if not isinstance(messages, list):
        return
    for message in messages:
        if isinstance(message, dict) and "content" in message:
            message["content"] = "[REDACTED_PROMPT]"


def _copy_jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _copy_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_jsonable(item) for item in value]
    return value
