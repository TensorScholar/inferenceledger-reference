from __future__ import annotations

import json
from uuid import uuid4

import pytest

from inference_engine.domain.models.execution import (
    AttemptOutcome,
    CostEvidenceKind,
    ProviderAttempt,
)
from inference_engine.domain.models.response import CacheInfo, InferenceResponse, UsageMetrics
from inference_engine.domain.models.routing import (
    ModelConfig,
    ModelTier,
    RoutingDecision,
    RoutingStrategy,
)
from inference_engine.infrastructure.models.errors import ProviderError, ProviderErrorType
from inference_engine.infrastructure.telemetry.request_log import (
    JsonlRequestLog,
    JsonlRouteLog,
    RequestTrace,
    RouteTrace,
)


def _known_attempt(*, attempt_index: int = 1, cost: float = 0.00002) -> ProviderAttempt:
    return ProviderAttempt(
        attempt_index=attempt_index,
        provider="test-provider",
        model="test-model",
        outcome=AttemptOutcome.SUCCEEDED,
        latency_ms=20,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cached_tokens=0,
        calculated_cost_usd=cost,
        cost_evidence=CostEvidenceKind.CALCULATED_FROM_USAGE,
        pricing_table_version="test",
        pricing_record_id="test-provider:test-model:2026-09-03",
        pricing_observed_at="2026-09-03",
        pricing_source_url="https://pricing.example/test-model",
    )


def _unknown_failed_attempt(*, attempt_index: int = 1) -> ProviderAttempt:
    return ProviderAttempt(
        attempt_index=attempt_index,
        provider="test-provider",
        model="test-model",
        outcome=AttemptOutcome.FAILED,
        latency_ms=10,
        error_type="rate_limit",
        status_code=429,
    )


def test_jsonl_request_log_round_trips_success_trace_and_pricing_provenance(tmp_path) -> None:
    response = InferenceResponse(
        request_id=uuid4(),
        text="ok",
        model_used="test-model",
        usage=UsageMetrics(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.00002,
        ),
        cache_info=CacheInfo(hit=False),
        latency_ms=123,
        provider_attempts=(_known_attempt(),),
    )
    request_log = JsonlRequestLog(tmp_path / "ledger.jsonl")

    request_log.append(RequestTrace.from_response(provider="test-provider", response=response))

    traces = request_log.read_all()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.request_id == str(response.request_id)
    assert trace.model == "test-model"
    assert trace.estimated_cost_usd == pytest.approx(0.00002)
    assert trace.cost_evidence_complete is True
    assert trace.pricing_table_version == "test"
    assert trace.error_type is None
    assert trace.provider_attempt_count == 1
    assert trace.provider_retry_count == 0
    assert trace.provider_attempts == response.provider_attempts
    assert trace.provider_attempts[0].pricing_record_id == "test-provider:test-model:2026-09-03"
    assert trace.provider_attempts[0].pricing_source_url == "https://pricing.example/test-model"


def test_success_after_unknown_retry_has_unknown_total_execution_cost() -> None:
    response = InferenceResponse(
        request_id=uuid4(),
        text="ok",
        model_used="test-model",
        usage=UsageMetrics(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.00002,
        ),
        cache_info=CacheInfo(hit=False),
        latency_ms=123,
        provider_attempts=(
            _unknown_failed_attempt(attempt_index=1),
            _known_attempt(attempt_index=2),
        ),
    )

    trace = RequestTrace.from_response(provider="test-provider", response=response)

    assert trace.provider_attempt_count == 2
    assert trace.provider_retry_count == 1
    assert trace.estimated_cost_usd is None
    assert trace.cost_evidence_complete is False
    assert trace.pricing_table_version == "incomplete:test"


def test_legacy_multi_attempt_response_without_attempt_details_is_incomplete() -> None:
    response = InferenceResponse(
        request_id=uuid4(),
        text="ok",
        model_used="test-model",
        usage=UsageMetrics(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost_usd=0.00002,
        ),
        cache_info=CacheInfo(hit=False),
        latency_ms=123,
        provider_attempt_count=2,
        provider_retry_count=1,
    )

    trace = RequestTrace.from_response(provider="test-provider", response=response)

    assert trace.estimated_cost_usd is None
    assert trace.cost_evidence_complete is False


def test_jsonl_request_log_round_trips_error_trace_as_unknown_cost(tmp_path) -> None:
    request_id = uuid4()
    request_log = JsonlRequestLog(tmp_path / "ledger.jsonl")
    attempts = (
        _unknown_failed_attempt(attempt_index=1),
        _unknown_failed_attempt(attempt_index=2),
    )
    error = ProviderError(
        ProviderErrorType.RATE_LIMIT,
        "rate limited",
        provider="test-provider",
        retryable=True,
        status_code=429,
        provider_attempts=attempts,
    )

    request_log.append(
        RequestTrace.from_error(
            request_id=request_id,
            provider="test-provider",
            model="test-model",
            latency_ms=42,
            error=error,
        )
    )

    traces = request_log.read_all()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.request_id == str(request_id)
    assert trace.latency_ms == 42
    assert trace.error_type == "rate_limit"
    assert trace.estimated_cost_usd is None
    assert trace.cost_evidence_complete is False
    assert trace.pricing_table_version == "unpriced"
    assert trace.provider_attempt_count == 2
    assert trace.provider_retry_count == 1
    assert trace.provider_attempts == attempts


def test_jsonl_reader_downgrades_legacy_retry_cost_to_unknown(tmp_path) -> None:
    ledger_path = tmp_path / "legacy.jsonl"
    ledger_path.write_text(
        "{"
        '"request_id":"request-1","provider":"openai","model":"test-model",'
        '"latency_ms":42,"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,'
        '"estimated_cost_usd":0.00002,"pricing_table_version":"legacy","cache_hit":false,'
        '"error_type":null,"error_message":null,"timestamp":"2026-01-01T00:00:00+00:00",'
        '"provider_attempt_count":2,"provider_retry_count":1}'
        "\n",
        encoding="utf-8",
    )

    trace = JsonlRequestLog(ledger_path).read_all()[0]

    assert trace.provider_attempt_count == 2
    assert trace.provider_retry_count == 1
    assert trace.estimated_cost_usd is None
    assert trace.cost_evidence_complete is False


def test_jsonl_reader_downgrades_pre_provenance_calculated_attempt(tmp_path) -> None:
    ledger_path = tmp_path / "legacy-attempt.jsonl"
    raw = {
        "request_id": "request-1",
        "provider": "openai",
        "model": "test-model",
        "latency_ms": 42,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "estimated_cost_usd": 0.00002,
        "pricing_table_version": "legacy",
        "cache_hit": False,
        "error_type": None,
        "error_message": None,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "provider_attempt_count": 1,
        "provider_retry_count": 0,
        "cost_evidence_complete": True,
        "provider_attempts": [
            {
                "attempt_index": 1,
                "provider": "openai",
                "model": "test-model",
                "outcome": "succeeded",
                "latency_ms": 42,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cached_tokens": 0,
                "calculated_cost_usd": 0.00002,
                "cost_evidence": "calculated_from_usage",
                "pricing_table_version": "legacy",
                "error_type": None,
                "status_code": None,
            }
        ],
    }
    ledger_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    trace = JsonlRequestLog(ledger_path).read_all()[0]

    assert trace.estimated_cost_usd is None
    assert trace.cost_evidence_complete is False
    assert trace.pricing_table_version == "unpriced"
    assert trace.provider_attempts[0].cost_evidence == CostEvidenceKind.UNKNOWN
    assert trace.provider_attempts[0].calculated_cost_usd is None


def test_request_trace_allows_known_zero_cost_when_no_provider_attempt_occurred() -> None:
    trace = RequestTrace(
        request_id="request-1",
        provider="openai",
        model="test-model",
        latency_ms=0,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        estimated_cost_usd=0.0,
        pricing_table_version="not_charged",
        cache_hit=False,
        error_type="budget_violation",
        error_message="rejected before provider execution",
        timestamp="2026-01-01T00:00:00+00:00",
        provider_attempt_count=0,
        provider_retry_count=0,
        cost_evidence_complete=True,
    )

    assert trace.estimated_cost_usd == 0.0
    assert trace.cost_evidence_complete is True
    assert trace.provider_attempt_count == 0


def test_request_trace_rejects_numeric_total_when_cost_is_incomplete() -> None:
    with pytest.raises(ValueError, match="incomplete cost evidence"):
        RequestTrace(
            request_id="request-1",
            provider="openai",
            model="test-model",
            latency_ms=10,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0.0,
            pricing_table_version="test",
            cache_hit=False,
            error_type="rate_limit",
            error_message="rate limited",
            timestamp="2026-01-01T00:00:00+00:00",
            cost_evidence_complete=False,
        )


def test_jsonl_route_log_round_trips_route_trace(tmp_path) -> None:
    request_id = uuid4()
    model = ModelConfig(
        id="test-model",
        name="Test Model",
        tier=ModelTier.STANDARD,
        max_context_length=4096,
    )
    decision = RoutingDecision(
        request_id=request_id,
        selected_model=model,
        strategy=RoutingStrategy.SINGLE_MODEL,
        complexity_estimate=None,
        estimated_cost=0.001,
        estimated_latency_ms=250,
        estimated_quality_score=0.7,
        decision_reason="test route",
        fallback_models=[],
        considered_models=["test-model"],
    )
    route_log = JsonlRouteLog(tmp_path / "routes.jsonl")

    route_log.append(
        RouteTrace.from_decision(
            decision,
            max_estimated_cost_usd=0.0005,
            budget_violation_reason="too expensive",
        )
    )

    traces = route_log.read_all()
    assert len(traces) == 1
    assert traces[0].request_id == str(request_id)
    assert traces[0].selected_model == "test-model"
    assert traces[0].budget_violation is True
    assert traces[0].budget_violation_reason == "too expensive"
