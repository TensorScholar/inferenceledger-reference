from __future__ import annotations

from datetime import date

import pytest

from inference_engine.benchmarking.segmentation import (
    BenchmarkRequestContext,
    empirical_nearest_rank,
    summarize_segments,
)
from inference_engine.domain.cost.pricing import PricingQuote
from inference_engine.domain.models.execution import (
    AttemptOutcome,
    CostEvidenceKind,
    ProviderAttempt,
)
from inference_engine.infrastructure.telemetry.request_log import RequestTrace, RouteTrace

_PRICING_DATE = "2026-09-03"


def _context(
    request_id: str,
    workload_item_id: str,
    tags: dict[str, str],
) -> BenchmarkRequestContext:
    return BenchmarkRequestContext.from_tags(
        request_id=request_id,
        workload_item_id=workload_item_id,
        tags=tags,
    )


def _quote(request_id: str, *, model: str = "model-a") -> PricingQuote:
    del request_id
    observed_at = date.fromisoformat(_PRICING_DATE)
    return PricingQuote(
        amount_usd=0.001,
        provider="openai",
        model=model,
        input_tokens=10,
        output_tokens=5,
        cached_input_tokens=0,
        input_per_million=50.0,
        output_per_million=100.0,
        cached_input_per_million=None,
        pricing_record_id=f"openai:{model}:{_PRICING_DATE}",
        pricing_table_version="test-v1",
        pricing_observed_at=observed_at,
        pricing_source_url="https://pricing.example/test",
    )


def _route(request_id: str, *, model: str = "model-a") -> RouteTrace:
    quote = _quote(request_id, model=model)
    return RouteTrace(
        request_id=request_id,
        strategy="policy",
        selected_model=model,
        estimated_cost_usd=quote.amount_usd,
        estimated_latency_ms=100,
        decision_reason="test",
        considered_models=[model],
        fallback_models=[],
        max_estimated_cost_usd=None,
        budget_violation=False,
        budget_violation_reason=None,
        timestamp="2026-09-03T00:00:00+00:00",
        cost_evidence_complete=True,
        cost_quote=quote,
    )


def _attempt(
    *,
    index: int,
    outcome: AttemptOutcome,
    cost: float,
    model: str = "model-a",
) -> ProviderAttempt:
    return ProviderAttempt(
        attempt_index=index,
        provider="openai",
        model=model,
        outcome=outcome,
        latency_ms=10,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cached_tokens=0,
        calculated_cost_usd=cost,
        cost_evidence=CostEvidenceKind.CALCULATED_FROM_USAGE,
        pricing_table_version="test-v1",
        pricing_record_id=f"openai:{model}:{_PRICING_DATE}",
        pricing_observed_at=_PRICING_DATE,
        pricing_source_url="https://pricing.example/test",
        error_type="rate_limit" if outcome == AttemptOutcome.FAILED else None,
        status_code=429 if outcome == AttemptOutcome.FAILED else 200,
    )


def _trace(
    request_id: str,
    *,
    latency_ms: int,
    cost: float | None = 0.001,
    error_type: str | None = None,
    quality_passed: bool | None = True,
    model: str = "model-a",
) -> RequestTrace:
    if cost is None:
        return RequestTrace(
            request_id=request_id,
            provider="openai",
            model=model,
            latency_ms=latency_ms,
            prompt_tokens=0 if error_type else 10,
            completion_tokens=0 if error_type else 5,
            total_tokens=0 if error_type else 15,
            estimated_cost_usd=None,
            pricing_table_version="unpriced",
            cache_hit=False,
            error_type=error_type,
            error_message="failed" if error_type else None,
            timestamp="2026-09-03T00:00:01+00:00",
            quality_passed=quality_passed if error_type is None else None,
            quality_score=1.0 if quality_passed else None,
            provider_attempt_count=1,
            provider_retry_count=0,
            cost_evidence_complete=False,
        )

    outcome = AttemptOutcome.FAILED if error_type is not None else AttemptOutcome.SUCCEEDED
    attempts = (_attempt(index=1, outcome=outcome, cost=cost, model=model),)
    return RequestTrace(
        request_id=request_id,
        provider="openai",
        model=model,
        latency_ms=latency_ms,
        prompt_tokens=0 if error_type else 10,
        completion_tokens=0 if error_type else 5,
        total_tokens=0 if error_type else 15,
        estimated_cost_usd=cost,
        pricing_table_version="test-v1",
        cache_hit=False,
        error_type=error_type,
        error_message="failed" if error_type else None,
        timestamp="2026-09-03T00:00:01+00:00",
        quality_passed=quality_passed if error_type is None else None,
        quality_score=1.0 if quality_passed else None,
        provider_attempt_count=1,
        provider_retry_count=0,
        cost_evidence_complete=True,
        provider_attempts=attempts,
    )


def test_segment_evidence_uses_dynamic_tags_and_failed_spend_in_unit_cost() -> None:
    contexts = [
        _context("request-1", "item-1", {"task": "code", "risk": "high"}),
        _context("request-2", "item-2", {"task": "code", "risk": "low"}),
    ]
    traces = [
        _trace("request-1", latency_ms=10, quality_passed=True),
        _trace(
            "request-2",
            latency_ms=100,
            error_type="provider_unavailable",
            quality_passed=None,
        ),
    ]
    routes = [_route("request-1"), _route("request-2")]

    summary = summarize_segments(request_contexts=contexts, traces=traces, routes=routes)

    assert summary.available is True
    assert summary.request_count == 2
    assert summary.tagged_request_count == 2
    assert summary.segment_count == 3
    task = next(
        segment
        for segment in summary.segments
        if segment.tag_key == "task" and segment.tag_value == "code"
    )
    assert task.request_count == 2
    assert task.success_count == 1
    assert task.failure_count == 1
    assert task.error_rate == pytest.approx(0.5)
    assert task.latency_sample_count == 1
    assert task.latency_p50_ms == 10
    assert task.latency_p95_ms == 10
    assert task.latency_p99_ms == 10
    assert task.estimated_cost_usd == pytest.approx(0.002)
    assert task.cost_per_success_usd == pytest.approx(0.002)
    assert task.quality_count == 1
    assert task.quality_coverage == pytest.approx(1.0)
    assert task.quality_pass_rate == pytest.approx(1.0)
    assert task.cost_per_accepted_outcome_usd == pytest.approx(0.002)


def test_partial_quality_coverage_suppresses_cost_per_accepted_outcome() -> None:
    contexts = [
        _context("request-1", "item-1", {"task": "qa"}),
        _context("request-2", "item-2", {"task": "qa"}),
    ]
    traces = [
        _trace("request-1", latency_ms=10, quality_passed=True),
        _trace("request-2", latency_ms=20, quality_passed=None),
    ]

    summary = summarize_segments(request_contexts=contexts, traces=traces)
    segment = summary.segments[0]

    assert segment.quality_count == 1
    assert segment.quality_coverage == pytest.approx(0.5)
    assert segment.cost_evidence_complete is True
    assert segment.cost_per_success_usd == pytest.approx(0.001)
    assert segment.cost_per_accepted_outcome_usd is None


def test_unknown_cost_suppresses_segment_cost_and_unit_economics() -> None:
    contexts = [
        _context("request-1", "item-1", {"task": "qa"}),
        _context("request-2", "item-2", {"task": "qa"}),
    ]
    traces = [
        _trace("request-1", latency_ms=10),
        _trace("request-2", latency_ms=20, cost=None),
    ]

    segment = summarize_segments(request_contexts=contexts, traces=traces).segments[0]

    assert segment.cost_evidence_complete is False
    assert segment.estimated_cost_usd is None
    assert segment.cost_per_success_usd is None
    assert segment.cost_per_accepted_outcome_usd is None


def test_segment_with_no_success_has_no_latency_claim() -> None:
    contexts = [_context("request-1", "item-1", {"task": "qa"})]
    traces = [
        _trace(
            "request-1",
            latency_ms=1,
            error_type="provider_unavailable",
            quality_passed=None,
        )
    ]

    segment = summarize_segments(request_contexts=contexts, traces=traces).segments[0]

    assert segment.latency_sample_count == 0
    assert segment.latency_p50_ms is None
    assert segment.latency_p95_ms is None
    assert segment.latency_p99_ms is None


def test_empty_contexts_are_explicitly_unavailable_for_legacy_runs() -> None:
    summary = summarize_segments(
        request_contexts=[],
        traces=[_trace("request-1", latency_ms=10)],
    )

    assert summary.available is False
    assert summary.segment_count == 0
    assert summary.unavailable_reason is not None


def test_partial_context_coverage_fails_closed() -> None:
    with pytest.raises(ValueError, match="exact request-context coverage"):
        summarize_segments(
            request_contexts=[_context("request-1", "item-1", {"task": "qa"})],
            traces=[
                _trace("request-1", latency_ms=10),
                _trace("request-2", latency_ms=20),
            ],
        )


def test_duplicate_workload_item_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate benchmark workload_item_id"):
        summarize_segments(
            request_contexts=[
                _context("request-1", "item-1", {"task": "qa"}),
                _context("request-2", "item-1", {"task": "qa"}),
            ],
            traces=[
                _trace("request-1", latency_ms=10),
                _trace("request-2", latency_ms=20),
            ],
        )


def test_empirical_nearest_rank_is_explicit_and_conservative_for_small_tails() -> None:
    values = [10, 20, 30, 40]

    assert empirical_nearest_rank(values, 0) == 10
    assert empirical_nearest_rank(values, 50) == 20
    assert empirical_nearest_rank(values, 95) == 40
    assert empirical_nearest_rank(values, 99) == 40
    with pytest.raises(ValueError, match="at least one"):
        empirical_nearest_rank([], 95)
