from __future__ import annotations

from datetime import date

import pytest

from inference_engine.benchmarking.reconciliation import (
    CostDeviationDirection,
    ReconciliationStatus,
    reconcile_request_cost,
    reconcile_run_costs,
)
from inference_engine.domain.cost.pricing import PricingQuote
from inference_engine.domain.models.execution import (
    AttemptOutcome,
    CostEvidenceKind,
    ProviderAttempt,
)
from inference_engine.infrastructure.telemetry.request_log import (
    RequestTrace,
    RouteTrace,
)

_PRICING_DATE = "2026-09-03"
_PRICING_SOURCE = "https://pricing.example/model-a"


def _quote(
    *,
    provider: str = "openai",
    model: str = "model-a",
    amount_usd: float = 0.0005,
) -> PricingQuote:
    observed_at = date.fromisoformat(_PRICING_DATE)
    if amount_usd == 0.0005:
        input_tokens = 5_000
        output_tokens = 2_500
        input_rate = 0.05
        output_rate = 0.1
    elif amount_usd == 0.001:
        input_tokens = 10_000
        output_tokens = 5_000
        input_rate = 0.05
        output_rate = 0.1
    else:
        raise ValueError("unsupported test quote amount")
    return PricingQuote(
        amount_usd=amount_usd,
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=0,
        input_per_million=input_rate,
        output_per_million=output_rate,
        cached_input_per_million=None,
        pricing_record_id=f"{provider}:{model}:{_PRICING_DATE}",
        pricing_table_version="test-pricing-v1",
        pricing_observed_at=observed_at,
        pricing_source_url=_PRICING_SOURCE,
    )


def _route(
    request_id: str = "request-1",
    *,
    provider: str = "openai",
    model: str = "model-a",
    amount_usd: float = 0.0005,
) -> RouteTrace:
    quote = _quote(provider=provider, model=model, amount_usd=amount_usd)
    return RouteTrace(
        request_id=request_id,
        strategy="policy",
        selected_model=model,
        estimated_cost_usd=quote.amount_usd,
        estimated_latency_ms=250,
        decision_reason="policy.balanced_score",
        considered_models=[model],
        fallback_models=[],
        max_estimated_cost_usd=None,
        budget_violation=False,
        budget_violation_reason=None,
        timestamp="2026-09-03T00:00:00+00:00",
        cost_evidence_complete=True,
        cost_quote=quote,
    )


def _priced_attempt(
    *,
    index: int,
    outcome: AttemptOutcome,
    cost: float,
    provider: str = "openai",
    model: str = "model-a",
) -> ProviderAttempt:
    return ProviderAttempt(
        attempt_index=index,
        provider=provider,
        model=model,
        outcome=outcome,
        latency_ms=50,
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cached_tokens=0,
        calculated_cost_usd=cost,
        cost_evidence=CostEvidenceKind.CALCULATED_FROM_USAGE,
        pricing_table_version="test-pricing-v1",
        pricing_record_id=f"{provider}:{model}:{_PRICING_DATE}",
        pricing_observed_at=_PRICING_DATE,
        pricing_source_url=_PRICING_SOURCE,
        error_type="rate_limit" if outcome == AttemptOutcome.FAILED else None,
        status_code=429 if outcome == AttemptOutcome.FAILED else 200,
    )


def _execution(
    request_id: str = "request-1",
    *,
    attempts: tuple[ProviderAttempt, ...],
    model: str = "model-a",
    error_type: str | None = None,
) -> RequestTrace:
    total_cost = sum(attempt.calculated_cost_usd or 0.0 for attempt in attempts)
    return RequestTrace(
        request_id=request_id,
        provider=attempts[-1].provider,
        model=model,
        latency_ms=sum(attempt.latency_ms for attempt in attempts),
        prompt_tokens=100 if error_type is None else 0,
        completion_tokens=50 if error_type is None else 0,
        total_tokens=150 if error_type is None else 0,
        estimated_cost_usd=total_cost,
        pricing_table_version="test-pricing-v1",
        cache_hit=False,
        error_type=error_type,
        error_message="provider failed" if error_type is not None else None,
        timestamp="2026-09-03T00:00:01+00:00",
        provider_attempt_count=len(attempts),
        provider_retry_count=max(len(attempts) - 1, 0),
        cost_evidence_complete=True,
        provider_attempts=attempts,
    )


def test_reconcile_success_exposes_underestimate_and_retry_amplification() -> None:
    route = _route()
    execution = _execution(
        attempts=(
            _priced_attempt(index=1, outcome=AttemptOutcome.FAILED, cost=0.00025),
            _priced_attempt(index=2, outcome=AttemptOutcome.SUCCEEDED, cost=0.00075),
        )
    )

    result = reconcile_request_cost(route=route, execution=execution)

    assert result.status == ReconciliationStatus.COMPARABLE_SUCCESS
    assert result.comparable is True
    assert result.route_estimated_cost_usd == pytest.approx(0.0005)
    assert result.observed_execution_cost_usd == pytest.approx(0.001)
    assert result.cost_delta_usd == pytest.approx(0.0005)
    assert result.absolute_cost_deviation_usd == pytest.approx(0.0005)
    assert result.relative_cost_delta_percent == pytest.approx(100.0)
    assert result.execution_to_route_ratio == pytest.approx(2.0)
    assert result.deviation_direction == CostDeviationDirection.UNDERESTIMATED
    assert result.successful_final_attempt_cost_usd == pytest.approx(0.00075)
    assert result.non_final_attempt_cost_usd == pytest.approx(0.00025)
    assert result.retry_amplification_ratio == pytest.approx(4 / 3)
    assert result.provider_attempt_count == 2
    assert result.provider_retry_count == 1
    assert result.execution_path_diverged is False


def test_budget_rejection_is_not_treated_as_estimation_error() -> None:
    route = _route()
    execution = RequestTrace(
        request_id=route.request_id,
        provider="openai",
        model=route.selected_model,
        latency_ms=0,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        estimated_cost_usd=0.0,
        pricing_table_version="not_charged",
        cache_hit=False,
        error_type="budget_violation",
        error_message="blocked before provider execution",
        timestamp="2026-09-03T00:00:01+00:00",
        provider_attempt_count=0,
        provider_retry_count=0,
        cost_evidence_complete=True,
    )

    result = reconcile_request_cost(route=route, execution=execution)

    assert result.status == ReconciliationStatus.NOT_EXECUTED
    assert result.comparable is False
    assert result.route_estimated_cost_usd == pytest.approx(0.0005)
    assert result.observed_execution_cost_usd == pytest.approx(0.0)
    assert result.cost_delta_usd is None
    assert result.deviation_direction == CostDeviationDirection.NOT_COMPARABLE
    assert result.execution_succeeded is None


def test_unknown_attempt_cost_blocks_reconciliation_math() -> None:
    route = _route()
    unknown_attempt = ProviderAttempt(
        attempt_index=1,
        provider="openai",
        model="model-a",
        outcome=AttemptOutcome.FAILED,
        latency_ms=50,
        cost_evidence=CostEvidenceKind.UNKNOWN,
        error_type="rate_limit",
        status_code=429,
    )
    execution = RequestTrace(
        request_id=route.request_id,
        provider="openai",
        model="model-a",
        latency_ms=50,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        estimated_cost_usd=None,
        pricing_table_version="unpriced",
        cache_hit=False,
        error_type="rate_limit",
        error_message="rate limited",
        timestamp="2026-09-03T00:00:01+00:00",
        provider_attempt_count=1,
        provider_retry_count=0,
        cost_evidence_complete=False,
        provider_attempts=(unknown_attempt,),
    )

    result = reconcile_request_cost(route=route, execution=execution)

    assert result.status == ReconciliationStatus.EXECUTION_COST_INCOMPLETE
    assert result.observed_execution_cost_usd is None
    assert result.cost_delta_usd is None


def test_fallback_path_divergence_is_preserved_in_reconciliation() -> None:
    route = _route(model="model-a")
    execution = _execution(
        attempts=(
            _priced_attempt(index=1, outcome=AttemptOutcome.FAILED, cost=0.0002),
            _priced_attempt(
                index=2,
                outcome=AttemptOutcome.SUCCEEDED,
                cost=0.0003,
                provider="other-provider",
                model="model-b",
            ),
        ),
        model="model-b",
    )

    result = reconcile_request_cost(route=route, execution=execution)

    assert result.status == ReconciliationStatus.COMPARABLE_SUCCESS
    assert result.execution_path == ("openai/model-a", "other-provider/model-b")
    assert result.execution_path_diverged is True
    assert result.deviation_direction == CostDeviationDirection.MATCHED


def test_known_failed_execution_remains_comparable_but_has_no_success_amplification() -> None:
    route = _route()
    execution = _execution(
        attempts=(
            _priced_attempt(index=1, outcome=AttemptOutcome.FAILED, cost=0.0002),
            _priced_attempt(index=2, outcome=AttemptOutcome.FAILED, cost=0.0003),
        ),
        error_type="provider_unavailable",
    )

    result = reconcile_request_cost(route=route, execution=execution)

    assert result.status == ReconciliationStatus.COMPARABLE_FAILURE
    assert result.observed_execution_cost_usd == pytest.approx(0.0005)
    assert result.deviation_direction == CostDeviationDirection.MATCHED
    assert result.successful_final_attempt_cost_usd is None
    assert result.non_final_attempt_cost_usd is None
    assert result.retry_amplification_ratio is None


def test_run_summary_counts_evidence_gaps_and_retry_tax_conservatively() -> None:
    comparable_route = _route("request-1")
    comparable_execution = _execution(
        "request-1",
        attempts=(
            _priced_attempt(index=1, outcome=AttemptOutcome.FAILED, cost=0.00025),
            _priced_attempt(index=2, outcome=AttemptOutcome.SUCCEEDED, cost=0.00075),
        ),
    )
    budget_route = _route("request-2")
    budget_execution = RequestTrace(
        request_id="request-2",
        provider="openai",
        model="model-a",
        latency_ms=0,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        estimated_cost_usd=0.0,
        pricing_table_version="not_charged",
        cache_hit=False,
        error_type="budget_violation",
        error_message="blocked",
        timestamp="2026-09-03T00:00:01+00:00",
        provider_attempt_count=0,
        provider_retry_count=0,
        cost_evidence_complete=True,
    )
    missing_execution_route = _route("request-3")

    summary = reconcile_run_costs(
        routes=[comparable_route, budget_route, missing_execution_route],
        executions=[comparable_execution, budget_execution],
    )

    assert summary.request_count == 3
    assert summary.paired_request_count == 2
    assert summary.comparable_request_count == 1
    assert summary.comparable_coverage == pytest.approx(1 / 3)
    assert summary.not_executed_count == 1
    assert summary.missing_execution_count == 1
    assert summary.comparable_route_estimated_cost_usd == pytest.approx(0.0005)
    assert summary.comparable_observed_execution_cost_usd == pytest.approx(0.001)
    assert summary.comparable_cost_delta_usd == pytest.approx(0.0005)
    assert summary.comparable_cost_delta_percent == pytest.approx(100.0)
    assert summary.mean_absolute_cost_deviation_usd == pytest.approx(0.0005)
    assert summary.median_absolute_cost_deviation_usd == pytest.approx(0.0005)
    assert summary.p95_absolute_cost_deviation_usd == pytest.approx(0.0005)
    assert summary.underestimation_rate == pytest.approx(1.0)
    assert summary.overestimation_rate == pytest.approx(0.0)
    assert summary.matched_rate == pytest.approx(0.0)
    assert summary.execution_path_divergence_count == 0
    assert summary.execution_path_divergence_rate == pytest.approx(0.0)
    assert summary.retry_amplification_eligible_request_count == 1
    assert summary.non_final_attempt_cost_usd == pytest.approx(0.00025)
    assert summary.retry_amplification_share == pytest.approx(0.25)


def test_run_reconciliation_rejects_duplicate_request_ids() -> None:
    route = _route()

    with pytest.raises(ValueError, match="duplicate route request_id"):
        reconcile_run_costs(routes=[route, route], executions=[])
