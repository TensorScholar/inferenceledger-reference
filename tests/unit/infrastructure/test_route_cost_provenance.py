from __future__ import annotations

import json
from datetime import date
from uuid import UUID

import pytest

from inference_engine.domain.cost.pricing import PricingQuote
from inference_engine.domain.models.routing import (
    ModelConfig,
    ModelTier,
    RoutingDecision,
    RoutingStrategy,
)
from inference_engine.infrastructure.telemetry.request_log import JsonlRouteLog, RouteTrace


def _quote() -> PricingQuote:
    observed_at = date(2026, 9, 3)
    return PricingQuote(
        amount_usd=0.0004,
        provider="test-provider",
        model="test-model",
        input_tokens=10,
        output_tokens=5,
        cached_input_tokens=0,
        input_per_million=20.0,
        output_per_million=40.0,
        cached_input_per_million=None,
        pricing_record_id=f"test-provider:test-model:{observed_at.isoformat()}",
        pricing_table_version="test-route-v1",
        pricing_observed_at=observed_at,
        pricing_source_url="https://pricing.example/test-model",
    )


def _decision() -> RoutingDecision:
    quote = _quote()
    return RoutingDecision(
        request_id=UUID("00000000-0000-0000-0000-000000000001"),
        selected_model=ModelConfig(
            id="test-model",
            name="Test Model",
            tier=ModelTier.STANDARD,
            max_context_length=4096,
        ),
        strategy=RoutingStrategy.SINGLE_MODEL,
        complexity_estimate=None,
        estimated_cost=quote.amount_usd,
        estimated_latency_ms=250,
        estimated_quality_score=0.7,
        decision_reason="test route",
        cost_quote=quote,
        considered_models=["test-model"],
    )


def test_route_trace_jsonl_round_trip_preserves_reconstructable_quote(tmp_path) -> None:
    path = tmp_path / "routes.jsonl"
    route = RouteTrace.from_decision(_decision(), max_estimated_cost_usd=0.01)
    log = JsonlRouteLog(path)

    log.append(route)
    stored = log.read_all()[0]

    assert stored == route
    assert stored.cost_evidence_complete is True
    assert stored.cost_quote == _quote()
    assert stored.cost_quote is not None
    assert stored.cost_quote.reconstructed_amount_usd == pytest.approx(stored.estimated_cost_usd)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["cost_quote"]["pricing_observed_at"] == "2026-09-03"
    assert raw["cost_quote"]["input_per_million"] == 20.0
    assert raw["cost_quote"]["output_per_million"] == 40.0


def test_route_jsonl_reader_downgrades_legacy_numeric_cost_without_quote(tmp_path) -> None:
    path = tmp_path / "routes.jsonl"
    path.write_text(
        json.dumps(
            {
                "request_id": "request-1",
                "strategy": "single_model",
                "selected_model": "test-model",
                "estimated_cost_usd": 0.0004,
                "estimated_latency_ms": 250,
                "decision_reason": "legacy route",
                "considered_models": ["test-model"],
                "fallback_models": [],
                "max_estimated_cost_usd": 0.01,
                "budget_violation": False,
                "budget_violation_reason": None,
                "timestamp": "2026-09-03T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    stored = JsonlRouteLog(path).read_all()[0]

    assert stored.estimated_cost_usd is None
    assert stored.cost_evidence_complete is False
    assert stored.cost_quote is None


def test_pricing_quote_rejects_tampered_amount() -> None:
    quote = _quote()

    with pytest.raises(ValueError, match="token and rate assumptions"):
        PricingQuote(
            amount_usd=quote.amount_usd + 0.01,
            provider=quote.provider,
            model=quote.model,
            input_tokens=quote.input_tokens,
            output_tokens=quote.output_tokens,
            cached_input_tokens=quote.cached_input_tokens,
            input_per_million=quote.input_per_million,
            output_per_million=quote.output_per_million,
            cached_input_per_million=quote.cached_input_per_million,
            pricing_record_id=quote.pricing_record_id,
            pricing_table_version=quote.pricing_table_version,
            pricing_observed_at=quote.pricing_observed_at,
            pricing_source_url=quote.pricing_source_url,
        )


def test_routing_decision_rejects_quote_for_different_model() -> None:
    quote = _quote()
    other = PricingQuote(
        amount_usd=quote.amount_usd,
        provider=quote.provider,
        model="other-model",
        input_tokens=quote.input_tokens,
        output_tokens=quote.output_tokens,
        cached_input_tokens=quote.cached_input_tokens,
        input_per_million=quote.input_per_million,
        output_per_million=quote.output_per_million,
        cached_input_per_million=quote.cached_input_per_million,
        pricing_record_id="test-provider:other-model:2026-09-03",
        pricing_table_version=quote.pricing_table_version,
        pricing_observed_at=quote.pricing_observed_at,
        pricing_source_url=quote.pricing_source_url,
    )

    with pytest.raises(ValueError, match="selected model"):
        RoutingDecision(
            request_id=UUID("00000000-0000-0000-0000-000000000002"),
            selected_model=ModelConfig(
                id="test-model",
                name="Test Model",
                tier=ModelTier.STANDARD,
                max_context_length=4096,
            ),
            strategy=RoutingStrategy.SINGLE_MODEL,
            complexity_estimate=None,
            estimated_cost=other.amount_usd,
            estimated_latency_ms=250,
            estimated_quality_score=0.7,
            decision_reason="tampered route",
            cost_quote=other,
        )
