from __future__ import annotations

from datetime import date

import pytest

from inference_engine.domain.cost.pricing import PricingQuote
from inference_engine.domain.models.request import InferenceRequest, ModelParameters
from inference_engine.domain.models.routing import ModelConfig, ModelTier, RoutingStrategy
from inference_engine.domain.routing.baseline import BaselineRouter, BaselineRoutingModeError
from inference_engine.domain.routing.capability import (
    required_context_tokens,
    supports_request_context,
)
from inference_engine.domain.routing.complexity import ComplexityEstimator
from inference_engine.domain.routing.cost_aware import CostAwareRouter
from inference_engine.domain.routing.load_balanced import LoadBalancedRouter
from inference_engine.domain.routing.policy import PolicyRouter


class AdversarialCostEstimator:
    def quote(
        self,
        *,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> PricingQuote:
        target_amount = {"too-small": 0.000001, "capable": 100.0}[model_id]
        total_tokens = input_tokens + output_tokens
        if total_tokens <= 0:
            raise ValueError("adversarial test quote requires positive token volume")
        rate_per_million = target_amount * 1_000_000 / total_tokens
        observed_at = date(2026, 9, 3)
        return PricingQuote(
            amount_usd=target_amount,
            provider="test-provider",
            model=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=0,
            input_per_million=rate_per_million,
            output_per_million=rate_per_million,
            cached_input_per_million=None,
            pricing_record_id=f"test-provider:{model_id}:{observed_at.isoformat()}",
            pricing_table_version="test-context-v1",
            pricing_observed_at=observed_at,
            pricing_source_url=f"https://pricing.example/{model_id}",
        )

    def estimate(
        self,
        *,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        return self.quote(
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ).amount_usd


def _request() -> InferenceRequest:
    return InferenceRequest(
        prompt="small prompt",
        parameters=ModelParameters(max_tokens=100),
    )


def _models() -> list[ModelConfig]:
    return [
        ModelConfig(
            id="too-small",
            name="Too Small",
            tier=ModelTier.ECONOMY,
            max_context_length=64,
            avg_latency_ms=1,
        ),
        ModelConfig(
            id="capable",
            name="Capable",
            tier=ModelTier.STANDARD,
            max_context_length=1024,
            avg_latency_ms=1000,
        ),
    ]


def test_context_capability_uses_prompt_plus_maximum_requested_output() -> None:
    request = _request()
    small, capable = _models()

    assert required_context_tokens(request) == request.estimated_input_tokens + 100
    assert supports_request_context(small, request) is False
    assert supports_request_context(capable, request) is True


@pytest.mark.asyncio
async def test_single_model_baseline_rejects_context_insufficient_configured_model() -> None:
    router = BaselineRouter(
        _models(),
        ComplexityEstimator(),
        AdversarialCostEstimator(),
        mode=RoutingStrategy.SINGLE_MODEL,
        single_model_id="too-small",
    )

    with pytest.raises(BaselineRoutingModeError, match="cannot satisfy request context"):
        await router.route(_request())


@pytest.mark.asyncio
async def test_rule_based_baseline_never_chooses_cheaper_context_insufficient_model() -> None:
    router = BaselineRouter(
        _models(),
        ComplexityEstimator(),
        AdversarialCostEstimator(),
        mode=RoutingStrategy.RULE_BASED,
    )

    decision = await router.route(_request())

    assert decision.selected_model.id == "capable"
    assert decision.cost_quote is not None
    assert decision.considered_models == ["capable"]
    assert decision.fallback_models == []


@pytest.mark.asyncio
async def test_load_balanced_router_excludes_context_insufficient_model() -> None:
    router = LoadBalancedRouter(_models(), AdversarialCostEstimator())

    first = await router.route(_request())
    second = await router.route(_request())

    assert first.selected_model.id == "capable"
    assert second.selected_model.id == "capable"
    assert first.cost_quote is not None
    assert second.cost_quote is not None
    assert first.considered_models == ["capable"]


@pytest.mark.asyncio
async def test_policy_router_excludes_context_insufficient_model_before_scoring() -> None:
    router = PolicyRouter(
        _models(),
        ComplexityEstimator(),
        AdversarialCostEstimator(),
    )

    decision = await router.route(_request())

    assert decision.selected_model.id == "capable"
    assert decision.cost_quote is not None
    assert decision.considered_models == ["capable"]


@pytest.mark.asyncio
async def test_cost_aware_router_excludes_context_insufficient_model_before_scoring() -> None:
    router = CostAwareRouter(
        _models(),
        ComplexityEstimator(),
        AdversarialCostEstimator(),
        cost_weight=1.0,
    )

    decision = await router.route(_request())

    assert decision.selected_model.id == "capable"
    assert decision.cost_quote is not None
    assert decision.considered_models == ["capable"]
    assert decision.estimated_cost == pytest.approx(100.0)
