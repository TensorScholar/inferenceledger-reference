from __future__ import annotations

from datetime import date

import pytest

from inference_engine.domain.cost.pricing import PricingQuote
from inference_engine.domain.models.request import InferenceRequest, ModelParameters
from inference_engine.domain.models.routing import ComplexityEstimate, ModelConfig, ModelTier
from inference_engine.domain.routing.cost_aware import CostAwareRouter


class PremiumComplexityEstimator:
    async def estimate(self, _request: InferenceRequest) -> ComplexityEstimate:
        return ComplexityEstimate(
            score=0.9,
            factors={"test": 1.0},
            input_length=100,
            estimated_reasoning_steps=4,
            requires_context=False,
            domain_specific=True,
        )


class AdversarialCostEstimator:
    """Make the economy model unrealistically cheap to prove capability gating wins first."""

    def quote(
        self,
        *,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> PricingQuote:
        target_amount = {"economy": 0.000001, "premium": 100.0}[model_id]
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
            pricing_table_version="test-adversarial-v1",
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


def _premium_request() -> InferenceRequest:
    return InferenceRequest(
        prompt="adversarial premium request",
        parameters=ModelParameters(max_tokens=64),
    )


@pytest.mark.asyncio
async def test_premium_complexity_excludes_economy_before_cost_scoring() -> None:
    models = [
        ModelConfig(
            id="economy",
            name="Economy",
            tier=ModelTier.ECONOMY,
            max_context_length=128_000,
        ),
        ModelConfig(
            id="premium",
            name="Premium",
            tier=ModelTier.PREMIUM,
            max_context_length=128_000,
        ),
    ]
    router = CostAwareRouter(
        models,
        PremiumComplexityEstimator(),
        AdversarialCostEstimator(),
        cost_weight=1.0,
    )

    decision = await router.route(_premium_request())

    assert decision.selected_model.id == "premium"
    assert decision.considered_models == ["premium"]
    assert decision.cost_quote is not None
    assert decision.estimated_cost == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_cost_aware_router_does_not_fallback_to_incapable_healthy_model() -> None:
    models = [
        ModelConfig(
            id="economy",
            name="Economy",
            tier=ModelTier.ECONOMY,
            max_context_length=128_000,
        ),
        ModelConfig(
            id="premium",
            name="Premium",
            tier=ModelTier.PREMIUM,
            max_context_length=128_000,
            healthy=False,
        ),
    ]
    router = CostAwareRouter(
        models,
        PremiumComplexityEstimator(),
        AdversarialCostEstimator(),
    )

    with pytest.raises(RuntimeError, match="capability constraints"):
        await router.route(_premium_request())


@pytest.mark.parametrize("cost_weight", [-0.01, 1.01])
def test_cost_aware_router_rejects_invalid_cost_weight(cost_weight: float) -> None:
    with pytest.raises(ValueError, match="cost_weight"):
        CostAwareRouter(
            [],
            PremiumComplexityEstimator(),
            AdversarialCostEstimator(),
            cost_weight=cost_weight,
        )
