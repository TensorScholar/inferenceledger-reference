"""Unit tests for cost management."""

import pytest

from inference_engine.domain.cost.calculator import CostCalculator
from inference_engine.domain.cost.pricing import (
    DEFAULT_PRICING,
    PRICING_TABLE_VERSION,
    ModelPricing,
    PricingTable,
    UnknownModelPricingError,
)
from inference_engine.domain.cost.routing_estimator import ProviderPricingCostEstimator
from inference_engine.domain.models.cost import CostBreakdown


def _pricing_table(*, cached_input_per_million: float | None = None) -> PricingTable:
    pricing = ModelPricing(
        provider="test-provider",
        model="test-model",
        input_per_million=1.00,
        output_per_million=2.00,
        cached_input_per_million=cached_input_per_million,
        source_url="https://pricing.example/test-model",
    )
    return PricingTable(
        prices={(pricing.provider, pricing.model): pricing},
        version="test",
    )


class TestCostCalculator:
    """Tests for canonical provider pricing."""

    def test_quote_provider_usage_preserves_pricing_provenance(self):
        calculator = CostCalculator(_pricing_table())

        quote = calculator.quote_provider_usage(
            provider="test-provider",
            model_name="test-model",
            input_tokens=1_000,
            output_tokens=500,
        )

        assert quote.amount_usd == pytest.approx(0.002)
        assert quote.provider == "test-provider"
        assert quote.model == "test-model"
        assert quote.pricing_table_version == "test"
        assert quote.pricing_record_id.startswith("test-provider:test-model:")
        assert quote.pricing_source_url == "https://pricing.example/test-model"

    def test_quote_provider_usage_uses_cached_input_rate(self):
        calculator = CostCalculator(_pricing_table(cached_input_per_million=0.25))

        quote = calculator.quote_provider_usage(
            provider="test-provider",
            model_name="test-model",
            input_tokens=1_000,
            output_tokens=500,
            cached_input_tokens=400,
        )

        assert quote.amount_usd == pytest.approx(0.0017)

    def test_quote_provider_usage_rejects_unknown_provider_even_for_known_model(self):
        calculator = CostCalculator(_pricing_table())

        with pytest.raises(UnknownModelPricingError):
            calculator.quote_provider_usage(
                provider="different-provider",
                model_name="test-model",
                input_tokens=1,
                output_tokens=1,
            )

    def test_routing_estimator_uses_same_quote_path_as_execution_accounting(self):
        calculator = CostCalculator(_pricing_table())
        estimator = ProviderPricingCostEstimator(
            provider="test-provider",
            cost_calculator=calculator,
        )

        estimated = estimator.estimate(
            model_id="test-model",
            input_tokens=1_000,
            output_tokens=500,
        )
        execution_quote = calculator.quote_provider_usage(
            provider="test-provider",
            model_name="test-model",
            input_tokens=1_000,
            output_tokens=500,
        )

        assert estimated == execution_quote.amount_usd
        assert estimated == pytest.approx(0.002)

    def test_routing_estimator_validates_candidates_against_canonical_table(self):
        estimator = ProviderPricingCostEstimator(
            provider="test-provider",
            cost_calculator=CostCalculator(_pricing_table()),
        )

        estimator.validate_model("test-model")
        with pytest.raises(UnknownModelPricingError):
            estimator.validate_model("missing-model")

    def test_default_pricing_includes_sourced_openai_standard_record(self):
        pricing = DEFAULT_PRICING[("openai", "gpt-5.4-mini")]

        assert PRICING_TABLE_VERSION == "openai-standard-2026-09-03"
        assert pricing.provider == "openai"
        assert pricing.input_per_million == 0.75
        assert pricing.output_per_million == 4.50
        assert pricing.cached_input_per_million == 0.075
        assert pricing.source_url


class TestCostBreakdown:
    """Tests for CostBreakdown."""

    def test_savings_rate(self):
        breakdown = CostBreakdown(
            inference_cost=100.0,
            compute_cost=20.0,
            cache_savings=30.0,
            optimization_savings=50.0,
        )

        assert breakdown.savings_rate == pytest.approx(80 / 120)

    def test_net_cost(self):
        breakdown = CostBreakdown(
            inference_cost=100.0,
            compute_cost=20.0,
            cache_savings=30.0,
            optimization_savings=50.0,
        )

        assert breakdown.net_cost == 40.0
