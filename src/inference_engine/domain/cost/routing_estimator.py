from __future__ import annotations

from .calculator import CostCalculator
from .pricing import PricingQuote


class ProviderPricingCostEstimator:
    """Routing estimator backed by the same provider pricing authority as execution accounting."""

    def __init__(
        self,
        *,
        provider: str,
        cost_calculator: CostCalculator | None = None,
    ) -> None:
        if not provider.strip():
            raise ValueError("provider must be non-empty")
        self.provider = provider
        self.cost_calculator = cost_calculator or CostCalculator()

    def validate_model(self, model_id: str) -> None:
        """Fail fast when a router candidate has no canonical pricing record."""
        self.cost_calculator.pricing_table.get(provider=self.provider, model=model_id)

    def quote(
        self,
        *,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> PricingQuote:
        return self.cost_calculator.quote_provider_usage(
            provider=self.provider,
            model_name=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def estimate(
        self,
        *,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Compatibility view for callers that only need the amount, not route evidence."""
        return self.quote(
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ).amount_usd
