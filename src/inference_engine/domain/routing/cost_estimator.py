from __future__ import annotations

from typing import Protocol

from ..cost.pricing import PricingQuote


class RoutingCostEstimator(Protocol):
    """Pre-execution pricing quote provider consumed by routing policies.

    Routing owns selection policy, not tariff data. Implementations must source monetary assumptions
    from the canonical pricing authority and return the complete quote used by the decision.
    """

    def quote(
        self,
        *,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> PricingQuote:
        """Return the auditable pricing quote used for one candidate model."""
        ...
