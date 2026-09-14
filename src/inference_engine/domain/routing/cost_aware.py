from __future__ import annotations

import structlog

from ..cost.pricing import PricingQuote
from ..models.request import InferenceRequest
from ..models.routing import (
    ComplexityEstimate,
    ModelConfig,
    ModelTier,
    RoutingDecision,
    RoutingStrategy,
)
from .base import AbstractRouter
from .capability import supports_request_context
from .complexity import ComplexityEstimator
from .cost_estimator import RoutingCostEstimator

logger = structlog.get_logger()


class CostAwareRouter(AbstractRouter):
    """Route to the lowest-scoring capable model using request-specific canonical cost quotes."""

    def __init__(
        self,
        models: list[ModelConfig],
        complexity_estimator: ComplexityEstimator,
        cost_estimator: RoutingCostEstimator,
        cost_weight: float = 0.7,
    ) -> None:
        if not 0 <= cost_weight <= 1:
            raise ValueError("cost_weight must be between 0 and 1")
        self.models = {model.id: model for model in models}
        self.complexity_estimator = complexity_estimator
        self.cost_estimator = cost_estimator
        self.cost_weight = cost_weight

    async def route(self, request: InferenceRequest) -> RoutingDecision:
        complexity = await self.complexity_estimator.estimate(request)
        available = [
            model
            for model in self.models.values()
            if model.is_available and self._can_handle_request(model, request, complexity)
        ]
        if not available:
            raise RuntimeError("No available model can satisfy request capability constraints")

        selected, cost_quote, candidate_quotes = self._select_optimal_model(
            available,
            request,
            complexity,
        )
        fallback_models = [
            model
            for model in sorted(
                available,
                key=lambda model: (
                    candidate_quotes[model.id].amount_usd,
                    model.avg_latency_ms,
                    model.id,
                ),
            )
            if model.id != selected.id
        ][:3]
        decision = RoutingDecision(
            request_id=request.id,
            selected_model=selected,
            fallback_models=fallback_models,
            strategy=RoutingStrategy.COST_OPTIMAL,
            complexity_estimate=complexity,
            estimated_cost=cost_quote.amount_usd,
            estimated_latency_ms=selected.avg_latency_ms,
            estimated_quality_score=self._estimate_quality(selected, complexity),
            decision_reason=self._generate_reason(selected, complexity),
            cost_quote=cost_quote,
            considered_models=sorted(model.id for model in available),
        )
        logger.debug("routing_decision", selected_model=selected.id, cost=cost_quote.amount_usd)
        return decision

    def _can_handle_request(
        self, model: ModelConfig, request: InferenceRequest, complexity: ComplexityEstimate
    ) -> bool:
        if not supports_request_context(model, request):
            return False
        return not (
            complexity.recommended_tier == ModelTier.PREMIUM and model.tier == ModelTier.ECONOMY
        )

    def _select_optimal_model(
        self,
        models: list[ModelConfig],
        request: InferenceRequest,
        complexity: ComplexityEstimate,
    ) -> tuple[ModelConfig, PricingQuote, dict[str, PricingQuote]]:
        quote_by_model = {
            model.id: self.cost_estimator.quote(
                model_id=model.id,
                input_tokens=request.estimated_input_tokens,
                output_tokens=request.parameters.max_tokens,
            )
            for model in models
        }
        costs = [quote.amount_usd for quote in quote_by_model.values()]
        min_cost = min(costs)
        max_cost = max(costs)
        scored: list[tuple[float, str, ModelConfig]] = []
        for model in models:
            normalized_cost = _normalize(quote_by_model[model.id].amount_usd, min_cost, max_cost)
            quality_score = self._estimate_quality(model, complexity)
            score = self.cost_weight * normalized_cost + (1 - self.cost_weight) * (
                1 - quality_score
            )
            score += model.current_load * 0.2
            scored.append((score, model.id, model))
        selected = min(scored, key=lambda item: (item[0], item[1]))[2]
        return selected, quote_by_model[selected.id], quote_by_model

    def _estimate_quality(self, model: ModelConfig, complexity: ComplexityEstimate) -> float:
        tier_scores = {
            ModelTier.ECONOMY: 0.4,
            ModelTier.STANDARD: 0.7,
            ModelTier.PREMIUM: 1.0,
        }
        base_score = tier_scores[model.tier]
        recommended = complexity.recommended_tier
        if model.tier == recommended:
            return base_score
        if model.tier.rank > recommended.rank:
            return min(1.0, base_score + 0.1)
        return max(0.0, base_score - 0.2)

    def _generate_reason(self, model: ModelConfig, complexity: ComplexityEstimate) -> str:
        return (
            f"Selected {model.name} ({model.tier.value}) with complexity "
            f"{complexity.score:.2f}"
        )


def _normalize(value: float, minimum: float, maximum: float) -> float:
    span = maximum - minimum
    if span <= 0:
        return 0.0
    return (value - minimum) / span
