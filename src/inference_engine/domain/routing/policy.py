from __future__ import annotations

from dataclasses import dataclass

from ..cost.pricing import PricingQuote
from ..models.request import InferenceRequest
from ..models.routing import (
    ComplexityEstimate,
    ModelConfig,
    ModelTier,
    RoutingDecision,
    RoutingReason,
    RoutingStrategy,
)
from .base import AbstractRouter
from .capability import supports_request_context
from .complexity import ComplexityEstimator
from .cost_estimator import RoutingCostEstimator


@dataclass(frozen=True)
class PolicyRouterConfig:
    """Deterministic policy knobs used by benchmark routing experiments."""

    max_estimated_cost_usd: float | None = None
    latency_slo_ms: int | None = None
    min_quality_score: float | None = None
    cost_weight: float = 0.55
    latency_weight: float = 0.25
    quality_weight: float = 0.20

    def __post_init__(self) -> None:
        if self.max_estimated_cost_usd is not None and self.max_estimated_cost_usd < 0:
            raise ValueError("max_estimated_cost_usd must be non-negative")
        if self.latency_slo_ms is not None and self.latency_slo_ms < 1:
            raise ValueError("latency_slo_ms must be positive")
        if self.min_quality_score is not None and not 0 <= self.min_quality_score <= 1:
            raise ValueError("min_quality_score must be between 0 and 1")
        if min(self.cost_weight, self.latency_weight, self.quality_weight) < 0:
            raise ValueError("policy weights must be non-negative")
        if self.cost_weight + self.latency_weight + self.quality_weight <= 0:
            raise ValueError("at least one policy weight must be positive")


@dataclass(frozen=True)
class _Candidate:
    model: ModelConfig
    cost_quote: PricingQuote
    estimated_latency_ms: int
    estimated_quality_score: float

    @property
    def estimated_cost(self) -> float:
        return self.cost_quote.amount_usd


class PolicyRouter(AbstractRouter):
    """SLO-aware deterministic router with auditable reason codes and cost quotes."""

    def __init__(
        self,
        models: list[ModelConfig],
        complexity_estimator: ComplexityEstimator,
        cost_estimator: RoutingCostEstimator,
        config: PolicyRouterConfig | None = None,
    ) -> None:
        self.models = {model.id: model for model in models}
        self.complexity_estimator = complexity_estimator
        self.cost_estimator = cost_estimator
        self.config = config or PolicyRouterConfig()

    async def route(self, request: InferenceRequest) -> RoutingDecision:
        complexity = await self.complexity_estimator.estimate(request)
        candidates = self._candidates(request, complexity)
        if not candidates:
            raise RuntimeError("No available model can satisfy request context length")

        preferred = self._preferred_candidate(request, candidates)
        if preferred is not None and self._meets_constraints(preferred):
            return self._decision(
                request=request,
                selected=preferred,
                candidates=candidates,
                complexity=complexity,
                reason=RoutingReason.POLICY_PREFERRED_MODEL,
            )

        constrained, reason = self._apply_constraints(candidates)
        selected = self._select_best(constrained)
        return self._decision(
            request=request,
            selected=selected,
            candidates=candidates,
            complexity=complexity,
            reason=reason,
        )

    def _candidates(
        self, request: InferenceRequest, complexity: ComplexityEstimate
    ) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for model in self.models.values():
            if not model.is_available or not supports_request_context(model, request):
                continue
            candidates.append(
                _Candidate(
                    model=model,
                    cost_quote=self.cost_estimator.quote(
                        model_id=model.id,
                        input_tokens=request.estimated_input_tokens,
                        output_tokens=request.parameters.max_tokens,
                    ),
                    estimated_latency_ms=model.avg_latency_ms,
                    estimated_quality_score=_estimate_quality(model.tier, complexity),
                )
            )
        return sorted(candidates, key=lambda candidate: candidate.model.id)

    def _preferred_candidate(
        self, request: InferenceRequest, candidates: list[_Candidate]
    ) -> _Candidate | None:
        if request.preferred_model is None:
            return None
        return next(
            (candidate for candidate in candidates if candidate.model.id == request.preferred_model),
            None,
        )

    def _meets_constraints(self, candidate: _Candidate) -> bool:
        if (
            self.config.max_estimated_cost_usd is not None
            and candidate.estimated_cost > self.config.max_estimated_cost_usd
        ):
            return False
        if (
            self.config.latency_slo_ms is not None
            and candidate.estimated_latency_ms > self.config.latency_slo_ms
        ):
            return False
        return not (
            self.config.min_quality_score is not None
            and candidate.estimated_quality_score < self.config.min_quality_score
        )

    def _apply_constraints(
        self, candidates: list[_Candidate]
    ) -> tuple[list[_Candidate], RoutingReason]:
        remaining = candidates
        reason = RoutingReason.POLICY_BALANCED_SCORE

        if self.config.min_quality_score is not None:
            quality_candidates = [
                candidate
                for candidate in remaining
                if candidate.estimated_quality_score >= self.config.min_quality_score
            ]
            if quality_candidates:
                remaining = quality_candidates
                reason = RoutingReason.POLICY_QUALITY_FLOOR
            else:
                return remaining, RoutingReason.POLICY_NO_CANDIDATE_MEETS_QUALITY_FLOOR

        if self.config.latency_slo_ms is not None:
            latency_candidates = [
                candidate
                for candidate in remaining
                if candidate.estimated_latency_ms <= self.config.latency_slo_ms
            ]
            if latency_candidates:
                remaining = latency_candidates
                reason = RoutingReason.POLICY_LATENCY_WITHIN_SLO
            else:
                return remaining, RoutingReason.POLICY_NO_CANDIDATE_WITHIN_LATENCY_SLO

        if self.config.max_estimated_cost_usd is not None:
            budget_candidates = [
                candidate
                for candidate in remaining
                if candidate.estimated_cost <= self.config.max_estimated_cost_usd
            ]
            if budget_candidates:
                remaining = budget_candidates
                reason = RoutingReason.POLICY_COST_WITHIN_BUDGET
            else:
                return remaining, RoutingReason.POLICY_NO_CANDIDATE_WITHIN_BUDGET

        return remaining, reason

    def _select_best(self, candidates: list[_Candidate]) -> _Candidate:
        min_cost = min(candidate.estimated_cost for candidate in candidates)
        max_cost = max(candidate.estimated_cost for candidate in candidates)
        min_latency = min(candidate.estimated_latency_ms for candidate in candidates)
        max_latency = max(candidate.estimated_latency_ms for candidate in candidates)

        def score(candidate: _Candidate) -> tuple[float, float, int, str]:
            normalized_cost = _normalize(candidate.estimated_cost, min_cost, max_cost)
            normalized_latency = _normalize(
                candidate.estimated_latency_ms,
                min_latency,
                max_latency,
            )
            quality_penalty = 1 - candidate.estimated_quality_score
            weighted_score = (
                self.config.cost_weight * normalized_cost
                + self.config.latency_weight * normalized_latency
                + self.config.quality_weight * quality_penalty
                + candidate.model.current_load * 0.1
            )
            return (
                weighted_score,
                candidate.estimated_cost,
                candidate.estimated_latency_ms,
                candidate.model.id,
            )

        return min(candidates, key=score)

    def _decision(
        self,
        *,
        request: InferenceRequest,
        selected: _Candidate,
        candidates: list[_Candidate],
        complexity: ComplexityEstimate,
        reason: RoutingReason,
    ) -> RoutingDecision:
        fallback_models = [
            candidate.model
            for candidate in sorted(
                candidates,
                key=lambda candidate: (
                    candidate.estimated_cost,
                    candidate.estimated_latency_ms,
                    candidate.model.id,
                ),
            )
            if candidate.model.id != selected.model.id
        ][:3]
        return RoutingDecision(
            request_id=request.id,
            selected_model=selected.model,
            strategy=RoutingStrategy.POLICY,
            complexity_estimate=complexity,
            estimated_cost=selected.estimated_cost,
            estimated_latency_ms=selected.estimated_latency_ms,
            estimated_quality_score=selected.estimated_quality_score,
            decision_reason=reason.value,
            cost_quote=selected.cost_quote,
            fallback_models=fallback_models,
            considered_models=[candidate.model.id for candidate in candidates],
        )


def _estimate_quality(tier: ModelTier, complexity: ComplexityEstimate) -> float:
    base = {
        ModelTier.ECONOMY: 0.45,
        ModelTier.STANDARD: 0.72,
        ModelTier.PREMIUM: 0.92,
    }[tier]
    if tier.rank < complexity.recommended_tier.rank:
        return max(0.0, base - 0.22)
    if tier.rank > complexity.recommended_tier.rank:
        return min(1.0, base + 0.05)
    return base


def _normalize(value: float, minimum: float, maximum: float) -> float:
    span = maximum - minimum
    if span <= 0:
        return 0.0
    return (value - minimum) / span
