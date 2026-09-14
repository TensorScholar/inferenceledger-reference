from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from math import isclose
from uuid import UUID, uuid4

from ...utils.time import utc_now
from ..cost.pricing import PricingQuote


class ModelTier(StrEnum):
    ECONOMY = "economy"
    STANDARD = "standard"
    PREMIUM = "premium"

    @property
    def rank(self) -> int:
        return {
            ModelTier.ECONOMY: 1,
            ModelTier.STANDARD: 2,
            ModelTier.PREMIUM: 3,
        }[self]


class RoutingStrategy(StrEnum):
    SINGLE_MODEL = "single_model"
    RULE_BASED = "rule_based"
    POLICY = "policy"
    COST_OPTIMAL = "cost_optimal"
    LATENCY_OPTIMAL = "latency_optimal"
    QUALITY_OPTIMAL = "quality_optimal"
    BALANCED = "balanced"
    ROUND_ROBIN = "round_robin"


class RoutingReason(StrEnum):
    POLICY_PREFERRED_MODEL = "policy.preferred_model"
    POLICY_COST_WITHIN_BUDGET = "policy.cost_within_budget"
    POLICY_LATENCY_WITHIN_SLO = "policy.latency_within_slo"
    POLICY_QUALITY_FLOOR = "policy.quality_floor"
    POLICY_BALANCED_SCORE = "policy.balanced_score"
    POLICY_NO_CANDIDATE_WITHIN_BUDGET = "policy.no_candidate_within_budget"
    POLICY_NO_CANDIDATE_WITHIN_LATENCY_SLO = "policy.no_candidate_within_latency_slo"
    POLICY_NO_CANDIDATE_MEETS_QUALITY_FLOOR = "policy.no_candidate_meets_quality_floor"


@dataclass(frozen=True)
class ModelConfig:
    """Operational and capability metadata used by routers.

    Tariff data deliberately does not live here. Routing cost estimates must come from an injected
    pricing-backed estimator so model metadata cannot drift from execution accounting.
    """

    id: str
    name: str
    tier: ModelTier
    max_context_length: int
    supports_streaming: bool = True
    supports_batching: bool = True
    avg_latency_ms: int = 500
    max_throughput_rps: int = 100
    tokens_per_second: int = 50
    max_replicas: int = 3
    current_load: float = 0.0
    healthy: bool = True
    circuit_breaker_open: bool = False

    @property
    def is_available(self) -> bool:
        return self.healthy and not self.circuit_breaker_open and self.current_load < 0.95


@dataclass(frozen=True)
class ComplexityEstimate:
    score: float
    factors: dict[str, float]
    input_length: int
    estimated_reasoning_steps: int
    requires_context: bool
    domain_specific: bool

    @property
    def recommended_tier(self) -> ModelTier:
        if self.score > 0.7:
            return ModelTier.PREMIUM
        if self.score > 0.3:
            return ModelTier.STANDARD
        return ModelTier.ECONOMY


@dataclass(frozen=True)
class RoutingDecision:
    request_id: UUID
    selected_model: ModelConfig
    strategy: RoutingStrategy
    complexity_estimate: ComplexityEstimate | None
    estimated_cost: float
    estimated_latency_ms: int
    estimated_quality_score: float
    decision_reason: str
    cost_quote: PricingQuote | None = None
    fallback_models: list[ModelConfig] = field(default_factory=list)
    considered_models: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=utc_now)
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if self.estimated_cost < 0:
            raise ValueError("estimated_cost must be non-negative")
        if self.cost_quote is None:
            return
        if self.cost_quote.model != self.selected_model.id:
            raise ValueError("routing cost quote model must match selected model")
        if not isclose(
            self.estimated_cost,
            self.cost_quote.amount_usd,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("routing estimated_cost must equal cost quote amount")
