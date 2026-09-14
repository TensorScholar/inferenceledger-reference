from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from ...utils.time import utc_now


@dataclass(frozen=True)
class CostBreakdown:
    """Detailed cost breakdown."""

    inference_cost: float
    compute_cost: float
    cache_savings: float
    optimization_savings: float

    @property
    def total_cost(self) -> float:
        return self.inference_cost + self.compute_cost

    @property
    def net_cost(self) -> float:
        return self.total_cost - self.cache_savings - self.optimization_savings

    @property
    def savings_rate(self) -> float:
        if self.total_cost == 0:
            return 0.0
        return (self.cache_savings + self.optimization_savings) / self.total_cost


@dataclass(frozen=True)
class CostAttribution:
    request_id: UUID
    user_id: str | None = None
    feature_name: str | None = None
    experiment_id: str | None = None
    application: str = "default"
    cost_breakdown: CostBreakdown = field(default_factory=lambda: CostBreakdown(0, 0, 0, 0))
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0
    latency_ms: int = 0
    timestamp: datetime = field(default_factory=utc_now)


@dataclass
class CostMetrics:
    period_start: datetime
    period_end: datetime
    total_requests: int = 0
    total_cost_usd: float = 0.0
    total_savings_usd: float = 0.0
    cost_by_user: dict[str, float] = field(default_factory=dict)
    cost_by_feature: dict[str, float] = field(default_factory=dict)
    cost_by_model: dict[str, float] = field(default_factory=dict)
    avg_cost_per_request: float = 0.0
    avg_cost_per_1k_tokens: float = 0.0
    cache_hit_rate: float = 0.0

    @property
    def savings_rate(self) -> float:
        denominator = self.total_cost_usd + self.total_savings_usd
        if denominator == 0:
            return 0.0
        return self.total_savings_usd / denominator


@dataclass(frozen=True)
class CostAlert:
    severity: str
    title: str
    description: str
    current_cost: float
    expected_cost: float
    variance: float
    affected_users: list[str] = field(default_factory=list)
    affected_features: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=utc_now)
    id: UUID = field(default_factory=uuid4)
