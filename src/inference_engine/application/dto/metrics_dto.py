from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricsSummaryDTO:
    total_requests: int
    total_cost_usd: float
    avg_latency_ms: float
    cache_hit_rate: float

