"""Attempt-aware inference cost model (reference).

Rules:
- Unknown/partial monetary evidence is never rewritten as 0.
- Retries are first-class attempts, not hidden in a single request row.
- Aggregates expose known cost AND unknown attempt counts separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Attempt:
    attempt_id: str
    provider: str
    model: str
    success: bool
    cost_usd: Optional[float]
    latency_ms: Optional[float]
    is_retry: bool = False
    error: Optional[str] = None


@dataclass
class Aggregate:
    known_cost_usd: float
    unknown_attempts: int
    success_count: int
    total_attempts: int
    retry_count: int
    providers: dict[str, float] = field(default_factory=dict)


def aggregate_attempts(attempts: list[Attempt]) -> Aggregate:
    known = 0.0
    unknown = 0
    successes = 0
    retries = 0
    by_provider: dict[str, float] = {}
    for a in attempts:
        if a.cost_usd is None:
            unknown += 1
        else:
            known += a.cost_usd
            by_provider[a.provider] = by_provider.get(a.provider, 0.0) + a.cost_usd
        if a.success:
            successes += 1
        if a.is_retry:
            retries += 1
    return Aggregate(
        known_cost_usd=round(known, 6),
        unknown_attempts=unknown,
        success_count=successes,
        total_attempts=len(attempts),
        retry_count=retries,
        providers=by_provider,
    )


def success_rate(agg: Aggregate) -> float:
    if agg.total_attempts == 0:
        return 0.0
    return agg.success_count / agg.total_attempts


def non_compensatory_gate(
    *,
    baseline_success_rate: float,
    candidate_success_rate: float,
    min_success_rate: float,
    baseline_known_cost: float,
    candidate_known_cost: float,
    candidate_unknown_attempts: int,
    baseline_p95_latency_ms: float | None = None,
    candidate_p95_latency_ms: float | None = None,
    max_latency_regression_ms: float = 200.0,
) -> str:
    """SHIP | REVIEW | NO-GO. Cost cannot offset a required reliability miss."""
    if candidate_success_rate < min_success_rate:
        return "NO-GO"
    if candidate_success_rate < baseline_success_rate:
        return "NO-GO"
    if (
        baseline_p95_latency_ms is not None
        and candidate_p95_latency_ms is not None
        and (candidate_p95_latency_ms - baseline_p95_latency_ms) > max_latency_regression_ms
    ):
        return "NO-GO"
    if candidate_unknown_attempts > 0:
        return "REVIEW"
    if candidate_known_cost < baseline_known_cost and candidate_success_rate >= baseline_success_rate:
        return "SHIP"
    return "REVIEW"
