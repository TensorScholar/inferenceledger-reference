from __future__ import annotations

from uuid import uuid4

import pytest

from inference_engine.benchmarking.budget import BudgetViolation, enforce_estimated_cost_budget
from inference_engine.domain.models.routing import (
    ModelConfig,
    ModelTier,
    RoutingDecision,
    RoutingStrategy,
)


def _decision(cost: float) -> RoutingDecision:
    model = ModelConfig(
        id="test-model",
        name="Test Model",
        tier=ModelTier.STANDARD,
        max_context_length=4096,
    )
    return RoutingDecision(
        request_id=uuid4(),
        selected_model=model,
        strategy=RoutingStrategy.SINGLE_MODEL,
        complexity_estimate=None,
        estimated_cost=cost,
        estimated_latency_ms=250,
        estimated_quality_score=0.7,
        decision_reason="test route",
        fallback_models=[],
        considered_models=["test-model"],
    )


def test_budget_allows_route_under_limit() -> None:
    enforce_estimated_cost_budget(_decision(0.001), 0.002)


def test_budget_rejects_route_over_limit() -> None:
    with pytest.raises(BudgetViolation) as exc_info:
        enforce_estimated_cost_budget(_decision(0.003), 0.002)

    assert exc_info.value.estimated_cost_usd == 0.003
    assert exc_info.value.max_estimated_cost_usd == 0.002


def test_budget_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        enforce_estimated_cost_budget(_decision(0.001), -1.0)
