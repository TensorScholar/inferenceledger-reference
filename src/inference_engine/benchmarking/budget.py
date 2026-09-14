from __future__ import annotations

from dataclasses import dataclass

from ..domain.models.routing import RoutingDecision


@dataclass(frozen=True)
class BudgetViolation(Exception):
    """Raised when a route decision exceeds a configured estimated cost budget."""

    request_id: str
    selected_model: str
    estimated_cost_usd: float
    max_estimated_cost_usd: float

    @property
    def message(self) -> str:
        return (
            f"estimated route cost ${self.estimated_cost_usd:.8f} exceeds budget "
            f"${self.max_estimated_cost_usd:.8f}"
        )

    def __str__(self) -> str:
        return self.message


def enforce_estimated_cost_budget(
    decision: RoutingDecision,
    max_estimated_cost_usd: float | None,
) -> None:
    if max_estimated_cost_usd is None:
        return
    if max_estimated_cost_usd < 0:
        raise ValueError("max_estimated_cost_usd must be non-negative")
    if decision.estimated_cost > max_estimated_cost_usd:
        raise BudgetViolation(
            request_id=str(decision.request_id),
            selected_model=decision.selected_model.id,
            estimated_cost_usd=decision.estimated_cost,
            max_estimated_cost_usd=max_estimated_cost_usd,
        )
