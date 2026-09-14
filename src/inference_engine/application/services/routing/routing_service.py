import structlog

from ....domain.models.request import InferenceRequest
from ....domain.models.routing import RoutingDecision
from ....domain.routing.base import AbstractRouter

logger = structlog.get_logger()


class RoutingService:
    """Service for intelligent model routing."""

    def __init__(self, routers: dict[str, AbstractRouter]) -> None:
        self.routers = routers

    async def route(self, request: InferenceRequest) -> RoutingDecision:
        """
        Route request to optimal model.

        Args:
            request: Inference request

        Returns:
            Routing decision with selected model
        """
        # Select router based on configured strategy
        strategy = "cost_optimal"  # Would come from config
        router = self.routers.get(strategy)

        if not router:
            raise ValueError(f"Unknown routing strategy: {strategy}")

        decision = await router.route(request)

        logger.info(
            "routing_completed",
            request_id=str(request.id),
            model=decision.selected_model.id,
            strategy=strategy,
        )

        return decision
