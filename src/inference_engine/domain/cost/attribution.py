
import structlog

from ..models.cost import CostAttribution, CostBreakdown

logger = structlog.get_logger()


class CostAttributor:
    """
    Attributes costs to different dimensions (user, feature, experiment).

    Enables cost analysis and optimization at granular levels.
    """

    def __init__(self) -> None:
        self.attributions: dict[str, list[CostAttribution]] = {}

    def attribute(
        self,
        request_id: str,
        user_id: str | None,
        feature_name: str | None,
        experiment_id: str | None,
        application: str,
        cost_breakdown: CostBreakdown,
        input_tokens: int,
        output_tokens: int,
        cache_hits: int,
        latency_ms: int,
    ) -> CostAttribution:
        """
        Create cost attribution record.

        Returns:
            CostAttribution object with all attribution information
        """
        from uuid import UUID
        attribution = CostAttribution(
            request_id=UUID(request_id) if isinstance(request_id, str) else request_id,
            user_id=user_id,
            feature_name=feature_name,
            experiment_id=experiment_id,
            application=application,
            cost_breakdown=cost_breakdown,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hits=cache_hits,
            latency_ms=latency_ms,
        )

        # Store for later analysis
        if user_id:
            self.attributions.setdefault(user_id, []).append(attribution)

        logger.info(
            "cost_attributed",
            request_id=request_id,
            user_id=user_id,
            feature_name=feature_name,
            net_cost=cost_breakdown.net_cost,
        )

        return attribution

    def get_user_costs(self, user_id: str) -> float:
        """Get total costs for a user."""
        return sum(a.cost_breakdown.net_cost for a in self.attributions.get(user_id, []))

    def get_feature_costs(self, feature_name: str) -> list[CostAttribution]:
        """Get all attributions for a feature."""
        return [
            a for attributions in self.attributions.values() for a in attributions if a.feature_name == feature_name
        ]

