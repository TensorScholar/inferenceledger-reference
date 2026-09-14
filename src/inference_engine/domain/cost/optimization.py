import structlog

from ..models.cost import CostMetrics

logger = structlog.get_logger()


class CostOptimizer:
    """
    Analyzes costs and provides optimization recommendations.

    Identifies opportunities for cost reduction through data-driven analysis.
    """

    def __init__(self) -> None:
        pass

    def analyze_trends(self, metrics_list: list[CostMetrics]) -> dict[str, float]:
        """
        Analyze cost trends over time.

        Returns:
            Dictionary with trend statistics
        """
        if not metrics_list:
            return {}

        total_costs = [m.total_cost_usd for m in metrics_list]
        total_savings = [m.total_savings_usd for m in metrics_list]

        return {
            "avg_daily_cost": sum(total_costs) / len(total_costs),
            "avg_daily_savings": sum(total_savings) / len(total_savings),
            "avg_savings_rate": sum(m.savings_rate for m in metrics_list) / len(metrics_list),
            "total_cost": sum(total_costs),
            "total_savings": sum(total_savings),
        }

    def get_top_cost_drivers(self, metrics: CostMetrics, limit: int = 10) -> list[dict[str, object]]:
        """
        Identify top cost drivers.

        Returns:
            List of cost driver dictionaries
        """
        drivers: list[dict[str, object]] = []

        # Top users
        for user, cost in sorted(metrics.cost_by_user.items(), key=lambda x: x[1], reverse=True)[:limit]:
            drivers.append({"type": "user", "id": user, "cost": cost})

        # Top features
        for feature, cost in sorted(metrics.cost_by_feature.items(), key=lambda x: x[1], reverse=True)[:limit]:
            drivers.append({"type": "feature", "id": feature, "cost": cost})

        # Top models
        for model, cost in sorted(metrics.cost_by_model.items(), key=lambda x: x[1], reverse=True)[:limit]:
            drivers.append({"type": "model", "id": model, "cost": cost})

        def cost_value(item: dict[str, object]) -> float:
            value = item["cost"]
            if isinstance(value, int | float):
                return float(value)
            raise TypeError(f"Cost driver has non-numeric cost: {value!r}")

        return sorted(drivers, key=cost_value, reverse=True)[:limit]

    def recommend_optimizations(self, metrics: CostMetrics) -> list[str]:
        """
        Generate optimization recommendations based on cost analysis.

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        if metrics.cache_hit_rate < 0.4:
            recommendations.append("Cache hit rate is low. Consider enabling semantic caching or reviewing prompts.")

        if len(metrics.cost_by_user) > 0:
            top_user_cost = max(metrics.cost_by_user.values())
            avg_user_cost = metrics.total_cost_usd / len(metrics.cost_by_user)
            if top_user_cost > avg_user_cost * 5:
                recommendations.append("High variance in user costs detected. Implement per-user throttling.")

        if metrics.cost_by_model:
            most_expensive_model = max(metrics.cost_by_model.items(), key=lambda x: x[1])[0]
            recommendations.append(f"Consider routing more requests away from {most_expensive_model} to cheaper models.")

        logger.info(
            "optimization_recommendations",
            count=len(recommendations),
            recommendations=recommendations,
        )

        return recommendations
