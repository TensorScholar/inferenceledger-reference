import structlog

from .pricing import PricingQuote, PricingTable

logger = structlog.get_logger()


class CostCalculator:
    """Provider-aware monetary calculation backed by the canonical pricing table."""

    def __init__(self, pricing_table: PricingTable | None = None) -> None:
        self.pricing_table = pricing_table or PricingTable()

    def quote_provider_usage(
        self,
        *,
        provider: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> PricingQuote:
        """Calculate cost and preserve the exact provider pricing provenance."""
        quote = self.pricing_table.quote(
            provider=provider,
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
        )

        logger.debug(
            "cost_calculated_from_provider_usage",
            provider=provider,
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            pricing_table_version=quote.pricing_table_version,
            pricing_record_id=quote.pricing_record_id,
            total_cost=quote.amount_usd,
        )
        return quote
