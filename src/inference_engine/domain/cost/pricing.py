from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isclose

PRICING_TABLE_VERSION = "openai-standard-2026-09-03"
PRICING_OBSERVED_AT = date(2026, 9, 3)


@dataclass(frozen=True)
class ModelPricing:
    """One observed provider/model pricing record in USD per one million tokens."""

    provider: str
    model: str
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None
    observed_at: date = PRICING_OBSERVED_AT
    source_url: str = "https://developers.openai.com/api/docs/pricing"

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("pricing provider must be non-empty")
        if not self.model.strip():
            raise ValueError("pricing model must be non-empty")
        if min(self.input_per_million, self.output_per_million) < 0:
            raise ValueError("pricing rates must be non-negative")
        if self.cached_input_per_million is not None and self.cached_input_per_million < 0:
            raise ValueError("cached input pricing must be non-negative")
        if not self.source_url.strip():
            raise ValueError("pricing source_url must be non-empty")

    @property
    def record_id(self) -> str:
        return f"{self.provider}:{self.model}:{self.observed_at.isoformat()}"


@dataclass(frozen=True)
class PricingQuote:
    """Self-contained, reconstructable cost quote with immutable pricing assumptions."""

    amount_usd: float
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None
    pricing_record_id: str
    pricing_table_version: str
    pricing_observed_at: date
    pricing_source_url: str

    def __post_init__(self) -> None:
        if self.amount_usd < 0:
            raise ValueError("pricing quote amount_usd must be non-negative")
        if min(self.input_tokens, self.output_tokens, self.cached_input_tokens) < 0:
            raise ValueError("pricing quote token counts must be non-negative")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("pricing quote cached_input_tokens cannot exceed input_tokens")
        if min(self.input_per_million, self.output_per_million) < 0:
            raise ValueError("pricing quote rates must be non-negative")
        if self.cached_input_per_million is not None and self.cached_input_per_million < 0:
            raise ValueError("pricing quote cached input rate must be non-negative")
        string_fields = (
            self.provider,
            self.model,
            self.pricing_record_id,
            self.pricing_table_version,
            self.pricing_source_url,
        )
        if any(not value.strip() for value in string_fields):
            raise ValueError("pricing quote provenance fields must be non-empty")
        expected_record_id = (
            f"{self.provider}:{self.model}:{self.pricing_observed_at.isoformat()}"
        )
        if self.pricing_record_id != expected_record_id:
            raise ValueError("pricing_record_id must bind provider, model, and observation date")
        expected_amount = self.reconstructed_amount_usd
        if not isclose(self.amount_usd, expected_amount, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("pricing quote amount must match token and rate assumptions")

    @property
    def reconstructed_amount_usd(self) -> float:
        billable_input_tokens = self.input_tokens - self.cached_input_tokens
        input_cost = billable_input_tokens * self.input_per_million / 1_000_000
        cached_rate = (
            self.cached_input_per_million
            if self.cached_input_per_million is not None
            else self.input_per_million
        )
        cached_cost = self.cached_input_tokens * cached_rate / 1_000_000
        output_cost = self.output_tokens * self.output_per_million / 1_000_000
        return input_cost + cached_cost + output_cost


_OPENAI_MODEL_DOC = "https://developers.openai.com/api/docs/models"


def _openai_pricing(
    model: str,
    *,
    input_per_million: float,
    output_per_million: float,
    cached_input_per_million: float | None = None,
) -> ModelPricing:
    return ModelPricing(
        provider="openai",
        model=model,
        input_per_million=input_per_million,
        output_per_million=output_per_million,
        cached_input_per_million=cached_input_per_million,
        source_url=f"{_OPENAI_MODEL_DOC}/{model}",
    )


DEFAULT_PRICING: dict[tuple[str, str], ModelPricing] = {
    ("openai", "gpt-4o-mini"): _openai_pricing(
        "gpt-4o-mini",
        input_per_million=0.15,
        output_per_million=0.60,
        cached_input_per_million=0.075,
    ),
    ("openai", "gpt-4o"): _openai_pricing(
        "gpt-4o",
        input_per_million=2.50,
        output_per_million=10.00,
        cached_input_per_million=1.25,
    ),
    ("openai", "gpt-3.5-turbo"): _openai_pricing(
        "gpt-3.5-turbo",
        input_per_million=0.50,
        output_per_million=1.50,
    ),
    ("openai", "gpt-5.5"): _openai_pricing(
        "gpt-5.5",
        input_per_million=5.00,
        output_per_million=30.00,
        cached_input_per_million=0.50,
    ),
    ("openai", "gpt-5.4"): _openai_pricing(
        "gpt-5.4",
        input_per_million=2.50,
        output_per_million=15.00,
        cached_input_per_million=0.25,
    ),
    ("openai", "gpt-5.4-mini"): _openai_pricing(
        "gpt-5.4-mini",
        input_per_million=0.75,
        output_per_million=4.50,
        cached_input_per_million=0.075,
    ),
    ("openai", "gpt-5.3-codex"): _openai_pricing(
        "gpt-5.3-codex",
        input_per_million=1.75,
        output_per_million=14.00,
        cached_input_per_million=0.175,
    ),
}


class UnknownModelPricingError(ValueError):
    """Raised when cost is requested without a matching pricing assumption."""


class PricingTable:
    """Versioned provider/model pricing authority for calculated execution cost."""

    def __init__(
        self,
        prices: dict[tuple[str, str], ModelPricing] | None = None,
        version: str = PRICING_TABLE_VERSION,
    ) -> None:
        self.prices = DEFAULT_PRICING if prices is None else prices
        self.version = version

    def get(
        self,
        *,
        provider: str,
        model: str,
        as_of: date | None = None,
    ) -> ModelPricing:
        try:
            pricing = self.prices[(provider, model)]
        except KeyError as exc:
            raise UnknownModelPricingError(
                f"Missing pricing for provider/model '{provider}/{model}' in pricing table {self.version}"
            ) from exc

        if as_of is not None and pricing.observed_at > as_of:
            raise UnknownModelPricingError(
                f"Pricing record {pricing.record_id} was observed after requested date {as_of.isoformat()}"
            )
        return pricing

    def quote(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        as_of: date | None = None,
    ) -> PricingQuote:
        if min(input_tokens, output_tokens, cached_input_tokens) < 0:
            raise ValueError("token counts must be non-negative")
        if cached_input_tokens > input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")

        pricing = self.get(provider=provider, model=model, as_of=as_of)
        billable_input_tokens = input_tokens - cached_input_tokens
        input_cost = billable_input_tokens * pricing.input_per_million / 1_000_000
        cached_rate = pricing.cached_input_per_million
        cached_cost = (
            cached_input_tokens * cached_rate / 1_000_000
            if cached_rate is not None
            else cached_input_tokens * pricing.input_per_million / 1_000_000
        )
        output_cost = output_tokens * pricing.output_per_million / 1_000_000
        return PricingQuote(
            amount_usd=input_cost + cached_cost + output_cost,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            input_per_million=pricing.input_per_million,
            output_per_million=pricing.output_per_million,
            cached_input_per_million=pricing.cached_input_per_million,
            pricing_record_id=pricing.record_id,
            pricing_table_version=self.version,
            pricing_observed_at=pricing.observed_at,
            pricing_source_url=pricing.source_url,
        )
