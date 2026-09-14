from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from .execution import CostEvidenceKind

EXACT_MONEY_FORMAT_VERSION = "decimal-string-v1"


def canonical_decimal_text(value: object) -> str:
    """Return one canonical finite non-negative decimal string.

    This function intentionally converts through ``str`` rather than binary arithmetic. When the
    source is already a decimal string, its numeric value is preserved exactly. When an adapter is
    handed a Python float by an upstream library, the canonical string is the exact decimal value
    exposed by that library at the adapter boundary; it is not promoted to provider-invoice truth.
    """
    if value is None or isinstance(value, bool):
        raise ValueError("exact monetary value must be a decimal-compatible scalar")
    try:
        resolved = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("exact monetary value must be a valid decimal") from exc
    if not resolved.is_finite():
        raise ValueError("exact monetary value must be finite")
    if resolved < 0:
        raise ValueError("exact monetary value must be non-negative")
    if resolved == 0:
        return "0"
    return format(resolved.normalize(), "f")


def decimal_from_text(value: str) -> Decimal:
    canonical = canonical_decimal_text(value)
    return Decimal(canonical)


@dataclass(frozen=True)
class ExactRateEvidence:
    """Exact per-million-token tariff evidence for a calculated attempt cost."""

    input_per_million: str
    output_per_million: str
    cached_input_per_million: str | None
    pricing_record_id: str
    pricing_table_version: str
    pricing_observed_at: str
    pricing_source_url: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_per_million",
            canonical_decimal_text(self.input_per_million),
        )
        object.__setattr__(
            self,
            "output_per_million",
            canonical_decimal_text(self.output_per_million),
        )
        if self.cached_input_per_million is not None:
            object.__setattr__(
                self,
                "cached_input_per_million",
                canonical_decimal_text(self.cached_input_per_million),
            )
        string_fields = (
            self.pricing_record_id,
            self.pricing_table_version,
            self.pricing_observed_at,
            self.pricing_source_url,
        )
        if any(not value.strip() for value in string_fields):
            raise ValueError("exact pricing provenance fields must be non-empty")


@dataclass(frozen=True)
class ExactAttemptCost:
    """Authoritative decimal-string monetary evidence for one provider invocation."""

    attempt_index: int
    currency: str
    evidence_kind: CostEvidenceKind
    amount_decimal: str | None
    source: str | None
    source_record_id: str | None
    unknown_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    rate_evidence: ExactRateEvidence | None = None

    def __post_init__(self) -> None:
        if self.attempt_index < 1:
            raise ValueError("exact attempt index must be at least 1")
        currency = self.currency.strip().upper()
        if len(currency) != 3 or not currency.isalpha():
            raise ValueError("exact attempt currency must be a three-letter code")
        object.__setattr__(self, "currency", currency)

        token_values = (self.prompt_tokens, self.completion_tokens, self.cached_tokens)
        if any(value is not None and value < 0 for value in token_values):
            raise ValueError("exact attempt token counts must be non-negative when present")
        if (
            self.cached_tokens is not None
            and self.prompt_tokens is not None
            and self.cached_tokens > self.prompt_tokens
        ):
            raise ValueError("cached tokens cannot exceed prompt tokens")

        if self.evidence_kind == CostEvidenceKind.UNKNOWN:
            if self.amount_decimal is not None:
                raise ValueError("unknown exact cost evidence must not carry an amount")
            if self.rate_evidence is not None:
                raise ValueError("unknown exact cost evidence must not carry rate evidence")
            if self.unknown_reason is None or not self.unknown_reason.strip():
                raise ValueError("unknown exact cost evidence requires an explicit reason")
            return

        if self.amount_decimal is None:
            raise ValueError("known exact cost evidence requires amount_decimal")
        canonical = canonical_decimal_text(self.amount_decimal)
        object.__setattr__(self, "amount_decimal", canonical)
        if self.source is None or not self.source.strip():
            raise ValueError("known exact cost evidence requires a source")
        if self.source_record_id is None or not self.source_record_id.strip():
            raise ValueError("known exact cost evidence requires a source record id")
        if self.unknown_reason is not None:
            raise ValueError("known exact cost evidence must not carry an unknown reason")

        if self.evidence_kind == CostEvidenceKind.CALCULATED_FROM_USAGE:
            if self.rate_evidence is None:
                raise ValueError("calculated exact cost requires exact rate evidence")
            if self.prompt_tokens is None or self.completion_tokens is None:
                raise ValueError("calculated exact cost requires prompt/completion token counts")
            cached = self.cached_tokens or 0
            rate = self.rate_evidence
            cached_rate = (
                decimal_from_text(rate.cached_input_per_million)
                if rate.cached_input_per_million is not None
                else decimal_from_text(rate.input_per_million)
            )
            expected = (
                Decimal(self.prompt_tokens - cached) * decimal_from_text(rate.input_per_million)
                + Decimal(cached) * cached_rate
                + Decimal(self.completion_tokens) * decimal_from_text(rate.output_per_million)
            ) / Decimal(1_000_000)
            if decimal_from_text(canonical) != expected:
                raise ValueError("calculated exact cost must equal exact usage-rate reconstruction")
        elif self.rate_evidence is not None:
            raise ValueError("externally reported exact cost must not carry pricing-rate evidence")

    @property
    def known(self) -> bool:
        return self.amount_decimal is not None


@dataclass(frozen=True)
class ExactRequestCostEvidence:
    """Exact attempt-chain cost evidence for one request."""

    request_id: str
    currency: str
    attempts: tuple[ExactAttemptCost, ...]
    complete: bool
    total_decimal: str | None

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("exact request cost requires request_id")
        currency = self.currency.strip().upper()
        if len(currency) != 3 or not currency.isalpha():
            raise ValueError("exact request currency must be a three-letter code")
        object.__setattr__(self, "currency", currency)
        expected_indexes = tuple(range(1, len(self.attempts) + 1))
        indexes = tuple(attempt.attempt_index for attempt in self.attempts)
        if indexes != expected_indexes:
            raise ValueError("exact attempt indexes must be contiguous and start at 1")
        if any(attempt.currency != currency for attempt in self.attempts):
            raise ValueError("all exact attempt costs must use the request currency")

        all_known = all(attempt.known for attempt in self.attempts)
        if self.complete != all_known:
            raise ValueError("exact request completeness must match attempt cost completeness")
        if self.complete:
            if self.total_decimal is None:
                raise ValueError("complete exact request cost requires a total")
            total = sum(
                (decimal_from_text(attempt.amount_decimal or "0") for attempt in self.attempts),
                Decimal(0),
            )
            canonical_total = canonical_decimal_text(self.total_decimal)
            if decimal_from_text(canonical_total) != total:
                raise ValueError("exact request total must equal the exact attempt sum")
            object.__setattr__(self, "total_decimal", canonical_total)
        elif self.total_decimal is not None:
            raise ValueError("incomplete exact request cost must not expose a numeric total")

    @classmethod
    def from_attempts(
        cls,
        *,
        request_id: str,
        currency: str,
        attempts: tuple[ExactAttemptCost, ...],
    ) -> ExactRequestCostEvidence:
        complete = all(attempt.known for attempt in attempts)
        total = None
        if complete:
            total_decimal = sum(
                (decimal_from_text(attempt.amount_decimal or "0") for attempt in attempts),
                Decimal(0),
            )
            total = canonical_decimal_text(total_decimal)
        return cls(
            request_id=request_id,
            currency=currency,
            attempts=attempts,
            complete=complete,
            total_decimal=total,
        )
