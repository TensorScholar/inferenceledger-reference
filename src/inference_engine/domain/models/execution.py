from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from math import isfinite


class AttemptOutcome(StrEnum):
    """Observed outcome of one provider invocation."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class CostEvidenceKind(StrEnum):
    """How the monetary amount for one provider attempt was obtained."""

    CALCULATED_FROM_USAGE = "calculated_from_usage"
    REPORTED_BY_EXECUTION_STACK = "reported_by_execution_stack"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderAttempt:
    """One actual provider invocation in an inference execution chain.

    A retry is a new attempt. A future fallback to another model/provider must also be represented
    as another attempt rather than being collapsed into the final request outcome.

    Cost origin is explicit. ``calculated_cost_usd`` is reserved for amounts reconstructed from
    usage using InferenceLedger pricing evidence. ``reported_cost_usd`` is reserved for amounts
    reported by an external execution stack and must carry source provenance instead of pricing
    provenance. The two evidence paths are intentionally mutually exclusive.
    """

    attempt_index: int
    provider: str
    model: str
    outcome: AttemptOutcome
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    calculated_cost_usd: float | None = None
    reported_cost_usd: float | None = None
    cost_evidence: CostEvidenceKind = CostEvidenceKind.UNKNOWN
    pricing_table_version: str | None = None
    pricing_record_id: str | None = None
    pricing_observed_at: str | None = None
    pricing_source_url: str | None = None
    cost_source: str | None = None
    cost_source_record_id: str | None = None
    error_type: str | None = None
    status_code: int | None = None

    def __post_init__(self) -> None:
        if self.attempt_index < 1:
            raise ValueError("attempt_index must be at least 1")
        if not self.provider.strip():
            raise ValueError("provider must be non-empty")
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")

        token_values = (
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
            self.cached_tokens,
        )
        if any(value is not None and value < 0 for value in token_values):
            raise ValueError("token counts must be non-negative when present")
        if self.calculated_cost_usd is not None and not isfinite(self.calculated_cost_usd):
            raise ValueError("calculated_cost_usd must be finite when present")
        if self.calculated_cost_usd is not None and self.calculated_cost_usd < 0:
            raise ValueError("calculated_cost_usd must be non-negative when present")
        if self.reported_cost_usd is not None and not isfinite(self.reported_cost_usd):
            raise ValueError("reported_cost_usd must be finite when present")
        if self.reported_cost_usd is not None and self.reported_cost_usd < 0:
            raise ValueError("reported_cost_usd must be non-negative when present")
        if self.calculated_cost_usd is not None and self.reported_cost_usd is not None:
            raise ValueError("one provider attempt cannot carry both calculated and reported cost")

        pricing_provenance = (
            self.pricing_table_version,
            self.pricing_record_id,
            self.pricing_observed_at,
            self.pricing_source_url,
        )
        external_provenance = (self.cost_source, self.cost_source_record_id)

        if self.cost_evidence == CostEvidenceKind.CALCULATED_FROM_USAGE:
            if self.calculated_cost_usd is None:
                raise ValueError("calculated cost evidence requires calculated_cost_usd")
            if self.reported_cost_usd is not None:
                raise ValueError("calculated cost evidence must not carry reported_cost_usd")
            if any(value is None or not value.strip() for value in pricing_provenance):
                raise ValueError("calculated cost evidence requires complete pricing provenance")
            if any(value is not None for value in external_provenance):
                raise ValueError("calculated cost evidence must not carry external source provenance")
            assert self.pricing_observed_at is not None
            assert self.pricing_record_id is not None
            try:
                date.fromisoformat(self.pricing_observed_at)
            except ValueError as exc:
                raise ValueError("pricing_observed_at must be an ISO date") from exc
            expected_record_id = f"{self.provider}:{self.model}:{self.pricing_observed_at}"
            if self.pricing_record_id != expected_record_id:
                raise ValueError(
                    "pricing_record_id must bind the observed provider, model, and pricing date"
                )
        elif self.cost_evidence == CostEvidenceKind.REPORTED_BY_EXECUTION_STACK:
            if self.reported_cost_usd is None:
                raise ValueError("externally reported cost evidence requires reported_cost_usd")
            if self.calculated_cost_usd is not None:
                raise ValueError("externally reported cost evidence must not carry calculated cost")
            if any(value is not None for value in pricing_provenance):
                raise ValueError("externally reported cost evidence must not carry pricing provenance")
            if any(value is None or not value.strip() for value in external_provenance):
                raise ValueError(
                    "externally reported cost evidence requires source and source record id"
                )
        else:
            if self.calculated_cost_usd is not None or self.reported_cost_usd is not None:
                raise ValueError("unknown cost evidence must not carry a monetary amount")
            if any(value is not None for value in pricing_provenance):
                raise ValueError("unknown cost evidence must not carry pricing provenance")
            if any(value is not None for value in external_provenance):
                raise ValueError("unknown cost evidence must not carry external source provenance")

    @property
    def cost_usd(self) -> float | None:
        """Return the known amount without erasing how that amount was obtained."""
        if self.cost_evidence == CostEvidenceKind.CALCULATED_FROM_USAGE:
            return self.calculated_cost_usd
        if self.cost_evidence == CostEvidenceKind.REPORTED_BY_EXECUTION_STACK:
            return self.reported_cost_usd
        return None

    @property
    def cost_is_known(self) -> bool:
        return self.cost_usd is not None
