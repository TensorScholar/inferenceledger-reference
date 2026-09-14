from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from math import isclose
from pathlib import Path
from typing import Any
from uuid import UUID

from ...domain.cost.pricing import PRICING_TABLE_VERSION, PricingQuote
from ...domain.models.execution import (
    AttemptOutcome,
    CostEvidenceKind,
    ProviderAttempt,
)
from ...domain.models.response import InferenceResponse
from ...domain.models.routing import RoutingDecision
from ...infrastructure.models.errors import ProviderError
from ...utils.time import utc_now


@dataclass(frozen=True)
class RequestTrace:
    """One append-only request execution record.

    `estimated_cost_usd` is `None` whenever the full executed attempt chain does not have complete
    cost evidence. In particular, a failed or retried provider call must not silently become a
    zero-cost execution. A locally rejected request with zero provider attempts may, however,
    carry complete zero-cost evidence.

    `pricing_table_version` is a request-level compatibility/summary field. Canonical monetary
    provenance lives on each `ProviderAttempt`; request-level values such as ``unpriced``,
    ``not_charged``, ``external_reported:<source>``, or ``mixed`` must not be interpreted as a
    pricing-record identifier.
    """

    request_id: str
    provider: str
    model: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None
    pricing_table_version: str
    cache_hit: bool | None
    error_type: str | None
    error_message: str | None
    timestamp: str
    quality_passed: bool | None = None
    quality_score: float | None = None
    quality_reason: str | None = None
    eval_type: str | None = None
    provider_attempt_count: int = 1
    provider_retry_count: int = 0
    cost_evidence_complete: bool = True
    provider_attempts: tuple[ProviderAttempt, ...] = ()

    def __post_init__(self) -> None:
        if self.provider_attempt_count < 0 or self.provider_retry_count < 0:
            raise ValueError("provider attempt counts must be non-negative")

        if self.provider_attempts:
            attempt_count = len(self.provider_attempts)
            retry_count = max(attempt_count - 1, 0)
            object.__setattr__(self, "provider_attempt_count", attempt_count)
            object.__setattr__(self, "provider_retry_count", retry_count)

            all_attempt_costs_known = all(attempt.cost_is_known for attempt in self.provider_attempts)
            if all_attempt_costs_known:
                expected_cost = sum(attempt.cost_usd or 0.0 for attempt in self.provider_attempts)
                if not self.cost_evidence_complete or self.estimated_cost_usd is None:
                    raise ValueError(
                        "complete provider-attempt cost evidence requires a total execution cost"
                    )
                if not isclose(self.estimated_cost_usd, expected_cost, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(
                        "request execution cost must equal the sum of known provider-attempt costs"
                    )
            elif self.cost_evidence_complete or self.estimated_cost_usd is not None:
                raise ValueError(
                    "unknown provider-attempt cost requires incomplete request cost evidence"
                )
        else:
            ambiguous_legacy_execution = self.provider_attempt_count > 0 and (
                self.error_type is not None
                or self.provider_retry_count > 0
                or self.provider_attempt_count > 1
            )
            if ambiguous_legacy_execution and self.cost_evidence_complete:
                raise ValueError(
                    "retry/failure trace without provider-attempt evidence cannot claim complete cost evidence"
                )

        if self.provider_attempt_count == 0 and self.provider_retry_count != 0:
            raise ValueError("zero provider attempts cannot contain provider retries")
        if self.provider_attempt_count == 0 and self.provider_attempts:
            raise ValueError("zero provider attempts cannot contain provider-attempt evidence")

        if self.cost_evidence_complete and self.estimated_cost_usd is None:
            raise ValueError("complete cost evidence requires an execution cost")
        if not self.cost_evidence_complete and self.estimated_cost_usd is not None:
            raise ValueError("incomplete cost evidence must not expose a total execution cost")

    @classmethod
    def from_response(
        cls,
        *,
        provider: str,
        response: InferenceResponse,
        pricing_table_version: str | None = None,
    ) -> RequestTrace:
        attempts = response.provider_attempts
        execution_cost, cost_complete = _execution_cost_from_attempts(
            attempts,
            fallback_final_cost=response.usage.cost_usd,
            legacy_attempt_count=response.provider_attempt_count,
        )
        if pricing_table_version is not None:
            request_pricing_context = pricing_table_version
        elif response.provider_attempt_count == 0:
            request_pricing_context = "not_charged"
        else:
            request_pricing_context = _request_pricing_context(attempts)
        return cls(
            request_id=str(response.request_id),
            provider=provider,
            model=response.model_used,
            latency_ms=response.latency_ms,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            estimated_cost_usd=execution_cost,
            pricing_table_version=request_pricing_context,
            cache_hit=response.cache_info.hit,
            error_type=None,
            error_message=None,
            timestamp=response.timestamp.isoformat(),
            provider_attempt_count=response.provider_attempt_count,
            provider_retry_count=response.provider_retry_count,
            cost_evidence_complete=cost_complete,
            provider_attempts=attempts,
        )

    @classmethod
    def from_error(
        cls,
        *,
        request_id: UUID,
        provider: str,
        model: str,
        latency_ms: int,
        error: ProviderError,
        pricing_table_version: str | None = None,
    ) -> RequestTrace:
        attempts = error.provider_attempts
        execution_cost, cost_complete = _execution_cost_from_attempts(
            attempts,
            fallback_final_cost=None,
            legacy_attempt_count=error.provider_attempt_count,
        )
        return cls(
            request_id=str(request_id),
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            estimated_cost_usd=execution_cost,
            pricing_table_version=(
                pricing_table_version
                if pricing_table_version is not None
                else _request_pricing_context(attempts)
            ),
            cache_hit=False,
            error_type=error.error_type.value,
            error_message=error.message,
            timestamp=utc_now().isoformat(),
            provider_attempt_count=error.provider_attempt_count,
            provider_retry_count=error.provider_retry_count,
            cost_evidence_complete=cost_complete,
            provider_attempts=attempts,
        )


class JsonlRequestLog:
    """Append-only JSONL request ledger for local smoke and benchmark runs."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, trace: RequestTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(trace), sort_keys=True) + "\n")

    def read_all(self) -> list[RequestTrace]:
        if not self.path.exists():
            return []

        traces: list[RequestTrace] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                traces.append(_request_trace_from_dict(json.loads(line)))
        return traces


@dataclass(frozen=True)
class RouteTrace:
    """One routing decision record with fail-closed pricing evidence.

    New route traces carry the exact `PricingQuote` used during selection. Historical route rows
    that contain only a numeric estimate are retained as raw history but are not exposed as complete
    monetary evidence until matching pricing provenance exists.
    """

    request_id: str
    strategy: str
    selected_model: str
    estimated_cost_usd: float | None
    estimated_latency_ms: int
    decision_reason: str
    considered_models: list[str]
    fallback_models: list[str]
    max_estimated_cost_usd: float | None
    budget_violation: bool
    budget_violation_reason: str | None
    timestamp: str
    cost_evidence_complete: bool = False
    cost_quote: PricingQuote | None = None

    def __post_init__(self) -> None:
        if self.cost_evidence_complete:
            if self.estimated_cost_usd is None or self.cost_quote is None:
                raise ValueError("complete route cost evidence requires a pricing quote and cost")
            if self.cost_quote.model != self.selected_model:
                raise ValueError("route pricing quote model must match selected model")
            if not isclose(
                self.estimated_cost_usd,
                self.cost_quote.amount_usd,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("route estimated cost must equal pricing quote amount")
        elif self.estimated_cost_usd is not None or self.cost_quote is not None:
            raise ValueError("incomplete route cost evidence must not expose cost or pricing quote")

    @classmethod
    def from_decision(
        cls,
        decision: RoutingDecision,
        *,
        max_estimated_cost_usd: float | None = None,
        budget_violation_reason: str | None = None,
    ) -> RouteTrace:
        budget_violation = budget_violation_reason is not None
        quote = decision.cost_quote
        return cls(
            request_id=str(decision.request_id),
            strategy=decision.strategy.value,
            selected_model=decision.selected_model.id,
            estimated_cost_usd=quote.amount_usd if quote is not None else None,
            estimated_latency_ms=decision.estimated_latency_ms,
            decision_reason=decision.decision_reason,
            considered_models=decision.considered_models,
            fallback_models=[model.id for model in decision.fallback_models],
            max_estimated_cost_usd=max_estimated_cost_usd,
            budget_violation=budget_violation,
            budget_violation_reason=budget_violation_reason,
            timestamp=decision.timestamp.isoformat(),
            cost_evidence_complete=quote is not None,
            cost_quote=quote,
        )


class JsonlRouteLog:
    """Append-only JSONL route decision ledger for local benchmark runs."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, trace: RouteTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_route_trace_to_dict(trace), sort_keys=True) + "\n")

    def read_all(self) -> list[RouteTrace]:
        if not self.path.exists():
            return []

        traces: list[RouteTrace] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                traces.append(_route_trace_from_dict(json.loads(line)))
        return traces


def _execution_cost_from_attempts(
    attempts: tuple[ProviderAttempt, ...],
    *,
    fallback_final_cost: float | None,
    legacy_attempt_count: int,
) -> tuple[float | None, bool]:
    if attempts:
        if all(attempt.cost_is_known for attempt in attempts):
            return sum(attempt.cost_usd or 0.0 for attempt in attempts), True
        return None, False

    if legacy_attempt_count == 1 and fallback_final_cost is not None:
        return fallback_final_cost, True
    if legacy_attempt_count == 0:
        return 0.0, True
    return None, False


def _request_pricing_context(attempts: tuple[ProviderAttempt, ...]) -> str:
    """Summarize attempt-level monetary origin without fabricating a pricing record."""
    if not attempts:
        return PRICING_TABLE_VERSION

    versions = sorted(
        {
            attempt.pricing_table_version
            for attempt in attempts
            if attempt.pricing_table_version is not None
        }
    )
    external_sources = sorted(
        {attempt.cost_source for attempt in attempts if attempt.cost_source is not None}
    )
    if all(attempt.cost_is_known for attempt in attempts):
        if not external_sources and len(versions) == 1:
            return versions[0]
        if not versions and len(external_sources) == 1:
            return f"external_reported:{external_sources[0]}"
        return "mixed"

    contexts = [*versions, *(f"external:{source}" for source in external_sources)]
    if not contexts:
        return "unpriced"
    return "incomplete:" + ",".join(contexts)


def _request_trace_from_dict(raw: dict[str, Any]) -> RequestTrace:
    raw = dict(raw)
    raw.setdefault("cache_hit", False)
    if raw["cache_hit"] is not None and not isinstance(raw["cache_hit"], bool):
        raise ValueError("cache_hit must be boolean or null")
    raw_attempts = raw.pop("provider_attempts", [])
    attempts = tuple(_provider_attempt_from_dict(item) for item in raw_attempts)
    raw["provider_attempts"] = attempts

    legacy_attempt_count = int(raw.get("provider_attempt_count", 1))
    legacy_retry_count = int(raw.get("provider_retry_count", 0))
    legacy_ambiguous = (
        not attempts
        and legacy_attempt_count > 0
        and (
            raw.get("error_type") is not None
            or legacy_retry_count > 0
            or legacy_attempt_count > 1
        )
    )
    incomplete_attempt_evidence = attempts and not all(attempt.cost_is_known for attempt in attempts)
    if legacy_ambiguous or incomplete_attempt_evidence:
        raw["cost_evidence_complete"] = False
        raw["estimated_cost_usd"] = None
        if incomplete_attempt_evidence:
            raw["pricing_table_version"] = _request_pricing_context(attempts)
    elif attempts:
        raw["estimated_cost_usd"] = sum(attempt.cost_usd or 0.0 for attempt in attempts)
        raw["cost_evidence_complete"] = True
        raw["pricing_table_version"] = _request_pricing_context(attempts)
    elif "cost_evidence_complete" not in raw:
        raw["cost_evidence_complete"] = True

    return RequestTrace(**raw)


def _route_trace_to_dict(trace: RouteTrace) -> dict[str, Any]:
    raw = asdict(trace)
    quote = trace.cost_quote
    if quote is not None:
        raw_quote = raw["cost_quote"]
        assert isinstance(raw_quote, dict)
        raw_quote["pricing_observed_at"] = quote.pricing_observed_at.isoformat()
    return raw


def _route_trace_from_dict(raw: dict[str, Any]) -> RouteTrace:
    normalized = dict(raw)
    raw_quote = normalized.pop("cost_quote", None)
    if raw_quote is None:
        normalized["estimated_cost_usd"] = None
        normalized["cost_evidence_complete"] = False
        normalized["cost_quote"] = None
        return RouteTrace(**normalized)
    if not isinstance(raw_quote, dict):
        raise ValueError("route cost_quote must be an object")

    quote = _pricing_quote_from_dict(raw_quote)
    normalized["cost_quote"] = quote
    normalized["cost_evidence_complete"] = True
    if normalized.get("estimated_cost_usd") is None:
        normalized["estimated_cost_usd"] = quote.amount_usd
    return RouteTrace(**normalized)


def _pricing_quote_from_dict(raw: dict[str, Any]) -> PricingQuote:
    return PricingQuote(
        amount_usd=float(raw["amount_usd"]),
        provider=str(raw["provider"]),
        model=str(raw["model"]),
        input_tokens=int(raw["input_tokens"]),
        output_tokens=int(raw["output_tokens"]),
        cached_input_tokens=int(raw["cached_input_tokens"]),
        input_per_million=float(raw["input_per_million"]),
        output_per_million=float(raw["output_per_million"]),
        cached_input_per_million=_optional_float(raw.get("cached_input_per_million")),
        pricing_record_id=str(raw["pricing_record_id"]),
        pricing_table_version=str(raw["pricing_table_version"]),
        pricing_observed_at=date.fromisoformat(str(raw["pricing_observed_at"])),
        pricing_source_url=str(raw["pricing_source_url"]),
    )


def _provider_attempt_from_dict(raw: dict[str, Any]) -> ProviderAttempt:
    cost_evidence = CostEvidenceKind(
        str(raw.get("cost_evidence", CostEvidenceKind.UNKNOWN.value))
    )
    pricing_table_version = _optional_str(raw.get("pricing_table_version"))
    pricing_record_id = _optional_str(raw.get("pricing_record_id"))
    pricing_observed_at = _optional_str(raw.get("pricing_observed_at"))
    pricing_source_url = _optional_str(raw.get("pricing_source_url"))
    calculated_cost_usd = _optional_float(raw.get("calculated_cost_usd"))
    reported_cost_usd = _optional_float(raw.get("reported_cost_usd"))
    cost_source = _optional_str(raw.get("cost_source"))
    cost_source_record_id = _optional_str(raw.get("cost_source_record_id"))

    pricing_provenance = (
        pricing_table_version,
        pricing_record_id,
        pricing_observed_at,
        pricing_source_url,
    )
    external_provenance = (cost_source, cost_source_record_id)
    malformed_calculated = cost_evidence == CostEvidenceKind.CALCULATED_FROM_USAGE and (
        calculated_cost_usd is None
        or reported_cost_usd is not None
        or any(value is None or not value.strip() for value in pricing_provenance)
        or any(value is not None for value in external_provenance)
    )
    malformed_reported = cost_evidence == CostEvidenceKind.REPORTED_BY_EXECUTION_STACK and (
        reported_cost_usd is None
        or calculated_cost_usd is not None
        or any(value is not None for value in pricing_provenance)
        or any(value is None or not value.strip() for value in external_provenance)
    )
    malformed_unknown = cost_evidence == CostEvidenceKind.UNKNOWN and (
        calculated_cost_usd is not None
        or reported_cost_usd is not None
        or any(value is not None for value in pricing_provenance)
        or any(value is not None for value in external_provenance)
    )
    if malformed_calculated or malformed_reported or malformed_unknown:
        cost_evidence = CostEvidenceKind.UNKNOWN
        calculated_cost_usd = None
        reported_cost_usd = None
        pricing_table_version = None
        pricing_record_id = None
        pricing_observed_at = None
        pricing_source_url = None
        cost_source = None
        cost_source_record_id = None

    return ProviderAttempt(
        attempt_index=int(raw["attempt_index"]),
        provider=str(raw["provider"]),
        model=str(raw["model"]),
        outcome=AttemptOutcome(str(raw["outcome"])),
        latency_ms=int(raw["latency_ms"]),
        prompt_tokens=_optional_int(raw.get("prompt_tokens")),
        completion_tokens=_optional_int(raw.get("completion_tokens")),
        total_tokens=_optional_int(raw.get("total_tokens")),
        cached_tokens=_optional_int(raw.get("cached_tokens")),
        calculated_cost_usd=calculated_cost_usd,
        reported_cost_usd=reported_cost_usd,
        cost_evidence=cost_evidence,
        pricing_table_version=pricing_table_version,
        pricing_record_id=pricing_record_id,
        pricing_observed_at=pricing_observed_at,
        pricing_source_url=pricing_source_url,
        cost_source=cost_source,
        cost_source_record_id=cost_source_record_id,
        error_type=_optional_str(raw.get("error_type")),
        status_code=_optional_int(raw.get("status_code")),
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, int)):
        return int(value)
    if isinstance(value, (str, bytes, bytearray, float)):
        return int(value)
    raise TypeError(f"expected integer-compatible value, got {type(value).__name__}")


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return float(value)
    if isinstance(value, (str, bytes, bytearray)):
        return float(value)
    raise TypeError(f"expected float-compatible value, got {type(value).__name__}")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
