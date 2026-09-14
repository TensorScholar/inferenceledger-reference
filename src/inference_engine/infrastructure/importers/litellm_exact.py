from __future__ import annotations

from collections.abc import Mapping, Sequence

from ...domain.models.economics import (
    ExactAttemptCost,
    ExactRequestCostEvidence,
    canonical_decimal_text,
)
from ...domain.models.execution import CostEvidenceKind
from .litellm import LITELLM_STANDARD_LOGGING_SOURCE, import_litellm_standard_logging_chain


def extract_litellm_exact_cost_evidence(
    payloads: Sequence[Mapping[str, object]],
    *,
    currency: str = "USD",
) -> ExactRequestCostEvidence:
    """Extract exact decimal-string attempt costs from one validated LiteLLM trace.

    The regular importer is invoked first so trace identity, ordering, duplicate ids, overlapping
    attempts, and terminal-success semantics share one validation contract. Monetary values are then
    re-read from the original payload objects rather than from the compatibility floats carried by
    ``RequestTrace``.

    ``response_cost`` remains execution-stack-reported evidence. Preserving the exact decimal text
    does not promote the value to provider-ledger or final-invoice truth.
    """
    trace = import_litellm_standard_logging_chain(payloads)
    ordered = sorted(
        payloads,
        key=lambda payload: (
            _numeric(payload, "startTime"),
            _numeric(payload, "endTime"),
            _text(payload, "id"),
        ),
    )
    attempts = tuple(
        _exact_attempt(index=index, payload=payload, currency=currency)
        for index, payload in enumerate(ordered, 1)
    )
    if len(attempts) != trace.provider_attempt_count:
        raise ValueError("exact LiteLLM attempt evidence disagrees with canonical attempt count")
    return ExactRequestCostEvidence.from_attempts(
        request_id=trace.request_id,
        currency=currency,
        attempts=attempts,
    )


def _exact_attempt(
    *,
    index: int,
    payload: Mapping[str, object],
    currency: str,
) -> ExactAttemptCost:
    record_id = _text(payload, "id")
    cost_failure = payload.get("response_cost_failure_debug_info")
    raw_cost = payload.get("response_cost")
    if cost_failure is not None:
        return ExactAttemptCost(
            attempt_index=index,
            currency=currency,
            evidence_kind=CostEvidenceKind.UNKNOWN,
            amount_decimal=None,
            source=None,
            source_record_id=None,
            unknown_reason="LiteLLM reported response_cost_failure_debug_info",
        )
    if raw_cost is None:
        return ExactAttemptCost(
            attempt_index=index,
            currency=currency,
            evidence_kind=CostEvidenceKind.UNKNOWN,
            amount_decimal=None,
            source=None,
            source_record_id=None,
            unknown_reason="LiteLLM StandardLoggingPayload omitted response_cost",
        )
    return ExactAttemptCost(
        attempt_index=index,
        currency=currency,
        evidence_kind=CostEvidenceKind.REPORTED_BY_EXECUTION_STACK,
        amount_decimal=canonical_decimal_text(raw_cost),
        source=LITELLM_STANDARD_LOGGING_SOURCE,
        source_record_id=record_id,
    )


def _text(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"LiteLLM exact evidence requires non-empty {field}")
    return value.strip()


def _numeric(payload: Mapping[str, object], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"LiteLLM exact evidence requires numeric {field}")
    return float(value)
