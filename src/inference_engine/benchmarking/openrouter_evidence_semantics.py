"""Mission 6C OpenRouter evidence-semantics contract.

This module does not rewrite Mission 6B-OR. Frozen historical R1 remains
``reconcile_tokens_r1`` (v1) in ``openrouter_billing.py``. v2 distinguishes field
presence from semantic comparability and must not treat undocumented integer
token fields as billed equivalents.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any

from .openrouter_billing import (
    EvidenceClass,
    OpenRouterBillingError,
    ResponseUsage,
    RuleResult,
    allowed_upstream_provider,
    combine_required_statuses,
    compare_optional_int,
    optional_decimal,
)

RECONCILIATION_SEMANTICS_V1 = "v1"
RECONCILIATION_SEMANTICS_V2 = "v2"
HISTORICAL_6B_RECONCILIATION_SEMANTICS = RECONCILIATION_SEMANTICS_V1
PROVIDER_METADATA_CONTRACT_ANOMALY = "PROVIDER_METADATA_CONTRACT_ANOMALY"

_COMPARABLE_PAIRS = frozenset(
    {
        frozenset({"response.usage.prompt_tokens", "generation.native_tokens_prompt"}),
        frozenset({"response.usage.completion_tokens", "generation.native_tokens_completion"}),
        frozenset(
            {
                "response.usage.prompt_tokens_details.cached_tokens",
                "generation.native_tokens_cached",
            }
        ),
    }
)


class TokenFieldSemantics(StrEnum):
    """Provenance labels for OpenRouter token fields. Not a general ontology."""

    PROVIDER_NATIVE = "PROVIDER_NATIVE"
    GATEWAY_NORMALIZED = "GATEWAY_NORMALIZED"
    BILLING_OBSERVED = "BILLING_OBSERVED"
    UNKNOWN = "UNKNOWN"


class StreamedFieldClassification(StrEnum):
    PROVIDER_METADATA_CONTRACT_ANOMALY = PROVIDER_METADATA_CONTRACT_ANOMALY
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class StreamedFieldInterpretation:
    classification: StreamedFieldClassification
    proves_client_visible_streaming: bool
    requested_stream: bool
    response_object: str | None
    generation_streamed: bool | None
    reason: str


@dataclass(frozen=True)
class UpstreamCostSurfaces:
    response_cost_details_upstream_inference_cost: Decimal | None
    generation_upstream_inference_cost: Decimal | None
    required_financial_gate: bool = False


def resolve_reconciliation_semantics(spec: Mapping[str, Any]) -> str:
    """Default v1 so frozen 6B specs remain replayable without a new field."""

    raw = spec.get("reconciliation_semantics")
    if raw is None:
        return RECONCILIATION_SEMANTICS_V1
    text = str(raw)
    if text not in {RECONCILIATION_SEMANTICS_V1, RECONCILIATION_SEMANTICS_V2}:
        raise OpenRouterBillingError("reconciliation_semantics must be v1 or v2")
    return text


def token_field_semantics(field: str) -> TokenFieldSemantics:
    """Official-docs-backed labels. GATEWAY_NORMALIZED is not assigned without docs."""

    known = {
        "response.usage.prompt_tokens": TokenFieldSemantics.BILLING_OBSERVED,
        "response.usage.completion_tokens": TokenFieldSemantics.BILLING_OBSERVED,
        "response.usage.total_tokens": TokenFieldSemantics.BILLING_OBSERVED,
        "response.usage.prompt_tokens_details.cached_tokens": TokenFieldSemantics.BILLING_OBSERVED,
        "response.usage.cost": TokenFieldSemantics.BILLING_OBSERVED,
        "generation.native_tokens_prompt": TokenFieldSemantics.PROVIDER_NATIVE,
        "generation.native_tokens_completion": TokenFieldSemantics.PROVIDER_NATIVE,
        "generation.native_tokens_cached": TokenFieldSemantics.PROVIDER_NATIVE,
        "generation.tokens_prompt": TokenFieldSemantics.UNKNOWN,
        "generation.tokens_completion": TokenFieldSemantics.UNKNOWN,
        "generation.total_cost": TokenFieldSemantics.BILLING_OBSERVED,
        "generation.usage": TokenFieldSemantics.BILLING_OBSERVED,
        "generation.upstream_inference_cost": TokenFieldSemantics.UNKNOWN,
        "response.usage.cost_details.upstream_inference_cost": TokenFieldSemantics.UNKNOWN,
    }
    if field not in known:
        raise OpenRouterBillingError(f"no semantics label registered for {field}")
    return known[field]


def token_fields_semantically_comparable(left_field: str, right_field: str) -> bool:
    """Presence of two integer token fields is not comparability."""

    return frozenset({left_field, right_field}) in _COMPARABLE_PAIRS


def generation_streamed_proves_client_visible_streaming(generation_streamed: object) -> bool:
    """generation.streamed must never prove client-visible streaming by itself."""

    del generation_streamed
    return False


def interpret_generation_streamed_field(
    *,
    requested_stream: bool,
    response_object: str | None,
    generation_streamed: bool | None,
) -> StreamedFieldInterpretation:
    """Official schema says only 'Whether the response was streamed'.

    It does not distinguish client delivery from upstream/internal transport.
    """

    client_non_streaming_response = response_object == "chat.completion" and requested_stream is False
    if generation_streamed is True and client_non_streaming_response:
        return StreamedFieldInterpretation(
            classification=StreamedFieldClassification.PROVIDER_METADATA_CONTRACT_ANOMALY,
            proves_client_visible_streaming=False,
            requested_stream=requested_stream,
            response_object=response_object,
            generation_streamed=generation_streamed,
            reason=(
                "generation.streamed=true while Chat Completions requested stream=false and "
                "returned object=chat.completion. Official docs do not establish whether "
                "streamed describes client delivery or upstream transport."
            ),
        )
    return StreamedFieldInterpretation(
        classification=StreamedFieldClassification.INSUFFICIENT_EVIDENCE,
        proves_client_visible_streaming=False,
        requested_stream=requested_stream,
        response_object=response_object,
        generation_streamed=generation_streamed,
        reason=(
            "Official OpenAPI description is only 'Whether the response was streamed'. "
            "That is not sufficient to treat generation.streamed as client-visible streaming."
        ),
    )


def split_upstream_cost_surfaces(
    *,
    response_cost_details_upstream: Decimal | None,
    generation_upstream: Decimal | None,
) -> UpstreamCostSurfaces:
    """Keep the two OpenRouter upstream-cost fields separate. Neither is a required gate."""

    return UpstreamCostSurfaces(
        response_cost_details_upstream_inference_cost=response_cost_details_upstream,
        generation_upstream_inference_cost=generation_upstream,
        required_financial_gate=False,
    )


def upstream_cost_surfaces_from_payloads(
    *,
    chat_payload: Mapping[str, Any] | None,
    generation_payload: Mapping[str, Any] | None,
) -> UpstreamCostSurfaces:
    response_upstream = None
    if chat_payload is not None:
        usage = chat_payload.get("usage")
        if isinstance(usage, Mapping):
            details = usage.get("cost_details")
            if isinstance(details, Mapping) and "upstream_inference_cost" in details:
                response_upstream = optional_decimal(
                    details.get("upstream_inference_cost"),
                    field="usage.cost_details.upstream_inference_cost",
                )
    generation_upstream = None
    data = generation_payload.get("data") if isinstance(generation_payload, Mapping) else None
    source = data if isinstance(data, Mapping) else generation_payload
    if isinstance(source, Mapping) and "upstream_inference_cost" in source:
        generation_upstream = optional_decimal(
            source.get("upstream_inference_cost"),
            field="generation.upstream_inference_cost",
        )
    return split_upstream_cost_surfaces(
        response_cost_details_upstream=response_upstream,
        generation_upstream=generation_upstream,
    )


def reconcile_tokens_r1_v2(
    *,
    response: ResponseUsage | None,
    generation: Mapping[str, Any] | None,
) -> RuleResult:
    """Compare only officially comparable native/billing token pairs.

    generation.tokens_prompt remains UNKNOWN and is NOT_COMPARABLE with response
    prompt_tokens even when both integers are present.
    """

    if response is None or generation is None:
        return RuleResult(
            rule_id="R1",
            required=True,
            status=EvidenceClass.INCONCLUSIVE,
            reason="response usage or generation metadata is missing",
        )
    native_prompt = _optional_int(generation.get("native_tokens_prompt"))
    native_completion = _optional_int(generation.get("native_tokens_completion"))
    native_cached = _optional_int(generation.get("native_tokens_cached"))
    tokens_prompt = _optional_int(generation.get("tokens_prompt"))
    provider = generation.get("provider_name") if isinstance(generation.get("provider_name"), str) else None
    comparable = [
        _native_pair_status(
            response.prompt_tokens,
            native_prompt,
            provider_name=provider,
            left_field="response.usage.prompt_tokens",
            right_field="generation.native_tokens_prompt",
        ),
        _native_pair_status(
            response.completion_tokens,
            native_completion,
            provider_name=provider,
            left_field="response.usage.completion_tokens",
            right_field="generation.native_tokens_completion",
        ),
    ]
    if response.cached_input_tokens is not None and native_cached is not None:
        comparable.append(
            _native_pair_status(
                response.cached_input_tokens,
                native_cached,
                provider_name=provider,
                left_field="response.usage.prompt_tokens_details.cached_tokens",
                right_field="generation.native_tokens_cached",
            )
        )
    status = combine_required_statuses(comparable)
    tokens_prompt_note = (
        "generation.tokens_prompt is UNKNOWN and NOT_COMPARABLE with response "
        "prompt_tokens; integer presence is not billed-field equivalence"
    )
    if tokens_prompt is not None and tokens_prompt != response.prompt_tokens:
        tokens_prompt_note += "; observed tokens_prompt divergence was not classified as R1 MISMATCH"
    return RuleResult(
        rule_id="R1",
        required=True,
        status=status,
        reason=(
            "v2 compares provider-native generation token fields with response usage "
            f"when provenance allows; {tokens_prompt_note}"
        ),
        left=str(response.prompt_tokens),
        right=None if native_prompt is None else str(native_prompt),
    )


def select_r1_reconciler(semantics: str) -> Callable[..., RuleResult]:
    from .openrouter_billing import reconcile_tokens_r1

    if semantics == RECONCILIATION_SEMANTICS_V1:
        return reconcile_tokens_r1
    if semantics == RECONCILIATION_SEMANTICS_V2:
        return reconcile_tokens_r1_v2
    raise OpenRouterBillingError("reconciliation_semantics must be v1 or v2")


def _native_pair_status(
    response_value: int | None,
    native_value: int | None,
    *,
    provider_name: str | None,
    left_field: str,
    right_field: str,
) -> EvidenceClass:
    if not token_fields_semantically_comparable(left_field, right_field):
        return EvidenceClass.NOT_COMPARABLE
    if response_value is None or native_value is None:
        return EvidenceClass.INCONCLUSIVE
    if provider_name is None or not allowed_upstream_provider(provider_name):
        return EvidenceClass.NOT_COMPARABLE
    return compare_optional_int(response_value, native_value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


__all__ = [
    "HISTORICAL_6B_RECONCILIATION_SEMANTICS",
    "PROVIDER_METADATA_CONTRACT_ANOMALY",
    "RECONCILIATION_SEMANTICS_V1",
    "RECONCILIATION_SEMANTICS_V2",
    "StreamedFieldClassification",
    "StreamedFieldInterpretation",
    "TokenFieldSemantics",
    "UpstreamCostSurfaces",
    "generation_streamed_proves_client_visible_streaming",
    "interpret_generation_streamed_field",
    "reconcile_tokens_r1_v2",
    "resolve_reconciliation_semantics",
    "select_r1_reconciler",
    "split_upstream_cost_surfaces",
    "token_field_semantics",
    "token_fields_semantically_comparable",
    "upstream_cost_surfaces_from_payloads",
]
