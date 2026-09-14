from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from decimal import Decimal

from ..domain.models.economics import (
    ExactRequestCostEvidence,
    canonical_decimal_text,
    decimal_from_text,
)
from ..domain.models.execution import CostEvidenceKind, ProviderAttempt
from ..infrastructure.telemetry.request_log import RequestTrace
from .pilot_contract import (
    AcquisitionSemantics,
    MeasurementSemantics,
    PilotInstanceContract,
    SemanticEnforcementMode,
)


@dataclass(frozen=True)
class PilotArmEvidenceSummary:
    acquisition: AcquisitionSemantics
    request_count: int
    provider_attempt_count: int
    unknown_cost_attempt_count: int
    exact_cost_complete: bool
    exact_total_decimal: str | None
    currency: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PilotSemanticValidation:
    valid_for_full_semantics: bool
    violations: tuple[str, ...]
    baseline: PilotArmEvidenceSummary
    candidate: PilotArmEvidenceSummary

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_pilot_semantics(
    *,
    contract: PilotInstanceContract,
    baseline_acquisition: AcquisitionSemantics,
    candidate_acquisition: AcquisitionSemantics,
    baseline_traces: Sequence[RequestTrace],
    candidate_traces: Sequence[RequestTrace],
    baseline_exact_costs: Sequence[ExactRequestCostEvidence],
    candidate_exact_costs: Sequence[ExactRequestCostEvidence],
) -> PilotSemanticValidation:
    """Validate the claim-bearing semantics of one frozen comparative pilot evidence set."""
    violations: list[str] = []
    if baseline_acquisition != contract.baseline_acquisition:
        violations.append("stored baseline acquisition semantics disagree with frozen pilot contract")
    if candidate_acquisition != contract.candidate_acquisition:
        violations.append("stored candidate acquisition semantics disagree with frozen pilot contract")

    baseline_summary = _validate_arm(
        arm="baseline",
        acquisition=baseline_acquisition,
        traces=baseline_traces,
        exact_costs=baseline_exact_costs,
        currency=contract.economic.currency,
        rate_semantics=contract.economic.rate_semantics,
        violations=violations,
    )
    candidate_summary = _validate_arm(
        arm="candidate",
        acquisition=candidate_acquisition,
        traces=candidate_traces,
        exact_costs=candidate_exact_costs,
        currency=contract.economic.currency,
        rate_semantics=contract.economic.rate_semantics,
        violations=violations,
    )

    if baseline_acquisition.measurement != MeasurementSemantics.OBSERVED:
        violations.append(
            "baseline measurement is not OBSERVED; pilot quality/reliability/latency/economic approval is unsupported"
        )
    if candidate_acquisition.measurement != MeasurementSemantics.OBSERVED:
        violations.append(
            "candidate measurement is not OBSERVED; pilot quality/reliability/latency/economic approval is unsupported"
        )

    return PilotSemanticValidation(
        valid_for_full_semantics=not violations,
        violations=tuple(violations),
        baseline=baseline_summary,
        candidate=candidate_summary,
    )


def derive_statistical_cost_traces(
    traces: Sequence[RequestTrace],
    exact_costs: Sequence[ExactRequestCostEvidence],
) -> list[RequestTrace]:
    """Derive numeric analysis traces from authoritative exact monetary evidence.

    Statistical primitives currently consume floats. Conversion occurs only here after exact values
    have been validated and persisted. Attempt-level compatibility values are rebuilt from the same
    exact sidecar so ``RequestTrace``'s attempt-sum invariant continues to hold even when the exact
    decimal contains precision not preserved by a legacy float.
    """
    exact_by_id = {item.request_id: item for item in exact_costs}
    result: list[RequestTrace] = []
    for trace in traces:
        exact = exact_by_id.get(trace.request_id)
        if exact is None:
            result.append(replace(trace, estimated_cost_usd=None, cost_evidence_complete=False))
            continue
        derived_attempts = _derive_attempts(trace.provider_attempts, exact)
        value = (
            float(decimal_from_text(exact.total_decimal))
            if exact.total_decimal is not None
            else None
        )
        result.append(
            replace(
                trace,
                estimated_cost_usd=value,
                cost_evidence_complete=exact.complete,
                provider_attempts=derived_attempts,
            )
        )
    return result


def study_decision_for_gate(
    *,
    gate_decision: str,
    contract: PilotInstanceContract,
    validation: PilotSemanticValidation,
    mode: SemanticEnforcementMode,
) -> str:
    """Map the internal Change Gate into the preregistered study vocabulary.

    FULL_INFERENCELEDGER fails closed at the study-claim boundary. The semantics ablation receives
    the same evidence and gate result but bypasses only this semantic-validity override.
    """
    if (
        mode == SemanticEnforcementMode.FULL_INFERENCELEDGER
        and not validation.valid_for_full_semantics
    ):
        return "ABSTAIN"
    return contract.study_outputs.map_gate_decision(gate_decision)


def _derive_attempts(
    provider_attempts: tuple[ProviderAttempt, ...],
    exact: ExactRequestCostEvidence,
) -> tuple[ProviderAttempt, ...]:
    if not provider_attempts or len(provider_attempts) != len(exact.attempts):
        return provider_attempts
    derived: list[ProviderAttempt] = []
    for legacy, authoritative in zip(provider_attempts, exact.attempts, strict=True):
        if authoritative.evidence_kind == CostEvidenceKind.UNKNOWN:
            derived.append(
                replace(
                    legacy,
                    calculated_cost_usd=None,
                    reported_cost_usd=None,
                    cost_evidence=CostEvidenceKind.UNKNOWN,
                    pricing_table_version=None,
                    pricing_record_id=None,
                    pricing_observed_at=None,
                    pricing_source_url=None,
                    cost_source=None,
                    cost_source_record_id=None,
                )
            )
            continue
        assert authoritative.amount_decimal is not None
        amount = float(decimal_from_text(authoritative.amount_decimal))
        if authoritative.evidence_kind == CostEvidenceKind.REPORTED_BY_EXECUTION_STACK:
            derived.append(
                replace(
                    legacy,
                    calculated_cost_usd=None,
                    reported_cost_usd=amount,
                    cost_evidence=CostEvidenceKind.REPORTED_BY_EXECUTION_STACK,
                    pricing_table_version=None,
                    pricing_record_id=None,
                    pricing_observed_at=None,
                    pricing_source_url=None,
                    cost_source=authoritative.source,
                    cost_source_record_id=authoritative.source_record_id,
                )
            )
        else:
            derived.append(
                replace(
                    legacy,
                    calculated_cost_usd=amount,
                    reported_cost_usd=None,
                    cost_evidence=CostEvidenceKind.CALCULATED_FROM_USAGE,
                    cost_source=None,
                    cost_source_record_id=None,
                )
            )
    return tuple(derived)


def _validate_arm(
    *,
    arm: str,
    acquisition: AcquisitionSemantics,
    traces: Sequence[RequestTrace],
    exact_costs: Sequence[ExactRequestCostEvidence],
    currency: str,
    rate_semantics: str,
    violations: list[str],
) -> PilotArmEvidenceSummary:
    trace_by_id = {trace.request_id: trace for trace in traces}
    if len(trace_by_id) != len(traces):
        violations.append(f"{arm} traces contain duplicate request ids")
    exact_by_id = {item.request_id: item for item in exact_costs}
    if len(exact_by_id) != len(exact_costs):
        violations.append(f"{arm} exact-cost evidence contains duplicate request ids")
    if set(trace_by_id) != set(exact_by_id):
        violations.append(
            f"{arm} exact monetary coverage differs from trace coverage: "
            f"trace_only={sorted(set(trace_by_id) - set(exact_by_id))}, "
            f"cost_only={sorted(set(exact_by_id) - set(trace_by_id))}"
        )

    unknown_attempts = 0
    attempt_count = 0
    all_complete = True
    total = Decimal(0)
    observed_cost_kinds: set[CostEvidenceKind] = set()
    for request_id in sorted(set(trace_by_id) & set(exact_by_id)):
        trace = trace_by_id[request_id]
        exact = exact_by_id[request_id]
        if exact.currency != currency:
            violations.append(
                f"{arm} request {request_id} exact currency {exact.currency} != frozen {currency}"
            )
        if trace.provider_attempt_count != len(exact.attempts):
            violations.append(
                f"{arm} request {request_id} exact attempt count does not match canonical trace"
            )
        if trace.provider_attempt_count > 0 and not trace.provider_attempts:
            violations.append(
                f"{arm} request {request_id} has provider attempts but lacks attempt-chain evidence"
            )
        if trace.provider_attempts and len(trace.provider_attempts) == len(exact.attempts):
            for legacy, authoritative in zip(trace.provider_attempts, exact.attempts, strict=True):
                if legacy.cost_evidence != authoritative.evidence_kind:
                    violations.append(
                        f"{arm} request {request_id} attempt {authoritative.attempt_index} cost-evidence kind disagrees between canonical and exact surfaces"
                    )
                if (
                    authoritative.source_record_id is not None
                    and legacy.cost_source_record_id != authoritative.source_record_id
                ):
                    violations.append(
                        f"{arm} request {request_id} attempt {authoritative.attempt_index} source record id disagrees between canonical and exact surfaces"
                    )
        attempt_count += len(exact.attempts)
        for attempt in exact.attempts:
            observed_cost_kinds.add(attempt.evidence_kind)
            if not attempt.known:
                unknown_attempts += 1
        if not exact.complete or exact.total_decimal is None:
            all_complete = False
        else:
            total += decimal_from_text(exact.total_decimal)

    if (
        CostEvidenceKind.CALCULATED_FROM_USAGE in observed_cost_kinds
        and rate_semantics != "EXACT_RATE_STRINGS"
    ):
        violations.append(
            f"{arm} uses calculated exact costs but frozen rate semantics are not EXACT_RATE_STRINGS"
        )
    if (
        CostEvidenceKind.REPORTED_BY_EXECUTION_STACK in observed_cost_kinds
        and rate_semantics != "NOT_APPLICABLE_REPORTED_CHARGE"
    ):
        violations.append(
            f"{arm} uses execution-stack-reported charges but frozen rate semantics do not declare them non-reconstructed"
        )
    if unknown_attempts:
        violations.append(
            f"{arm} contains {unknown_attempts} economically relevant attempt(s) with unknown exact cost"
        )
    if acquisition.measurement == MeasurementSemantics.ESTIMATED:
        observed_attempts = sum(trace.provider_attempt_count for trace in traces)
        if observed_attempts:
            violations.append(
                f"{arm} is ESTIMATED but contains {observed_attempts} observed provider attempt(s)"
            )
        if any(trace.latency_ms != 0 for trace in traces):
            violations.append(f"{arm} is ESTIMATED but carries measured-looking latency values")
        if any(trace.error_type is not None for trace in traces):
            violations.append(f"{arm} is ESTIMATED but carries observed failure outcomes")
        if any(exact.total_decimal is not None for exact in exact_costs):
            violations.append(f"{arm} is ESTIMATED but carries observed/reported monetary totals")

    exact_total = canonical_decimal_text(total) if all_complete else None
    return PilotArmEvidenceSummary(
        acquisition=acquisition,
        request_count=len(traces),
        provider_attempt_count=attempt_count,
        unknown_cost_attempt_count=unknown_attempts,
        exact_cost_complete=all_complete,
        exact_total_decimal=exact_total,
        currency=currency,
    )
