from __future__ import annotations

from pathlib import Path

import pytest

from inference_engine.benchmarking.evidence_manifest import (
    build_manifest,
    committed_manifest_entry,
    private_reference_manifest_entry,
    verify_manifest,
    write_manifest,
)
from inference_engine.benchmarking.pilot_analysis import (
    ReferenceOutcome,
    StudyDecision,
    StudyWorkflow,
    WorkflowDecisionRecord,
    summarize_workflow_outcomes,
)
from inference_engine.benchmarking.pilot_contract import (
    PILOT_EVIDENCE_CONTRACT_VERSION,
    PILOT_MANIFEST_VERSION,
    PILOT_PROTOCOL_ID,
    AcquisitionSemantics,
    ArtifactRetention,
    ArtifactRetentionPolicy,
    AssignmentSemantics,
    AttemptVisibilityContract,
    CollectionSemantics,
    ComparatorParityPlan,
    EvaluatorRecord,
    ExactMoneySemantics,
    ExternalOwnerRecord,
    MeasurementSemantics,
    PairingPolicy,
    PilotInstanceContract,
    PositiveControl,
    SemanticEnforcementMode,
    StudyOutputContract,
    WorkloadFreezeRecord,
)
from inference_engine.benchmarking.pilot_evidence_store import SQLitePilotEvidenceStore
from inference_engine.benchmarking.pilot_semantics import (
    derive_statistical_cost_traces,
    study_decision_for_gate,
    validate_pilot_semantics,
)
from inference_engine.domain.models.economics import (
    ExactAttemptCost,
    ExactRateEvidence,
    ExactRequestCostEvidence,
    canonical_decimal_text,
)
from inference_engine.domain.models.execution import (
    AttemptOutcome,
    CostEvidenceKind,
    ProviderAttempt,
)
from inference_engine.infrastructure.importers.litellm_exact import (
    extract_litellm_exact_cost_evidence,
)
from inference_engine.infrastructure.telemetry.request_log import RequestTrace

_REQUIRED_ENDPOINTS = (
    "harmful_approval",
    "unsupported_approval",
    "correct_approval",
    "unnecessary_rejection",
    "abstention",
    "decision_coverage",
    "integration_effort",
    "evidence_collection_effort",
    "review_effort",
    "decision_delay",
)


def _contract() -> PilotInstanceContract:
    acquisition = AcquisitionSemantics(
        measurement=MeasurementSemantics.OBSERVED,
        collection=CollectionSemantics.CONTROLLED_REPLAY,
        assignment=AssignmentSemantics.CONTROLLED,
    )
    return PilotInstanceContract(
        version=PILOT_EVIDENCE_CONTRACT_VERSION,
        protocol_id=PILOT_PROTOCOL_ID,
        external_owner=ExternalOwnerRecord(
            organization_id_or_pseudonym="external-org-pseudonym",
            accountable_role="inference deployment owner",
            owns_or_approves_decision=True,
            consent_confirmed=True,
            requirements_approved_before_execution=True,
        ),
        workload=WorkloadFreezeRecord(
            source="externally owned workload",
            selection_procedure="prospective frozen sample",
            version="v1",
            item_count=1,
            sha256="a" * 64,
            retention=ArtifactRetention.COMMITTED_SANITIZED,
            critical_segments=("segment=critical",),
        ),
        baseline_acquisition=acquisition,
        candidate_acquisition=acquisition,
        economic=ExactMoneySemantics(
            accounting_basis="gateway-reported request charge",
            currency="USD",
            representation="DECIMAL_STRING",
            rate_semantics="NOT_APPLICABLE_REPORTED_CHARGE",
            rounding="NONE",
            aggregation="EXACT_SUM",
            equality="EXACT_DECIMAL",
            legacy_float_role="DERIVED_NON_AUTHORITATIVE",
        ),
        attempt_visibility=AttemptVisibilityContract(
            provider_invocations="ATTEMPT_OR_EXPLICIT_UNKNOWN",
            ancillary_billable_calls="ATTEMPT_OR_EXPLICIT_UNKNOWN",
            health_checks="NON_INFERENCE_ONLY",
        ),
        evaluator=EvaluatorRecord(
            identity="external-evaluator",
            version="v1",
            measures="owner-approved task correctness",
            population="frozen workload",
            positive_controls=(
                PositiveControl("good", "PASS", "known-good output"),
                PositiveControl("bad", "FAIL", "known-bad output"),
            ),
            calibration_status="PASSED",
            missing_outcome_policy="FAIL_CLOSED",
            independent_adjudication="independent owner-approved adjudicator",
        ),
        pairing=PairingPolicy(
            key="workload_item_id",
            missing_outcome_policy="FAIL_CLOSED",
            duplicate_policy="FAIL_CLOSED",
        ),
        comparator=ComparatorParityPlan(
            workflow_id="strong-conventional-v1",
            parity_requirement="MATERIAL_EQUIVALENCE_REQUIRED",
            evidence_surfaces=("attempts", "quality", "latency", "cost", "segments"),
        ),
        study_outputs=StudyOutputContract(
            decision_mapping=(
                ("INCONCLUSIVE", "ABSTAIN"),
                ("NO_GO", "REJECT"),
                ("REVIEW", "ABSTAIN"),
                ("SHIP", "APPROVE"),
            ),
            endpoints=_REQUIRED_ENDPOINTS,
            effort_time_unit="minutes",
            delay_start_event="frozen evidence available",
            delay_end_event="workflow decision frozen",
        ),
        ablations=(
            "FULL_INFERENCELEDGER",
            "SEMANTICS_ABLATION",
            "STRONG_CONVENTIONAL",
        ),
        artifacts=ArtifactRetentionPolicy(
            workload=ArtifactRetention.COMMITTED_SANITIZED,
            raw_provider_evidence=ArtifactRetention.PRIVATE_RAW_HASH_REFERENCED,
            sanitized_provider_evidence=ArtifactRetention.COMMITTED_SANITIZED,
            private_raw_location="approved-private-store",
            manifest_version=PILOT_MANIFEST_VERSION,
        ),
    )


def _reported_attempt(index: int, amount: float | None = 0.000000001) -> ProviderAttempt:
    if amount is None:
        return ProviderAttempt(
            attempt_index=index,
            provider="provider",
            model="model",
            outcome=AttemptOutcome.SUCCEEDED,
            latency_ms=10,
            cost_evidence=CostEvidenceKind.UNKNOWN,
        )
    return ProviderAttempt(
        attempt_index=index,
        provider="provider",
        model="model",
        outcome=AttemptOutcome.SUCCEEDED,
        latency_ms=10,
        reported_cost_usd=amount,
        cost_evidence=CostEvidenceKind.REPORTED_BY_EXECUTION_STACK,
        cost_source="litellm_standard_logging_payload",
        cost_source_record_id=f"record-{index}",
    )


def _trace(request_id: str = "request-1", amount: float = 0.000000001) -> RequestTrace:
    attempt = _reported_attempt(1, amount)
    return RequestTrace(
        request_id=request_id,
        provider="provider",
        model="model",
        latency_ms=10,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        estimated_cost_usd=amount,
        pricing_table_version="external_reported:litellm_standard_logging_payload",
        cache_hit=False,
        error_type=None,
        error_message=None,
        timestamp="2026-09-13T00:00:00+00:00",
        quality_passed=True,
        quality_score=1.0,
        quality_reason="fixture",
        eval_type="exact_match",
        provider_attempts=(attempt,),
    )


def _exact_request(
    request_id: str = "request-1", amount: str = "0.000000001"
) -> ExactRequestCostEvidence:
    attempt = ExactAttemptCost(
        attempt_index=1,
        currency="USD",
        evidence_kind=CostEvidenceKind.REPORTED_BY_EXECUTION_STACK,
        amount_decimal=amount,
        source="litellm_standard_logging_payload",
        source_record_id="record-1",
    )
    return ExactRequestCostEvidence.from_attempts(
        request_id=request_id,
        currency="USD",
        attempts=(attempt,),
    )


def test_exact_decimal_strings_preserve_sub_micro_values_and_sum_exactly() -> None:
    first = ExactAttemptCost(
        attempt_index=1,
        currency="usd",
        evidence_kind=CostEvidenceKind.REPORTED_BY_EXECUTION_STACK,
        amount_decimal="0.000000001",
        source="gateway",
        source_record_id="a",
    )
    second = ExactAttemptCost(
        attempt_index=2,
        currency="USD",
        evidence_kind=CostEvidenceKind.REPORTED_BY_EXECUTION_STACK,
        amount_decimal="0.0000000020",
        source="gateway",
        source_record_id="b",
    )
    evidence = ExactRequestCostEvidence.from_attempts(
        request_id="r",
        currency="USD",
        attempts=(first, second),
    )
    assert first.amount_decimal == "0.000000001"
    assert second.amount_decimal == "0.000000002"
    assert evidence.total_decimal == "0.000000003"
    assert canonical_decimal_text("7.08450E-4") == "0.00070845"


def test_unknown_exact_cost_is_not_numeric_zero() -> None:
    unknown = ExactAttemptCost(
        attempt_index=1,
        currency="USD",
        evidence_kind=CostEvidenceKind.UNKNOWN,
        amount_decimal=None,
        source=None,
        source_record_id=None,
        unknown_reason="provider omitted charge",
    )
    evidence = ExactRequestCostEvidence.from_attempts(
        request_id="r",
        currency="USD",
        attempts=(unknown,),
    )
    assert evidence.complete is False
    assert evidence.total_decimal is None


def test_calculated_exact_cost_requires_exact_rate_reconstruction() -> None:
    rates = ExactRateEvidence(
        input_per_million="0.15",
        output_per_million="0.60",
        cached_input_per_million="0.075",
        pricing_record_id="provider:model:2026-09-13",
        pricing_table_version="v1",
        pricing_observed_at="2026-09-13",
        pricing_source_url="https://example.test/pricing",
    )
    exact = ExactAttemptCost(
        attempt_index=1,
        currency="USD",
        evidence_kind=CostEvidenceKind.CALCULATED_FROM_USAGE,
        amount_decimal="0.000000675",
        source="pricing_reconstruction",
        source_record_id="provider:model:2026-09-13",
        prompt_tokens=1,
        completion_tokens=1,
        cached_tokens=1,
        rate_evidence=rates,
    )
    assert exact.amount_decimal == "0.000000675"
    with pytest.raises(ValueError, match="exact usage-rate reconstruction"):
        ExactAttemptCost(
            attempt_index=1,
            currency="USD",
            evidence_kind=CostEvidenceKind.CALCULATED_FROM_USAGE,
            amount_decimal="0.000000676",
            source="pricing_reconstruction",
            source_record_id="provider:model:2026-09-13",
            prompt_tokens=1,
            completion_tokens=1,
            cached_tokens=1,
            rate_evidence=rates,
        )


def test_litellm_retry_chain_extracts_exact_reported_attempt_costs() -> None:
    payloads: list[dict[str, object]] = [
        {
            "id": "a",
            "trace_id": "trace",
            "custom_llm_provider": "provider-a",
            "model": "model-a",
            "status": "failure",
            "startTime": 1.0,
            "endTime": 2.0,
            "response_time": 1.0,
            "prompt_tokens": 1,
            "completion_tokens": 0,
            "total_tokens": 1,
            "response_cost": 0.000000001,
            "cache_hit": False,
        },
        {
            "id": "b",
            "trace_id": "trace",
            "custom_llm_provider": "provider-b",
            "model": "model-b",
            "status": "success",
            "startTime": 3.0,
            "endTime": 4.0,
            "response_time": 1.0,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "response_cost": 0.000000002,
            "cache_hit": False,
        },
    ]
    exact = extract_litellm_exact_cost_evidence(payloads)
    assert len(exact.attempts) == 2
    assert exact.total_decimal == "0.000000003"
    assert [attempt.source_record_id for attempt in exact.attempts] == ["a", "b"]


def test_pilot_store_round_trips_acquisition_and_exact_decimal_text(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    store = SQLitePilotEvidenceStore(path)
    acquisition = _contract().baseline_acquisition
    evidence = _exact_request(amount="0.0000000012300")
    store.record_run(
        run_id="baseline",
        acquisition=acquisition,
        exact_costs=[evidence],
        expected_request_ids=["request-1"],
    )
    restored = store.get_exact_costs("baseline")
    assert store.get_acquisition("baseline") == acquisition
    assert restored[0].total_decimal == "0.00000000123"
    assert restored[0].attempts[0].amount_decimal == "0.00000000123"


def test_semantic_validation_controls_study_decision_but_ablation_uses_same_gate() -> None:
    contract = _contract()
    trace = _trace()
    unknown = ExactRequestCostEvidence.from_attempts(
        request_id="request-1",
        currency="USD",
        attempts=(
            ExactAttemptCost(
                attempt_index=1,
                currency="USD",
                evidence_kind=CostEvidenceKind.UNKNOWN,
                amount_decimal=None,
                source=None,
                source_record_id=None,
                unknown_reason="charge unavailable",
            ),
        ),
    )
    validation = validate_pilot_semantics(
        contract=contract,
        baseline_acquisition=contract.baseline_acquisition,
        candidate_acquisition=contract.candidate_acquisition,
        baseline_traces=[trace],
        candidate_traces=[trace],
        baseline_exact_costs=[unknown],
        candidate_exact_costs=[unknown],
    )
    assert validation.valid_for_full_semantics is False
    assert study_decision_for_gate(
        gate_decision="SHIP",
        contract=contract,
        validation=validation,
        mode=SemanticEnforcementMode.FULL_INFERENCELEDGER,
    ) == "ABSTAIN"
    assert study_decision_for_gate(
        gate_decision="SHIP",
        contract=contract,
        validation=validation,
        mode=SemanticEnforcementMode.SEMANTICS_ABLATION,
    ) == "APPROVE"


def test_exact_evidence_is_converted_only_to_derived_statistical_float() -> None:
    trace = _trace(amount=0.000000001)
    derived = derive_statistical_cost_traces([trace], [_exact_request(amount="0.00000000123")])
    assert derived[0].estimated_cost_usd == pytest.approx(0.00000000123)
    assert _exact_request(amount="0.00000000123").total_decimal == "0.00000000123"


def test_manifest_is_deterministic_distinguishes_private_refs_and_detects_tamper(
    tmp_path: Path,
) -> None:
    first = tmp_path / "b.json"
    second = tmp_path / "a.json"
    first.write_text("b\n", encoding="utf-8")
    second.write_text("a\n", encoding="utf-8")
    entries = [
        committed_manifest_entry(root=tmp_path, path=first, role="b"),
        private_reference_manifest_entry(
            logical_path="private/raw.json",
            role="private_raw_provider_evidence",
            sha256_hex="c" * 64,
            byte_size=123,
        ),
        committed_manifest_entry(root=tmp_path, path=second, role="a"),
    ]
    manifest = build_manifest(entries)
    assert [entry.logical_path for entry in manifest.entries] == [
        "a.json",
        "b.json",
        "private/raw.json",
    ]
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest, manifest_path)
    assert "generated_at" not in manifest_path.read_text(encoding="utf-8")
    verification = verify_manifest(manifest, root=tmp_path)
    assert verification.valid is True
    assert verification.checked_local_count == 2
    assert verification.private_reference_count == 1

    first.write_text("tampered\n", encoding="utf-8")
    verification = verify_manifest(manifest, root=tmp_path)
    assert verification.valid is False
    assert any("mismatch" in error for error in verification.errors)


def test_study_summary_keeps_approval_errors_abstention_and_coverage_separate() -> None:
    records = [
        WorkflowDecisionRecord(
            decision_unit_id="1",
            workflow=StudyWorkflow.FULL_INFERENCELEDGER,
            decision=StudyDecision.APPROVE,
            reference_outcome=ReferenceOutcome.SUPPORTED,
            evidence_parity_satisfied=True,
            evidence_parity_note="same attempt, quality, latency and cost exports",
            integration_effort_minutes=1,
            evidence_collection_effort_minutes=2,
            review_effort_minutes=3,
            decision_delay_minutes=4,
            manual_semantic_reconciliations=0,
            missing_or_incomparable_surfaces_found=1,
        ),
        WorkflowDecisionRecord(
            decision_unit_id="2",
            workflow=StudyWorkflow.FULL_INFERENCELEDGER,
            decision=StudyDecision.ABSTAIN,
            reference_outcome=ReferenceOutcome.UNSUPPORTED_HARM,
            evidence_parity_satisfied=True,
            evidence_parity_note="same evidence",
            integration_effort_minutes=1,
            evidence_collection_effort_minutes=1,
            review_effort_minutes=1,
            decision_delay_minutes=1,
            manual_semantic_reconciliations=1,
            missing_or_incomparable_surfaces_found=2,
        ),
    ]
    summary = summarize_workflow_outcomes(records, workflow=StudyWorkflow.FULL_INFERENCELEDGER)
    assert summary.correct_approvals == 1
    assert summary.harmful_approvals == 0
    assert summary.abstentions == 1
    assert summary.decision_coverage == 0.5
    assert summary.review_effort_minutes == 4


def test_contract_rejects_unapproved_external_owner() -> None:
    with pytest.raises(ValueError, match="consent"):
        ExternalOwnerRecord(
            organization_id_or_pseudonym="org",
            accountable_role="owner",
            owns_or_approves_decision=True,
            consent_confirmed=False,
            requirements_approved_before_execution=True,
        )
