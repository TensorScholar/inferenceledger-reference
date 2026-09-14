from __future__ import annotations

from typing import Any

from inference_engine.benchmarking.evidence_completeness import (
    EvidencePresence,
    assess_decision_evidence_completeness,
)


def _complete_payload() -> dict[str, Any]:
    return {
        "experiment_id": "exp-1",
        "classification": "PUBLISHED_REFERENCE",
        "final_decision": "inconclusive",
        "workload": {
            "path": "benchmarks/workloads/example.jsonl",
            "sha256": "abc123",
            "item_count": 2,
        },
        "baseline_definition": {"run_id": "baseline"},
        "candidate_definition": {"run_id": "candidate"},
        "runtime": {"provider": "example"},
        "execution_counts": {
            "baseline_requests": 2,
            "baseline_successes": 2,
            "baseline_failures": 0,
            "candidate_requests": 2,
            "candidate_successes": 2,
            "candidate_failures": 0,
        },
        "pairing_audit": {
            "matched_pair_count": 2,
            "coverage": 1.0,
            "ambiguous": False,
        },
        "quality": {
            "baseline_quality_count": 2,
            "baseline_quality_pass_count": 1,
            "candidate_quality_count": 2,
            "candidate_quality_pass_count": 2,
        },
        "cost_evidence": {
            "baseline": {
                "request_cost_evidence_complete": True,
                "unknown_attempt_count": 0,
                "reported_total_usd": 0.01,
            },
            "candidate": {
                "request_cost_evidence_complete": True,
                "unknown_attempt_count": 0,
                "reported_total_usd": 0.009,
            },
        },
        "reconciliation": {
            "baseline_missing_route_count": 0,
            "candidate_missing_route_count": 0,
            "baseline_comparable_request_count": 2,
            "candidate_comparable_request_count": 2,
        },
        "statistical_evidence": {
            "available": True,
            "unavailable_reason": None,
        },
        "change_gate_result": {
            "decision": "inconclusive",
            "checks": [{"check_id": "overall:cost"}],
        },
        "experiment_provenance": {
            "spec_sha256": "def456",
            "spec_commit_sha": "0123456789abcdef",
            "spec_committed_and_clean": True,
        },
        "pre_registration": {
            "evidence_class": "GIT_PROVEN",
            "reason": "spec commit predates execution",
            "git_proven_before_execution": True,
        },
        "limitations": ["not production evidence"],
        "unsupported_claims": ["not invoice validation"],
    }


def _status_by_id(payload: dict[str, Any]) -> dict[str, EvidencePresence]:
    result = assess_decision_evidence_completeness(payload)
    return {check.check_id: check.status for check in result.checks}


def test_complete_payload_satisfies_representation_contract() -> None:
    result = assess_decision_evidence_completeness(_complete_payload())

    assert result.contract_satisfied is True
    assert result.missing_count == 0
    assert result.explicitly_unavailable_count == 0
    assert result.present_count == len(result.checks)


def test_explicitly_unavailable_evidence_is_fail_closed_but_not_missing() -> None:
    payload = _complete_payload()
    baseline_cost = payload["cost_evidence"]["baseline"]
    baseline_cost["request_cost_evidence_complete"] = False
    baseline_cost["reported_total_usd"] = None
    payload["pre_registration"] = {
        "evidence_class": "INSUFFICIENT_EVIDENCE",
        "reason": "historical spec was not Git-proven before execution",
        "git_proven_before_execution": False,
    }

    result = assess_decision_evidence_completeness(payload)

    assert result.contract_satisfied is True
    assert "cost_evidence" in result.explicitly_unavailable_checks
    assert "pre_registration" in result.explicitly_unavailable_checks
    assert result.missing_count == 0


def test_missing_workload_hash_breaks_contract() -> None:
    payload = _complete_payload()
    del payload["workload"]["sha256"]

    result = assess_decision_evidence_completeness(payload)

    assert result.contract_satisfied is False
    assert result.missing_checks == ("workload_identity",)


def test_ambiguous_pairing_is_explicitly_unavailable_not_silent_missing() -> None:
    payload = _complete_payload()
    payload["pairing_audit"]["ambiguous"] = True
    payload["pairing_audit"]["coverage"] = 0.5

    result = assess_decision_evidence_completeness(payload)
    statuses = _status_by_id(payload)

    assert result.contract_satisfied is True
    assert statuses["pairing"] == EvidencePresence.EXPLICITLY_UNAVAILABLE


def test_pairing_rejects_nonfinite_or_out_of_range_coverage() -> None:
    for invalid_coverage in (float("nan"), float("inf"), -0.1, 1.1):
        payload = _complete_payload()
        payload["pairing_audit"]["coverage"] = invalid_coverage

        result = assess_decision_evidence_completeness(payload)

        assert result.contract_satisfied is False
        assert "pairing" in result.missing_checks


def test_zero_quality_trials_are_explicitly_unavailable() -> None:
    payload = _complete_payload()
    payload["quality"].update(
        {
            "baseline_quality_count": 0,
            "baseline_quality_pass_count": 0,
            "candidate_quality_count": 0,
            "candidate_quality_pass_count": 0,
        }
    )

    statuses = _status_by_id(payload)

    assert statuses["quality"] == EvidencePresence.EXPLICITLY_UNAVAILABLE


def test_malformed_cost_completeness_boolean_breaks_contract() -> None:
    payload = _complete_payload()
    payload["cost_evidence"]["baseline"]["request_cost_evidence_complete"] = "false"

    result = assess_decision_evidence_completeness(payload)

    assert result.contract_satisfied is False
    assert "cost_evidence" in result.missing_checks


def test_complete_cost_cannot_contain_unknown_attempts() -> None:
    payload = _complete_payload()
    payload["cost_evidence"]["candidate"]["unknown_attempt_count"] = 1

    result = assess_decision_evidence_completeness(payload)

    assert result.contract_satisfied is False
    assert "cost_evidence" in result.missing_checks


def test_complete_cost_requires_finite_nonnegative_total() -> None:
    for invalid_total in (float("nan"), float("inf"), -0.001, "0.01"):
        payload = _complete_payload()
        payload["cost_evidence"]["candidate"]["reported_total_usd"] = invalid_total

        result = assess_decision_evidence_completeness(payload)

        assert result.contract_satisfied is False
        assert "cost_evidence" in result.missing_checks


def test_gate_decision_conflict_breaks_contract() -> None:
    payload = _complete_payload()
    payload["change_gate_result"]["decision"] = "ship"

    result = assess_decision_evidence_completeness(payload)

    assert result.contract_satisfied is False
    assert "change_gate" in result.missing_checks


def test_unavailable_statistics_require_an_explicit_reason() -> None:
    payload = _complete_payload()
    payload["statistical_evidence"] = {
        "available": False,
        "unavailable_reason": "pairing population is insufficient",
    }

    statuses = _status_by_id(payload)

    assert statuses["statistical_evidence"] == EvidencePresence.EXPLICITLY_UNAVAILABLE

    payload["statistical_evidence"]["unavailable_reason"] = None
    result = assess_decision_evidence_completeness(payload)

    assert result.contract_satisfied is False
    assert "statistical_evidence" in result.missing_checks
