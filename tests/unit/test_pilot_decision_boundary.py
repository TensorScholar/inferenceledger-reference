from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from inference_engine.benchmarking.decision_report import write_decision_markdown
from inference_engine.benchmarking.pilot_contract import load_optional_pilot_contract
from inference_engine.benchmarking.pilot_decision import _validate_frozen_workload_identity


def _workload_file(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "workload.jsonl"
    path.write_text(
        '{"id":"item-1","prompt":"hello","tags":{"segment":"critical"}}\n',
        encoding="utf-8",
    )
    return path, sha256(path.read_bytes()).hexdigest()


def _experiment(path: Path, digest: str, *, item_count: int = 1) -> dict[str, object]:
    pilot_contract = {
        "version": "pilot-evidence-contract-v1",
        "protocol_id": "comparative-external-pilot-v1",
        "external_owner": {
            "organization_id_or_pseudonym": "external-org",
            "accountable_role": "deployment owner",
            "owns_or_approves_decision": True,
            "consent_confirmed": True,
            "requirements_approved_before_execution": True,
        },
        "workload": {
            "source": "external workload",
            "selection_procedure": "prospective frozen sample",
            "version": "v1",
            "item_count": item_count,
            "sha256": digest,
            "retention": "COMMITTED_SANITIZED",
            "critical_segments": ["segment=critical"],
        },
        "arms": {
            "baseline": {
                "measurement": "OBSERVED",
                "collection": "CONTROLLED_REPLAY",
                "assignment": "CONTROLLED",
            },
            "candidate": {
                "measurement": "OBSERVED",
                "collection": "CONTROLLED_REPLAY",
                "assignment": "CONTROLLED",
            },
        },
        "economic": {
            "accounting_basis": "gateway-reported request charge",
            "currency": "USD",
            "representation": "DECIMAL_STRING",
            "rate_semantics": "NOT_APPLICABLE_REPORTED_CHARGE",
            "rounding": "NONE",
            "aggregation": "EXACT_SUM",
            "equality": "EXACT_DECIMAL",
            "legacy_float_role": "DERIVED_NON_AUTHORITATIVE",
        },
        "attempt_visibility": {
            "provider_invocations": "ATTEMPT_OR_EXPLICIT_UNKNOWN",
            "ancillary_billable_calls": "ATTEMPT_OR_EXPLICIT_UNKNOWN",
            "health_checks": "NON_INFERENCE_ONLY",
        },
        "evaluator": {
            "identity": "external-evaluator",
            "version": "v1",
            "measures": "task correctness",
            "population": "frozen workload",
            "positive_controls": [
                {"control_id": "good", "expected": "PASS", "description": "known-good"},
                {"control_id": "bad", "expected": "FAIL", "description": "known-bad"},
            ],
            "calibration_status": "PASSED",
            "missing_outcome_policy": "FAIL_CLOSED",
            "independent_adjudication": "owner-approved adjudicator",
        },
        "pairing": {
            "key": "workload_item_id",
            "missing_outcome_policy": "FAIL_CLOSED",
            "duplicate_policy": "FAIL_CLOSED",
        },
        "comparator": {
            "workflow_id": "strong-conventional-v1",
            "parity_requirement": "MATERIAL_EQUIVALENCE_REQUIRED",
            "evidence_surfaces": ["attempts", "quality", "latency", "cost", "segments"],
        },
        "study_outputs": {
            "decision_mapping": {
                "SHIP": "APPROVE",
                "REVIEW": "ABSTAIN",
                "NO_GO": "REJECT",
                "INCONCLUSIVE": "ABSTAIN",
            },
            "endpoints": [
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
            ],
            "effort_time_unit": "minutes",
            "delay_start_event": "frozen evidence available",
            "delay_end_event": "workflow decision frozen",
        },
        "ablations": [
            "FULL_INFERENCELEDGER",
            "SEMANTICS_ABLATION",
            "STRONG_CONVENTIONAL",
        ],
        "artifacts": {
            "workload": "COMMITTED_SANITIZED",
            "raw_provider_evidence": "PRIVATE_RAW_HASH_REFERENCED",
            "sanitized_provider_evidence": "COMMITTED_SANITIZED",
            "private_raw_location": "approved-private-store",
            "manifest_version": "pilot-evidence-manifest-v1",
        },
    }
    return {
        "protocol_id": "comparative-external-pilot-v1",
        "workload": {
            "path": str(path),
            "sha256": digest,
            "item_count": item_count,
        },
        "pilot_contract": pilot_contract,
    }


def _contract(experiment: dict[str, object]):
    contract = load_optional_pilot_contract(experiment)
    assert contract is not None
    return contract


def test_decision_boundary_rejects_workload_sha_disagreement_with_contract(tmp_path: Path) -> None:
    path, digest = _workload_file(tmp_path)
    experiment = _experiment(path, digest)
    workload = experiment["workload"]
    assert isinstance(workload, dict)
    workload["sha256"] = "b" * 64

    with pytest.raises(ValueError, match="SHA-256 disagrees with frozen pilot contract"):
        _validate_frozen_workload_identity(
            experiment=experiment,
            contract=_contract(_experiment(path, digest)),
            baseline_traces=[],
            candidate_traces=[],
        )


def test_decision_boundary_rejects_workload_bytes_not_matching_frozen_hash(tmp_path: Path) -> None:
    path, _ = _workload_file(tmp_path)
    frozen_digest = "a" * 64
    experiment = _experiment(path, frozen_digest)

    with pytest.raises(ValueError, match="workload bytes disagree with frozen pilot contract"):
        _validate_frozen_workload_identity(
            experiment=experiment,
            contract=_contract(experiment),
            baseline_traces=[],
            candidate_traces=[],
        )


def test_decision_boundary_rejects_actual_workload_cardinality_mismatch(tmp_path: Path) -> None:
    path, digest = _workload_file(tmp_path)
    experiment = _experiment(path, digest, item_count=2)

    with pytest.raises(ValueError, match="workload cardinality disagrees"):
        _validate_frozen_workload_identity(
            experiment=experiment,
            contract=_contract(experiment),
            baseline_traces=[],
            candidate_traces=[],
        )


def test_markdown_surfaces_study_acquisition_and_exact_money_semantics(tmp_path: Path) -> None:
    path = tmp_path / "decision.md"
    payload = {
        "experiment_id": "pilot-1",
        "final_decision": "SHIP",
        "generated_at_utc": "2026-09-13T00:00:00+00:00",
        "classification": "CUSTOMER",
        "workload": {"path": "workload.jsonl", "sha256": "a" * 64, "item_count": 1},
        "runtime": {},
        "pairing_audit": {},
        "execution_counts": {},
        "quality": {},
        "latency": {},
        "cost_evidence": {},
        "change_gate_result": {"checks": []},
        "limitations": [],
        "unsupported_claims": [],
        "pilot_evidence": {
            "contract_version": "pilot-evidence-contract-v1",
            "protocol_id": "comparative-external-pilot-v1",
            "semantic_enforcement_mode": "FULL_INFERENCELEDGER",
            "study_decision": "APPROVE",
            "acquisition": {
                "baseline": {
                    "measurement": "OBSERVED",
                    "collection": "CONTROLLED_REPLAY",
                    "assignment": "CONTROLLED",
                },
                "candidate": {
                    "measurement": "OBSERVED",
                    "collection": "CONTROLLED_REPLAY",
                    "assignment": "CONTROLLED",
                },
            },
            "exact_money_policy": {
                "currency": "USD",
                "representation": "DECIMAL_STRING",
                "rate_semantics": "NOT_APPLICABLE_REPORTED_CHARGE",
                "rounding": "NONE",
                "aggregation": "EXACT_SUM",
                "equality": "EXACT_DECIMAL",
            },
            "exact_money_summary": {
                "baseline": {"exact_cost_complete": True, "exact_total_decimal": "0.001"},
                "candidate": {"exact_cost_complete": True, "exact_total_decimal": "0.0008"},
            },
            "semantic_validation": {
                "valid_for_full_semantics": True,
                "violations": (),
            },
        },
    }

    write_decision_markdown(payload, path)
    rendered = path.read_text(encoding="utf-8")
    assert "**Study decision:** `APPROVE`" in rendered
    assert "## Pilot evidence contract" in rendered
    assert "measurement=`OBSERVED`" in rendered
    assert "representation=`DECIMAL_STRING`" in rendered
    assert "total=`0.0008`" in rendered
