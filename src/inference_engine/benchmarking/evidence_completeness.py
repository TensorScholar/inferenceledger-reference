"""Fail-closed representational completeness for migration-decision evidence bundles.

Completeness here means that required evidence surfaces are explicitly represented. It does
not mean that every surface is favorable, externally validated, or sufficient to authorize
SHIP. An explicitly unavailable surface satisfies the representation contract while remaining
unusable for the underlying substantive claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any, TypeGuard

CONTRACT_VERSION = "decision-evidence-completeness-v1"


class EvidencePresence(StrEnum):
    """Presence state for one required decision-evidence surface."""

    PRESENT = "present"
    EXPLICITLY_UNAVAILABLE = "explicitly_unavailable"
    MISSING = "missing"


@dataclass(frozen=True)
class CompletenessCheck:
    """One deterministic completeness check."""

    check_id: str
    status: EvidencePresence
    reason: str


@dataclass(frozen=True)
class DecisionEvidenceCompleteness:
    """Representational completeness of one baseline-to-candidate decision artifact."""

    contract_version: str
    contract_satisfied: bool
    present_count: int
    explicitly_unavailable_count: int
    missing_count: int
    missing_checks: tuple[str, ...]
    explicitly_unavailable_checks: tuple[str, ...]
    checks: tuple[CompletenessCheck, ...]
    interpretation: str


def assess_decision_evidence_completeness(
    payload: Mapping[str, Any],
) -> DecisionEvidenceCompleteness:
    """Assess whether required evidence surfaces are present or explicitly unavailable.

    The contract is satisfied when no required surface is silently missing. A surface may be
    ``EXPLICITLY_UNAVAILABLE`` and still satisfy this representation contract; callers must not
    interpret that state as substantive evidence for the associated claim.
    """
    checks = (
        _check_identity(payload),
        _check_workload_identity(payload),
        _check_run_definitions(payload),
        _check_runtime(payload),
        _check_execution_accounting(payload),
        _check_pairing(payload),
        _check_quality(payload),
        _check_cost_evidence(payload),
        _check_reconciliation(payload),
        _check_statistical_evidence(payload),
        _check_change_gate(payload),
        _check_experiment_provenance(payload),
        _check_pre_registration(payload),
        _check_claim_boundaries(payload),
    )
    missing = tuple(
        check.check_id for check in checks if check.status == EvidencePresence.MISSING
    )
    unavailable = tuple(
        check.check_id
        for check in checks
        if check.status == EvidencePresence.EXPLICITLY_UNAVAILABLE
    )
    present_count = sum(check.status == EvidencePresence.PRESENT for check in checks)
    return DecisionEvidenceCompleteness(
        contract_version=CONTRACT_VERSION,
        contract_satisfied=not missing,
        present_count=present_count,
        explicitly_unavailable_count=len(unavailable),
        missing_count=len(missing),
        missing_checks=missing,
        explicitly_unavailable_checks=unavailable,
        checks=checks,
        interpretation=(
            "Completeness is representational, not a validity or readiness score. "
            "EXPLICITLY_UNAVAILABLE is fail-closed evidence and cannot support the underlying "
            "claim."
        ),
    )


def _check_identity(payload: Mapping[str, Any]) -> CompletenessCheck:
    required = ("experiment_id", "classification", "final_decision")
    if all(_nonempty_string(payload.get(key)) for key in required):
        return _present("identity", "experiment, classification, and final decision are explicit")
    return _missing("identity", "experiment_id, classification, or final_decision is missing")


def _check_workload_identity(payload: Mapping[str, Any]) -> CompletenessCheck:
    workload = _mapping(payload.get("workload"))
    if workload is None:
        return _missing("workload_identity", "workload block is missing")
    if not _nonempty_string(workload.get("path")):
        return _missing("workload_identity", "workload path is missing")
    if not _nonempty_string(workload.get("sha256")):
        return _missing("workload_identity", "workload SHA-256 is missing")
    if not _positive_int(workload.get("item_count")):
        return _missing("workload_identity", "positive workload item_count is missing")
    return _present("workload_identity", "workload path, hash, and cardinality are explicit")


def _check_run_definitions(payload: Mapping[str, Any]) -> CompletenessCheck:
    baseline = _mapping(payload.get("baseline_definition"))
    candidate = _mapping(payload.get("candidate_definition"))
    if baseline is None or candidate is None:
        return _missing("run_definitions", "baseline or candidate definition is missing")
    if not _nonempty_string(baseline.get("run_id")) or not _nonempty_string(
        candidate.get("run_id")
    ):
        return _missing("run_definitions", "baseline or candidate run_id is missing")
    return _present("run_definitions", "baseline and candidate identities are explicit")


def _check_runtime(payload: Mapping[str, Any]) -> CompletenessCheck:
    runtime = _mapping(payload.get("runtime"))
    if runtime is None:
        return _missing("runtime", "runtime block is missing")
    if not runtime:
        return _unavailable("runtime", "runtime block is explicit but contains no runtime facts")
    return _present("runtime", "runtime facts are explicitly represented")


def _check_execution_accounting(payload: Mapping[str, Any]) -> CompletenessCheck:
    counts = _mapping(payload.get("execution_counts"))
    if counts is None:
        return _missing("execution_accounting", "execution_counts block is missing")
    keys = (
        "baseline_requests",
        "baseline_successes",
        "baseline_failures",
        "candidate_requests",
        "candidate_successes",
        "candidate_failures",
    )
    if not all(_nonnegative_int(counts.get(key)) for key in keys):
        return _missing("execution_accounting", "required execution counts are missing")
    baseline_requests = int(counts["baseline_requests"])
    candidate_requests = int(counts["candidate_requests"])
    baseline_outcomes = int(counts["baseline_successes"]) + int(counts["baseline_failures"])
    candidate_outcomes = int(counts["candidate_successes"]) + int(counts["candidate_failures"])
    if baseline_outcomes != baseline_requests:
        return _missing("execution_accounting", "baseline success/failure counts are inconsistent")
    if candidate_outcomes != candidate_requests:
        return _missing("execution_accounting", "candidate success/failure counts are inconsistent")
    return _present("execution_accounting", "request outcomes reconcile to run cardinalities")


def _check_pairing(payload: Mapping[str, Any]) -> CompletenessCheck:
    pairing = _mapping(payload.get("pairing_audit"))
    if pairing is None:
        return _missing("pairing", "pairing_audit block is missing")
    if not _nonnegative_int(pairing.get("matched_pair_count")):
        return _missing("pairing", "matched_pair_count is missing")
    coverage = pairing.get("coverage")
    if not _finite_number(coverage):
        return _missing("pairing", "pair coverage is missing or non-finite")
    numeric_coverage = float(coverage)
    if numeric_coverage < 0.0 or numeric_coverage > 1.0:
        return _missing("pairing", "pair coverage must be within [0, 1]")
    if not isinstance(pairing.get("ambiguous"), bool):
        return _missing("pairing", "pair ambiguity state is missing")
    if bool(pairing["ambiguous"]) or numeric_coverage < 1.0:
        return _unavailable(
            "pairing",
            "pairing is explicitly ambiguous or incomplete; paired claims must fail closed",
        )
    return _present("pairing", "pairing is explicit, unambiguous, and has full coverage")


def _check_quality(payload: Mapping[str, Any]) -> CompletenessCheck:
    quality = _mapping(payload.get("quality"))
    if quality is None:
        return _missing("quality", "quality block is missing")
    keys = (
        "baseline_quality_count",
        "baseline_quality_pass_count",
        "candidate_quality_count",
        "candidate_quality_pass_count",
    )
    if not all(_nonnegative_int(quality.get(key)) for key in keys):
        return _missing("quality", "quality trial/pass counts are missing")
    baseline_count = int(quality["baseline_quality_count"])
    candidate_count = int(quality["candidate_quality_count"])
    if int(quality["baseline_quality_pass_count"]) > baseline_count:
        return _missing("quality", "baseline quality pass count exceeds trial count")
    if int(quality["candidate_quality_pass_count"]) > candidate_count:
        return _missing("quality", "candidate quality pass count exceeds trial count")
    if baseline_count == 0 or candidate_count == 0:
        return _unavailable(
            "quality",
            "quality coverage is explicitly empty for baseline or candidate",
        )
    return _present("quality", "quality trial and pass counts are explicitly represented")


def _check_cost_evidence(payload: Mapping[str, Any]) -> CompletenessCheck:
    cost = _mapping(payload.get("cost_evidence"))
    if cost is None:
        return _missing("cost_evidence", "cost_evidence block is missing")
    baseline = _mapping(cost.get("baseline"))
    candidate = _mapping(cost.get("candidate"))
    if baseline is None or candidate is None:
        return _missing("cost_evidence", "baseline or candidate cost summary is missing")
    required = (
        "request_cost_evidence_complete",
        "unknown_attempt_count",
        "reported_total_usd",
    )
    if not all(key in baseline for key in required):
        return _missing("cost_evidence", "required baseline cost provenance fields are missing")
    if not all(key in candidate for key in required):
        return _missing("cost_evidence", "required candidate cost provenance fields are missing")

    for label, summary in (("baseline", baseline), ("candidate", candidate)):
        complete_value = summary["request_cost_evidence_complete"]
        unknown_attempt_count = summary["unknown_attempt_count"]
        total = summary["reported_total_usd"]
        if not isinstance(complete_value, bool):
            return _missing(
                "cost_evidence",
                f"{label} request_cost_evidence_complete must be a boolean",
            )
        if not _nonnegative_int(unknown_attempt_count):
            return _missing(
                "cost_evidence",
                f"{label} unknown_attempt_count must be a non-negative integer",
            )
        if complete_value and int(unknown_attempt_count) != 0:
            return _missing(
                "cost_evidence",
                f"{label} cost evidence is marked complete but has unknown attempts",
            )
        if complete_value and not _finite_nonnegative_number(total):
            return _missing(
                "cost_evidence",
                f"{label} complete cost evidence requires a finite non-negative total",
            )
        if not complete_value and total is not None and not _finite_nonnegative_number(total):
            return _missing(
                "cost_evidence",
                f"{label} reported total is malformed",
            )

    complete = bool(baseline["request_cost_evidence_complete"]) and bool(
        candidate["request_cost_evidence_complete"]
    )
    totals_present = (
        baseline["reported_total_usd"] is not None
        and candidate["reported_total_usd"] is not None
    )
    if not complete or not totals_present:
        return _unavailable(
            "cost_evidence",
            "unknown or incomplete request cost evidence is explicitly represented",
        )
    return _present("cost_evidence", "request cost evidence is explicit for both runs")


def _check_reconciliation(payload: Mapping[str, Any]) -> CompletenessCheck:
    reconciliation = _mapping(payload.get("reconciliation"))
    if reconciliation is None:
        return _missing("reconciliation", "reconciliation block is missing")
    required = (
        "baseline_missing_route_count",
        "candidate_missing_route_count",
        "baseline_comparable_request_count",
        "candidate_comparable_request_count",
    )
    if not all(_nonnegative_int(reconciliation.get(key)) for key in required):
        return _missing("reconciliation", "route reconciliation counts are missing")
    return _present(
        "reconciliation",
        "route/cost reconciliation coverage is explicitly represented, including zeros",
    )


def _check_statistical_evidence(payload: Mapping[str, Any]) -> CompletenessCheck:
    evidence = _mapping(payload.get("statistical_evidence"))
    if evidence is None:
        return _missing("statistical_evidence", "statistical_evidence block is missing")
    available = evidence.get("available")
    if not isinstance(available, bool):
        return _missing("statistical_evidence", "statistical evidence availability is missing")
    if available:
        return _present("statistical_evidence", "paired statistical evidence is available")
    if _nonempty_string(evidence.get("unavailable_reason")):
        return _unavailable(
            "statistical_evidence",
            "statistical evidence is explicitly unavailable with a reason",
        )
    return _missing(
        "statistical_evidence",
        "statistical evidence is unavailable but no fail-closed reason is recorded",
    )


def _check_change_gate(payload: Mapping[str, Any]) -> CompletenessCheck:
    gate = _mapping(payload.get("change_gate_result"))
    if gate is None:
        return _missing("change_gate", "change_gate_result block is missing")
    if not _nonempty_string(gate.get("decision")):
        return _missing("change_gate", "change-gate decision is missing")
    if not _sequence(gate.get("checks")):
        return _missing("change_gate", "change-gate checks are missing")
    final_decision = payload.get("final_decision")
    if str(gate["decision"]).lower() != str(final_decision).lower():
        return _missing("change_gate", "change-gate decision conflicts with final_decision")
    return _present("change_gate", "gate decision and check inventory are explicit and consistent")


def _check_experiment_provenance(payload: Mapping[str, Any]) -> CompletenessCheck:
    provenance = _mapping(payload.get("experiment_provenance"))
    if provenance is None:
        return _missing("experiment_provenance", "experiment_provenance block is missing")
    required = ("spec_sha256", "spec_commit_sha", "spec_committed_and_clean")
    if not all(key in provenance for key in required):
        return _missing("experiment_provenance", "spec provenance fields are missing")
    if not _nonempty_string(provenance.get("spec_sha256")):
        return _missing("experiment_provenance", "spec hash is empty")
    if not _nonempty_string(provenance.get("spec_commit_sha")):
        return _missing("experiment_provenance", "spec commit SHA is empty")
    committed_and_clean = provenance.get("spec_committed_and_clean")
    if not isinstance(committed_and_clean, bool):
        return _missing(
            "experiment_provenance",
            "spec_committed_and_clean must be a boolean",
        )
    if committed_and_clean:
        return _present("experiment_provenance", "committed spec provenance is explicit")
    return _unavailable(
        "experiment_provenance",
        "spec provenance is explicit but not committed-and-clean",
    )


def _check_pre_registration(payload: Mapping[str, Any]) -> CompletenessCheck:
    registration = _mapping(payload.get("pre_registration"))
    if registration is None:
        return _missing("pre_registration", "pre_registration block is missing")
    if not _nonempty_string(registration.get("evidence_class")):
        return _missing("pre_registration", "pre-registration evidence class is missing")
    if not _nonempty_string(registration.get("reason")):
        return _missing("pre_registration", "pre-registration reason is missing")
    evidence_class = str(registration["evidence_class"]).upper()
    git_proven = registration.get("git_proven_before_execution")
    if not isinstance(git_proven, bool):
        return _missing("pre_registration", "git_proven_before_execution is missing")
    if git_proven:
        if evidence_class != "GIT_PROVEN":
            return _missing(
                "pre_registration",
                "git_proven_before_execution conflicts with evidence_class",
            )
        return _present("pre_registration", "Git-proven pre-registration evidence is explicit")
    if evidence_class == "GIT_PROVEN":
        return _missing(
            "pre_registration",
            "GIT_PROVEN evidence_class conflicts with git_proven_before_execution=false",
        )
    return _unavailable(
        "pre_registration",
        "pre-registration is explicitly not Git-proven; no stronger chronology claim is allowed",
    )


def _check_claim_boundaries(payload: Mapping[str, Any]) -> CompletenessCheck:
    if "limitations" not in payload or "unsupported_claims" not in payload:
        return _missing("claim_boundaries", "limitations or unsupported_claims is missing")
    if not isinstance(payload.get("limitations"), list):
        return _missing("claim_boundaries", "limitations must be an explicit list")
    if not isinstance(payload.get("unsupported_claims"), list):
        return _missing("claim_boundaries", "unsupported_claims must be an explicit list")
    return _present("claim_boundaries", "limitations and unsupported claims are explicit")


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) > 0
    )


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _nonnegative_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_int(value: object) -> TypeGuard[int]:
    return _nonnegative_int(value) and value > 0


def _finite_number(value: object) -> TypeGuard[int | float]:
    return _number(value) and isfinite(float(value))


def _finite_nonnegative_number(value: object) -> TypeGuard[int | float]:
    return _finite_number(value) and float(value) >= 0.0


def _number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _present(check_id: str, reason: str) -> CompletenessCheck:
    return CompletenessCheck(
        check_id=check_id,
        status=EvidencePresence.PRESENT,
        reason=reason,
    )


def _unavailable(check_id: str, reason: str) -> CompletenessCheck:
    return CompletenessCheck(
        check_id=check_id,
        status=EvidencePresence.EXPLICITLY_UNAVAILABLE,
        reason=reason,
    )


def _missing(check_id: str, reason: str) -> CompletenessCheck:
    return CompletenessCheck(
        check_id=check_id,
        status=EvidencePresence.MISSING,
        reason=reason,
    )
