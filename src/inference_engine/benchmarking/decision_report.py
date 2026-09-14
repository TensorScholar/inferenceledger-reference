from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..domain.models.execution import CostEvidenceKind
from ..infrastructure.telemetry.request_log import RequestTrace
from .change_gate import ChangeGatePolicy, ChangeGateResult
from .harness import BenchmarkReport
from .paired_comparison import PairedRunStatisticalEvidence, PairingAudit
from .reconciliation import RunCostReconciliation
from .segmentation import SegmentEvidenceSummary


def build_decision_payload(
    *,
    experiment: Mapping[str, Any],
    policy: ChangeGatePolicy,
    pairing_audit: PairingAudit,
    evidence: PairedRunStatisticalEvidence,
    gate: ChangeGateResult,
    baseline_report: BenchmarkReport,
    candidate_report: BenchmarkReport,
    baseline_segments: SegmentEvidenceSummary,
    candidate_segments: SegmentEvidenceSummary,
    baseline_traces: Sequence[RequestTrace],
    candidate_traces: Sequence[RequestTrace],
    baseline_reconciliation: RunCostReconciliation,
    candidate_reconciliation: RunCostReconciliation,
    runtime: Mapping[str, Any],
    experiment_provenance: Mapping[str, Any] | None = None,
    pre_registration: Mapping[str, Any] | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Assemble a reconstructable baseline -> candidate decision artifact."""
    payload: dict[str, Any] = {
        "experiment_id": experiment.get("experiment_id"),
        "generated_at_utc": generated_at_utc or datetime.now(tz=UTC).isoformat(),
        "classification": experiment.get("classification"),
        "workload": experiment.get("workload"),
        "baseline_definition": experiment.get("baseline"),
        "candidate_definition": experiment.get("candidate"),
        "expected_change": experiment.get("expected_change"),
        "runtime": dict(runtime),
        "quality_evaluator": experiment.get("quality_evaluator"),
        "execution_counts": {
            "baseline_requests": baseline_report.request_count,
            "candidate_requests": candidate_report.request_count,
            "baseline_successes": baseline_report.success_count,
            "candidate_successes": candidate_report.success_count,
            "baseline_failures": baseline_report.failure_count,
            "candidate_failures": candidate_report.failure_count,
            "baseline_attempts": baseline_report.provider_attempt_count,
            "candidate_attempts": candidate_report.provider_attempt_count,
            "baseline_retries": baseline_report.provider_retry_count,
            "candidate_retries": candidate_report.provider_retry_count,
        },
        "pairing_audit": asdict(pairing_audit),
        "segments": {
            "baseline": asdict(baseline_segments),
            "candidate": asdict(candidate_segments),
        },
        "quality": {
            "baseline_quality_count": baseline_report.quality_count,
            "candidate_quality_count": candidate_report.quality_count,
            "baseline_quality_pass_count": baseline_report.quality_pass_count,
            "candidate_quality_pass_count": candidate_report.quality_pass_count,
            "baseline_quality_pass_rate": baseline_report.quality_pass_rate,
            "candidate_quality_pass_rate": candidate_report.quality_pass_rate,
            "measures": _quality_measures(experiment),
        },
        "latency": {
            "baseline_p50_ms": baseline_report.latency_p50_ms,
            "candidate_p50_ms": candidate_report.latency_p50_ms,
            "baseline_p95_ms": baseline_report.latency_p95_ms,
            "candidate_p95_ms": candidate_report.latency_p95_ms,
            "p95_note": (
                "Observed nearest-rank percentiles only. This report does not emit inferential "
                "p95/p99 confidence intervals."
            ),
        },
        "cost_evidence": {
            "baseline": _cost_evidence_summary(baseline_traces),
            "candidate": _cost_evidence_summary(candidate_traces),
            "classification_rule": (
                "LiteLLM response_cost remains REPORTED_BY_EXECUTION_STACK. Local numeric zero "
                "is not provider-invoice evidence."
            ),
        },
        "reconciliation": {
            "baseline_missing_route_count": baseline_reconciliation.missing_route_count,
            "candidate_missing_route_count": candidate_reconciliation.missing_route_count,
            "baseline_comparable_request_count": baseline_reconciliation.comparable_request_count,
            "candidate_comparable_request_count": candidate_reconciliation.comparable_request_count,
            "note": (
                "External-stack executions have no InferenceLedger route decision. "
                "MISSING_ROUTE is expected and is not silently converted into a cost delta."
            ),
        },
        "statistical_evidence": asdict(evidence),
        "change_gate_policy": asdict(policy),
        "change_gate_result": asdict(gate),
        "final_decision": gate.decision.value,
        "limitations": list(gate.limitations),
        "unsupported_claims": _unsupported_claims(experiment),
    }
    if experiment_provenance is not None:
        payload["experiment_provenance"] = dict(experiment_provenance)
    if pre_registration is not None:
        payload["pre_registration"] = dict(pre_registration)
    return payload


def _quality_measures(experiment: Mapping[str, Any]) -> str:
    evaluator = experiment.get("quality_evaluator")
    if isinstance(evaluator, Mapping) and isinstance(evaluator.get("measures"), str):
        return str(evaluator["measures"])
    return "deterministic workload-declared evaluation; not semantic quality"


def _unsupported_claims(experiment: Mapping[str, Any]) -> list[str]:
    claims = [str(item) for item in list(experiment.get("unsupported_claims_before_results") or [])]
    claims.extend(
        [
            "SHIP is not production readiness",
            "local reported zero cost is not invoice validation",
        ]
    )
    classification = experiment.get("classification")
    if classification == "SYNTHETIC":
        claims.append("synthetic workload is not a customer workload")
    else:
        claims.append("this workload is not a customer or production workload")
    return claims


def write_decision_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_decision_markdown(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_markdown(payload), encoding="utf-8")


def _cost_evidence_summary(traces: Sequence[RequestTrace]) -> dict[str, Any]:
    kinds: dict[str, int] = {}
    sources: dict[str, int] = {}
    unknown_count = 0
    for trace in traces:
        if not trace.provider_attempts:
            unknown_count += 1
            kinds[CostEvidenceKind.UNKNOWN.value] = kinds.get(CostEvidenceKind.UNKNOWN.value, 0) + 1
            continue
        for attempt in trace.provider_attempts:
            kind = attempt.cost_evidence.value
            kinds[kind] = kinds.get(kind, 0) + 1
            if attempt.cost_source:
                sources[attempt.cost_source] = sources.get(attempt.cost_source, 0) + 1
            if attempt.cost_evidence == CostEvidenceKind.UNKNOWN:
                unknown_count += 1
    complete = all(trace.cost_evidence_complete for trace in traces)
    total = (
        sum(trace.estimated_cost_usd or 0.0 for trace in traces)
        if complete and traces
        else None
    )
    return {
        "attempt_cost_kinds": dict(sorted(kinds.items())),
        "cost_sources": dict(sorted(sources.items())),
        "unknown_attempt_count": unknown_count,
        "request_cost_evidence_complete": complete,
        "reported_total_usd": total,
    }


def _render_markdown(payload: Mapping[str, Any]) -> str:
    experiment_id = payload.get("experiment_id")
    decision = str(payload.get("final_decision", "unknown")).upper()
    workload = _as_dict(payload.get("workload"))
    runtime = _as_dict(payload.get("runtime"))
    pairing = _as_dict(payload.get("pairing_audit"))
    counts = _as_dict(payload.get("execution_counts"))
    quality = _as_dict(payload.get("quality"))
    latency = _as_dict(payload.get("latency"))
    cost = _as_dict(payload.get("cost_evidence"))
    gate = _as_dict(payload.get("change_gate_result"))
    pilot = _as_dict(payload.get("pilot_evidence"))
    checks = _as_list(gate.get("checks"))
    limitations = _as_list(payload.get("limitations"))
    unsupported = _as_list(payload.get("unsupported_claims"))
    lines = [
        f"# Migration Decision `{experiment_id}`",
        "",
        f"**Final decision:** `{decision}`",
    ]
    if pilot:
        lines.extend(
            [
                f"**Study decision:** `{pilot.get('study_decision')}`",
                f"**Semantic enforcement mode:** `{pilot.get('semantic_enforcement_mode')}`",
            ]
        )
    lines.extend(
        [
            "",
            "This artifact reconstructs one baseline → candidate Change Gate decision from paired",
            "workload items, imported executions, and attempt-level cost evidence.",
            "",
            "## Identity",
            "",
            f"- Generated at (UTC): `{payload.get('generated_at_utc')}`",
            f"- Classification: `{payload.get('classification')}`",
            f"- Workload: `{workload.get('path')}`",
            f"- Workload SHA-256: `{workload.get('sha256')}`",
            f"- Workload items: `{workload.get('item_count')}`",
            "",
            "## Runtime",
            "",
        ]
    )
    for key, value in runtime.items():
        lines.append(f"- {key}: `{_format_runtime_value(value)}`")
    pre_registration = _as_dict(payload.get("pre_registration"))
    if pre_registration:
        lines.extend(
            [
                "",
                "## Pre-registration",
                "",
                f"- Declared status: `{pre_registration.get('declared_status')}`",
                f"- Declared locked_at_utc: `{pre_registration.get('declared_locked_at_utc')}`",
                f"- Git-proven before execution: `{pre_registration.get('git_proven_before_execution')}`",
                f"- Evidence class: `{pre_registration.get('evidence_class')}`",
                f"- Reason: {pre_registration.get('reason')}",
            ]
        )
    if pilot:
        acquisition = _as_dict(pilot.get("acquisition"))
        baseline_acquisition = _as_dict(acquisition.get("baseline"))
        candidate_acquisition = _as_dict(acquisition.get("candidate"))
        money_policy = _as_dict(pilot.get("exact_money_policy"))
        money_summary = _as_dict(pilot.get("exact_money_summary"))
        baseline_money = _as_dict(money_summary.get("baseline"))
        candidate_money = _as_dict(money_summary.get("candidate"))
        semantic_validation = _as_dict(pilot.get("semantic_validation"))
        violations = _as_sequence(semantic_validation.get("violations"))
        lines.extend(
            [
                "",
                "## Pilot evidence contract",
                "",
                f"- Contract version: `{pilot.get('contract_version')}`",
                f"- Protocol: `{pilot.get('protocol_id')}`",
                f"- Study decision: `{pilot.get('study_decision')}`",
                f"- Semantic enforcement mode: `{pilot.get('semantic_enforcement_mode')}`",
                f"- Full-semantics evidence valid: `{semantic_validation.get('valid_for_full_semantics')}`",
                "- Baseline acquisition: "
                f"measurement=`{baseline_acquisition.get('measurement')}`, "
                f"collection=`{baseline_acquisition.get('collection')}`, "
                f"assignment=`{baseline_acquisition.get('assignment')}`",
                "- Candidate acquisition: "
                f"measurement=`{candidate_acquisition.get('measurement')}`, "
                f"collection=`{candidate_acquisition.get('collection')}`, "
                f"assignment=`{candidate_acquisition.get('assignment')}`",
                "- Exact money policy: "
                f"currency=`{money_policy.get('currency')}`, "
                f"representation=`{money_policy.get('representation')}`, "
                f"rate_semantics=`{money_policy.get('rate_semantics')}`, "
                f"rounding=`{money_policy.get('rounding')}`, "
                f"aggregation=`{money_policy.get('aggregation')}`, "
                f"equality=`{money_policy.get('equality')}`",
                "- Baseline exact money: "
                f"complete=`{baseline_money.get('exact_cost_complete')}`, "
                f"total=`{baseline_money.get('exact_total_decimal')}`",
                "- Candidate exact money: "
                f"complete=`{candidate_money.get('exact_cost_complete')}`, "
                f"total=`{candidate_money.get('exact_total_decimal')}`",
            ]
        )
        if violations:
            lines.append("- Semantic violations:")
            for violation in violations:
                lines.append(f"  - {violation}")
    baseline_def = payload.get("baseline_definition")
    candidate_def = payload.get("candidate_definition")
    lines.extend(
        [
            "",
            "## Baseline",
            "",
            _definition_block(baseline_def),
            "",
            "## Candidate",
            "",
            _definition_block(candidate_def),
            "",
            "## Execution counts",
            "",
            f"- Baseline requests: {counts.get('baseline_requests')}",
            f"- Candidate requests: {counts.get('candidate_requests')}",
            f"- Baseline successes/failures: {counts.get('baseline_successes')}/"
            f"{counts.get('baseline_failures')}",
            f"- Candidate successes/failures: {counts.get('candidate_successes')}/"
            f"{counts.get('candidate_failures')}",
            f"- Baseline attempts/retries: {counts.get('baseline_attempts')}/"
            f"{counts.get('baseline_retries')}",
            f"- Candidate attempts/retries: {counts.get('candidate_attempts')}/"
            f"{counts.get('candidate_retries')}",
            "",
            "## Pairing audit",
            "",
            "- Pairing key: `workload_item_id`",
            f"- Baseline count: {pairing.get('baseline_count')}",
            f"- Candidate count: {pairing.get('candidate_count')}",
            f"- Matched pairs: {pairing.get('matched_pair_count')}",
            f"- Coverage: {pairing.get('coverage')}",
            f"- Unmatched baseline ids: `{pairing.get('unmatched_baseline_item_ids')}`",
            f"- Unmatched candidate ids: `{pairing.get('unmatched_candidate_item_ids')}`",
            f"- Duplicate ids present: "
            f"`{(pairing.get('duplicate_baseline_item_ids') or pairing.get('duplicate_candidate_item_ids'))}`",
            f"- Ambiguous: `{pairing.get('ambiguous')}`",
            "",
            "## Quality",
            "",
            f"- Evaluator: `{quality.get('measures')}`",
            f"- Baseline pass rate: {quality.get('baseline_quality_pass_rate')} "
            f"({quality.get('baseline_quality_pass_count')}/"
            f"{quality.get('baseline_quality_count')})",
            f"- Candidate pass rate: {quality.get('candidate_quality_pass_rate')} "
            f"({quality.get('candidate_quality_pass_count')}/"
            f"{quality.get('candidate_quality_count')})",
            "",
            "## Latency",
            "",
            f"- Baseline p50/p95: {latency.get('baseline_p50_ms')} / {latency.get('baseline_p95_ms')} ms",
            f"- Candidate p50/p95: {latency.get('candidate_p50_ms')} / {latency.get('candidate_p95_ms')} ms",
            f"- Note: {latency.get('p95_note')}",
            "",
            "## Cost evidence",
            "",
            f"- Baseline: `{cost.get('baseline')}`",
            f"- Candidate: `{cost.get('candidate')}`",
            f"- Rule: {cost.get('classification_rule')}",
            "",
            "## Change Gate",
            "",
            f"- Decision: `{decision}`",
            f"- Pass/fail/review/inconclusive: "
            f"{gate.get('pass_count')}/{gate.get('fail_count')}/"
            f"{gate.get('review_count')}/{gate.get('inconclusive_count')}",
            "",
        ]
    )
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        lines.append(
            f"- `{check.get('check_id')}` [{check.get('status')}] {check.get('reason')} "
            f"(observed={check.get('observed_value')}, bound={check.get('confidence_bound')}, "
            f"threshold={check.get('threshold')})"
        )
    lines.extend(["", "## Limitations", ""])
    for item in limitations:
        lines.append(f"- {item}")
    lines.extend(["", "## Unsupported claims", ""])
    for item in unsupported:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _as_dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_sequence(value: object) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _format_runtime_value(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _definition_block(raw: object) -> str:
    if not isinstance(raw, Mapping):
        return "- unavailable"
    return "\n".join(
        [
            f"- Run id: `{raw.get('run_id')}`",
            f"- Definition: {raw.get('definition')}",
            f"- Rationale: {raw.get('rationale')}",
        ]
    )
