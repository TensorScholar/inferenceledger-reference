from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ....benchmarking.change_gate import (
    evaluate_change_gate,
    load_change_gate_policy,
    write_change_gate_result,
)
from ....benchmarking.context_store import SQLiteBenchmarkContextStore
from ....benchmarking.decision_report import (
    build_decision_payload,
    write_decision_json,
    write_decision_markdown,
)
from ....benchmarking.experiment_provenance import (
    ExperimentProvenanceError,
    classify_pre_registration,
    execution_interval_from_standard_logging_payloads,
    load_standard_logging_payloads,
    merge_execution_intervals,
    pre_registration_as_dict,
    require_committed_clean_experiment_spec,
    snapshot_as_dict,
    withdraw_invalid_captured_at,
)
from ....benchmarking.paired_comparison import (
    audit_workload_pairing,
    compare_paired_runs,
    write_paired_run_evidence,
    write_pairing_audit,
)
from ....benchmarking.pilot_contract import SemanticEnforcementMode
from ....benchmarking.pilot_decision import (
    augment_decision_payload,
    load_pilot_decision_evidence,
    prepare_pilot_bundle_identity,
    write_and_verify_pilot_manifest,
    write_pilot_evidence_snapshot,
)
from ....benchmarking.reconciliation import reconcile_run_costs
from ....benchmarking.segmentation import summarize_segments
from ....benchmarking.sqlite_ledger import SQLiteBenchmarkLedger
from ....benchmarking.statistics import PairedBootstrapConfig
from ....infrastructure.telemetry.request_log import RequestTrace


@dataclass(frozen=True)
class MigrationDecisionRequest:
    """Inputs to the authoritative stored-evidence migration decision use case."""

    experiment_path: Path
    sqlite_ledger_path: Path
    output_dir: Path
    runtime_json_path: Path | None = None
    semantic_mode: SemanticEnforcementMode = SemanticEnforcementMode.FULL_INFERENCELEDGER


@dataclass(frozen=True)
class MigrationDecisionResult:
    """Deterministic application result returned to thin CLI adapters."""

    exit_code: int
    message: str
    decision: str | None = None
    study_decision: str | None = None
    matched_pairs: int | None = None
    coverage: float | None = None


def decide_stored_change(request: MigrationDecisionRequest) -> MigrationDecisionResult:
    """Build one baseline→candidate decision bundle from canonical stored run evidence.

    This is the pilot-authoritative comparison/report owner. It deliberately consumes stored run
    evidence rather than executing providers, so execution adapters and external importers share
    the same pairing, statistical, segmentation, reconciliation, gate, provenance, and report
    semantics after their evidence crosses the run-persistence boundary.
    """
    experiment_path = request.experiment_path
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if not isinstance(experiment, dict):
        raise ValueError("experiment file must contain a JSON object")

    try:
        snapshot = require_committed_clean_experiment_spec(experiment_path)
    except ExperimentProvenanceError as exc:
        return MigrationDecisionResult(
            exit_code=2,
            message=f"experiment_spec_provenance_error={exc}",
        )

    policy = load_change_gate_policy(experiment_path)
    output_dir = request.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ledger = SQLiteBenchmarkLedger(request.sqlite_ledger_path)
    contexts = SQLiteBenchmarkContextStore(request.sqlite_ledger_path)
    baseline_run_id = str(experiment["baseline"]["run_id"])
    candidate_run_id = str(experiment["candidate"]["run_id"])
    baseline_contexts = contexts.get_contexts(baseline_run_id)
    candidate_contexts = contexts.get_contexts(candidate_run_id)

    pairing_audit = audit_workload_pairing(
        baseline_contexts=baseline_contexts,
        candidate_contexts=candidate_contexts,
    )
    write_pairing_audit(pairing_audit, output_dir / "pairing-audit.json")
    if pairing_audit.ambiguous:
        return MigrationDecisionResult(
            exit_code=2,
            message=f"pairing_ambiguous=true reason={pairing_audit.fail_closed_reason}",
            matched_pairs=pairing_audit.matched_pair_count,
            coverage=pairing_audit.coverage,
        )

    baseline_report = ledger.get_report(baseline_run_id)
    candidate_report = ledger.get_report(candidate_run_id)
    baseline_traces = ledger.get_traces(baseline_run_id)
    candidate_traces = ledger.get_traces(candidate_run_id)

    try:
        pilot = load_pilot_decision_evidence(
            experiment=experiment,
            sqlite_ledger_path=request.sqlite_ledger_path,
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            baseline_traces=baseline_traces,
            candidate_traces=candidate_traces,
            mode=request.semantic_mode,
        )
    except ValueError as exc:
        return MigrationDecisionResult(
            exit_code=2,
            message=f"pilot_evidence_error={exc}",
            matched_pairs=pairing_audit.matched_pair_count,
            coverage=pairing_audit.coverage,
        )

    analysis_baseline_traces = (
        list(pilot.baseline_statistical_traces) if pilot is not None else baseline_traces
    )
    analysis_candidate_traces = (
        list(pilot.candidate_statistical_traces) if pilot is not None else candidate_traces
    )

    raw_statistics = experiment.get("statistics")
    statistics: dict[str, Any] = dict(raw_statistics) if isinstance(raw_statistics, dict) else {}
    evidence = compare_paired_runs(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        baseline_contexts=baseline_contexts,
        candidate_contexts=candidate_contexts,
        baseline_traces=analysis_baseline_traces,
        candidate_traces=analysis_candidate_traces,
        bootstrap_config=PairedBootstrapConfig(
            confidence_level=float(statistics.get("confidence_level", policy.confidence_level)),
            bootstrap_iterations=int(
                statistics.get("bootstrap_iterations", policy.minimum_bootstrap_iterations)
            ),
            minimum_samples=int(statistics.get("minimum_samples", 30)),
            seed=int(statistics.get("seed", 0)),
        ),
    )
    write_paired_run_evidence(evidence, output_dir / "paired-evidence.json")

    baseline_routes = ledger.get_routes(baseline_run_id)
    candidate_routes = ledger.get_routes(candidate_run_id)
    baseline_segments = summarize_segments(
        request_contexts=baseline_contexts,
        traces=analysis_baseline_traces,
        routes=baseline_routes,
    )
    candidate_segments = summarize_segments(
        request_contexts=candidate_contexts,
        traces=analysis_candidate_traces,
        routes=candidate_routes,
    )
    (output_dir / "baseline-segments.json").write_text(
        json.dumps(asdict(baseline_segments), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "candidate-segments.json").write_text(
        json.dumps(asdict(candidate_segments), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    gate = evaluate_change_gate(
        evidence=evidence,
        policy=policy,
        baseline_segments=baseline_segments,
        candidate_segments=candidate_segments,
        candidate_quality_pass_count=candidate_report.quality_pass_count,
        candidate_quality_trial_count=candidate_report.quality_count,
        candidate_request_count=candidate_report.request_count,
    )
    write_change_gate_result(gate, output_dir / "change-gate.json")

    runtime = _load_runtime(request.runtime_json_path)
    raw_workload = experiment.get("workload")
    workload: dict[str, Any] = dict(raw_workload) if isinstance(raw_workload, dict) else {}
    workload_sha256 = str(workload.get("sha256") or "")
    intervals = []
    for run_id in (baseline_run_id, candidate_run_id):
        payload_path = output_dir / f"{run_id}.sanitized-payloads.jsonl"
        if payload_path.is_file():
            intervals.append(
                execution_interval_from_standard_logging_payloads(
                    load_standard_logging_payloads(payload_path)
                )
            )

    earliest_start: str | None = None
    if intervals:
        merged_interval = merge_execution_intervals(intervals)
        earliest_start = merged_interval.earliest_start_utc
        runtime = {
            **runtime,
            "execution_interval": {
                "source": merged_interval.source,
                "start_field": merged_interval.start_field,
                "end_field": merged_interval.end_field,
                "earliest_start_utc": merged_interval.earliest_start_utc,
                "latest_end_utc": merged_interval.latest_end_utc,
                "payload_count": merged_interval.payload_count,
            },
        }
        interval_map = runtime.get("execution_interval")
        if isinstance(interval_map, dict):
            runtime = withdraw_invalid_captured_at(
                runtime,
                earliest_start_utc=str(interval_map["earliest_start_utc"]),
                latest_end_utc=str(interval_map["latest_end_utc"]),
                generated_at_utc=None,
            )

    try:
        pre_registration = classify_pre_registration(
            declared_status=str(experiment["status"]) if experiment.get("status") is not None else None,
            declared_locked_at_utc=(
                str(experiment["locked_at_utc"])
                if experiment.get("locked_at_utc") is not None
                else None
            ),
            snapshot=snapshot,
            workload_sha256=workload_sha256,
            earliest_start_utc=earliest_start,
            require_spec_precedes_execution=bool(intervals),
        )
    except ExperimentProvenanceError as exc:
        return MigrationDecisionResult(
            exit_code=2,
            message=f"experiment_spec_provenance_error={exc}",
            matched_pairs=pairing_audit.matched_pair_count,
            coverage=pairing_audit.coverage,
        )

    payload = build_decision_payload(
        experiment=experiment,
        policy=policy,
        pairing_audit=pairing_audit,
        evidence=evidence,
        gate=gate,
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        baseline_segments=baseline_segments,
        candidate_segments=candidate_segments,
        baseline_traces=analysis_baseline_traces,
        candidate_traces=analysis_candidate_traces,
        baseline_reconciliation=reconcile_run_costs(
            routes=baseline_routes,
            executions=analysis_baseline_traces,
        ),
        candidate_reconciliation=reconcile_run_costs(
            routes=candidate_routes,
            executions=analysis_candidate_traces,
        ),
        runtime=runtime,
        experiment_provenance=snapshot_as_dict(snapshot),
        pre_registration=pre_registration_as_dict(pre_registration),
    )
    _replace_cost_classification_rule(
        payload,
        baseline_traces=baseline_traces,
        candidate_traces=candidate_traces,
    )

    study_decision: str | None = None
    if pilot is not None:
        prepare_pilot_bundle_identity(
            contract=pilot.contract,
            experiment_path=experiment_path,
            output_dir=output_dir,
        )
        study_decision = augment_decision_payload(
            payload,
            pilot=pilot,
            gate_decision=gate.decision.value,
        )
        write_pilot_evidence_snapshot(pilot=pilot, output_dir=output_dir)

    write_decision_json(payload, output_dir / "decision.json")
    write_decision_markdown(payload, output_dir / "decision.md")

    if pilot is not None:
        try:
            write_and_verify_pilot_manifest(
                pilot=pilot,
                experiment_path=experiment_path,
                output_dir=output_dir,
                baseline_run_id=baseline_run_id,
                candidate_run_id=candidate_run_id,
            )
        except ValueError as exc:
            return MigrationDecisionResult(
                exit_code=2,
                message=f"pilot_manifest_error={exc}",
                decision=gate.decision.value,
                study_decision="ABSTAIN",
                matched_pairs=pairing_audit.matched_pair_count,
                coverage=pairing_audit.coverage,
            )

    message_parts = [
        f"decision={gate.decision.value}",
        f"matched_pairs={pairing_audit.matched_pair_count}",
        f"coverage={pairing_audit.coverage}",
        f"pass={gate.pass_count}",
        f"fail={gate.fail_count}",
        f"review={gate.review_count}",
        f"inconclusive={gate.inconclusive_count}",
        f"output_dir={output_dir}",
    ]
    if study_decision is not None:
        message_parts.insert(1, f"study_decision={study_decision}")
        message_parts.insert(2, f"semantic_mode={request.semantic_mode.value}")
    return MigrationDecisionResult(
        exit_code=0,
        message=" ".join(message_parts),
        decision=gate.decision.value,
        study_decision=study_decision,
        matched_pairs=pairing_audit.matched_pair_count,
        coverage=pairing_audit.coverage,
    )


def _load_runtime(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _replace_cost_classification_rule(
    payload: dict[str, Any],
    *,
    baseline_traces: list[RequestTrace],
    candidate_traces: list[RequestTrace],
) -> None:
    cost_evidence = payload.get("cost_evidence")
    if not isinstance(cost_evidence, dict):
        return
    kinds: set[str] = set()
    sources: set[str] = set()
    for trace in [*baseline_traces, *candidate_traces]:
        for attempt in trace.provider_attempts:
            kinds.add(attempt.cost_evidence.value)
            if attempt.cost_source:
                sources.add(attempt.cost_source)
    kind_text = ", ".join(sorted(kinds)) if kinds else "none"
    source_text = ", ".join(sorted(sources)) if sources else "none"
    cost_evidence["classification_rule"] = (
        f"Observed attempt cost-evidence kinds: {kind_text}; observed cost sources: {source_text}. "
        "REPORTED_BY_EXECUTION_STACK means the named execution stack reported the amount; "
        "CALCULATED_FROM_USAGE means reconstruction from observed usage plus identified pricing "
        "provenance; UNKNOWN remains unresolved. Pilot economic claims use the separate exact "
        "decimal-string evidence surface when a pilot contract is active. No attempt-cost kind is "
        "automatically provider-ledger or final-invoice truth."
    )
