from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..domain.models.economics import ExactRequestCostEvidence
from ..infrastructure.telemetry.request_log import RequestTrace
from .evidence_manifest import (
    EvidenceManifestEntry,
    build_manifest,
    committed_manifest_entry,
    private_reference_manifest_entry,
    verify_manifest,
    write_manifest,
)
from .harness import load_workload
from .pilot_contract import (
    ArtifactRetention,
    PilotInstanceContract,
    SemanticEnforcementMode,
    load_optional_pilot_contract,
)
from .pilot_evidence_store import SQLitePilotEvidenceStore
from .pilot_semantics import (
    PilotSemanticValidation,
    derive_statistical_cost_traces,
    study_decision_for_gate,
    validate_pilot_semantics,
)


@dataclass(frozen=True)
class LoadedPilotDecisionEvidence:
    contract: PilotInstanceContract
    mode: SemanticEnforcementMode
    baseline_exact_costs: tuple[ExactRequestCostEvidence, ...]
    candidate_exact_costs: tuple[ExactRequestCostEvidence, ...]
    validation: PilotSemanticValidation
    baseline_statistical_traces: tuple[RequestTrace, ...]
    candidate_statistical_traces: tuple[RequestTrace, ...]


def load_pilot_decision_evidence(
    *,
    experiment: Mapping[str, Any],
    sqlite_ledger_path: Path,
    baseline_run_id: str,
    candidate_run_id: str,
    baseline_traces: Sequence[RequestTrace],
    candidate_traces: Sequence[RequestTrace],
    mode: SemanticEnforcementMode,
) -> LoadedPilotDecisionEvidence | None:
    contract = load_optional_pilot_contract(experiment)
    if contract is None:
        return None

    _validate_frozen_workload_identity(
        experiment=experiment,
        contract=contract,
        baseline_traces=baseline_traces,
        candidate_traces=candidate_traces,
    )

    store = SQLitePilotEvidenceStore(sqlite_ledger_path)
    try:
        baseline_acquisition = store.get_acquisition(baseline_run_id)
        candidate_acquisition = store.get_acquisition(candidate_run_id)
        baseline_exact = tuple(store.get_exact_costs(baseline_run_id))
        candidate_exact = tuple(store.get_exact_costs(candidate_run_id))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"pilot evidence store is incomplete or invalid: {exc}") from exc

    validation = validate_pilot_semantics(
        contract=contract,
        baseline_acquisition=baseline_acquisition,
        candidate_acquisition=candidate_acquisition,
        baseline_traces=baseline_traces,
        candidate_traces=candidate_traces,
        baseline_exact_costs=baseline_exact,
        candidate_exact_costs=candidate_exact,
    )
    return LoadedPilotDecisionEvidence(
        contract=contract,
        mode=mode,
        baseline_exact_costs=baseline_exact,
        candidate_exact_costs=candidate_exact,
        validation=validation,
        baseline_statistical_traces=tuple(
            derive_statistical_cost_traces(baseline_traces, baseline_exact)
        ),
        candidate_statistical_traces=tuple(
            derive_statistical_cost_traces(candidate_traces, candidate_exact)
        ),
    )


def augment_decision_payload(
    payload: dict[str, Any],
    *,
    pilot: LoadedPilotDecisionEvidence,
    gate_decision: str,
) -> str:
    study_decision = study_decision_for_gate(
        gate_decision=gate_decision,
        contract=pilot.contract,
        validation=pilot.validation,
        mode=pilot.mode,
    )
    payload["pilot_evidence"] = {
        "contract_version": pilot.contract.version,
        "protocol_id": pilot.contract.protocol_id,
        "semantic_enforcement_mode": pilot.mode.value,
        "study_decision": study_decision,
        "acquisition": {
            "baseline": asdict(pilot.validation.baseline.acquisition),
            "candidate": asdict(pilot.validation.candidate.acquisition),
        },
        "exact_money_policy": asdict(pilot.contract.economic),
        "exact_money_summary": {
            "baseline": pilot.validation.baseline.as_dict(),
            "candidate": pilot.validation.candidate.as_dict(),
        },
        "semantic_validation": pilot.validation.as_dict(),
        "evaluator": asdict(pilot.contract.evaluator),
        "pairing_policy": asdict(pilot.contract.pairing),
        "comparator_evidence_parity_plan": asdict(pilot.contract.comparator),
        "study_output_contract": asdict(pilot.contract.study_outputs),
        "ablations": list(pilot.contract.ablations),
        "attempt_visibility": asdict(pilot.contract.attempt_visibility),
        "artifact_policy": asdict(pilot.contract.artifacts),
    }
    payload["study_decision"] = study_decision
    payload["evidence_manifest"] = {
        "version": pilot.contract.artifacts.manifest_version,
        "path": "evidence-manifest.json",
    }
    if not pilot.validation.valid_for_full_semantics:
        limitations = list(payload.get("limitations") or [])
        limitations.extend(
            f"pilot semantic validation: {violation}" for violation in pilot.validation.violations
        )
        payload["limitations"] = limitations
    return study_decision


def write_pilot_evidence_snapshot(
    *,
    pilot: LoadedPilotDecisionEvidence,
    output_dir: Path,
) -> Path:
    path = output_dir / "pilot-evidence.json"
    payload = {
        "contract_version": pilot.contract.version,
        "semantic_enforcement_mode": pilot.mode.value,
        "semantic_validation": pilot.validation.as_dict(),
        "baseline_exact_request_costs": [asdict(item) for item in pilot.baseline_exact_costs],
        "candidate_exact_request_costs": [asdict(item) for item in pilot.candidate_exact_costs],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def prepare_pilot_bundle_identity(
    *,
    contract: PilotInstanceContract,
    experiment_path: Path,
    output_dir: Path,
) -> tuple[Path, Path | None]:
    """Freeze the exact pilot-instance bytes and, when allowed, workload bytes in the bundle."""
    instance_path = output_dir / "pilot-instance.json"
    instance_path.write_bytes(experiment_path.read_bytes())
    raw = json.loads(experiment_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("pilot experiment must contain a JSON object")
    workload = raw.get("workload")
    if not isinstance(workload, dict) or not isinstance(workload.get("path"), str):
        raise ValueError("pilot experiment workload.path is required")
    workload_path = Path(str(workload["path"]))
    if contract.artifacts.workload == ArtifactRetention.COMMITTED_SANITIZED:
        bundle_workload = output_dir / "workload.jsonl"
        shutil.copyfile(workload_path, bundle_workload)
        return instance_path, bundle_workload
    return instance_path, None


def write_and_verify_pilot_manifest(
    *,
    pilot: LoadedPilotDecisionEvidence,
    experiment_path: Path,
    output_dir: Path,
    baseline_run_id: str,
    candidate_run_id: str,
) -> Path:
    entries: list[EvidenceManifestEntry] = []
    required_local = {
        "pilot-instance.json": "pilot_instance_spec",
        "pilot-evidence.json": "pilot_semantic_and_exact_evidence",
        "pairing-audit.json": "pairing_audit",
        "paired-evidence.json": "paired_statistical_evidence",
        "baseline-segments.json": "baseline_segment_evidence",
        "candidate-segments.json": "candidate_segment_evidence",
        "change-gate.json": "change_gate",
        "decision.json": "canonical_decision_json",
        "decision.md": "canonical_decision_markdown",
    }
    for run_id in (baseline_run_id, candidate_run_id):
        required_local.update(
            {
                f"{run_id}.report.json": f"{run_id}_run_report",
                f"{run_id}.segments.json": f"{run_id}_segment_report",
                f"{run_id}.sanitized-payloads.jsonl": f"{run_id}_sanitized_provider_evidence",
                f"{run_id}.raw-payload-sha256.json": f"{run_id}_private_raw_hash_index",
                f"{run_id}.provenance.json": f"{run_id}_execution_provenance",
            }
        )

    for relative, role in sorted(required_local.items()):
        path = output_dir / relative
        if not path.is_file():
            raise ValueError(f"required pilot bundle artifact is missing: {relative}")
        entries.append(committed_manifest_entry(root=output_dir, path=path, role=role))

    workload_path = _workload_path_from_experiment(experiment_path)
    if pilot.contract.artifacts.workload == ArtifactRetention.COMMITTED_SANITIZED:
        bundle_workload = output_dir / "workload.jsonl"
        if not bundle_workload.is_file():
            raise ValueError("committed/sanitized pilot workload copy is missing")
        entries.append(
            committed_manifest_entry(
                root=output_dir,
                path=bundle_workload,
                role="frozen_workload",
            )
        )
    elif pilot.contract.artifacts.workload == ArtifactRetention.PRIVATE_RAW_HASH_REFERENCED:
        data = workload_path.read_bytes()
        entries.append(
            private_reference_manifest_entry(
                logical_path=f"private_workload/{workload_path.name}",
                role="frozen_workload",
                sha256_hex=pilot.contract.workload.sha256,
                byte_size=len(data),
            )
        )

    for run_id in (baseline_run_id, candidate_run_id):
        entries.extend(_private_raw_entries(output_dir / f"{run_id}.raw-payload-sha256.json"))

    manifest = build_manifest(entries)
    path = output_dir / "evidence-manifest.json"
    write_manifest(manifest, path)
    verification = verify_manifest(manifest, root=output_dir)
    if not verification.valid:
        raise ValueError(
            "pilot evidence manifest verification failed: " + "; ".join(verification.errors)
        )
    return path


def _validate_frozen_workload_identity(
    *,
    experiment: Mapping[str, Any],
    contract: PilotInstanceContract,
    baseline_traces: Sequence[RequestTrace],
    candidate_traces: Sequence[RequestTrace],
) -> None:
    workload = experiment.get("workload")
    if not isinstance(workload, Mapping):
        raise ValueError("pilot experiment workload must be an object")

    path_value = workload.get("path")
    sha_value = workload.get("sha256")
    item_count_value = workload.get("item_count")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("pilot experiment workload.path is required")
    if not isinstance(sha_value, str) or not sha_value.strip():
        raise ValueError("pilot experiment workload.sha256 is required")
    if isinstance(item_count_value, bool) or not isinstance(item_count_value, int):
        raise ValueError("pilot experiment workload.item_count must be an integer")

    if sha_value.strip().lower() != contract.workload.sha256:
        raise ValueError("pilot experiment workload SHA-256 disagrees with frozen pilot contract")
    if item_count_value != contract.workload.item_count:
        raise ValueError("pilot experiment workload item_count disagrees with frozen pilot contract")

    workload_path = Path(path_value)
    data = workload_path.read_bytes()
    actual_sha = sha256(data).hexdigest()
    if actual_sha != contract.workload.sha256:
        raise ValueError("pilot workload bytes disagree with frozen pilot contract SHA-256")

    actual_items = load_workload(workload_path)
    if len(actual_items) != contract.workload.item_count:
        raise ValueError(
            "pilot workload cardinality disagrees with frozen pilot contract: "
            f"expected={contract.workload.item_count}, actual={len(actual_items)}"
        )

    for arm, traces in (("baseline", baseline_traces), ("candidate", candidate_traces)):
        if len(traces) != contract.workload.item_count:
            raise ValueError(
                f"pilot {arm} trace cardinality disagrees with frozen workload item_count: "
                f"expected={contract.workload.item_count}, actual={len(traces)}"
            )


def _private_raw_entries(hash_index_path: Path) -> list[EvidenceManifestEntry]:
    raw = json.loads(hash_index_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{hash_index_path.name} must contain an array")
    entries: list[EvidenceManifestEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"{hash_index_path.name} entries must be objects")
        logical = item.get("logical_path")
        digest = item.get("sha256")
        byte_size = item.get("byte_size")
        if not isinstance(logical, str) or not isinstance(digest, str):
            raise ValueError(f"{hash_index_path.name} lacks private raw logical_path/SHA-256")
        if isinstance(byte_size, bool) or not isinstance(byte_size, int):
            raise ValueError(f"{hash_index_path.name} lacks private raw byte_size")
        entries.append(
            private_reference_manifest_entry(
                logical_path=logical,
                role="private_raw_provider_evidence",
                sha256_hex=digest,
                byte_size=byte_size,
            )
        )
    return entries


def _workload_path_from_experiment(experiment_path: Path) -> Path:
    raw = json.loads(experiment_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("pilot experiment must contain a JSON object")
    workload = raw.get("workload")
    if not isinstance(workload, dict) or not isinstance(workload.get("path"), str):
        raise ValueError("pilot workload.path is required")
    path = Path(str(workload["path"]))
    data = path.read_bytes()
    if sha256(data).hexdigest() != str(workload.get("sha256")):
        raise ValueError("pilot workload bytes disagree with frozen experiment hash")
    return path
