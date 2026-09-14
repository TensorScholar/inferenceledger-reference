from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

PILOT_PROTOCOL_ID = "comparative-external-pilot-v1"
PILOT_EVIDENCE_CONTRACT_VERSION = "pilot-evidence-contract-v1"
PILOT_MANIFEST_VERSION = "pilot-evidence-manifest-v1"


class MeasurementSemantics(StrEnum):
    OBSERVED = "OBSERVED"
    ESTIMATED = "ESTIMATED"


class CollectionSemantics(StrEnum):
    CONTROLLED_REPLAY = "CONTROLLED_REPLAY"
    SHADOW = "SHADOW"
    LIVE = "LIVE"
    OFFLINE_IMPORT = "OFFLINE_IMPORT"


class AssignmentSemantics(StrEnum):
    CONTROLLED = "CONTROLLED"
    OBSERVATIONAL = "OBSERVATIONAL"


class SemanticEnforcementMode(StrEnum):
    FULL_INFERENCELEDGER = "FULL_INFERENCELEDGER"
    SEMANTICS_ABLATION = "SEMANTICS_ABLATION"


class ArtifactRetention(StrEnum):
    COMMITTED_SANITIZED = "COMMITTED_SANITIZED"
    PRIVATE_RAW_HASH_REFERENCED = "PRIVATE_RAW_HASH_REFERENCED"
    EXPLICITLY_UNAVAILABLE = "EXPLICITLY_UNAVAILABLE"


@dataclass(frozen=True)
class AcquisitionSemantics:
    measurement: MeasurementSemantics
    collection: CollectionSemantics
    assignment: AssignmentSemantics


@dataclass(frozen=True)
class ExternalOwnerRecord:
    organization_id_or_pseudonym: str
    accountable_role: str
    owns_or_approves_decision: bool
    consent_confirmed: bool
    requirements_approved_before_execution: bool

    def __post_init__(self) -> None:
        if not self.organization_id_or_pseudonym.strip():
            raise ValueError("pilot external owner requires organization id or approved pseudonym")
        if not self.accountable_role.strip():
            raise ValueError("pilot external owner requires an accountable role")
        if not self.owns_or_approves_decision:
            raise ValueError("pilot owner must own, approve, or be accountable for the decision")
        if not self.consent_confirmed:
            raise ValueError("pilot workload consent must be confirmed before execution")
        if not self.requirements_approved_before_execution:
            raise ValueError("pilot requirements must be approved before execution")


@dataclass(frozen=True)
class WorkloadFreezeRecord:
    source: str
    selection_procedure: str
    version: str
    item_count: int
    sha256: str
    retention: ArtifactRetention
    critical_segments: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.selection_procedure.strip() or not self.version.strip():
            raise ValueError("pilot workload source, selection procedure, and version are required")
        if self.item_count < 1:
            raise ValueError("pilot workload item_count must be positive")
        digest = self.sha256.strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("pilot workload sha256 must be a 64-character hex digest")
        object.__setattr__(self, "sha256", digest)
        if not self.critical_segments:
            raise ValueError("pilot workload requires at least one predeclared critical segment")
        if any(not item.strip() for item in self.critical_segments):
            raise ValueError("pilot critical segment identifiers must be non-empty")


@dataclass(frozen=True)
class ExactMoneySemantics:
    accounting_basis: str
    currency: str
    representation: str
    rate_semantics: str
    rounding: str
    aggregation: str
    equality: str
    legacy_float_role: str

    def __post_init__(self) -> None:
        if not self.accounting_basis.strip():
            raise ValueError("pilot economic accounting_basis is required")
        currency = self.currency.strip().upper()
        if len(currency) != 3 or not currency.isalpha():
            raise ValueError("pilot economic currency must be a three-letter code")
        object.__setattr__(self, "currency", currency)
        required = {
            "representation": (self.representation, "DECIMAL_STRING"),
            "rounding": (self.rounding, "NONE"),
            "aggregation": (self.aggregation, "EXACT_SUM"),
            "equality": (self.equality, "EXACT_DECIMAL"),
            "legacy_float_role": (self.legacy_float_role, "DERIVED_NON_AUTHORITATIVE"),
        }
        for field, (observed, expected) in required.items():
            if observed != expected:
                raise ValueError(f"pilot economic {field} must be {expected}")
        if self.rate_semantics not in {
            "EXACT_RATE_STRINGS",
            "NOT_APPLICABLE_REPORTED_CHARGE",
        }:
            raise ValueError(
                "pilot rate_semantics must be EXACT_RATE_STRINGS or NOT_APPLICABLE_REPORTED_CHARGE"
            )


@dataclass(frozen=True)
class AttemptVisibilityContract:
    provider_invocations: str
    ancillary_billable_calls: str
    health_checks: str

    def __post_init__(self) -> None:
        if self.provider_invocations != "ATTEMPT_OR_EXPLICIT_UNKNOWN":
            raise ValueError("provider invocations must be attempt-visible or explicitly unknown")
        if self.ancillary_billable_calls != "ATTEMPT_OR_EXPLICIT_UNKNOWN":
            raise ValueError("ancillary billable calls must be visible or explicitly unknown")
        if self.health_checks != "NON_INFERENCE_ONLY":
            raise ValueError("pilot health checks must be non-inference by construction")


@dataclass(frozen=True)
class PositiveControl:
    control_id: str
    expected: str
    description: str

    def __post_init__(self) -> None:
        if not self.control_id.strip() or not self.description.strip():
            raise ValueError("evaluator positive controls require id and description")
        if self.expected not in {"PASS", "FAIL"}:
            raise ValueError("evaluator positive-control expected result must be PASS or FAIL")


@dataclass(frozen=True)
class EvaluatorRecord:
    identity: str
    version: str
    measures: str
    population: str
    positive_controls: tuple[PositiveControl, ...]
    calibration_status: str
    missing_outcome_policy: str
    independent_adjudication: str

    def __post_init__(self) -> None:
        string_fields = (
            self.identity,
            self.version,
            self.measures,
            self.population,
            self.missing_outcome_policy,
            self.independent_adjudication,
        )
        if any(not value.strip() for value in string_fields):
            raise ValueError("pilot evaluator record fields must be non-empty")
        if self.calibration_status != "PASSED":
            raise ValueError("pilot evaluator positive-control calibration must be PASSED")
        expected_results = {control.expected for control in self.positive_controls}
        if expected_results != {"PASS", "FAIL"}:
            raise ValueError("pilot evaluator requires both known-good and known-bad positive controls")


@dataclass(frozen=True)
class PairingPolicy:
    key: str
    missing_outcome_policy: str
    duplicate_policy: str

    def __post_init__(self) -> None:
        if self.key != "workload_item_id":
            raise ValueError("pilot pairing key must be workload_item_id")
        if self.missing_outcome_policy != "FAIL_CLOSED":
            raise ValueError("pilot missing outcomes must fail closed in v1")
        if self.duplicate_policy != "FAIL_CLOSED":
            raise ValueError("pilot duplicate workload identities must fail closed in v1")


@dataclass(frozen=True)
class ComparatorParityPlan:
    workflow_id: str
    parity_requirement: str
    evidence_surfaces: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.workflow_id.strip():
            raise ValueError("pilot comparator workflow_id is required")
        if self.parity_requirement != "MATERIAL_EQUIVALENCE_REQUIRED":
            raise ValueError("pilot comparator requires materially equivalent evidence access")
        if not self.evidence_surfaces or any(not item.strip() for item in self.evidence_surfaces):
            raise ValueError("pilot comparator requires explicit evidence-parity surfaces")


_REQUIRED_STUDY_ENDPOINTS = {
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
}


@dataclass(frozen=True)
class StudyOutputContract:
    decision_mapping: tuple[tuple[str, str], ...]
    endpoints: tuple[str, ...]
    effort_time_unit: str
    delay_start_event: str
    delay_end_event: str

    def __post_init__(self) -> None:
        mapping = dict(self.decision_mapping)
        required_mapping = {
            "SHIP": "APPROVE",
            "REVIEW": "ABSTAIN",
            "NO_GO": "REJECT",
            "INCONCLUSIVE": "ABSTAIN",
        }
        if mapping != required_mapping:
            raise ValueError("pilot v1 decision mapping must preserve the frozen study vocabulary")
        if not _REQUIRED_STUDY_ENDPOINTS.issubset(set(self.endpoints)):
            missing = sorted(_REQUIRED_STUDY_ENDPOINTS - set(self.endpoints))
            raise ValueError(f"pilot study output contract is missing endpoints: {missing}")
        if self.effort_time_unit != "minutes":
            raise ValueError("pilot effort time unit must be minutes")
        if not self.delay_start_event.strip() or not self.delay_end_event.strip():
            raise ValueError("pilot decision-delay start/end events must be explicit")

    def map_gate_decision(self, gate_decision: str) -> str:
        try:
            return dict(self.decision_mapping)[gate_decision]
        except KeyError as exc:
            raise ValueError(f"unmapped Change Gate decision: {gate_decision}") from exc


@dataclass(frozen=True)
class ArtifactRetentionPolicy:
    workload: ArtifactRetention
    raw_provider_evidence: ArtifactRetention
    sanitized_provider_evidence: ArtifactRetention
    private_raw_location: str | None
    manifest_version: str

    def __post_init__(self) -> None:
        if self.raw_provider_evidence == ArtifactRetention.COMMITTED_SANITIZED:
            raise ValueError("raw provider evidence must not be mislabeled as sanitized committed data")
        if (
            self.raw_provider_evidence == ArtifactRetention.PRIVATE_RAW_HASH_REFERENCED
            and (self.private_raw_location is None or not self.private_raw_location.strip())
        ):
            raise ValueError("private raw evidence requires an approved private location")
        if self.sanitized_provider_evidence != ArtifactRetention.COMMITTED_SANITIZED:
            raise ValueError("pilot v1 requires sanitized provider evidence in the committed bundle")
        if self.manifest_version != PILOT_MANIFEST_VERSION:
            raise ValueError(f"pilot manifest version must be {PILOT_MANIFEST_VERSION}")


_REQUIRED_ABLATIONS = {
    "FULL_INFERENCELEDGER",
    "SEMANTICS_ABLATION",
    "STRONG_CONVENTIONAL",
}


@dataclass(frozen=True)
class PilotInstanceContract:
    version: str
    protocol_id: str
    external_owner: ExternalOwnerRecord
    workload: WorkloadFreezeRecord
    baseline_acquisition: AcquisitionSemantics
    candidate_acquisition: AcquisitionSemantics
    economic: ExactMoneySemantics
    attempt_visibility: AttemptVisibilityContract
    evaluator: EvaluatorRecord
    pairing: PairingPolicy
    comparator: ComparatorParityPlan
    study_outputs: StudyOutputContract
    ablations: tuple[str, ...]
    artifacts: ArtifactRetentionPolicy

    def __post_init__(self) -> None:
        if self.version != PILOT_EVIDENCE_CONTRACT_VERSION:
            raise ValueError(f"pilot contract version must be {PILOT_EVIDENCE_CONTRACT_VERSION}")
        if self.protocol_id != PILOT_PROTOCOL_ID:
            raise ValueError(f"pilot protocol_id must be {PILOT_PROTOCOL_ID}")
        if set(self.ablations) != _REQUIRED_ABLATIONS:
            raise ValueError(
                "pilot ablations must contain FULL_INFERENCELEDGER, SEMANTICS_ABLATION, and STRONG_CONVENTIONAL"
            )

    def acquisition_for_arm(self, arm: str) -> AcquisitionSemantics:
        if arm == "baseline":
            return self.baseline_acquisition
        if arm == "candidate":
            return self.candidate_acquisition
        raise ValueError("pilot arm must be baseline or candidate")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_optional_pilot_contract(experiment: Mapping[str, Any]) -> PilotInstanceContract | None:
    raw = experiment.get("pilot_contract")
    if raw is None:
        if experiment.get("protocol_id") == PILOT_PROTOCOL_ID:
            raise ValueError("comparative external pilot experiment requires pilot_contract")
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("pilot_contract must be an object")
    return _parse_contract(raw)


def _parse_contract(raw: Mapping[str, Any]) -> PilotInstanceContract:
    arms = _mapping(raw, "arms")
    controls = tuple(
        PositiveControl(
            control_id=_text(item, "control_id"),
            expected=_text(item, "expected"),
            description=_text(item, "description"),
        )
        for item in _mapping_list(_mapping(raw, "evaluator"), "positive_controls")
    )
    output = _mapping(raw, "study_outputs")
    raw_mapping = _mapping(output, "decision_mapping")
    decision_mapping = tuple(sorted((str(key), str(value)) for key, value in raw_mapping.items()))
    workload = _mapping(raw, "workload")
    return PilotInstanceContract(
        version=_text(raw, "version"),
        protocol_id=_text(raw, "protocol_id"),
        external_owner=ExternalOwnerRecord(
            organization_id_or_pseudonym=_text(
                _mapping(raw, "external_owner"), "organization_id_or_pseudonym"
            ),
            accountable_role=_text(_mapping(raw, "external_owner"), "accountable_role"),
            owns_or_approves_decision=_bool(
                _mapping(raw, "external_owner"), "owns_or_approves_decision"
            ),
            consent_confirmed=_bool(_mapping(raw, "external_owner"), "consent_confirmed"),
            requirements_approved_before_execution=_bool(
                _mapping(raw, "external_owner"), "requirements_approved_before_execution"
            ),
        ),
        workload=WorkloadFreezeRecord(
            source=_text(workload, "source"),
            selection_procedure=_text(workload, "selection_procedure"),
            version=_text(workload, "version"),
            item_count=_int(workload, "item_count"),
            sha256=_text(workload, "sha256"),
            retention=ArtifactRetention(_text(workload, "retention")),
            critical_segments=_text_tuple(workload, "critical_segments"),
        ),
        baseline_acquisition=_parse_acquisition(_mapping(arms, "baseline")),
        candidate_acquisition=_parse_acquisition(_mapping(arms, "candidate")),
        economic=_parse_economic(_mapping(raw, "economic")),
        attempt_visibility=AttemptVisibilityContract(
            provider_invocations=_text(_mapping(raw, "attempt_visibility"), "provider_invocations"),
            ancillary_billable_calls=_text(
                _mapping(raw, "attempt_visibility"), "ancillary_billable_calls"
            ),
            health_checks=_text(_mapping(raw, "attempt_visibility"), "health_checks"),
        ),
        evaluator=EvaluatorRecord(
            identity=_text(_mapping(raw, "evaluator"), "identity"),
            version=_text(_mapping(raw, "evaluator"), "version"),
            measures=_text(_mapping(raw, "evaluator"), "measures"),
            population=_text(_mapping(raw, "evaluator"), "population"),
            positive_controls=controls,
            calibration_status=_text(_mapping(raw, "evaluator"), "calibration_status"),
            missing_outcome_policy=_text(_mapping(raw, "evaluator"), "missing_outcome_policy"),
            independent_adjudication=_text(
                _mapping(raw, "evaluator"), "independent_adjudication"
            ),
        ),
        pairing=PairingPolicy(
            key=_text(_mapping(raw, "pairing"), "key"),
            missing_outcome_policy=_text(_mapping(raw, "pairing"), "missing_outcome_policy"),
            duplicate_policy=_text(_mapping(raw, "pairing"), "duplicate_policy"),
        ),
        comparator=ComparatorParityPlan(
            workflow_id=_text(_mapping(raw, "comparator"), "workflow_id"),
            parity_requirement=_text(_mapping(raw, "comparator"), "parity_requirement"),
            evidence_surfaces=_text_tuple(_mapping(raw, "comparator"), "evidence_surfaces"),
        ),
        study_outputs=StudyOutputContract(
            decision_mapping=decision_mapping,
            endpoints=_text_tuple(output, "endpoints"),
            effort_time_unit=_text(output, "effort_time_unit"),
            delay_start_event=_text(output, "delay_start_event"),
            delay_end_event=_text(output, "delay_end_event"),
        ),
        ablations=_text_tuple(raw, "ablations"),
        artifacts=ArtifactRetentionPolicy(
            workload=ArtifactRetention(_text(_mapping(raw, "artifacts"), "workload")),
            raw_provider_evidence=ArtifactRetention(
                _text(_mapping(raw, "artifacts"), "raw_provider_evidence")
            ),
            sanitized_provider_evidence=ArtifactRetention(
                _text(_mapping(raw, "artifacts"), "sanitized_provider_evidence")
            ),
            private_raw_location=_optional_text(
                _mapping(raw, "artifacts").get("private_raw_location")
            ),
            manifest_version=_text(_mapping(raw, "artifacts"), "manifest_version"),
        ),
    )


def _parse_acquisition(raw: Mapping[str, Any]) -> AcquisitionSemantics:
    return AcquisitionSemantics(
        measurement=MeasurementSemantics(_text(raw, "measurement")),
        collection=CollectionSemantics(_text(raw, "collection")),
        assignment=AssignmentSemantics(_text(raw, "assignment")),
    )


def _parse_economic(raw: Mapping[str, Any]) -> ExactMoneySemantics:
    return ExactMoneySemantics(
        accounting_basis=_text(raw, "accounting_basis"),
        currency=_text(raw, "currency"),
        representation=_text(raw, "representation"),
        rate_semantics=_text(raw, "rate_semantics"),
        rounding=_text(raw, "rounding"),
        aggregation=_text(raw, "aggregation"),
        equality=_text(raw, "equality"),
        legacy_float_role=_text(raw, "legacy_float_role"),
    )


def _mapping(raw: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = raw.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"pilot_contract.{field} must be an object")
    return value


def _mapping_list(raw: Mapping[str, Any], field: str) -> list[Mapping[str, Any]]:
    value = raw.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"pilot_contract.{field} must be a non-empty array")
    result: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"pilot_contract.{field} entries must be objects")
        result.append(item)
    return result


def _text(raw: Mapping[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"pilot_contract.{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("pilot_contract optional text values must be non-empty strings when present")
    return value.strip()


def _bool(raw: Mapping[str, Any], field: str) -> bool:
    value = raw.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"pilot_contract.{field} must be boolean")
    return value


def _int(raw: Mapping[str, Any], field: str) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"pilot_contract.{field} must be an integer")
    return value


def _text_tuple(raw: Mapping[str, Any], field: str) -> tuple[str, ...]:
    value = raw.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"pilot_contract.{field} must be a non-empty array")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"pilot_contract.{field} entries must be non-empty strings")
        result.append(item.strip())
    return tuple(result)
