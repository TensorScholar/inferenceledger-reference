from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum


class StudyDecision(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    ABSTAIN = "ABSTAIN"


class ReferenceOutcome(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED_HARM = "UNSUPPORTED_HARM"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNRESOLVED = "UNRESOLVED"


class StudyWorkflow(StrEnum):
    FULL_INFERENCELEDGER = "FULL_INFERENCELEDGER"
    SEMANTICS_ABLATION = "SEMANTICS_ABLATION"
    STRONG_CONVENTIONAL = "STRONG_CONVENTIONAL"


@dataclass(frozen=True)
class WorkflowDecisionRecord:
    decision_unit_id: str
    workflow: StudyWorkflow
    decision: StudyDecision
    reference_outcome: ReferenceOutcome
    evidence_parity_satisfied: bool
    evidence_parity_note: str
    integration_effort_minutes: float
    evidence_collection_effort_minutes: float
    review_effort_minutes: float
    decision_delay_minutes: float
    manual_semantic_reconciliations: int
    missing_or_incomparable_surfaces_found: int

    def __post_init__(self) -> None:
        if not self.decision_unit_id.strip():
            raise ValueError("study decision unit id must be non-empty")
        if not self.evidence_parity_note.strip():
            raise ValueError("study evidence parity note must be non-empty")
        numeric = (
            self.integration_effort_minutes,
            self.evidence_collection_effort_minutes,
            self.review_effort_minutes,
            self.decision_delay_minutes,
        )
        if any(value < 0 for value in numeric):
            raise ValueError("study effort and delay values must be non-negative")
        if self.manual_semantic_reconciliations < 0:
            raise ValueError("manual semantic reconciliations must be non-negative")
        if self.missing_or_incomparable_surfaces_found < 0:
            raise ValueError("missing/incomparable surface count must be non-negative")


@dataclass(frozen=True)
class WorkflowOutcomeSummary:
    workflow: StudyWorkflow
    total_units: int
    adjudicable_units: int
    harmful_approvals: int
    unsupported_approvals: int
    correct_approvals: int
    unnecessary_rejections: int
    abstentions: int
    non_abstaining_units: int
    decision_coverage: float
    parity_eligible_units: int
    integration_effort_minutes: float
    evidence_collection_effort_minutes: float
    review_effort_minutes: float
    decision_delay_minutes: float
    manual_semantic_reconciliations: int
    missing_or_incomparable_surfaces_found: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_workflow_outcomes(
    records: Iterable[WorkflowDecisionRecord],
    *,
    workflow: StudyWorkflow,
) -> WorkflowOutcomeSummary:
    selected = [record for record in records if record.workflow == workflow]
    adjudicable = [
        record
        for record in selected
        if record.reference_outcome
        in {
            ReferenceOutcome.SUPPORTED,
            ReferenceOutcome.UNSUPPORTED_HARM,
            ReferenceOutcome.INSUFFICIENT_EVIDENCE,
        }
    ]
    harmful = sum(
        record.decision == StudyDecision.APPROVE
        and record.reference_outcome == ReferenceOutcome.UNSUPPORTED_HARM
        for record in selected
    )
    unsupported = sum(
        record.decision == StudyDecision.APPROVE
        and record.reference_outcome == ReferenceOutcome.INSUFFICIENT_EVIDENCE
        for record in selected
    )
    correct = sum(
        record.decision == StudyDecision.APPROVE
        and record.reference_outcome == ReferenceOutcome.SUPPORTED
        for record in selected
    )
    unnecessary_rejections = sum(
        record.decision == StudyDecision.REJECT
        and record.reference_outcome == ReferenceOutcome.SUPPORTED
        for record in selected
    )
    abstentions = sum(record.decision == StudyDecision.ABSTAIN for record in selected)
    non_abstaining = sum(record.decision != StudyDecision.ABSTAIN for record in adjudicable)
    coverage = non_abstaining / len(adjudicable) if adjudicable else 0.0
    parity_eligible = sum(record.evidence_parity_satisfied for record in selected)
    return WorkflowOutcomeSummary(
        workflow=workflow,
        total_units=len(selected),
        adjudicable_units=len(adjudicable),
        harmful_approvals=harmful,
        unsupported_approvals=unsupported,
        correct_approvals=correct,
        unnecessary_rejections=unnecessary_rejections,
        abstentions=abstentions,
        non_abstaining_units=non_abstaining,
        decision_coverage=coverage,
        parity_eligible_units=parity_eligible,
        integration_effort_minutes=sum(record.integration_effort_minutes for record in selected),
        evidence_collection_effort_minutes=sum(
            record.evidence_collection_effort_minutes for record in selected
        ),
        review_effort_minutes=sum(record.review_effort_minutes for record in selected),
        decision_delay_minutes=sum(record.decision_delay_minutes for record in selected),
        manual_semantic_reconciliations=sum(
            record.manual_semantic_reconciliations for record in selected
        ),
        missing_or_incomparable_surfaces_found=sum(
            record.missing_or_incomparable_surfaces_found for record in selected
        ),
    )
