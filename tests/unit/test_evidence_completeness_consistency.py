from __future__ import annotations

import json
from pathlib import Path

from inference_engine.benchmarking.evidence_completeness import (
    assess_decision_evidence_completeness,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_proven_payload() -> dict[str, object]:
    path = (
        REPO_ROOT
        / "benchmarks"
        / "reports"
        / "ifstruct-json-mode-20260911"
        / "decision.json"
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_malformed_committed_clean_flag_breaks_contract() -> None:
    payload = _git_proven_payload()
    provenance = payload["experiment_provenance"]
    assert isinstance(provenance, dict)
    provenance["spec_committed_and_clean"] = "true"

    result = assess_decision_evidence_completeness(payload)

    assert result.contract_satisfied is False
    assert "experiment_provenance" in result.missing_checks


def test_git_proven_class_cannot_conflict_with_false_git_proven_flag() -> None:
    payload = _git_proven_payload()
    registration = payload["pre_registration"]
    assert isinstance(registration, dict)
    registration["git_proven_before_execution"] = False

    result = assess_decision_evidence_completeness(payload)

    assert result.contract_satisfied is False
    assert "pre_registration" in result.missing_checks


def test_true_git_proven_flag_requires_git_proven_evidence_class() -> None:
    payload = _git_proven_payload()
    registration = payload["pre_registration"]
    assert isinstance(registration, dict)
    registration["evidence_class"] = "INSUFFICIENT_EVIDENCE"

    result = assess_decision_evidence_completeness(payload)

    assert result.contract_satisfied is False
    assert "pre_registration" in result.missing_checks
